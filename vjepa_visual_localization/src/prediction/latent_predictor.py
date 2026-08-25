"""Online latent-dynamics adapter for V-JEPA/V-JEPA2 embeddings.

The frozen encoder remains responsible for semantic scene representations.
This module adds a causal constant-velocity dynamics head in that learned
space, producing explicit z(t+1), z(t+2), and z(t+3) rollouts.  It is small on
purpose: it runs asynchronously from control and can later be replaced by a
trained V-JEPA2 action-conditioned predictor without changing the logger or
evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LatentRollout:
    origin_index: int
    target_index: int
    horizon: int
    origin_timestamp: float
    scene: str
    origin_latent: np.ndarray
    predicted_latent: np.ndarray


@dataclass(frozen=True)
class LatentEvaluation:
    origin_index: int
    target_index: int
    horizon: int
    origin_timestamp: float
    actual_timestamp: float
    scene: str
    l1_latent_error: float
    cosine_similarity: float
    prediction_error_l2: float
    predicted_drift_l2: float
    actual_drift_l2: float
    prediction_drift_error: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "origin_index": self.origin_index,
            "target_index": self.target_index,
            "horizon": self.horizon,
            "origin_timestamp": self.origin_timestamp,
            "actual_timestamp": self.actual_timestamp,
            "scene": self.scene,
            "l1_latent_error": self.l1_latent_error,
            "cosine_similarity": self.cosine_similarity,
            "prediction_error_l2": self.prediction_error_l2,
            "predicted_drift_l2": self.predicted_drift_l2,
            "actual_drift_l2": self.actual_drift_l2,
            "prediction_drift_error": self.prediction_drift_error,
        }


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else vector.copy()


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


class LatentRolloutPredictor:
    """Predict and score one-, two-, and three-step future embeddings."""

    def __init__(
        self,
        *,
        horizons: tuple[int, ...] = (1, 2, 3),
        velocity_alpha: float = 0.65,
        normalize_rollouts: bool = True,
    ) -> None:
        if not horizons or any(int(value) <= 0 for value in horizons):
            raise ValueError("horizons must contain positive integers")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0,1]")
        self.horizons = tuple(sorted(set(int(value) for value in horizons)))
        self.velocity_alpha = float(velocity_alpha)
        self.normalize_rollouts = bool(normalize_rollouts)
        self.index = 0
        self.previous_latent: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.pending: dict[int, list[LatentRollout]] = {}

    def observe(
        self,
        latent: np.ndarray,
        *,
        timestamp: float,
        scene: str,
    ) -> tuple[list[LatentRollout], list[LatentEvaluation]]:
        current = np.asarray(latent, dtype=np.float32).reshape(-1)
        if current.size == 0 or not np.isfinite(current).all():
            raise ValueError("latent must be a non-empty finite vector")
        if self.previous_latent is not None and current.shape != self.previous_latent.shape:
            raise ValueError(
                f"latent dimension changed from {self.previous_latent.size} to {current.size}"
            )

        evaluations = [
            self._evaluate(rollout, current, float(timestamp))
            for rollout in self.pending.pop(self.index, [])
        ]

        if self.previous_latent is None:
            self.velocity = np.zeros_like(current)
        else:
            instantaneous = current - self.previous_latent
            assert self.velocity is not None
            self.velocity = (
                self.velocity_alpha * instantaneous
                + (1.0 - self.velocity_alpha) * self.velocity
            )

        assert self.velocity is not None
        predictions = []
        for horizon in self.horizons:
            predicted = current + float(horizon) * self.velocity
            if self.normalize_rollouts:
                predicted = _normalize(predicted)
            rollout = LatentRollout(
                origin_index=self.index,
                target_index=self.index + horizon,
                horizon=horizon,
                origin_timestamp=float(timestamp),
                scene=str(scene),
                origin_latent=current.copy(),
                predicted_latent=predicted.astype(np.float32, copy=False),
            )
            self.pending.setdefault(rollout.target_index, []).append(rollout)
            predictions.append(rollout)

        self.previous_latent = current.copy()
        self.index += 1
        return predictions, evaluations

    @staticmethod
    def _evaluate(
        rollout: LatentRollout,
        actual: np.ndarray,
        actual_timestamp: float,
    ) -> LatentEvaluation:
        residual = rollout.predicted_latent - actual
        predicted_drift = float(
            np.linalg.norm(rollout.predicted_latent - rollout.origin_latent)
        )
        actual_drift = float(np.linalg.norm(actual - rollout.origin_latent))
        return LatentEvaluation(
            origin_index=rollout.origin_index,
            target_index=rollout.target_index,
            horizon=rollout.horizon,
            origin_timestamp=rollout.origin_timestamp,
            actual_timestamp=float(actual_timestamp),
            scene=rollout.scene,
            l1_latent_error=float(np.mean(np.abs(residual))),
            cosine_similarity=_cosine(rollout.predicted_latent, actual),
            prediction_error_l2=float(np.linalg.norm(residual)),
            predicted_drift_l2=predicted_drift,
            actual_drift_l2=actual_drift,
            prediction_drift_error=abs(predicted_drift - actual_drift),
        )


def summarize_evaluations(
    evaluations: Iterable[LatentEvaluation],
) -> dict[str, object]:
    rows = list(evaluations)
    result: dict[str, object] = {"count": len(rows), "by_horizon": {}}
    for horizon in sorted({row.horizon for row in rows}):
        group = [row for row in rows if row.horizon == horizon]
        result["by_horizon"][str(horizon)] = {
            "count": len(group),
            "mean_l1_latent_error": float(
                np.mean([row.l1_latent_error for row in group])
            ),
            "mean_cosine_similarity": float(
                np.mean([row.cosine_similarity for row in group])
            ),
            "mean_prediction_drift_error": float(
                np.mean([row.prediction_drift_error for row in group])
            ),
        }
    scenes = {}
    for scene in sorted({row.scene for row in rows}):
        group = [row for row in rows if row.scene == scene]
        scenes[scene] = {
            "count": len(group),
            "mean_l1_latent_error": float(
                np.mean([row.l1_latent_error for row in group])
            ),
            "mean_cosine_similarity": float(
                np.mean([row.cosine_similarity for row in group])
            ),
            "mean_prediction_drift_error": float(
                np.mean([row.prediction_drift_error for row in group])
            ),
        }
    result["by_scene"] = scenes
    return result


def behavior_case(
    *,
    decision: str,
    scenario: str,
    previous_decision: str | None,
    path_occupied: bool = False,
) -> str | None:
    """Map runtime decisions to the three requested behavior demos."""
    if scenario == "human_1_static_until_close" and decision == "PASS" and previous_decision == "WAIT":
        return "human_leaves_path"
    if (
        scenario == "human_2_continuous_crossing"
        and (decision == "WAIT" or path_occupied)
    ):
        return "human_continues_crossing"
    if scenario == "human_2_continuous_crossing" and decision == "PASS":
        return "vehicle_can_safely_pass"
    return None
