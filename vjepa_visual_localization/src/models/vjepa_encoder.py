"""Configurable Hugging Face V-JEPA 2 video encoder wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pooling import l2_normalize, mean_pool_tokens


@dataclass(frozen=True)
class EncoderOutput:
    """Normalized global embedding and optional normalized local tokens."""

    global_embedding: np.ndarray
    local_tokens: np.ndarray | None
    token_shape: tuple[int, ...] | None = None


class VJEPAEncoder:
    """Expose V-JEPA through the stable ``encode_video`` API from the spec.

    The primary backend is Meta's V-JEPA 2 checkpoint as implemented by
    Hugging Face Transformers. Imports and weight loading are lazy so dataset,
    map database and metric utilities remain usable without a GPU process.
    """

    def __init__(
        self,
        *,
        checkpoint: str = "facebook/vjepa2-vitl-fpc64-256",
        device: str = "cuda",
        dtype: str = "float16",
        return_local_tokens: bool = True,
        normalize_embeddings: bool = True,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device_name = device
        self.dtype_name = dtype
        self.return_local_tokens = return_local_tokens
        self.normalize_embeddings = normalize_embeddings
        self._model = model
        self._processor = processor
        self._torch: Any | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "architecture": "V-JEPA 2",
            "backend": "huggingface_transformers",
            "checkpoint": self.checkpoint,
            "device": self.device_name,
            "dtype": self.dtype_name,
            "return_local_tokens": self.return_local_tokens,
            "normalize_embeddings": self.normalize_embeddings,
        }

    def _load(self) -> None:
        if self._torch is None:
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(
                    "PyTorch is required for V-JEPA inference; install requirements.txt"
                ) from error
            self._torch = torch
        torch = self._torch
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if self._model is None or self._processor is None:
            try:
                from transformers import AutoModel, AutoVideoProcessor
            except ImportError as error:
                raise RuntimeError(
                    "Transformers with V-JEPA 2 support is required; install requirements.txt"
                ) from error
            self._processor = AutoVideoProcessor.from_pretrained(self.checkpoint)
            self._model = AutoModel.from_pretrained(self.checkpoint)

        dtype = getattr(torch, self.dtype_name, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {self.dtype_name}")
        if self.device_name == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        self._device = torch.device(self.device_name)
        self._dtype = dtype
        self._model.to(device=self._device, dtype=dtype)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _validate_video(video: np.ndarray) -> np.ndarray:
        array = np.asarray(video)
        if array.ndim == 4:
            array = array[None, ...]
        if array.ndim != 5:
            raise ValueError(
                "video must have shape [T,H,W,C], [B,T,H,W,C], or [B,T,C,H,W]"
            )
        if array.shape[-1] == 3:
            return array
        if array.shape[2] == 3:
            return np.transpose(array, (0, 1, 3, 4, 2))
        raise ValueError(f"could not identify RGB channel in video shape {array.shape}")

    def _prepare_batch(self, videos: np.ndarray) -> Any:
        torch = self._torch
        encoded = []
        for clip in videos:
            frames = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
            prepared = self._processor(frames, return_tensors="pt")
            pixel_values = prepared["pixel_values_videos"]
            encoded.append(pixel_values)
        return torch.cat(encoded, dim=0).to(device=self._device, dtype=self._dtype)

    def _extract_tokens(self, pixel_values: Any) -> Any:
        model = self._model
        if hasattr(model, "get_vision_features"):
            output = model.get_vision_features(pixel_values_videos=pixel_values)
        else:
            output = model(pixel_values_videos=pixel_values)
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim < 3:
            raise RuntimeError(f"V-JEPA returned unexpected tensor shape {tuple(output.shape)}")
        batch, feature_dim = output.shape[0], output.shape[-1]
        return output.reshape(batch, -1, feature_dim), tuple(int(v) for v in output.shape[1:-1])

    def encode_video(self, video: np.ndarray) -> EncoderOutput:
        """Encode one clip or a batch into global and local representations."""

        self._load()
        torch = self._torch
        videos = self._validate_video(video)
        pixel_values = self._prepare_batch(videos)
        with torch.inference_mode():
            local_tensor, token_shape = self._extract_tokens(pixel_values)
        local = local_tensor.detach().float().cpu().numpy()
        global_embedding = mean_pool_tokens(local)
        if self.normalize_embeddings:
            global_embedding = l2_normalize(global_embedding)
            local = l2_normalize(local)
        if not np.isfinite(global_embedding).all() or not np.isfinite(local).all():
            raise RuntimeError("V-JEPA produced NaN or Inf embeddings")
        return EncoderOutput(
            global_embedding=global_embedding,
            local_tokens=local if self.return_local_tokens else None,
            token_shape=token_shape if self.return_local_tokens else None,
        )
