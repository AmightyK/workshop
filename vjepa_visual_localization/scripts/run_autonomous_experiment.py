#!/usr/bin/env python3
"""Orchestrate autonomous mapping, latent extraction and V-JEPA evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
WAREHOUSE_ROOT = WORKSPACE_ROOT / "warehouse_agv_demo"


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    stream: IO[str] | None = None

    def stop(self, timeout_sec: float = 8.0) -> None:
        if self.process.poll() is not None:
            if self.stream is not None:
                self.stream.close()
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3.0)
        finally:
            if self.stream is not None:
                self.stream.close()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mapping_run = _resolve(args.mapping_run)
        self.query_run = _resolve(args.query_run)
        self.map_dir = _resolve(args.map)
        self.results_dir = _resolve(args.results)
        self.config = _resolve(args.config)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.results_dir / "process_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.processes: list[ManagedProcess] = []
        self.uv = shutil.which("uv")
        if self.uv is None:
            raise RuntimeError("uv is required; install it before running this experiment")
        self.environment = os.environ.copy()
        self.environment["GZ_PARTITION"] = args.gz_partition
        self.environment["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
        # This orchestrator owns bridge/localizer/dashboard lifecycles itself.
        # Prevent the newly integrated run_demo entry point from duplicating
        # those processes during mapping and query phases.
        self.environment["WAREHOUSE_AUTOSTART_BRIDGE"] = "false"
        self.environment["WAREHOUSE_AUTOSTART_NAV2"] = "false"
        self.environment["WAREHOUSE_VJEPA_ENABLED"] = "false"

    def _uv_command(self, script: str, *arguments: str) -> list[str]:
        return [
            self.uv,
            "run",
            "--project",
            str(PROJECT_ROOT),
            "python",
            str(PROJECT_ROOT / "scripts" / script),
            *arguments,
        ]

    def start(
        self,
        name: str,
        command: list[str],
        *,
        log_to_file: bool = False,
    ) -> ManagedProcess:
        print(f"[PROCESS] start {name}: {' '.join(command)}", flush=True)
        stream: IO[str] | None = None
        output: int | IO[str] | None = None
        if log_to_file:
            stream = (self.logs_dir / f"{name}.log").open("w", encoding="utf-8")
            output = stream
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=self.environment,
            stdout=output,
            stderr=subprocess.STDOUT if log_to_file else None,
            text=True,
            start_new_session=True,
        )
        managed = ManagedProcess(name, process, stream)
        self.processes.append(managed)
        return managed

    def stop(self, process: ManagedProcess) -> None:
        print(f"[PROCESS] stop {process.name}", flush=True)
        process.stop()
        if process in self.processes:
            self.processes.remove(process)

    def run_command(self, name: str, command: list[str], *, allow_failure: bool = False) -> int:
        print(f"[PROCESS] run {name}: {' '.join(command)}", flush=True)
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=self.environment,
            check=False,
        )
        if result.returncode != 0 and not allow_failure:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")
        if result.returncode != 0:
            print(f"[WARN] {name} exited with {result.returncode}; continuing", flush=True)
        return result.returncode

    def launch_stack(self) -> None:
        demo_command = [str(WAREHOUSE_ROOT / "run_demo.sh")]
        if self.args.headless:
            demo_command.append("-s")
        self.start("gazebo", demo_command, log_to_file=True)
        time.sleep(3.0)
        self.start(
            "bridge", [str(WAREHOUSE_ROOT / "run_bridge.sh")], log_to_file=True
        )
        time.sleep(2.0)
        self.start(
            "nav2",
            [
                str(WAREHOUSE_ROOT / "run_nav2.sh"),
                "use_rviz:=False",
                "localization_source:=ground_truth",
            ],
            log_to_file=True,
        )
        print(
            f"[STACK] GZ_PARTITION={self.args.gz_partition}, "
            f"ROS_DOMAIN_ID={self.args.ros_domain_id}; warming up...",
            flush=True,
        )
        time.sleep(self.args.stack_warmup_sec)
        deadline = time.monotonic() + self.args.stack_ready_timeout_sec
        last_report = 0.0
        while time.monotonic() < deadline:
            for process in self.processes:
                if process.process.poll() is not None:
                    raise RuntimeError(
                        f"{process.name} stopped during startup; see "
                        f"{self.logs_dir / (process.name + '.log')}"
                    )
            state = subprocess.run(
                ["ros2", "lifecycle", "get", "/bt_navigator"],
                cwd=PROJECT_ROOT,
                env=self.environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if state.returncode == 0 and state.stdout.strip().lower().startswith("active"):
                print("[STACK] Nav2 lifecycle active; starting patrol", flush=True)
                return
            now = time.monotonic()
            if now - last_report >= 2.0:
                print("[STACK] waiting for bt_navigator lifecycle=active...", flush=True)
                last_report = now
            time.sleep(0.5)
        raise RuntimeError(
            f"Nav2 did not become active within {self.args.stack_ready_timeout_sec:.1f}s; "
            f"see {self.logs_dir / 'nav2.log'}"
        )

    def record_and_patrol(self, phase: str, run_dir: Path) -> int:
        recorder_arguments = [
            "--output",
            str(run_dir),
            "--fps",
            str(self.args.fps),
        ]
        if self.args.overwrite:
            recorder_arguments.append("--overwrite")
        if phase == "query":
            recorder_arguments.append("--keep-people")
        recorder = self.start(
            f"{phase}_recorder",
            self._uv_command("record_mapping_run.py", *recorder_arguments),
            log_to_file=True,
        )
        time.sleep(2.0)
        patrol_status = 1
        try:
            patrol_status = self.run_command(
                f"{phase}_patrol",
                self._uv_command(
                    "autonomous_patrol.py",
                    "--config",
                    str(self.config),
                    "--phase",
                    phase,
                    "--output",
                    str(self.results_dir),
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                ),
                allow_failure=True,
            )
        finally:
            self.stop(recorder)
        self.run_command(
            f"{phase}_camera_inspection",
            self._uv_command(
                "inspect_camera_pipeline.py",
                "--run",
                str(run_dir),
                "--expected-aspect",
                "16:9",
            ),
        )
        return patrol_status

    def run(self) -> None:
        if self.args.launch_stack:
            self.launch_stack()

        if self.args.skip_mapping:
            mapping_status: int | str = "reused"
            required_map_files = [
                self.map_dir / "global_embeddings.npy",
                self.map_dir / "metadata.json",
            ]
            missing = [str(path) for path in required_map_files if not path.exists()]
            if missing:
                raise RuntimeError(
                    "--skip-mapping requested but visual map files are missing: "
                    + ", ".join(missing)
                )
            print(
                "\n=== PHASES 1-2: REUSE EXISTING STATIC V-JEPA MAP ===",
                flush=True,
            )
            print(f"[MAP] reuse {self.map_dir}", flush=True)
        else:
            print("\n=== PHASE 1: AUTONOMOUS MAPPING ===", flush=True)
            mapping_status = self.record_and_patrol("mapping", self.mapping_run)
            if not (self.mapping_run / "video.mp4").exists():
                raise RuntimeError("mapping recorder did not produce video.mp4")

            print("\n=== PHASE 2: EXTRACT V-JEPA LATENTS ===", flush=True)
            self.run_command(
                "build_visual_map",
                self._uv_command(
                    "build_visual_map.py",
                    "--config",
                    str(self.config),
                    "--run",
                    str(self.mapping_run),
                    "--output",
                    str(self.map_dir),
                ),
            )

            if self.args.mapping_only:
                final = {
                    "mode": "mapping_only",
                    "mapping_patrol_exit_code": mapping_status,
                    "mapping_run": str(self.mapping_run),
                    "visual_map": str(self.map_dir),
                    "ros_domain_id": self.args.ros_domain_id,
                    "gz_partition": self.args.gz_partition,
                }
                (self.results_dir / "mapping_only_summary.json").write_text(
                    json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                print("\n=== DENSE LATENT MAP COMPLETE ===", flush=True)
                print(
                    json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
                return

        print("\n=== PHASE 3: QUERY PATROL + LIVE COMPARISON ===", flush=True)
        localizer = self.start(
            "live_localizer",
            self._uv_command(
                "run_live_localization.py",
                "--config",
                str(self.config),
                "--map",
                str(self.map_dir),
                "--ros-args",
                "-p",
                "use_sim_time:=true",
            ),
            log_to_file=True,
        )
        time.sleep(3.0)
        dashboard: ManagedProcess | None = None
        if not self.args.headless and not self.args.no_dashboard:
            dashboard = self.start(
                "localization_dashboard",
                self._uv_command(
                    "localization_dashboard.py",
                    "--inventory",
                    str(WAREHOUSE_ROOT / "config" / "inventory_locations.yaml"),
                    "--map-yaml",
                    str(WAREHOUSE_ROOT / "maps" / "warehouse_lidar.yaml"),
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                ),
                log_to_file=True,
            )
            time.sleep(1.0)
            if dashboard.process.poll() is not None:
                raise RuntimeError(
                    "localization dashboard stopped during startup; see "
                    f"{self.logs_dir / 'localization_dashboard.log'}"
                )
        try:
            query_status = self.record_and_patrol("query", self.query_run)
        finally:
            if dashboard is not None:
                self.stop(dashboard)
            self.stop(localizer)

        print("\n=== PHASE 4: OFFLINE SYNCHRONIZED EVALUATION ===", flush=True)
        predictions = self.results_dir / "offline_predictions.jsonl"
        metrics = self.results_dir / "offline_metrics.json"
        self.run_command(
            "offline_evaluation",
            self._uv_command(
                "run_global_baseline.py",
                "--config",
                str(self.config),
                "--query-run",
                str(self.query_run),
                "--map",
                str(self.map_dir),
                "--predictions",
                str(predictions),
                "--metrics",
                str(metrics),
            ),
        )
        final = {
            "mapping_patrol_exit_code": mapping_status,
            "query_patrol_exit_code": query_status,
            "mapping_run": str(self.mapping_run),
            "query_run": str(self.query_run),
            "visual_map": str(self.map_dir),
            "live_summary": str(self.results_dir / "query_summary.json"),
            "offline_metrics": json.loads(metrics.read_text(encoding="utf-8")),
            "ros_domain_id": self.args.ros_domain_id,
            "gz_partition": self.args.gz_partition,
            "dashboard_enabled": bool(
                not self.args.headless and not self.args.no_dashboard
            ),
        }
        (self.results_dir / "experiment_summary.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("\n=== COMPLETE ===", flush=True)
        print(json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True), flush=True)

    def cleanup(self) -> None:
        for process in reversed(self.processes.copy()):
            try:
                self.stop(process)
            except (ProcessLookupError, subprocess.SubprocessError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/warehouse_experiment.yaml")
    parser.add_argument("--mapping-run", default="data/autonomous_mapping_dense")
    parser.add_argument("--query-run", default="data/autonomous_query_dense")
    parser.add_argument("--map", default="outputs/autonomous_map_dense")
    parser.add_argument("--results", default="outputs/autonomous_experiment_dense")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-mapping",
        "--reuse-latents",
        action="store_true",
        help="reuse the existing latent map and run only query/evaluation",
    )
    parser.add_argument("--launch-stack", action="store_true")
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="record the expanded route and build its latent map, then stop before query",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="do not open the telemetry and warehouse-map windows",
    )
    parser.add_argument("--stack-warmup-sec", type=float, default=12.0)
    parser.add_argument("--stack-ready-timeout-sec", type=float, default=90.0)
    parser.add_argument("--gz-partition", default="warehouse_agv_vjepa_autonomous")
    parser.add_argument("--ros-domain-id", type=int, default=42)
    args = parser.parse_args()
    if (
        args.fps <= 0.0
        or args.stack_warmup_sec < 0.0
        or args.stack_ready_timeout_sec <= 0.0
    ):
        parser.error("fps/ready timeout must be positive and stack warmup cannot be negative")
    runner = ExperimentRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
