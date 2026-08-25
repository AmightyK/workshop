# V-JEPA visual localization

Camera-only visual localization following
`../vjepa_visual_localization_codex_spec.json`. The implementation contains the
Stage 0–4 global baseline plus an online temporal tracker:

1. synchronize uniformly sampled video clips with center-timestamp poses;
2. extract normalized global and local V-JEPA representations;
3. save a global visual map using NumPy arrays;
4. retrieve candidates with brute-force cosine similarity and estimate top-1 pose;
5. track the previous visual pose through ambiguous repeated racks;
6. report position, yaw and retrieval recall metrics with per-query debug logs.

There is no OCR, semantic landmark ID, detector, FAISS index or SLAM fallback.
Ground-truth pose is consumed only while building the map and evaluating a
separate query traversal. The encoder defaults to Meta's
`facebook/vjepa2-vitl-fpc64-256` checkpoint through Hugging Face Transformers.

## Environment

```bash
uv venv --python python3.12 --system-site-packages
uv sync --group dev
```

`pyproject.toml` pins the PyTorch CUDA 12.8 index and `uv.lock` makes the
environment reproducible. `--system-site-packages` exposes ROS 2 Jazzy's
system-installed Python packages to the UV environment. The default configuration
uses CUDA float16. Change `model.device` to `cpu`
and `model.dtype` to `float32` only when a CUDA GPU is unavailable. The first
real inference downloads the configured checkpoint into the Hugging Face cache.

## Data

Each mapping or evaluation traversal has this format:

```text
data/run_001/
├── video.mp4
└── poses.csv
```

`poses.csv` must contain:

```csv
timestamp,x,y,z,yaw
0.000,1.2,3.4,0.0,1.57
```

Timestamps are seconds on the video's time base. The dataset interpolates XYZ
and wrapped yaw at each clip center, but rejects clips whose nearest measured
pose exceeds `synchronization.pose_tolerance_sec`.

Inspect synchronization before spending GPU time:

```bash
uv run python scripts/inspect_dataset.py \
  --run data/run_001 --output outputs/run_001_sample.jpg
```

## Record a mapping traversal from Gazebo

Start the warehouse world and its `/camera` bridge, drive the AGV through all
areas that should be localizable, then record one mapping traversal:

```bash
GZ_PARTITION=warehouse_agv_demo ./run_record_mapping.sh \
  --output data/warehouse_map_001 --fps 10
```

This mapping-only command records `/camera` into `video.mp4` and Gazebo world
poses into `poses.csv`. Press `Ctrl+C` when the traversal is complete. Dynamic
people can be parked while constructing the persistent map; run with
`--keep-people` if they should remain active.

## Build and query the visual map

```bash
uv run python scripts/build_visual_map.py \
  --config configs/baseline.yaml \
  --run data/warehouse_map_001 \
  --output outputs/map

uv run python scripts/run_global_baseline.py \
  --config configs/baseline.yaml \
  --query-run data/run_002 \
  --map outputs/map
```

The map contains `global_embeddings.npy`, `poses.npy`, `timestamps.npy`,
`ids.npy` and `metadata.json`. Query output is written to
`outputs/baseline/predictions.jsonl`; every row preserves ground truth,
predicted pose, candidate IDs, similarities, candidate poses and errors.

Recompute metrics without running the encoder again:

```bash
uv run python scripts/evaluate_localization.py \
  --predictions outputs/baseline/predictions.jsonl
```

Reported metrics include mean/median/P95 position error, yaw error,
Recall@0.5/1/2/5 m and candidate Recall@1/5/20 within 1 m.

## Camera-only current pose

Once the visual map exists, run the online localizer while the DDS camera relay
is publishing. In the warehouse profile this node subscribes only to
`/vjepa/camera/image_raw`; it does not consume odometry, Gazebo pose or
evaluation ground truth:

```bash
./run_live_localization.sh --map outputs/map
```

For deployment on Jetson Orin connected directly to the Gazebo laptop by LAN,
use [`ORIN_DDS_LAN.md`](ORIN_DDS_LAN.md). The threaded PC relay publishes the
newest frame at 4 FPS using native ROS 2 DDS and the Orin returns `/vjepa_pose`
and `/vjepa_latent` through the same DDS domain.

`/vjepa_pose` is a map-frame visual position (GPS-like local warehouse
coordinates, not latitude/longitude). `/vjepa_latent` carries the current
1024-D query vector, while `/vjepa_localization/debug` carries the clip-center
timestamp, Orin hostname, inference latency, source ID, candidates, scores and
temporal state. The laptop dashboard combines these with its local static
latent map and timestamp-aligned Gazebo truth.

The estimated current pose is published on `/vjepa_pose`, with retrieval details
and confidence margin on `/vjepa_localization/debug`:

```bash
ros2 topic echo /vjepa_pose
ros2 topic echo /vjepa_localization/debug
```

The temporal tracker searches all saved candidates, rejects candidates that
cannot be reached from the previous accepted V-JEPA pose, and enforces forward
continuity in the recorded latent sequence. Image feature tracks provide a
camera-only motion cue: robust full-frame forward expansion accumulates fractional
route progress and must advance once a latent step is earned, while an in-place
turn or a worker moving through a small image region does not force progress.
This avoids staying pinned to one repeated-shelf latent when the AGV is visibly
moving. Because Gazebo always spawns this experiment at the
dock, the first retrieval is constrained to clips 0–3 at the start of the saved
closed route. This supplies the previous-position prior that a temporal tracker
does not yet have and prevents an ambiguous dock view from initializing at a
visually similar south-corridor clip. Every later retrieval searches the full
map. If no candidate passes the hard position/yaw gate, the output state becomes
`HOLDING` instead of teleporting to a similar rack. Its thresholds are under `temporal_tracking` in
`configs/warehouse_experiment.yaml`; `--no-temporal` is available only for
diagnostic comparisons.

The low-latency profile continuously keeps the newest 4 frames (4 FPS) from a
1-second rolling visual clip. Camera callbacks continue replacing old frames
while a serialized GPU query runs, so the next query uses the newest window
instead of processing a backlog. The pose message keeps the clip-center
timestamp, so the unavoidable raw visual delay is 0.5 seconds plus inference
time and metrics remain honest.
The dashboard draws that raw pose directly: `RAW ERR` uses timestamp-aligned
truth, while `NOW GAP` makes the expected delay visible. `/odom` is not
subscribed by default. Pass `--odom-projection` to the dashboard only when a
telemetry-only short-term projection is useful; it never changes `V-JEPA RAW`.
Gazebo truth is never used by the camera-only localizer.
baseline comparisons.

For the normal warehouse/pick workflow, the integrated demo owns this process
and the two dashboard windows:

```bash
cd ../warehouse_agv_demo
./run_demo.sh
# second terminal
./pick_box.sh --area A --color blue
```

`pick_box.sh` starts Nav2 if needed and reuses the already-running temporal
localizer. Before sending the first route goal it waits for V-JEPA GPU warm-up
and the initial dock-anchored `/vjepa_pose`; it does not rebuild the saved latent
map.

For a query directory containing only `video.mp4` and no `poses.csv`, use:

```bash
uv run python scripts/localize_video.py \
  --query-run data/query_only --map outputs/map
```

## Autonomous warehouse experiment

Vietnamese step-by-step instructions are available in
[`HUONG_DAN_CHAY_VJEPA_AGV.md`](HUONG_DAN_CHAY_VJEPA_AGV.md).

Run the full reproducible experiment with an isolated headless Gazebo, bridge
and Nav2 stack:

```bash
./run_autonomous_experiment.sh \
  --launch-stack --headless --overwrite
```

Omit `--headless` to open Gazebo plus two live comparison windows during the
query phase:

```bash
./run_autonomous_experiment.sh \
  --launch-stack --overwrite --reuse-latents
```

- **VL-JEPA Warehouse Streaming QA** follows the reference-video composition:
  live 16:9 camera at left and a PCA projection of the real saved/live
  1024-dimensional V-JEPA embeddings at right. It rotates eight prepared
  warehouse questions in the compact header; the former lower telemetry cards
  are hidden. With this window focused, hold `W/S` to move and add `A/D` at the
  same time to steer; `A/D` alone rotates in place. Release the keys to stop,
  use `Space` for an explicit stop, and close with `Esc`.
  The 640x360 camera, DDS relay, and dashboard run at 32 FPS by default; the
  V-JEPA encoder still uniformly samples four frames from each rolling second.
- **Warehouse Map - Truth GPS & Planning** draws only the LiDAR occupancy map,
  the Truth GPS trail/heading and the active planning path. V-JEPA remains in
  the streaming QA/latent views and is not drawn on this 2D navigation map.

The question list is editable in `configs/warehouse_live_questions.yaml`. Its
answers are prepared deterministic templates populated from V-JEPA pose,
temporal state, A*, LiDAR and the separate Gazebo evaluator. They are a
VL-JEPA-style streaming presentation, not an assertion that the V-JEPA 2
checkpoint includes a language decoder. The exact encoder vector used for
retrieval is also published on `/vjepa_latent` as a 1024-value
`Float32MultiArray`.

The dashboard is a separate evaluator: Gazebo truth is never subscribed to by
`run_live_localization.py` and cannot enter V-JEPA inference. To attach only the
two windows to an already-running localizer and warehouse stack, run:

```bash
./run_localization_dashboard.sh
```

Press `Q` or `Esc` to close both windows. Use `--no-dashboard` for a graphical
experiment without these windows, or `--headless` for server/CI execution.

To keep the existing latent map and rerun only query/evaluation (for example,
after changing Nav2 or the evaluator), add `--skip-mapping`:

```bash
./run_autonomous_experiment.sh \
  --launch-stack --headless --overwrite --skip-mapping
```

The closed route and thresholds are configured in
`configs/warehouse_experiment.yaml`. The command performs four phases:

1. Nav2 drives the mapping loop while the recorder parks random workers.
2. V-JEPA extracts and saves normalized 1024-dimensional map latents.
3. Nav2 repeats the loop with moving workers enabled while the camera-only
   localizer runs; the two optional dashboard windows compare its output with
   synchronized truth.
4. Live and offline predictions are compared against synchronized Gazebo pose
   only for clips where the AGV actually translates; stationary samples are
   reported and excluded from metrics.

During the route, the patrol prints Vietnamese events such as:

```text
[STATUS] khu đóng gói | pose Gazebo=(10.65,-9.64) | vật cản=có: thùng tĩnh 4
[AVOID] đang rẽ phải để né thùng tĩnh 1 (LiDAR 1.89 m)
[LOCALIZE] khu kệ T | Gazebo=(2.89,-13.62) V-JEPA=(3.02,-13.80) | lỗi=0.22 m
```

LiDAR decides whether an obstacle is present. Gazebo model names are used only
by the evaluator to annotate whether it is a static box or a moving worker;
they are never passed to V-JEPA. Important artifacts are:

```text
data/autonomous_mapping_dense/video.mp4 + poses.csv
data/autonomous_query_dense/video.mp4 + poses.csv
outputs/autonomous_map_dense/global_embeddings.npy
outputs/autonomous_experiment_dense/query_comparison.csv
outputs/autonomous_experiment_dense/query_events.jsonl
outputs/autonomous_experiment_dense/query_summary.json
outputs/autonomous_experiment_dense/offline_metrics.json
outputs/autonomous_experiment_dense/offline_predictions.jsonl
```

The full-stack command defaults to `ROS_DOMAIN_ID=42` and
`GZ_PARTITION=warehouse_agv_vjepa_autonomous`, keeping it isolated from an
already-open warehouse session. Omit `--launch-stack` to reuse a stack that is
already running in the current environment.

## Tests

```bash
uv run pytest -q
```

The tests use an injected deterministic encoder solely to verify data,
synchronization, storage, retrieval, pose and metric behavior without downloading
weights. Production CLIs always construct the configured V-JEPA encoder.

## Current boundary

The production online path now includes global retrieval, hard temporal gates,
latent-sequence continuity, camera-only motion estimation and smoothing. It
still intentionally omits local-token re-ranking, learned loop closure, FAISS
and SLAM fallback; Gazebo truth remains outside inference and is used only for
map construction and evaluation.
