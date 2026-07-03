# YOLO Edge-Runtime Profiler

**Catch YOLO tail latency, stage imbalance, and postprocess spikes from `Results.speed`.**

`YOLO Edge-Runtime Profiler` is a small, zero-intrusion runtime profiler for Ultralytics YOLO deployments. It is designed for edge and production users who already know that average FPS is not enough.

It answers the question:

```text
My average FPS looks fine. Why does the system still occasionally stall?
```

The profiler consumes Ultralytics `Results` objects and tracks:

```text
preprocess_ms
inference_ms
postprocess_ms
total_ms
p50 / p95 / p99 / max latency
tail-latency coefficient
postprocess spike residual
stage-imbalance ratio
box count
confidence entropy
class-distribution entropy
```

It does **not** modify YOLO, retrain models, hook neural-network layers, or require ROS 2.

## Why This Exists

YOLO benchmarking usually reports average speed or average FPS. That is useful, but it can hide tail latency.

In real deployments, the failure mode is often not:

```text
YOLO is always slow.
```

It is:

```text
YOLO is usually fast, but occasionally one stage spikes.
```

This profiler separates the runtime into:

```text
preprocess  -> image decode / resize / copy path
inference   -> model forward path
postprocess -> boxes / filtering / NMS / end-to-end output handling
```

Then it reports whether the runtime is:

```text
GREEN  -> stable
YELLOW -> tail latency or stage pressure rising
RED    -> severe tail latency or stage pressure
```

## Roadmap: Pressure-Triggered Hard Example Mining

YERP(YOLO Edge-Runtime Profiler) starts as a lightweight runtime profiler, but the long-term goal is larger: turn edge-runtime pressure into a data selection signal.

Most active learning pipelines select samples from model uncertainty alone. That is useful, but it misses a critical deployment question:

```text
Which frames actually made the edge system unstable?
```

YERP can detect frames associated with:

```text
tail-latency spikes
stage imbalance
postprocess pressure
high output entropy
dense detection bursts
runtime instability on edge devices
```

The next step is a local-first Pressure Data Bag:

```text
trigger:   YERP enters YELLOW or RED
capture:   current frame or short frame window
metadata:  latency, stage split, box count, entropy, dominant cause
storage:   local folder, private object store, or future platform integration
```

This creates a deployment-driven active learning loop:

```text
deploy YOLO at the edge
        ↓
detect runtime pressure
        ↓
capture high-value long-tail frames
        ↓
review / label / retrain
        ↓
redeploy a more stable model
```

YERP is local-first by design. Image capture and upload should be explicit, optional, and privacy-aware.

## Install

For local development:

```bash
git clone https://github.com/YOUR_NAME/yolo-edge-runtime-profiler.git
cd yolo-edge-runtime-profiler
pip install -e .
```

Optional dependencies for examples:

```bash
pip install ultralytics opencv-python
```

The core package only requires Python and NumPy.

## 5-Line Integration

```python
from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler

model = YOLO("yolov8n.pt")
profiler = YoloEdgeRuntimeProfiler(window_size=100)

for result in model("video.mp4", stream=True):
    status = profiler.update(result)
    print(status.to_compact_string())

profiler.export_json("runtime_report.json")
profiler.export_csv("runtime_frames.csv")
```

## CLI Usage

```bash
yolo-edge-profile --model yolov8n.pt --source video.mp4 --dashboard \
  --json runtime_report.json \
  --csv runtime_frames.csv
```

Camera example:

```bash
yolo-edge-profile --model yolov8n.pt --source 0 --dashboard
```

Synthetic demo without Ultralytics:

```bash
python examples/synthetic_spike_demo.py
```

## Runtime Metrics

### 1. Tail Latency Coefficient

```text
R_tail = rolling_p95_total_ms / rolling_p50_total_ms
```

This exposes cases where average FPS looks good but p95 / p99 latency is much worse.

### 2. Stage Imbalance

```text
postprocess_ratio = postprocess_ms / total_ms
```

This identifies whether preprocess, inference, or postprocess is dominating the frame.

### 3. Postprocess Spike Residual

```text
R_post = current_postprocess_ms / rolling_median_postprocess_ms
```

This catches sudden postprocess spikes, often associated with dense scenes or unusually high output pressure.

### 4. Output Pressure

```text
box_count
confidence entropy
class-distribution entropy
```

This helps correlate runtime spikes with scene complexity.

### 5. Low-Postprocess / End-to-End Path Detection

If the rolling postprocess time is near zero, the profiler de-emphasizes postprocess spike alarms and focuses on total tail latency and preprocess / inference stability.

Example log:

```text
[INFO] Low-postprocess path detected. Postprocess lag appears minimal; auditing total and stage tail latency instead.
```

This avoids false alarms for end-to-end or NMS-free-style pipelines.

## Recommended Thresholds

Default thresholds are conservative heuristics:

```text
YELLOW:
  p95/p50 total latency >= 2.0
  or postprocess ratio >= 0.30
  or postprocess spike >= 2.0x

RED:
  p95/p50 total latency >= 4.0
  or postprocess ratio >= 0.50
  or postprocess spike >= 4.0x
```

Tune them for your deployment target.

## Relation to OBIO(Offboard Boundary Integrity Observer)

This package is the model-runtime layer.

```text
YOLO Edge-Runtime Profiler
  -> audits YOLO runtime pressure

Resource-Aware YOLO Adapter
  -> consumes runtime or boundary pressure and throttles workload

OBIO Core
  -> audits the physical execution boundary in ROS 2 / PX4 systems
```

OBIO asks:

```text
Is the physical execution boundary still healthy?
```

This profiler asks:

```text
Which YOLO runtime stage is creating tail latency?
```

The two are complementary.

[Offboard Boundary Integrity Observer](https://github.com/ZC502/ai_flight_integrity_observer.git)

## What This Is Not

This tool does not improve mAP, retrain YOLO, replace TensorRT benchmarking, or certify safety. It is a lightweight runtime diagnostic tool for deployment debugging.

## License

Apache-2.0

