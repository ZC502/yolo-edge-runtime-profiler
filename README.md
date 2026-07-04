# YOLO Edge-Runtime Profiler

**Catch YOLO tail latency, stage imbalance, postprocess spikes, and pressure-triggered hard examples from `Results.speed`.**

`YOLO Edge-Runtime Profiler` (YERP) is a small, zero-intrusion runtime profiler for Ultralytics YOLO deployments. It is designed for edge and production users who already know that average FPS is not enough.

It answers two practical questions:

```text
My average FPS looks fine. Why does the system still occasionally stall?
Which frames should I keep for debugging, labeling, and retraining?
```

YERP consumes Ultralytics `Results` objects and tracks:

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

It can also run a **local-first hard-example recorder**: when runtime pressure or confidence uncertainty appears, YERP saves the frame plus a sidecar JSON file with the runtime residuals.

YERP does **not** modify YOLO, retrain models, hook neural-network layers, require ROS 2, or upload your data.

## Why This Exists

Traditional YOLO benchmarking usually reports average speed or average FPS. That is useful, but it can hide tail latency.

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

For low-postprocess or end-to-end models, YERP de-emphasizes postprocess-specific alarms and continues auditing total / preprocess / inference tail latency.

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

The core package only requires Python and NumPy. Image capture uses OpenCV or PIL when available.

## 5-Line Integration

```python
from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler

model = YOLO("yolov8n.pt")
profiler = YoloEdgeRuntimeProfiler(window_size=100)

for result in model("video.mp4", stream=True):
    status = profiler.update(result)
    print(status.to_compact_string())
```

## Local-First Hard Example Capture

YERP can optionally save high-value frames when runtime pressure or confidence uncertainty appears.

Instead of labeling random frames, collect the frames that actually made the edge runtime unstable（Zero I/O bottleneck: Built-in cooldowns and max-item limits ensure the profiler never crashes your edge device.）:

```text
tail-latency spikes
postprocess spikes
stage imbalance
dense detection bursts
high confidence entropy
low mean confidence
```

Example:

```python
from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler, LocalHardExampleRecorder

model = YOLO("yolov8n.pt")
profiler = YoloEdgeRuntimeProfiler(window_size=100)
recorder = LocalHardExampleRecorder(
    output_dir="hard_examples",
    selection_mode="pressure_or_confidence",
    trigger_states=("RED",),
    cooldown_sec=2.0,
    max_items=200,
    # Optional confidence dimension. Tune for your use case.
    min_confidence_entropy=None,
    max_confidence_mean=None,
)

for result in model("video.mp4", stream=True):
    status = profiler.update(result)
    frame = getattr(result, "orig_img", None)
    recorder.maybe_save(frame=frame, status=status)
```

Output:

```text
hard_examples/
├── frame_000127_1783000000123_RED_POSTPROCESS_SPIKE.jpg
└── frame_000127_1783000000123_RED_POSTPROCESS_SPIKE.json
```

The JSON sidecar stores the full runtime context:

```json
{
  "schema": "yolo-edge-runtime-profiler.hard-example.v0.1",
  "selection": {
    "reason": "runtime_pressure",
    "pressure_signal": {"state": "RED", "dominant_cause": "POSTPROCESS_SPIKE"},
    "confidence_signal": {"confidence_entropy": 2.74, "confidence_mean": 0.41}
  },
  "status": {
    "state": "RED",
    "dominant_cause": "POSTPROCESS_SPIKE",
    "stage_ms": {"preprocess": 2.1, "inference": 12.4, "postprocess": 76.8, "total": 91.3},
    "latency_ms": {"p50": 18.6, "p95": 47.2, "p99": 91.3},
    "residuals": {"tail_latency_coeff_p95_p50": 2.54, "postprocess_spike_coeff": 5.2}
  },
  "privacy_note": "Local-first capture. No upload was performed by YERP."
}
```

This is the local-first foundation for pressure-triggered hard-example mining. Nothing is uploaded unless you build or enable your own integration.

## CLI Usage

```bash
yolo-edge-profile --model yolov8n.pt --source video.mp4 --dashboard \
  --json runtime_report.json \
  --csv runtime_frames.csv
```

Enable local hard-example capture:

```bash
yolo-edge-profile --model yolov8n.pt --source video.mp4 --dashboard \
  --capture-dir hard_examples \
  --capture-states RED \
  --capture-cooldown-sec 2.0
```

Confidence-aware sampling can be added without changing the model:

```bash
yolo-edge-profile --model yolov8n.pt --source video.mp4 \
  --capture-dir hard_examples \
  --capture-selection-mode pressure_or_confidence \
  --capture-min-confidence-entropy 1.8
```

Camera example:

```bash
yolo-edge-profile --model yolov8n.pt --source 0 --dashboard --capture-dir hard_examples
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

This flags sudden postprocess pressure, especially in dense scenes or output-heavy frames.

### 4. Output Pressure

```text
box_count
box_pressure_coeff
confidence_entropy
class_entropy
```

This records whether the detection output stream is becoming dense or uncertain.

### 5. Runtime State

```text
GREEN / YELLOW / RED
```

Each state includes a `dominant_cause` and a human-readable reason.

## Design Notes

YERP is deliberately not a cloud platform. The current scope is:

```text
profile runtime stages
classify tail-latency and stage pressure
save local hard examples when pressure or uncertainty appears
export JSON / CSV reports
```

Future integrations can connect the local hard-example folder to a private dataset store, labeling workflow, or training platform.

## Relationship to OBIO

YERP is the pure-Python edge-runtime layer.

OBIO is the downstream physical-boundary observer for ROS 2 / PX4 / robot execution paths.

```text
YERP:
  Which YOLO runtime stage is creating pressure?

OBIO:
  Is that pressure reaching the physical execution boundary?

Resource-Aware Adapter:
  Use these signals to back off before vision load starves control.
```

Details：https://github.com/ZC502/ai_flight_integrity_observer.git

## License

Apache-2.0
