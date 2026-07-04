#!/usr/bin/env python3

from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler, LocalHardExampleRecorder


def main():
    model = YOLO("yolov8n.pt")
    profiler = YoloEdgeRuntimeProfiler(window_size=100)
    recorder = LocalHardExampleRecorder(
        output_dir="hard_examples",
        selection_mode="pressure_or_confidence",
        trigger_states=("RED",),
        cooldown_sec=2.0,
        max_items=100,
        # Optional confidence dimension. Tune for your use case.
        min_confidence_entropy=None,
        max_confidence_mean=None,
    )

    for result in model("video.mp4", stream=True):
        status = profiler.update(result)
        frame = getattr(result, "orig_img", None)
        capture = recorder.maybe_save(frame=frame, status=status)
        suffix = f" captured={capture.metadata_path}" if capture.saved else ""
        print(status.to_compact_string(color=True) + suffix)

    profiler.export_json("runtime_report.json")
    profiler.export_csv("runtime_frames.csv")


if __name__ == "__main__":
    main()
