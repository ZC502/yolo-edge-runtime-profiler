#!/usr/bin/env python3

from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler, LocalHardExampleRecorder


def main():
    model = YOLO("yolov8n.pt")
    profiler = YoloEdgeRuntimeProfiler(window_size=100)
    recorder = LocalHardExampleRecorder(output_dir="hard_examples", trigger_states=("RED",), cooldown_sec=2.0)

    for result in model(0, stream=True):
        status = profiler.update(result)
        frame = getattr(result, "orig_img", None)
        capture = recorder.maybe_save(frame=frame, status=status, source_info={"source": "webcam"})
        suffix = f" captured={capture.metadata_path}" if capture.saved else ""
        print(status.to_compact_string(color=True) + suffix)


if __name__ == "__main__":
    main()
