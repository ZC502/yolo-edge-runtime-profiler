#!/usr/bin/env python3

"""
Shows how to use YERP inside an existing OpenCV + Ultralytics loop.
"""

import cv2
from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler, LocalHardExampleRecorder


def main():
    model = YOLO("yolov8n.pt")
    profiler = YoloEdgeRuntimeProfiler(window_size=100)
    recorder = LocalHardExampleRecorder(output_dir="hard_examples", trigger_states=("RED",), cooldown_sec=2.0)

    cap = cv2.VideoCapture("video.mp4")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)
        status = profiler.update(results[0])
        capture = recorder.maybe_save(frame=frame, status=status)
        suffix = f" captured={capture.metadata_path}" if capture.saved else ""
        print(status.to_compact_string(color=True) + suffix)

    cap.release()
    profiler.export_json("runtime_report.json")
    profiler.export_csv("runtime_frames.csv")


if __name__ == "__main__":
    main()
