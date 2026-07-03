#!/usr/bin/env python3

from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler


def main():
    model = YOLO("yolov8n.pt")
    profiler = YoloEdgeRuntimeProfiler(window_size=100)

    for result in model("video.mp4", stream=True):
        status = profiler.update(result)
        print(status.to_compact_string(color=True))

    profiler.export_json("runtime_report.json")
    profiler.export_csv("runtime_frames.csv")


if __name__ == "__main__":
    main()
