#!/usr/bin/env python3

from ultralytics import YOLO
from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler

model = YOLO("yolov8n.pt")
profiler = YoloEdgeRuntimeProfiler(window_size=100)

for result in model("video.mp4", stream=True):
    status = profiler.update(result)

    if status.state != "GREEN":
        print(status.to_json())

profiler.export_json("runtime_report.json")
profiler.export_csv("runtime_frames.csv")
