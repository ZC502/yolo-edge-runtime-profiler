#!/usr/bin/env python3

"""
Synthetic demo that does not require Ultralytics.

It feeds fake speed dictionaries into the profiler to demonstrate:
GREEN -> YELLOW/RED when postprocess or tail latency spikes.
"""

import random
import time

from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler
from yolo_edge_runtime_profiler.dashboard import print_dashboard


def main():
    profiler = YoloEdgeRuntimeProfiler(window_size=50)

    for i in range(250):
        pre = 2.0 + random.uniform(-0.2, 0.2)
        inf = 12.0 + random.uniform(-0.8, 0.8)
        post = 2.0 + random.uniform(-0.2, 0.2)
        boxes = random.randint(1, 12)

        if 100 <= i < 140:
            post = 30.0 + random.uniform(0, 45.0)
            boxes = random.randint(120, 320)

        if 180 <= i < 195:
            inf = 60.0 + random.uniform(0, 25.0)
            boxes = random.randint(10, 30)

        status = profiler.update_from_parts(
            speed={"preprocess": pre, "inference": inf, "postprocess": post},
            box_count=boxes,
            confidences=[random.uniform(0.3, 0.95) for _ in range(boxes)],
            classes=[random.randint(0, 4) for _ in range(boxes)],
        )
        print_dashboard(status)
        time.sleep(0.05)

    profiler.export_json("synthetic_runtime_report.json")
    profiler.export_csv("synthetic_runtime_frames.csv")


if __name__ == "__main__":
    main()
