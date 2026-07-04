#!/usr/bin/env python3

"""
Synthetic demo that does not require Ultralytics.

It feeds fake speed dictionaries into the profiler to demonstrate:
GREEN -> YELLOW/RED when postprocess or tail latency spikes.
It also demonstrates local hard-example capture with synthetic frames.
"""

import random
import time

import numpy as np

from yolo_edge_runtime_profiler import YoloEdgeRuntimeProfiler, LocalHardExampleRecorder
from yolo_edge_runtime_profiler.dashboard import print_dashboard


def make_frame(i: int, boxes: int):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Encode a simple visual pattern so captured frames are not empty.
    frame[:, :, 1] = min(255, boxes)
    frame[10:40, 10:10 + (i % 250), 2] = 255
    return frame


def main():
    profiler = YoloEdgeRuntimeProfiler(window_size=50)
    recorder = LocalHardExampleRecorder(
        output_dir="synthetic_hard_examples",
        selection_mode="pressure_or_confidence",
        trigger_states=("YELLOW", "RED"),
        cooldown_sec=1.0,
        max_items=20,
        # Set min_confidence_entropy=1.8 to also capture high-entropy frames.
        min_confidence_entropy=None,
    )

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
        capture = recorder.maybe_save(frame=make_frame(i, boxes), status=status, source_info={"source": "synthetic"})
        print_dashboard(status)
        if capture.saved:
            print(f"[CAPTURE] {capture.metadata_path}")
        time.sleep(0.05)

    profiler.export_json("synthetic_runtime_report.json")
    profiler.export_csv("synthetic_runtime_frames.csv")


if __name__ == "__main__":
    main()
