#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO

from yolo_edge_runtime_profiler.adapters.yolo_adapter import context_from_yolo_result
from yolo_edge_runtime_profiler.engine.stutter_engine import StutterEngine, StutterEngineConfig
from yolo_edge_runtime_profiler.report_generator import export_json, export_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Edge CV stutter-frame diagnosis report.")

    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-frames", type=int, default=300)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument("--json", default="stutter_report.json")
    parser.add_argument("--markdown", default="stutter_report.md")

    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--min-window", type=int, default=10)

    parser.add_argument("--min-tail-latency-ms", type=float, default=30.0)
    parser.add_argument("--min-postprocess-ms", type=float, default=3.0)
    parser.add_argument("--min-target-count-postprocess", type=int, default=5)

    parser.add_argument("--demo-mode", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = args.source
    if not source.isdigit() and not source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        if not Path(source).exists():
            print(f"[ERROR] Source not found: {source}")
            return 1

    if args.demo_mode:
        min_tail_latency_ms = 15.0
        min_postprocess_ms = 1.0
        min_target_count = 8
        yellow_postprocess_ratio = 0.20
        red_postprocess_ratio = 0.45
    else:
        min_tail_latency_ms = args.min_tail_latency_ms
        min_postprocess_ms = args.min_postprocess_ms
        min_target_count = args.min_target_count_postprocess
        yellow_postprocess_ratio = 0.30
        red_postprocess_ratio = 0.50

    config = StutterEngineConfig(
        window_size=args.window_size,
        min_window=args.min_window,
        warmup_frames=args.warmup_frames,
        min_tail_latency_ms=min_tail_latency_ms,
        min_postprocess_ms=min_postprocess_ms,
        min_target_count_for_postprocess=min_target_count,
        yellow_postprocess_ratio=yellow_postprocess_ratio,
        red_postprocess_ratio=red_postprocess_ratio,
    )

    print("[YERP] Edge CV Stutter Frame Profiler")
    print(f"[YERP] model:  {args.model}")
    print(f"[YERP] source: {args.source}")
    print(f"[YERP] mode:   {'DEMO' if args.demo_mode else 'STRICT'}")
    print()

    model = YOLO(args.model)
    engine = StutterEngine(config)

    processed = 0
    last_processed_ts = None
    started = time.perf_counter()

    for result in model(
        args.source,
        stream=True,
        verbose=False,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    ):
        now = time.perf_counter()
        processed += 1

        processed_interval_ms = None
        if last_processed_ts is not None:
            processed_interval_ms = (now - last_processed_ts) * 1000.0
        last_processed_ts = now

        ctx = context_from_yolo_result(
            result,
            frame_id=processed,
            timestamp_sec=time.time(),
            source_type="file_replay" if not args.source.startswith(("rtsp://", "http://", "https://")) else "stream",
            source_name=args.source,
            processed_interval_ms=processed_interval_ms,
            custom_meta={
                "model": args.model,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
            },
        )

        event = engine.process(ctx)

        if event:
            print(
                f"[{event.state}] frame={event.frame_id} "
                f"cause={event.cause} "
                f"total={event.total_ms:.2f}ms "
                f"baseline={event.local_baseline_ms:.2f}ms "
                f"slowdown={event.slowdown_ratio:.2f}x "
                f"targets={event.target_count}"
            )
        elif processed % 50 == 0:
            print(f"[GREEN] processed={processed}")

        if args.max_frames > 0 and processed >= args.max_frames:
            break

    elapsed = time.perf_counter() - started
    summary = engine.summary()
    summary["elapsed_sec"] = elapsed
    summary["actual_fps"] = processed / elapsed if elapsed > 0 else 0.0

    export_json(
        args.json,
        events=engine.get_all_events(),
        summary=summary,
        frame_summaries=engine.get_frame_summaries(),
    )

    export_markdown(
        args.markdown,
        events=engine.get_all_events(),
        summary=summary,
        title="YERP Edge CV Stutter Frame Report",
    )

    print()
    print("[YERP] Done.")
    print(f"[YERP] Frames: {processed}")
    print(f"[YERP] Events: {len(engine.get_all_events())}")
    print(f"[YERP] JSON:   {Path(args.json).resolve()}")
    print(f"[YERP] MD:     {Path(args.markdown).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
