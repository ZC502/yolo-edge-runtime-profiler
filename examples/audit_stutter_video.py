#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from yolo_edge_runtime_profiler.adapters.yolo_adapter import context_from_yolo_result
from yolo_edge_runtime_profiler.engine.stutter_engine import StutterEngine, StutterEngineConfig
from yolo_edge_runtime_profiler.report_generator import export_json, export_markdown


STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")


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

    # Threshold knobs. These make YERP adaptable to different real-time budgets and scenes.
    parser.add_argument("--min-tail-latency-ms", type=float, default=30.0)
    parser.add_argument("--min-postprocess-ms", type=float, default=3.0)
    parser.add_argument("--min-target-count-postprocess", type=int, default=5)
    parser.add_argument("--yellow-slowdown-ratio", type=float, default=2.0)
    parser.add_argument("--red-slowdown-ratio", type=float, default=4.0)
    parser.add_argument("--yellow-robust-z", type=float, default=3.5)
    parser.add_argument("--red-robust-z", type=float, default=6.0)
    parser.add_argument("--yellow-postprocess-ratio", type=float, default=0.30)
    parser.add_argument("--red-postprocess-ratio", type=float, default=0.50)

    parser.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "file_replay", "stream", "camera", "ros_topic", "unknown"],
        help="Input source type. Default: auto.",
    )

    parser.add_argument(
        "--input-fps",
        type=float,
        default=None,
        help=(
            "Nominal source FPS. For file replay, this is used as the video cadence. "
            "For stream/camera, it is only used as a first-frame fallback and expected reference; "
            "observed arrival intervals are still measured to preserve jitter."
        ),
    )

    parser.add_argument(
        "--expected-input-fps",
        type=float,
        default=None,
        help=(
            "Expected production input FPS for queue/backlog diagnosis. "
            "Defaults to --input-fps when available."
        ),
    )

    parser.add_argument(
        "--ros-topic",
        default="",
        help="Optional ROS topic name to place into reserved report fields.",
    )

    parser.add_argument("--demo-mode", action="store_true")

    return parser.parse_args()


def source_exists_or_is_stream(source: str) -> bool:
    if source.isdigit():
        return True
    if source.startswith(STREAM_PREFIXES):
        return True
    return Path(source).exists()


def infer_source_type(source: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if source.isdigit():
        return "camera"
    if source.startswith(STREAM_PREFIXES):
        return "stream"
    if Path(source).exists():
        return "file_replay"
    return "unknown"


def detect_file_fps(source: str) -> Optional[float]:
    """
    Best-effort local-video FPS detection.

    OpenCV is imported lazily so YERP does not require cv2 for every use case.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps > 0 and fps < 1000:
            return fps
        return None
    except Exception:
        return None


def resolve_input_fps(source: str, source_type: str, user_input_fps: Optional[float]) -> Optional[float]:
    if user_input_fps is not None and user_input_fps > 0:
        return float(user_input_fps)

    if source_type == "file_replay" and Path(source).exists():
        return detect_file_fps(source)

    return None


def input_interval_ms_from_fps(fps: Optional[float]) -> Optional[float]:
    if fps is None or fps <= 0:
        return None
    return 1000.0 / float(fps)


def build_config(args: argparse.Namespace, expected_input_fps: Optional[float]) -> StutterEngineConfig:
    if args.demo_mode:
        # Recording/demo profile: intentionally more sensitive for short demos.
        min_tail_latency_ms = 15.0
        min_postprocess_ms = 1.0
        min_target_count = 8
        yellow_postprocess_ratio = 0.20
        red_postprocess_ratio = 0.45
        yellow_slowdown_ratio = min(args.yellow_slowdown_ratio, 1.50)
        red_slowdown_ratio = min(args.red_slowdown_ratio, 3.00)
    else:
        min_tail_latency_ms = args.min_tail_latency_ms
        min_postprocess_ms = args.min_postprocess_ms
        min_target_count = args.min_target_count_postprocess
        yellow_postprocess_ratio = args.yellow_postprocess_ratio
        red_postprocess_ratio = args.red_postprocess_ratio
        yellow_slowdown_ratio = args.yellow_slowdown_ratio
        red_slowdown_ratio = args.red_slowdown_ratio

    return StutterEngineConfig(
        window_size=args.window_size,
        min_window=args.min_window,
        warmup_frames=args.warmup_frames,
        min_tail_latency_ms=min_tail_latency_ms,
        min_postprocess_ms=min_postprocess_ms,
        min_target_count_for_postprocess=min_target_count,
        yellow_postprocess_ratio=yellow_postprocess_ratio,
        red_postprocess_ratio=red_postprocess_ratio,
        yellow_slowdown_ratio=yellow_slowdown_ratio,
        red_slowdown_ratio=red_slowdown_ratio,
        yellow_robust_z=args.yellow_robust_z,
        red_robust_z=args.red_robust_z,
        expected_input_fps=expected_input_fps,
    )


def main() -> int:
    args = parse_args()

    source = args.source
    if not source_exists_or_is_stream(source):
        print(f"[ERROR] Source not found: {source}")
        return 1

    source_type = infer_source_type(source, args.source_type)
    input_fps = resolve_input_fps(source, source_type, args.input_fps)
    expected_input_fps = args.expected_input_fps or input_fps

    # For file replay, the real-time budget is the original video cadence.
    # For stream/camera, the same value is only a nominal reference; observed intervals are still measured.
    nominal_input_interval_ms = input_interval_ms_from_fps(input_fps)

    config = build_config(args, expected_input_fps)

    print("[YERP] Edge CV Stutter Frame Profiler")
    print(f"[YERP] model:       {args.model}")
    print(f"[YERP] source:      {args.source}")
    print(f"[YERP] source_type: {source_type}")
    print(f"[YERP] input_fps:   {input_fps if input_fps else 'unknown'}")
    print(f"[YERP] input_dt:    {f'{nominal_input_interval_ms:.2f} ms' if nominal_input_interval_ms else 'unknown'}")
    print(f"[YERP] mode:        {'DEMO' if args.demo_mode else 'STRICT'}")
    print()

    if source_type == "file_replay" and nominal_input_interval_ms is None:
        print("[WARN] Could not determine video FPS. Queue/backlog diagnosis will be limited.")
        print("       Pass --input-fps 25 or --input-fps 30 if you know the video FPS.")
        print()

    if source_type != "file_replay" and nominal_input_interval_ms is not None:
        print("[YERP] Stream/camera mode:")
        print("       --input-fps is treated as nominal reference only.")
        print("       Observed arrival interval is still measured to preserve input jitter.")
        print()

    model = YOLO(args.model)
    engine = StutterEngine(config)

    processed = 0
    last_loop_ts: Optional[float] = None
    last_arrival_ts: Optional[float] = None
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
        if last_loop_ts is not None:
            processed_interval_ms = (now - last_loop_ts) * 1000.0
        last_loop_ts = now

        # Input cadence policy:
        # - file_replay: trust the video FPS and use the nominal frame interval.
        # - stream/camera/ros_topic: use observed arrival interval to preserve network/IO jitter.
        #   --input-fps is only used as the first-frame fallback and expected reference.
        if source_type == "file_replay":
            input_interval_ms = nominal_input_interval_ms
        else:
            input_interval_ms = None
            if last_arrival_ts is not None:
                input_interval_ms = (now - last_arrival_ts) * 1000.0
            elif nominal_input_interval_ms is not None:
                input_interval_ms = nominal_input_interval_ms
        last_arrival_ts = now

        ros_meta = {}
        if args.ros_topic:
            ros_meta["ros_topic"] = args.ros_topic
            ros_meta["ros_arrival_time_sec"] = time.time()
            ros_meta["ros_publish_interval_ms"] = input_interval_ms

        ctx = context_from_yolo_result(
            result,
            frame_id=processed,
            timestamp_sec=time.time(),
            source_type=source_type,
            source_name=args.source,
            input_interval_ms=input_interval_ms,
            processed_interval_ms=processed_interval_ms,
            ros_meta=ros_meta,
            custom_meta={
                "model": args.model,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "input_fps": input_fps,
                "expected_input_fps": expected_input_fps,
                "nominal_input_interval_ms": nominal_input_interval_ms,
                "input_interval_policy": (
                    "video_fps_for_file_replay"
                    if source_type == "file_replay"
                    else "observed_arrival_interval_for_stream"
                ),
                "demo_mode": bool(args.demo_mode),
            },
        )

        event = engine.process(ctx)

        if event:
            budget_ms = input_interval_ms_from_fps(expected_input_fps)
            budget = f"{budget_ms:.2f}ms" if budget_ms else "n/a"
            breach = "YES" if budget_ms and event.total_ms > budget_ms else "NO" if budget_ms else "n/a"

            print(
                f"[{event.state}] frame={event.frame_id} "
                f"cause={event.cause} "
                f"total={event.total_ms:.2f}ms "
                f"budget={budget} breach={breach} "
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
    summary["source_type"] = source_type
    summary["input_fps"] = input_fps
    summary["input_interval_ms"] = nominal_input_interval_ms
    summary["expected_input_fps"] = expected_input_fps
    summary["real_time_budget_ms"] = input_interval_ms_from_fps(expected_input_fps)
    summary["thresholds"] = {
        "min_tail_latency_ms": config.min_tail_latency_ms,
        "min_postprocess_ms": config.min_postprocess_ms,
        "min_target_count_for_postprocess": config.min_target_count_for_postprocess,
        "yellow_slowdown_ratio": config.yellow_slowdown_ratio,
        "red_slowdown_ratio": config.red_slowdown_ratio,
        "yellow_postprocess_ratio": config.yellow_postprocess_ratio,
        "red_postprocess_ratio": config.red_postprocess_ratio,
        "yellow_robust_z": config.yellow_robust_z,
        "red_robust_z": config.red_robust_z,
    }
    summary["notes"] = {
        "severity_vs_budget": (
            "RED/YELLOW are diagnostic severity levels. A RED event may indicate structural risk "
            "such as postprocess dominance even when it does not breach the current real-time budget."
        ),
        "input_interval_policy": (
            "file_replay uses video FPS as input cadence; stream/camera uses observed arrival intervals "
            "so RTSP/network/IO jitter is not hidden by --input-fps."
        ),
    }

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
