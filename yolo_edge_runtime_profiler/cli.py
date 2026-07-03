from __future__ import annotations

import argparse
import sys

from .profiler import YoloEdgeRuntimeProfiler, ProfilerConfig
from .dashboard import print_dashboard


def _parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yolo-edge-profile",
        description="Profile YOLO tail latency, stage imbalance, and postprocess spikes from Ultralytics Results.speed.",
    )
    p.add_argument("--model", required=True, help="YOLO model path/name, e.g. yolov8n.pt")
    p.add_argument("--source", required=True, help="Video/image/camera source. Use 0 for webcam.")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--min-window", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 = no limit.")
    p.add_argument("--dashboard", action="store_true", help="Show live ASCII dashboard.")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--json", default="", help="Export JSON report path.")
    p.add_argument("--csv", default="", help="Export CSV frame log path.")
    p.add_argument("--conf", type=float, default=None, help="Optional YOLO confidence threshold.")
    p.add_argument("--iou", type=float, default=None, help="Optional YOLO IoU/NMS threshold.")
    p.add_argument("--imgsz", type=int, default=None, help="Optional YOLO image size.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(
            "ERROR: ultralytics is required for the CLI.\n"
            "Install with: pip install ultralytics\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    cfg = ProfilerConfig(
        window_size=args.window_size,
        min_window=args.min_window,
        enable_color=not args.no_color,
    )
    profiler = YoloEdgeRuntimeProfiler(config=cfg)

    model = YOLO(args.model)
    source = _parse_source(args.source)

    predict_kwargs = {"source": source, "stream": True, "verbose": False}
    if args.conf is not None:
        predict_kwargs["conf"] = args.conf
    if args.iou is not None:
        predict_kwargs["iou"] = args.iou
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz

    frame_count = 0
    try:
        for result in model(**predict_kwargs):
            frame_count += 1
            status = profiler.update(result)

            if args.dashboard:
                print_dashboard(status)
            else:
                print(status.to_compact_string(color=not args.no_color), flush=True)

            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.json:
            profiler.export_json(args.json)
            print(f"[INFO] JSON report written: {args.json}")
        if args.csv:
            profiler.export_csv(args.csv)
            print(f"[INFO] CSV frame log written: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
