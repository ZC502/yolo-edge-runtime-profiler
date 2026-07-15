from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import time

from yolo_edge_runtime_profiler.structures import StutterEvent


CAUSE_ROOT_HINTS: Dict[str, str] = {
    "POSTPROCESS_PRESSURE": "High target density is increasing postprocess pressure.",
    "POSTPROCESS_DOMINANT": "Postprocess dominates this frame's runtime.",
    "POSTPROCESS_SPIKE": "Postprocess time spikes against the local baseline.",
    "POSTPROCESS_SPIKE_RISING": "Postprocess time is rising against the local baseline.",
    "TAIL_LATENCY_SPIKE": "This frame has a clear latency spike against the local baseline.",
    "TAIL_LATENCY_RISING": "This frame is slower than the local baseline and may become a latency spike.",
    "CASCADE_QUEUE_DELAY": "Previous slow frames or backlog hints suggest queue/cascade delay.",
    "IO_STREAM_BLOCKING": "Input, decode, stream, or storage wait may be blocking the pipeline.",
    "SYSTEM_WIDE_SLOWDOWN": "All stages appear slower without one dominant CV stage; system contention or throttling is suspected.",
    "ROS_EXECUTOR_DELAY": "ROS executor scheduling delay is suspected before callback execution.",
    "ROS_CALLBACK_BLOCKING": "ROS callback work is suspected to block frame processing.",
    "ROS_TF_WAIT": "Transform lookup / TF wait is suspected to block frame processing.",
    "ROS_MESSAGE_ARRIVAL_DELAY": "ROS message arrival delay or transport jitter is suspected.",
    "ROS_QOS_DROP_OR_BACKLOG": "ROS QoS queue drop, stale frame processing, or backlog is suspected.",
    "STABLE_RUNTIME": "No stutter cause detected.",
}

CAUSE_ACTIONS: Dict[str, str] = {
    "POSTPROCESS_PRESSURE": "Tune confidence threshold, max targets, input resolution, and NMS/postprocess path before blaming the model backbone.",
    "POSTPROCESS_DOMINANT": "Optimize postprocess first; check NMS, max targets, tracking, and output filtering.",
    "POSTPROCESS_SPIKE": "Inspect dense-scene frames and test lower max targets or a faster postprocess path.",
    "POSTPROCESS_SPIKE_RISING": "Watch dense scenes; consider tighter output filtering and a bounded target budget.",
    "TAIL_LATENCY_SPIKE": "Compare this frame with nearby normal frames and check scene complexity, target count, and system contention.",
    "TAIL_LATENCY_RISING": "Check whether this pattern repeats; tune thresholds or test a lighter model if it approaches the real-time budget.",
    "CASCADE_QUEUE_DELAY": "Use bounded queues, drop stale frames, lower input FPS, or add backpressure.",
    "IO_STREAM_BLOCKING": "Check RTSP/network jitter, decoder wait, disk writes, screenshot saving, and stream buffering.",
    "SYSTEM_WIDE_SLOWDOWN": "Check background processes, thermal throttling, power limits, CPU/GPU/NPU contention, and memory pressure.",
    "ROS_EXECUTOR_DELAY": "Check executor type, callback groups, long-running callbacks, and whether CV work blocks executor threads.",
    "ROS_CALLBACK_BLOCKING": "Move heavy CV work out of the subscription callback or use a bounded worker queue.",
    "ROS_TF_WAIT": "Check transform availability, timeout settings, and whether TF lookup blocks the frame callback.",
    "ROS_MESSAGE_ARRIVAL_DELAY": "Check camera driver timing, DDS transport, QoS settings, and network jitter.",
    "ROS_QOS_DROP_OR_BACKLOG": "Check QoS depth, reliability mode, queue size, and whether stale frames are being processed.",
}


def export_json(
    path: str,
    *,
    events: Iterable[StutterEvent],
    summary: Optional[Dict[str, Any]] = None,
    frame_summaries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "yerp.edge-cv-stutter-report.v0.2",
        "created_timestamp_sec": time.time(),
        "summary": summary or {},
        "events": [e.to_dict() for e in events],
        "frame_summaries": frame_summaries or [],
    }

    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_markdown(
    path: str,
    *,
    events: Iterable[StutterEvent],
    summary: Optional[Dict[str, Any]] = None,
    title: str = "YERP Edge CV Stutter Frame Report",
    max_events: int = 20,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    text = render_markdown(
        events=list(events),
        summary=summary or {},
        title=title,
        max_events=max_events,
    )

    p.write_text(text, encoding="utf-8")


def render_markdown(
    *,
    events: List[StutterEvent],
    summary: Dict[str, Any],
    title: str = "YERP Edge CV Stutter Frame Report",
    max_events: int = 20,
) -> str:
    events_sorted = sorted(events, key=lambda e: e.severity, reverse=True)
    top_events = events_sorted[:max_events]

    lines: List[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append("> YERP finds the exact frames that make an edge CV pipeline stutter, then explains the likely cause with frame-level runtime evidence.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Frames analyzed: **{summary.get('frames', 0)}**")
    lines.append(f"- Stutter events: **{summary.get('events', len(events))}**")
    lines.append(f"- Source type: **{summary.get('source_type', 'unknown')}**")
    lines.append(f"- Expected FPS: **{_fmt(summary.get('expected_input_fps'))}**")
    lines.append(f"- Real-time budget: **{_fmt_ms(_real_time_budget(summary))}**")
    lines.append("")

    notes = summary.get("notes", {})
    if notes.get("severity_vs_budget"):
        lines.append("> Note: RED/YELLOW are diagnostic severity levels. A RED event can indicate structural risk, such as postprocess dominance, even if the frame does not breach the current real-time budget.")
        lines.append("")

    state_counts = summary.get("state_counts", {})
    cause_counts = summary.get("cause_counts", {})

    lines.append("### State Counts")
    lines.append("")
    lines.append("| State | Frames |")
    lines.append("|---|---:|")
    for k, v in sorted(state_counts.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### Cause Counts")
    lines.append("")
    lines.append("| Cause Enum | Frames |")
    lines.append("|---|---:|")
    for k, v in sorted(cause_counts.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## Top Stutter Events")
    lines.append("")

    if not top_events:
        lines.append("No stutter events were detected.")
        lines.append("")
    else:
        lines.append("| Rank | Frame | Severity | Stutter Type | Likely Root Cause | Total Latency | Real-time Budget | Budget Breach | Slowdown | Suggested Action |")
        lines.append("|---:|---:|---|---|---|---:|---:|---|---:|---|")

        for i, e in enumerate(top_events, start=1):
            budget_ms = _event_budget_ms(e, summary)
            breach = _budget_breach(e.total_ms, budget_ms)
            lines.append(
                f"| {i} | {e.frame_id} | {e.state} | `{e.cause}` | "
                f"{_root_hint(e.cause)} | {_fmt_ms(e.total_ms)} | {_fmt_ms(budget_ms)} | "
                f"{breach} | {e.slowdown_ratio:.2f}x | {_short_action(e)} |"
            )
        lines.append("")

        for i, e in enumerate(top_events, start=1):
            lines.extend(_render_event_detail(i, e, summary))

    lines.append("## Thresholds")
    lines.append("")
    lines.append("YERP thresholds are intentionally configurable because real-time budgets vary by robot, camera FPS, hardware, and model size.")
    lines.append("")
    lines.append("| Parameter | Current Value | Purpose |")
    lines.append("|---|---:|---|")
    thresholds = summary.get("thresholds", {})
    if thresholds:
        for k, v in thresholds.items():
            lines.append(f"| `{k}` | {_fmt(v)} | {_threshold_meaning(k)} |")
    else:
        lines.append("| n/a | n/a | Threshold metadata was not exported. |")
    lines.append("")
    lines.append("Example tuning command:")
    lines.append("")
    lines.append("```bash")
    lines.append("python examples/audit_stutter_video.py \\")
    lines.append("  --model yolov8s.pt \\")
    lines.append("  --source your_video.mp4 \\")
    lines.append("  --input-fps 25 \\")
    lines.append("  --min-tail-latency-ms 30 \\")
    lines.append("  --yellow-slowdown-ratio 2.0 \\")
    lines.append("  --red-slowdown-ratio 4.0 \\")
    lines.append("  --min-postprocess-ms 3.0 \\")
    lines.append("  --min-target-count-postprocess 5")
    lines.append("```")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("YERP uses runtime residual auditing: each frame is compared against a robust local baseline rather than a global average.")
    lines.append("")
    lines.append("- Local baseline: rolling median of recent stable frames")
    lines.append("- Robust deviation: MAD-based robust z-score")
    lines.append("- Event diagnosis: heuristic rules over timing, stage ratios, target count, queue hints, IO hints, system hints, and reserved ROS fields")
    lines.append("- Budget check: real-time budget is derived from expected FPS when available")
    lines.append("")
    lines.append("This design is inspired by NARH-style residual auditing: a stutter frame is treated as a runtime residual event where timing, queue behavior, and scene pressure no longer agree with the local baseline.")
    lines.append("")

    lines.append("## Reserved Fields")
    lines.append("")
    lines.append("The v0.2 report schema reserves fields for future production probes:")
    lines.append("")
    lines.append("- RTSP / stream jitter: `read_wait_ms`, `decode_ms`, `input_interval_ms`")
    lines.append("- Queue backlog: `estimated_backlog_count`, `processed_interval_ms`")
    lines.append("- ROS: `ros_callback_ms`, `ros_executor_delay_ms`, `ros_tf_wait_ms`, `ros_dropped_frames_estimate`")
    lines.append("- Hardware/system: `system_context.temperature_c`, `system_context.cpu_percent`, `system_context.memory_mb`, `system_context.npu_utilization`")
    lines.append("")

    lines.append("## Diagnosis Code Cheat Sheet")
    lines.append("")
    lines.append("| Cause Enum | Meaning |")
    lines.append("|---|---|")
    for code, meaning in CAUSE_ROOT_HINTS.items():
        lines.append(f"| `{code}` | {meaning} |")
    lines.append("")

    return "\n".join(lines)


def _render_event_detail(rank: int, e: StutterEvent, summary: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    budget_ms = _event_budget_ms(e, summary)
    breach = _budget_breach(e.total_ms, budget_ms)

    lines.append(f"### Event #{rank}: Frame {e.frame_id}")
    lines.append("")
    lines.append(f"- Severity: **{e.state}**")
    lines.append(f"- Stutter type: **`{e.cause}`**")
    lines.append(f"- Likely root cause: **{_root_hint(e.cause)}**")
    lines.append(f"- Total latency: **{_fmt_ms(e.total_ms)}**")
    lines.append(f"- Real-time budget: **{_fmt_ms(budget_ms)}**")
    lines.append(f"- Budget breach: **{breach}**")
    lines.append(f"- Local baseline: **{_fmt_ms(e.local_baseline_ms)}**")
    lines.append(f"- Slowdown: **{e.slowdown_ratio:.2f}x**")
    lines.append(f"- Robust z-score: **{e.robust_z:.2f}**")
    lines.append(f"- Target count: **{e.target_count}**")
    lines.append(f"- Scene complexity: **{e.scene_complexity:.2f}**")
    lines.append("")

    if e.stage_ms:
        lines.append("#### Stage Timing")
        lines.append("")
        lines.append("| Stage | ms | Ratio |")
        lines.append("|---|---:|---:|")
        for stage, ms in e.stage_ms.items():
            ratio = e.stage_ratio.get(stage, 0.0)
            lines.append(f"| `{stage}` | {float(ms):.2f} | {ratio:.1%} |")
        lines.append("")

    if e.suggestions:
        lines.append("#### Suggested Action")
        lines.append("")
        for s in e.suggestions:
            lines.append(f"- {s}")
        lines.append("")
    else:
        action = _short_action(e)
        if action:
            lines.append("#### Suggested Action")
            lines.append("")
            lines.append(f"- {action}")
            lines.append("")

    queue = e.evidence.get("queue", {})
    io = e.evidence.get("io", {})
    ros = e.evidence.get("ros_reserved", {})
    system_context = e.evidence.get("system_context", {})

    if _has_ros_evidence(ros) or str(e.cause).startswith("ROS_"):
        lines.append("#### Robotics / ROS Evidence")
        lines.append("")
        lines.append("| Metric | Value | Meaning |")
        lines.append("|---|---:|---|")
        lines.append(f"| `ros_callback_ms` | {_fmt(ros.get('ros_callback_ms'))} | Callback execution time |")
        lines.append(f"| `ros_executor_delay_ms` | {_fmt(ros.get('ros_executor_delay_ms'))} | Scheduling delay before callback execution |")
        lines.append(f"| `ros_tf_wait_ms` | {_fmt(ros.get('ros_tf_wait_ms'))} | Time spent waiting for TF / transform |")
        lines.append(f"| `ros_arrival_delay_ms` | {_fmt(ros.get('ros_arrival_delay_ms'))} | Delay between message timestamp and local arrival |")
        lines.append(f"| `ros_dropped_frames_estimate` | {_fmt(ros.get('ros_dropped_frames_estimate'))} | Estimated dropped or stale frames |")
        lines.append("")
        if str(e.cause).startswith("ROS_"):
            lines.append("> Robotics diagnosis: this stutter appears related to ROS message scheduling, callback execution, TF wait, QoS backlog, or transport delay rather than pure CV inference.")
            lines.append("")

    lines.append("#### Diagnostic Evidence")
    lines.append("")
    lines.append("```json")
    compact = {
        "queue": queue,
        "io": io,
        "ros_reserved": ros,
        "system_context": system_context,
        "local_baseline": e.evidence.get("local_baseline", {}),
    }
    lines.append(json.dumps(compact, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    return lines


def _event_budget_ms(e: StutterEvent, summary: Dict[str, Any]) -> Optional[float]:
    queue = e.evidence.get("queue", {}) if e.evidence else {}
    candidates = [
        queue.get("input_interval_ms"),
        summary.get("real_time_budget_ms"),
        summary.get("input_interval_ms"),
    ]

    for v in candidates:
        try:
            if v is None:
                continue
            f = float(v)
            if f > 0:
                return f
        except Exception:
            continue

    return None


def _budget_breach(total_ms: float, budget_ms: Optional[float]) -> str:
    if budget_ms is None or budget_ms <= 0:
        return "N/A"
    return "YES" if float(total_ms) > float(budget_ms) else "NO"


def _root_hint(cause: str) -> str:
    return CAUSE_ROOT_HINTS.get(str(cause), "Frame-level runtime residual detected.")


def _short_action(e: StutterEvent) -> str:
    if e.suggestions:
        return _shorten(e.suggestions[0], 120)
    return _shorten(CAUSE_ACTIONS.get(str(e.cause), "Inspect this frame and compare it with nearby normal frames."), 120)


def _has_ros_evidence(ros: Dict[str, Any]) -> bool:
    for v in ros.values():
        if v not in (None, "", 0, 0.0):
            return True
    return False


def _real_time_budget(summary: Dict[str, Any]) -> Optional[float]:
    for key in ("real_time_budget_ms", "input_interval_ms"):
        v = summary.get(key)
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except Exception:
            pass
    return None


def _fmt_ms(value: Any) -> str:
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.2f} ms"
    except Exception:
        return "N/A"


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    except Exception:
        return "N/A"


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _threshold_meaning(name: str) -> str:
    mapping = {
        "min_tail_latency_ms": "Minimum absolute latency before tail-spike rules can fire.",
        "min_postprocess_ms": "Minimum postprocess time before postprocess pressure rules can fire.",
        "min_target_count_for_postprocess": "Minimum target count required for postprocess-pressure diagnosis.",
        "yellow_slowdown_ratio": "Slowdown ratio for YELLOW latency warnings.",
        "red_slowdown_ratio": "Slowdown ratio for RED latency spikes.",
        "yellow_postprocess_ratio": "Postprocess share for YELLOW postprocess pressure.",
        "red_postprocess_ratio": "Postprocess share for RED postprocess dominance.",
        "yellow_robust_z": "MAD-based robust z-score for YELLOW warnings.",
        "red_robust_z": "MAD-based robust z-score for RED warnings.",
    }
    return mapping.get(name, "YERP diagnostic threshold.")
