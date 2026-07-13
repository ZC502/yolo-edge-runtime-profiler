from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import time

from yolo_edge_runtime_profiler.structures import StutterEvent


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
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Frames analyzed: **{summary.get('frames', 0)}**")
    lines.append(f"- Stutter events: **{summary.get('events', len(events))}**")
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
    lines.append("| Cause | Frames |")
    lines.append("|---|---:|")
    for k, v in sorted(cause_counts.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Top Stutter Events")
    lines.append("")

    if not top_events:
        lines.append("No stutter events were detected.")
        lines.append("")
    else:
        lines.append("| Rank | Frame | State | Cause | Total ms | Local baseline ms | Slowdown | Target count |")
        lines.append("|---:|---:|---|---|---:|---:|---:|---:|")

        for i, e in enumerate(top_events, start=1):
            lines.append(
                f"| {i} | {e.frame_id} | {e.state} | {e.cause} | "
                f"{e.total_ms:.2f} | {e.local_baseline_ms:.2f} | "
                f"{e.slowdown_ratio:.2f}x | {e.target_count} |"
            )
        lines.append("")

        for i, e in enumerate(top_events, start=1):
            lines.extend(_render_event_detail(i, e))

    lines.append("## Methodology")
    lines.append("")
    lines.append("YERP uses runtime residual auditing: each frame is compared against a robust local baseline rather than a global average.")
    lines.append("")
    lines.append("- Local baseline: rolling median of recent stable frames")
    lines.append("- Robust deviation: MAD-based robust z-score")
    lines.append("- Event diagnosis: heuristic rules over timing, stage ratios, target count, queue hints, IO hints, and reserved system/ROS fields")
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

    return "\n".join(lines)


def _render_event_detail(rank: int, e: StutterEvent) -> List[str]:
    lines: List[str] = []

    lines.append(f"### Event #{rank}: Frame {e.frame_id}")
    lines.append("")
    lines.append(f"- State: **{e.state}**")
    lines.append(f"- Likely cause: **{e.cause}**")
    lines.append(f"- Total latency: **{e.total_ms:.2f} ms**")
    lines.append(f"- Local baseline: **{e.local_baseline_ms:.2f} ms**")
    lines.append(f"- Slowdown: **{e.slowdown_ratio:.2f}x**")
    lines.append(f"- Robust z-score: **{e.robust_z:.2f}**")
    lines.append(f"- Target count: **{e.target_count}**")
    lines.append(f"- Scene complexity: **{e.scene_complexity:.2f}**")
    lines.append("")

    if e.stage_ms:
        lines.append("Stage timing:")
        lines.append("")
        lines.append("| Stage | ms | ratio |")
        lines.append("|---|---:|---:|")
        for stage, ms in e.stage_ms.items():
            ratio = e.stage_ratio.get(stage, 0.0)
            lines.append(f"| {stage} | {float(ms):.2f} | {ratio:.1%} |")
        lines.append("")

    if e.suggestions:
        lines.append("Suggested action:")
        lines.append("")
        for s in e.suggestions:
            lines.append(f"- {s}")
        lines.append("")

    queue = e.evidence.get("queue", {})
    io = e.evidence.get("io", {})
    ros = e.evidence.get("ros_reserved", {})

    lines.append("Diagnostic evidence:")
    lines.append("")
    lines.append("```json")
    compact = {
        "queue": queue,
        "io": io,
        "ros_reserved": ros,
        "local_baseline": e.evidence.get("local_baseline", {}),
    }
    lines.append(json.dumps(compact, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    return lines
