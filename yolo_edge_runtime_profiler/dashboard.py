from __future__ import annotations

import shutil
from typing import Any


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _state_value(state: Any) -> str:
    return getattr(state, "value", str(state))


def _bar(value: float, max_value: float, width: int = 28) -> str:
    try:
        value = float(value)
        max_value = float(max_value)
    except Exception:
        value = 0.0
        max_value = 1.0

    if max_value <= 0:
        n = 0
    else:
        n = int(max(0, min(width, round((value / max_value) * width))))
    return "#" * n + "." * (width - n)


def _state_color(state: Any, text: str) -> str:
    s = _state_value(state).upper()
    if s == "GREEN":
        return f"\033[92m{text}\033[0m"
    if s == "YELLOW":
        return f"\033[93m{text}\033[0m"
    if s == "RED":
        return f"\033[91m{text}\033[0m"
    return text


def render_dashboard(status: Any) -> str:
    term_width = shutil.get_terminal_size((100, 24)).columns
    width = min(max(term_width, 80), 120)

    state_raw = _safe_get(status, "state", "UNKNOWN")
    state_text = _state_value(state_raw)
    state = _state_color(state_raw, state_text)

    dominant_cause = _safe_get(status, "dominant_cause", "UNKNOWN")
    reason = _safe_get(status, "reason", "")
    frame_index = _safe_get(status, "frame_index", 0)

    latency_ms = _safe_get(status, "latency_ms", {}) or {}
    stage_ms = _safe_get(status, "stage_ms", {}) or {}
    output_pressure = _safe_get(status, "output_pressure", {}) or {}

    cur = float(latency_ms.get("current", 0.0) or 0.0)
    p50 = float(latency_ms.get("p50", 0.0) or 0.0)
    p95 = float(latency_ms.get("p95", 0.0) or 0.0)
    p99 = float(latency_ms.get("p99", 0.0) or 0.0)
    max_ms = max(float(latency_ms.get("max", 0.0) or 0.0), cur, p99, 1.0)

    pre = float(stage_ms.get("preprocess", 0.0) or 0.0)
    inf = float(stage_ms.get("inference", 0.0) or 0.0)
    post = float(stage_ms.get("postprocess", 0.0) or 0.0)
    total = max(float(stage_ms.get("total", 0.0) or 0.0), pre + inf + post, 1.0)

    boxes = output_pressure.get("box_count", 0)
    conf_entropy = float(output_pressure.get("confidence_entropy", 0.0) or 0.0)
    cls_entropy = float(output_pressure.get("class_entropy", 0.0) or 0.0)
    tail = float(latency_ms.get("tail_coeff_p95_p50", 1.0) or 1.0)
    low_postprocess = _safe_get(status, "low_postprocess_path", False)

    def fit(s: str) -> str:
        return (s[: width - 3]).ljust(width - 2)

    lines = []
    lines.append("+" + "=" * (width - 2) + "+")
    lines.append("|" + " YOLO Edge-Runtime Profiler ".center(width - 2) + "|")
    lines.append("+" + "=" * (width - 2) + "+")
    lines.append("|" + fit(f" State: {state}   Cause: {dominant_cause}") + "|")
    lines.append("|" + fit(f" Frame: {frame_index}   Reason: {reason}") + "|")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("|" + fit(f" Total latency    {cur:8.2f} ms  [{_bar(cur, max_ms)}]") + "|")
    lines.append("|" + fit(f" p50 / p95 / p99  {p50:7.2f} / {p95:7.2f} / {p99:7.2f} ms    tail={tail:.2f}x") + "|")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("|" + fit(f" preprocess       {pre:8.2f} ms  [{_bar(pre, total)}]") + "|")
    lines.append("|" + fit(f" inference        {inf:8.2f} ms  [{_bar(inf, total)}]") + "|")
    lines.append("|" + fit(f" postprocess      {post:8.2f} ms  [{_bar(post, total)}]") + "|")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("|" + fit(
        f" boxes={boxes}   conf_entropy={conf_entropy:.3f}   "
        f"class_entropy={cls_entropy:.3f}   low_postprocess={low_postprocess}"
    ) + "|")
    lines.append("+" + "=" * (width - 2) + "+")
    return "\n".join(lines)


def print_dashboard(status: Any) -> None:
    print("\033[2J\033[H", end="")
    print(render_dashboard(status), flush=True)
