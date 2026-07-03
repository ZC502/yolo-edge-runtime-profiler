from __future__ import annotations

import shutil

from .status import RuntimeState, RuntimeStatus


def _bar(value: float, max_value: float, width: int = 28) -> str:
    if max_value <= 0:
        n = 0
    else:
        n = int(max(0, min(width, round((value / max_value) * width))))
    return "#" * n + "." * (width - n)


def _state_color(state: RuntimeState, text: str) -> str:
    if state == RuntimeState.GREEN:
        return f"\033[92m{text}\033[0m"
    if state == RuntimeState.YELLOW:
        return f"\033[93m{text}\033[0m"
    return f"\033[91m{text}\033[0m"


def render_dashboard(status: RuntimeStatus) -> str:
    term_width = shutil.get_terminal_size((100, 24)).columns
    width = min(max(term_width, 80), 120)

    state = _state_color(status.state, status.state.value)
    cur = status.latency_ms.get("current", 0.0)
    p50 = status.latency_ms.get("p50", 0.0)
    p95 = status.latency_ms.get("p95", 0.0)
    p99 = status.latency_ms.get("p99", 0.0)
    max_ms = max(status.latency_ms.get("max", 0.0), cur, p99, 1.0)

    pre = status.stage_ms.get("preprocess", 0.0)
    inf = status.stage_ms.get("inference", 0.0)
    post = status.stage_ms.get("postprocess", 0.0)
    total = max(status.stage_ms.get("total", 0.0), 1.0)

    boxes = status.output_pressure.get("box_count", 0)
    conf_entropy = status.output_pressure.get("confidence_entropy", 0.0)
    cls_entropy = status.output_pressure.get("class_entropy", 0.0)

    def fit(s: str) -> str:
        return (s[: width - 3]).ljust(width - 2)

    lines = []
    lines.append("+" + "=" * (width - 2) + "+")
    lines.append("|" + " YOLO Edge-Runtime Profiler ".center(width - 2) + "|")
    lines.append("+" + "=" * (width - 2) + "+")
    lines.append("|" + fit(f" State: {state}   Cause: {status.dominant_cause}") + "|")
    lines.append("|" + fit(f" Frame: {status.frame_index}   Reason: {status.reason}") + "|")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("|" + fit(f" Total latency    {cur:8.2f} ms  [{_bar(cur, max_ms)}]") + "|")
    lines.append("|" + fit(f" p50 / p95 / p99  {p50:7.2f} / {p95:7.2f} / {p99:7.2f} ms    tail={status.latency_ms.get('tail_coeff_p95_p50', 1.0):.2f}x") + "|")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("|" + fit(f" preprocess       {pre:8.2f} ms  [{_bar(pre, total)}]") + "|")
    lines.append("|" + fit(f" inference        {inf:8.2f} ms  [{_bar(inf, total)}]") + "|")
    lines.append("|" + fit(f" postprocess      {post:8.2f} ms  [{_bar(post, total)}]") + "|")
    lines.append("+" + "-" * (width - 2) + "+")
    lines.append("|" + fit(f" boxes={boxes}   conf_entropy={conf_entropy:.3f}   class_entropy={cls_entropy:.3f}   low_postprocess={status.low_postprocess_path}") + "|")
    lines.append("+" + "=" * (width - 2) + "+")
    return "\n".join(lines)


def print_dashboard(status: RuntimeStatus) -> None:
    print("\033[2J\033[H", end="")
    print(render_dashboard(status), flush=True)
