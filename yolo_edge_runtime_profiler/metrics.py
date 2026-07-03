from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Sequence
import math

import numpy as np


EPS = 1e-9


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return default
    return float(np.percentile(arr, q))


def median(values: Sequence[float], default: float = 0.0) -> float:
    return percentile(values, 50, default=default)


def entropy_from_values(values: Sequence[float], bins: int = 10) -> float:
    if values is None:
        return 0.0
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return 0.0
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vals = np.clip(vals, 0.0, 1.0)
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist.astype(float) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_from_labels(labels: Sequence[int]) -> float:
    if labels is None:
        return 0.0
    vals = list(labels)
    if not vals:
        return 0.0
    _, counts = np.unique(np.asarray(vals), return_counts=True)
    p = counts.astype(float) / float(counts.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@dataclass
class RollingWindow:
    maxlen: int
    total_ms: Deque[float] = field(init=False)
    preprocess_ms: Deque[float] = field(init=False)
    inference_ms: Deque[float] = field(init=False)
    postprocess_ms: Deque[float] = field(init=False)
    box_count: Deque[int] = field(init=False)

    def __post_init__(self) -> None:
        self.total_ms = deque(maxlen=self.maxlen)
        self.preprocess_ms = deque(maxlen=self.maxlen)
        self.inference_ms = deque(maxlen=self.maxlen)
        self.postprocess_ms = deque(maxlen=self.maxlen)
        self.box_count = deque(maxlen=self.maxlen)

    def append(
        self,
        *,
        preprocess_ms: float,
        inference_ms: float,
        postprocess_ms: float,
        total_ms: float,
        box_count: int,
    ) -> None:
        self.preprocess_ms.append(float(preprocess_ms))
        self.inference_ms.append(float(inference_ms))
        self.postprocess_ms.append(float(postprocess_ms))
        self.total_ms.append(float(total_ms))
        self.box_count.append(int(box_count))

    def __len__(self) -> int:
        return len(self.total_ms)

    def stats(self) -> Dict[str, float]:
        totals = list(self.total_ms)
        posts = list(self.postprocess_ms)
        boxes = list(self.box_count)

        p50 = percentile(totals, 50)
        p95 = percentile(totals, 95)
        p99 = percentile(totals, 99)
        max_total = max(totals) if totals else 0.0

        post_median = median(posts)
        post_p95 = percentile(posts, 95)
        box_median = median(boxes)
        box_p95 = percentile(boxes, 95)

        return {
            "total_p50": p50,
            "total_p95": p95,
            "total_p99": p99,
            "total_max": max_total,
            "tail_coeff_p95_p50": p95 / max(p50, EPS),
            "tail_coeff_p99_p50": p99 / max(p50, EPS),
            "postprocess_median": post_median,
            "postprocess_p95": post_p95,
            "box_median": box_median,
            "box_p95": box_p95,
        }


def normalize_speed_dict(speed: Dict[str, float] | None) -> Dict[str, float]:
    speed = speed or {}
    pre = safe_float(speed.get("preprocess", speed.get("preprocess_ms", 0.0)))
    inf = safe_float(speed.get("inference", speed.get("inference_ms", 0.0)))
    post = safe_float(speed.get("postprocess", speed.get("postprocess_ms", 0.0)))
    total = safe_float(speed.get("total", speed.get("total_ms", pre + inf + post)))
    if total <= 0:
        total = pre + inf + post
    return {
        "preprocess": pre,
        "inference": inf,
        "postprocess": post,
        "total": total,
    }
