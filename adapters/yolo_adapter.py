from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import time
import math

import numpy as np

from yolo_edge_runtime_profiler.structures import FrameInferenceContext


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _optional_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _to_float_list(x: Any) -> List[float]:
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        arr = np.asarray(x, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return [float(v) for v in arr.tolist()]
    except Exception:
        return []


def _entropy_from_values(values: Sequence[float], bins: int = 10) -> float:
    if not values:
        return 0.0

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    hist, _ = np.histogram(arr, bins=bins)
    total = hist.sum()
    if total <= 0:
        return 0.0

    p = hist.astype(float) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _entropy_from_labels(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0

    arr = np.asarray(labels, dtype=int).reshape(-1)
    if arr.size == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts.astype(float) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _extract_speed(result: Any) -> Dict[str, float]:
    raw = getattr(result, "speed", None) or {}

    preprocess = _safe_float(raw.get("preprocess", 0.0))
    inference = _safe_float(raw.get("inference", 0.0))
    postprocess = _safe_float(raw.get("postprocess", 0.0))

    total = _safe_float(
        raw.get("total", preprocess + inference + postprocess),
        preprocess + inference + postprocess,
    )

    return {
        "preprocess": preprocess,
        "inference": inference,
        "postprocess": postprocess,
        "total": total,
    }


def _extract_output_pressure(result: Any) -> Dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return {
            "target_count": 0,
            "confidence_mean": 0.0,
            "confidence_min": 0.0,
            "confidence_max": 0.0,
            "confidence_entropy": 0.0,
            "class_count": 0,
            "class_entropy": 0.0,
            "scene_complexity": 0.0,
        }

    try:
        target_count = int(len(boxes))
    except Exception:
        target_count = 0

    conf_values: List[float] = []
    cls_values: List[int] = []

    conf = getattr(boxes, "conf", None)
    if conf is not None:
        conf_values = _to_float_list(conf)

    cls = getattr(boxes, "cls", None)
    if cls is not None:
        cls_values = [int(x) for x in _to_float_list(cls)]

    confidence_mean = float(np.mean(conf_values)) if conf_values else 0.0
    confidence_min = float(np.min(conf_values)) if conf_values else 0.0
    confidence_max = float(np.max(conf_values)) if conf_values else 0.0
    confidence_entropy = _entropy_from_values(conf_values)
    class_entropy = _entropy_from_labels(cls_values)
    class_count = int(len(set(cls_values))) if cls_values else 0

    scene_complexity = float(confidence_entropy + 0.25 * math.log1p(max(0, target_count)))

    return {
        "target_count": target_count,
        "confidence_mean": confidence_mean,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
        "confidence_entropy": confidence_entropy,
        "class_count": class_count,
        "class_entropy": class_entropy,
        "scene_complexity": scene_complexity,
    }


def context_from_yolo_result(
    result: Any,
    *,
    frame_id: int,
    timestamp_sec: Optional[float] = None,
    source_type: str = "file_replay",
    source_name: str = "",
    input_interval_ms: Optional[float] = None,
    processed_interval_ms: Optional[float] = None,
    read_wait_ms: Optional[float] = None,
    decode_ms: Optional[float] = None,
    write_ms: Optional[float] = None,
    estimated_backlog_count: Optional[int] = None,
    ros_meta: Optional[Dict[str, Any]] = None,
    system_context: Optional[Dict[str, Any]] = None,
    custom_meta: Optional[Dict[str, Any]] = None,
) -> FrameInferenceContext:
    """
    Convert an Ultralytics YOLO result into YERP's generic FrameInferenceContext.

    This is the only place that should understand Ultralytics result objects.
    The stutter engine and report generator stay model-agnostic.
    """

    speed = _extract_speed(result)
    pressure = _extract_output_pressure(result)

    raw_frame = getattr(result, "orig_img", None)
    ros = ros_meta or {}

    meta = {
        "adapter": "ultralytics_yolo",
        "confidence_mean": pressure.get("confidence_mean", 0.0),
        "confidence_min": pressure.get("confidence_min", 0.0),
        "confidence_max": pressure.get("confidence_max", 0.0),
        "confidence_entropy": pressure.get("confidence_entropy", 0.0),
        "class_count": pressure.get("class_count", 0),
        "class_entropy": pressure.get("class_entropy", 0.0),
    }

    if custom_meta:
        meta.update(custom_meta)

    return FrameInferenceContext(
        frame_id=int(frame_id),
        timestamp_sec=float(timestamp_sec if timestamp_sec is not None else time.time()),
        total_ms=float(speed["total"]),
        stage_ms={
            "preprocess": float(speed["preprocess"]),
            "inference": float(speed["inference"]),
            "postprocess": float(speed["postprocess"]),
        },
        source_type=source_type,
        source_name=source_name,
        input_interval_ms=input_interval_ms,
        processed_interval_ms=processed_interval_ms,
        read_wait_ms=read_wait_ms,
        decode_ms=decode_ms,
        write_ms=write_ms,
        estimated_backlog_count=estimated_backlog_count,
        target_count=int(pressure.get("target_count", 0)),
        scene_complexity=float(pressure.get("scene_complexity", 0.0)),
        ros_topic=_optional_str(ros.get("ros_topic", ""), ""),
        ros_frame_id=_optional_str(ros.get("ros_frame_id", ""), ""),
        ros_header_stamp_sec=_optional_float(ros.get("ros_header_stamp_sec")),
        ros_arrival_time_sec=_optional_float(ros.get("ros_arrival_time_sec")),
        ros_arrival_delay_ms=_optional_float(ros.get("ros_arrival_delay_ms")),
        ros_publish_interval_ms=_optional_float(ros.get("ros_publish_interval_ms")),
        ros_callback_ms=_optional_float(ros.get("ros_callback_ms")),
        ros_executor_delay_ms=_optional_float(ros.get("ros_executor_delay_ms")),
        ros_tf_wait_ms=_optional_float(ros.get("ros_tf_wait_ms")),
        ros_qos_depth=_optional_int(ros.get("ros_qos_depth")),
        ros_dropped_frames_estimate=_optional_int(ros.get("ros_dropped_frames_estimate")),
        system_context=dict(system_context or {}),
        raw_frame=raw_frame,
        custom_meta=meta,
    )
