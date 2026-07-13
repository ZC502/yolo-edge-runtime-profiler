from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return str(value)


@dataclass
class FrameInferenceContext:
    """
    Generic per-frame CV runtime context.

    This is the core input contract of YERP v0.2.

    It is intentionally not YOLO-specific:
      - target_count can be boxes, keypoints, regions, tracks, defects, etc.
      - stage_ms can contain any stages exposed by the customer pipeline.
      - custom_meta can carry adapter-specific information.
    """

    frame_id: int
    timestamp_sec: float

    # Core timing
    total_ms: float
    stage_ms: Dict[str, float] = field(default_factory=dict)

    # Source information
    source_type: str = "unknown"       # file_replay / rtsp / camera / ros_topic / unknown
    source_name: str = ""

    # Input / stream / queue timing
    input_interval_ms: Optional[float] = None
    processed_interval_ms: Optional[float] = None
    read_wait_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    write_ms: Optional[float] = None
    estimated_backlog_count: Optional[int] = None

    # Generic scene pressure
    target_count: int = 0
    scene_complexity: float = 0.0

    # ROS / robotics reserved fields
    ros_topic: str = ""
    ros_frame_id: str = ""
    ros_header_stamp_sec: Optional[float] = None
    ros_arrival_time_sec: Optional[float] = None
    ros_arrival_delay_ms: Optional[float] = None
    ros_publish_interval_ms: Optional[float] = None
    ros_callback_ms: Optional[float] = None
    ros_executor_delay_ms: Optional[float] = None
    ros_tf_wait_ms: Optional[float] = None
    ros_qos_depth: Optional[int] = None
    ros_dropped_frames_estimate: Optional[int] = None

    # Future system / hardware context
    system_context: Dict[str, Any] = field(default_factory=dict)

    # Evidence
    raw_frame: Any = None

    # Adapter-specific metadata
    custom_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_raw_frame: bool = False) -> Dict[str, Any]:
        data = {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "total_ms": self.total_ms,
            "stage_ms": self.stage_ms,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "input_interval_ms": self.input_interval_ms,
            "processed_interval_ms": self.processed_interval_ms,
            "read_wait_ms": self.read_wait_ms,
            "decode_ms": self.decode_ms,
            "write_ms": self.write_ms,
            "estimated_backlog_count": self.estimated_backlog_count,
            "target_count": self.target_count,
            "scene_complexity": self.scene_complexity,
            "ros_topic": self.ros_topic,
            "ros_frame_id": self.ros_frame_id,
            "ros_header_stamp_sec": self.ros_header_stamp_sec,
            "ros_arrival_time_sec": self.ros_arrival_time_sec,
            "ros_arrival_delay_ms": self.ros_arrival_delay_ms,
            "ros_publish_interval_ms": self.ros_publish_interval_ms,
            "ros_callback_ms": self.ros_callback_ms,
            "ros_executor_delay_ms": self.ros_executor_delay_ms,
            "ros_tf_wait_ms": self.ros_tf_wait_ms,
            "ros_qos_depth": self.ros_qos_depth,
            "ros_dropped_frames_estimate": self.ros_dropped_frames_estimate,
            "system_context": self.system_context,
            "custom_meta": self.custom_meta,
        }

        if include_raw_frame:
            data["raw_frame"] = str(type(self.raw_frame)) if self.raw_frame is not None else None

        return _json_safe(data)


@dataclass
class StutterEvent:
    """
    Diagnostic output event.

    This is what report_generator consumes.
    It is also intentionally model-agnostic.
    """

    event_id: int
    frame_id: int
    timestamp_sec: float

    state: str                  # YELLOW / RED
    cause: str                  # POSTPROCESS_PRESSURE / CASCADE_QUEUE_DELAY / IO_STREAM_BLOCKING / ...
    severity: float             # higher = worse

    total_ms: float
    local_baseline_ms: float
    slowdown_ratio: float
    robust_z: float

    stage_ms: Dict[str, float] = field(default_factory=dict)
    stage_ratio: Dict[str, float] = field(default_factory=dict)

    target_count: int = 0
    scene_complexity: float = 0.0

    evidence: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    source_type: str = "unknown"
    source_name: str = ""

    image_path: str = ""
    metadata_path: str = ""

    context_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "event_id": self.event_id,
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "state": self.state,
            "cause": self.cause,
            "severity": self.severity,
            "total_ms": self.total_ms,
            "local_baseline_ms": self.local_baseline_ms,
            "slowdown_ratio": self.slowdown_ratio,
            "robust_z": self.robust_z,
            "stage_ms": self.stage_ms,
            "stage_ratio": self.stage_ratio,
            "target_count": self.target_count,
            "scene_complexity": self.scene_complexity,
            "evidence": self.evidence,
            "suggestions": self.suggestions,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "image_path": self.image_path,
            "metadata_path": self.metadata_path,
            "context_meta": self.context_meta,
        })
