from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Deque, Dict, List, Optional, Tuple
import math

from yolo_edge_runtime_profiler.structures import FrameInferenceContext, StutterEvent


EPS = 1e-9


@dataclass
class StutterEngineConfig:
    window_size: int = 100
    min_window: int = 10
    warmup_frames: int = 30

    # Tail / stutter detection
    min_tail_latency_ms: float = 30.0
    yellow_slowdown_ratio: float = 2.0
    red_slowdown_ratio: float = 4.0
    yellow_robust_z: float = 3.5
    red_robust_z: float = 6.0

    # Scene / postprocess pressure
    min_postprocess_ms: float = 3.0
    min_target_count_for_postprocess: int = 5
    yellow_postprocess_ratio: float = 0.30
    red_postprocess_ratio: float = 0.50
    yellow_postprocess_spike_ratio: float = 2.0
    red_postprocess_spike_ratio: float = 4.0

    # Queue / cascade
    cascade_lookback: int = 3
    cascade_slow_ratio: float = 1.10
    expected_input_fps: Optional[float] = None

    # IO / stream blocking
    io_wait_yellow_ms: float = 20.0
    io_wait_red_ms: float = 50.0

    # ROS / robotics diagnostics
    # These defaults are intentionally conservative. Tune them for the control-loop budget.
    ros_executor_yellow_ms: float = 10.0
    ros_executor_red_ms: float = 30.0
    ros_callback_yellow_ms: float = 10.0
    ros_callback_red_ms: float = 30.0
    ros_tf_wait_yellow_ms: float = 10.0
    ros_tf_wait_red_ms: float = 30.0
    ros_arrival_delay_yellow_ms: float = 50.0
    ros_arrival_delay_red_ms: float = 100.0
    ros_drop_yellow_count: int = 1
    ros_drop_red_count: int = 5

    # Baseline hygiene
    add_yellow_to_baseline: bool = False
    add_red_to_baseline: bool = False


class StutterEngine:
    """
    Model-agnostic stutter-frame diagnosis engine.

    Adapter -> FrameInferenceContext -> StutterEngine -> StutterEvent -> ReportGenerator
    """

    def __init__(self, config: Optional[StutterEngineConfig] = None) -> None:
        self.config = config or StutterEngineConfig()

        self._baseline: Deque[FrameInferenceContext] = deque(maxlen=self.config.window_size)
        self._recent_slow: Deque[bool] = deque(maxlen=max(1, self.config.cascade_lookback))

        self.frames_seen = 0
        self.events: List[StutterEvent] = []
        self.frame_summaries: List[Dict[str, Any]] = []

    def process(self, ctx: FrameInferenceContext) -> Optional[StutterEvent]:
        self.frames_seen += 1

        if self.frames_seen <= self.config.warmup_frames:
            self._record_frame(ctx, "GREEN", "WARMUP", None, None, 1.0, 0.0)
            return None

        if len(self._baseline) < self.config.min_window:
            self._baseline.append(ctx)
            self._record_frame(ctx, "GREEN", "BASELINE_COLLECTION", None, None, 1.0, 0.0)
            return None

        baseline_stats = self._baseline_stats()
        local_baseline_ms = baseline_stats["total_median"]
        local_mad_ms = baseline_stats["total_mad"]

        slowdown_ratio = ctx.total_ms / max(local_baseline_ms, EPS)
        robust_z = (ctx.total_ms - local_baseline_ms) / max(local_mad_ms, EPS)

        stage_ratio = _stage_ratio(ctx.stage_ms, ctx.total_ms)
        evidence = self._build_evidence(ctx, baseline_stats, stage_ratio, slowdown_ratio, robust_z)

        state, cause, severity = self._classify(
            ctx=ctx,
            baseline_stats=baseline_stats,
            stage_ratio=stage_ratio,
            slowdown_ratio=slowdown_ratio,
            robust_z=robust_z,
            evidence=evidence,
        )

        event: Optional[StutterEvent] = None

        if state in ("YELLOW", "RED"):
            suggestions = self._suggestions(cause, ctx, evidence)
            event = StutterEvent(
                event_id=len(self.events) + 1,
                frame_id=ctx.frame_id,
                timestamp_sec=ctx.timestamp_sec,
                state=state,
                cause=cause,
                severity=float(severity),
                total_ms=float(ctx.total_ms),
                local_baseline_ms=float(local_baseline_ms),
                slowdown_ratio=float(slowdown_ratio),
                robust_z=float(robust_z),
                stage_ms=dict(ctx.stage_ms),
                stage_ratio=stage_ratio,
                target_count=int(ctx.target_count),
                scene_complexity=float(ctx.scene_complexity),
                evidence=evidence,
                suggestions=suggestions,
                source_type=ctx.source_type,
                source_name=ctx.source_name,
                context_meta=dict(ctx.custom_meta),
            )
            self.events.append(event)

        self._record_frame(ctx, state, cause, local_baseline_ms, local_mad_ms, slowdown_ratio, robust_z)

        # Clean baseline policy:
        # Do not let confirmed stutters immediately redefine "normal".
        if state == "GREEN":
            self._baseline.append(ctx)
        elif state == "YELLOW" and self.config.add_yellow_to_baseline:
            self._baseline.append(ctx)
        elif state == "RED" and self.config.add_red_to_baseline:
            self._baseline.append(ctx)

        self._recent_slow.append(self._is_slow_for_queue(ctx, local_baseline_ms))

        return event

    def get_all_events(self) -> List[StutterEvent]:
        return list(self.events)

    def get_frame_summaries(self) -> List[Dict[str, Any]]:
        return list(self.frame_summaries)

    def summary(self) -> Dict[str, Any]:
        state_counts = Counter(row["state"] for row in self.frame_summaries)
        cause_counts = Counter(row["cause"] for row in self.frame_summaries)

        return {
            "schema": "yerp.edge-cv-stutter-report.v0.2",
            "frames": len(self.frame_summaries),
            "events": len(self.events),
            "state_counts": dict(state_counts),
            "cause_counts": dict(cause_counts),
            "top_causes": cause_counts.most_common(10),
            "config": self.config.__dict__,
        }

    def _baseline_stats(self) -> Dict[str, float]:
        totals = [float(x.total_ms) for x in self._baseline]
        post = [float(x.stage_ms.get("postprocess", 0.0)) for x in self._baseline]
        targets = [float(x.target_count) for x in self._baseline]

        total_median = _median(totals)
        total_mad = _mad(totals, total_median)

        post_median = _median(post)
        post_mad = _mad(post, post_median)

        return {
            "total_median": total_median,
            "total_mad": total_mad,
            "postprocess_median": post_median,
            "postprocess_mad": post_mad,
            "target_median": _median(targets),
        }

    def _classify(
        self,
        *,
        ctx: FrameInferenceContext,
        baseline_stats: Dict[str, float],
        stage_ratio: Dict[str, float],
        slowdown_ratio: float,
        robust_z: float,
        evidence: Dict[str, Any],
    ) -> Tuple[str, str, float]:
        cfg = self.config

        state = "GREEN"
        cause = "STABLE_RUNTIME"
        severity = 0.0

        def promote(new_state: str, new_cause: str, new_severity: float) -> None:
            nonlocal state, cause, severity
            # Higher state always wins. Within the same state, higher severity wins.
            if _state_rank(new_state) > _state_rank(state) or (
                _state_rank(new_state) == _state_rank(state) and new_severity > severity
            ):
                state = new_state
                cause = new_cause
                severity = float(new_severity)

        # 1. IO / stream blocking first, because it is an important "not the algorithm's fault" case.
        read_wait_ms = _none_to_zero(ctx.read_wait_ms)
        decode_ms = _none_to_zero(ctx.decode_ms)
        input_block_ms = max(read_wait_ms, decode_ms)

        if input_block_ms >= cfg.io_wait_red_ms:
            promote("RED", "IO_STREAM_BLOCKING", input_block_ms)
        elif input_block_ms >= cfg.io_wait_yellow_ms:
            promote("YELLOW", "IO_STREAM_BLOCKING", input_block_ms)

        # 2. ROS / robotics blocking.
        # This is the key layer that separates YERP from a simple CV timing script:
        # it can surface control-loop adjacent delays such as executor scheduling,
        # blocking callbacks, TF waits, message arrival delay, and QoS backlog/drop.
        self._classify_ros(ctx, promote)

        # 3. Cascade / queue delay.
        backlog = int(ctx.estimated_backlog_count or 0)
        previous_slow_count = sum(1 for x in self._recent_slow if x)

        if backlog > 0 or previous_slow_count >= cfg.cascade_lookback:
            # Mark as queue/cascade if the current frame is also not comfortably normal.
            if slowdown_ratio >= cfg.yellow_slowdown_ratio or backlog > 0:
                cascade_severity = max(slowdown_ratio, float(backlog + previous_slow_count))
                promote("YELLOW", "CASCADE_QUEUE_DELAY", cascade_severity)

        # 3. Tail latency / single-frame spike.
        if ctx.total_ms >= cfg.min_tail_latency_ms:
            if slowdown_ratio >= cfg.red_slowdown_ratio or robust_z >= cfg.red_robust_z:
                promote("RED", "TAIL_LATENCY_SPIKE", max(slowdown_ratio, robust_z))
            elif slowdown_ratio >= cfg.yellow_slowdown_ratio or robust_z >= cfg.yellow_robust_z:
                promote("YELLOW", "TAIL_LATENCY_RISING", max(slowdown_ratio, robust_z))

        # 4. Scene-triggered postprocess pressure.
        post_ms = float(ctx.stage_ms.get("postprocess", 0.0))
        post_ratio = float(stage_ratio.get("postprocess", 0.0))
        target_ok = int(ctx.target_count) >= cfg.min_target_count_for_postprocess
        post_ms_ok = post_ms >= cfg.min_postprocess_ms

        if target_ok and post_ms_ok:
            if post_ratio >= cfg.red_postprocess_ratio:
                promote("RED", "POSTPROCESS_DOMINANT", post_ratio * 10.0)
            elif post_ratio >= cfg.yellow_postprocess_ratio:
                promote("YELLOW", "POSTPROCESS_PRESSURE", post_ratio * 10.0)

            post_baseline = max(baseline_stats["postprocess_median"], EPS)
            post_spike_ratio = post_ms / post_baseline
            evidence["postprocess_spike_ratio"] = post_spike_ratio

            if post_spike_ratio >= cfg.red_postprocess_spike_ratio:
                promote("RED", "POSTPROCESS_SPIKE", post_spike_ratio)
            elif post_spike_ratio >= cfg.yellow_postprocess_spike_ratio:
                promote("YELLOW", "POSTPROCESS_SPIKE_RISING", post_spike_ratio)
        else:
            evidence["postprocess_gate"] = {
                "target_ok": target_ok,
                "post_ms_ok": post_ms_ok,
                "min_target_count_for_postprocess": cfg.min_target_count_for_postprocess,
                "min_postprocess_ms": cfg.min_postprocess_ms,
            }

        # 5. System-wide slowdown heuristic.
        # If total slows down but no single stage clearly dominates, suspect system contention or throttling.
        if state == "YELLOW" and cause == "TAIL_LATENCY_RISING":
            max_stage_ratio = max(stage_ratio.values()) if stage_ratio else 0.0
            if max_stage_ratio < 0.75:
                cause = "SYSTEM_WIDE_SLOWDOWN"

        if state == "RED" and cause == "TAIL_LATENCY_SPIKE":
            max_stage_ratio = max(stage_ratio.values()) if stage_ratio else 0.0
            if max_stage_ratio < 0.75:
                cause = "SYSTEM_WIDE_SLOWDOWN"

        return state, cause, severity

    def _classify_ros(self, ctx: FrameInferenceContext, promote) -> None:
        """
        Promote ROS/robotics causes based on reserved ROS timing fields.

        These fields are optional. If a non-ROS pipeline does not populate them,
        this method is effectively a no-op.
        """
        cfg = self.config

        ros_executor_ms = _none_to_zero(ctx.ros_executor_delay_ms)
        ros_callback_ms = _none_to_zero(ctx.ros_callback_ms)
        ros_tf_wait_ms = _none_to_zero(ctx.ros_tf_wait_ms)
        ros_arrival_delay_ms = _none_to_zero(ctx.ros_arrival_delay_ms)

        try:
            ros_dropped = int(ctx.ros_dropped_frames_estimate or 0)
        except Exception:
            ros_dropped = 0

        # Executor scheduling delay: the frame waits before the callback actually runs.
        if ros_executor_ms >= cfg.ros_executor_red_ms:
            promote("RED", "ROS_EXECUTOR_DELAY", ros_executor_ms)
        elif ros_executor_ms >= cfg.ros_executor_yellow_ms:
            promote("YELLOW", "ROS_EXECUTOR_DELAY", ros_executor_ms)

        # Callback blocking: heavy work inside callback or callback group contention.
        if ros_callback_ms >= cfg.ros_callback_red_ms:
            promote("RED", "ROS_CALLBACK_BLOCKING", ros_callback_ms)
        elif ros_callback_ms >= cfg.ros_callback_yellow_ms:
            promote("YELLOW", "ROS_CALLBACK_BLOCKING", ros_callback_ms)

        # TF wait: transform lookup blocks the frame path.
        if ros_tf_wait_ms >= cfg.ros_tf_wait_red_ms:
            promote("RED", "ROS_TF_WAIT", ros_tf_wait_ms)
        elif ros_tf_wait_ms >= cfg.ros_tf_wait_yellow_ms:
            promote("YELLOW", "ROS_TF_WAIT", ros_tf_wait_ms)

        # Message arrival delay: transport / DDS / camera driver / network jitter.
        if ros_arrival_delay_ms >= cfg.ros_arrival_delay_red_ms:
            promote("RED", "ROS_MESSAGE_ARRIVAL_DELAY", ros_arrival_delay_ms)
        elif ros_arrival_delay_ms >= cfg.ros_arrival_delay_yellow_ms:
            promote("YELLOW", "ROS_MESSAGE_ARRIVAL_DELAY", ros_arrival_delay_ms)

        # QoS / backlog / drops: stale frame processing risk.
        if ros_dropped >= cfg.ros_drop_red_count:
            promote("RED", "ROS_QOS_DROP_OR_BACKLOG", float(ros_dropped))
        elif ros_dropped >= cfg.ros_drop_yellow_count:
            promote("YELLOW", "ROS_QOS_DROP_OR_BACKLOG", float(ros_dropped))

    def _build_evidence(
        self,
        ctx: FrameInferenceContext,
        baseline_stats: Dict[str, float],
        stage_ratio: Dict[str, float],
        slowdown_ratio: float,
        robust_z: float,
    ) -> Dict[str, Any]:
        return {
            "local_baseline": {
                "total_median_ms": baseline_stats["total_median"],
                "total_mad_ms": baseline_stats["total_mad"],
                "postprocess_median_ms": baseline_stats["postprocess_median"],
                "postprocess_mad_ms": baseline_stats["postprocess_mad"],
                "target_median": baseline_stats["target_median"],
            },
            "slowdown_ratio": slowdown_ratio,
            "robust_z": robust_z,
            "stage_ratio": stage_ratio,
            "target_count": ctx.target_count,
            "scene_complexity": ctx.scene_complexity,
            "queue": {
                "input_interval_ms": ctx.input_interval_ms,
                "processed_interval_ms": ctx.processed_interval_ms,
                "estimated_backlog_count": ctx.estimated_backlog_count,
                "recent_slow_count": sum(1 for x in self._recent_slow if x),
            },
            "io": {
                "read_wait_ms": ctx.read_wait_ms,
                "decode_ms": ctx.decode_ms,
                "write_ms": ctx.write_ms,
            },
            "ros_reserved": {
                "ros_topic": ctx.ros_topic,
                "ros_frame_id": ctx.ros_frame_id,
                "ros_arrival_delay_ms": ctx.ros_arrival_delay_ms,
                "ros_callback_ms": ctx.ros_callback_ms,
                "ros_executor_delay_ms": ctx.ros_executor_delay_ms,
                "ros_tf_wait_ms": ctx.ros_tf_wait_ms,
                "ros_dropped_frames_estimate": ctx.ros_dropped_frames_estimate,
            },
            "system_context": ctx.system_context,
        }

    def _suggestions(self, cause: str, ctx: FrameInferenceContext, evidence: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []

        post_ratio = evidence.get("stage_ratio", {}).get("postprocess", 0.0)
        target_count = int(ctx.target_count)

        if cause in ("POSTPROCESS_PRESSURE", "POSTPROCESS_DOMINANT", "POSTPROCESS_SPIKE", "POSTPROCESS_SPIKE_RISING"):
            suggestions.append(
                "Postprocess appears to be the main pressure point. Consider limiting maximum targets, raising confidence threshold, lowering input resolution, or testing a faster postprocess/NMS path."
            )
            if target_count > 50:
                suggestions.append(
                    "The stutter frame has high target density. Check whether dense scenes cause postprocess or tracking cost to grow nonlinearly."
                )
            if post_ratio > 0.5:
                suggestions.append(
                    "Postprocess is more than 50% of current frame latency. Optimize postprocess before changing the model backbone."
                )

        elif cause in ("TAIL_LATENCY_SPIKE", "TAIL_LATENCY_RISING"):
            suggestions.append(
                "This looks like an isolated latency spike. Compare the saved frame with nearby normal frames and check whether target density or scene complexity changed abruptly."
            )

        elif cause == "CASCADE_QUEUE_DELAY":
            suggestions.append(
                "This looks like cascade queue delay. Consider dropping stale frames, lowering input FPS, adding backpressure, or making the processing queue bounded."
            )

        elif cause == "IO_STREAM_BLOCKING":
            suggestions.append(
                "Input or stream blocking is suspected. Check RTSP/network jitter, decoder wait time, camera delivery interval, and disk write stalls."
            )

        elif cause == "SYSTEM_WIDE_SLOWDOWN":
            suggestions.append(
                "Total runtime increased without one clear stage dominating. Check background processes, CPU/GPU/NPU contention, thermal throttling, and power limits."
            )

        elif cause == "ROS_EXECUTOR_DELAY":
            suggestions.append(
                "Likely ROS executor delay. Check executor type, callback group design, long-running callbacks, and whether image processing blocks the executor thread."
            )
            suggestions.append(
                "If using rclpy/rclcpp with heavy image processing, consider a MultiThreadedExecutor, separate callback groups, or moving CV work to a bounded worker queue."
            )

        elif cause == "ROS_CALLBACK_BLOCKING":
            suggestions.append(
                "Likely ROS callback blocking. Move heavy CV work out of the subscription callback or use a bounded worker queue."
            )
            suggestions.append(
                "Avoid doing synchronous disk writes, blocking TF lookups, or long GPU synchronization directly inside the image callback."
            )

        elif cause == "ROS_TF_WAIT":
            suggestions.append(
                "Likely TF wait blocking. Check transform availability, timeout settings, and whether TF lookup is blocking the frame callback."
            )
            suggestions.append(
                "Prefer non-blocking transform checks or cache transforms outside the hot image-processing path."
            )

        elif cause == "ROS_MESSAGE_ARRIVAL_DELAY":
            suggestions.append(
                "Likely ROS message arrival delay. Check camera driver timing, DDS transport, QoS settings, and network jitter."
            )
            suggestions.append(
                "Compare header timestamps with local arrival timestamps to separate transport delay from CV inference delay."
            )

        elif cause == "ROS_QOS_DROP_OR_BACKLOG":
            suggestions.append(
                "ROS QoS drop or backlog is suspected. Check QoS depth, reliability mode, queue size, and whether stale frames are being processed."
            )
            suggestions.append(
                "For real-time CV, consider a bounded queue and dropping stale frames instead of processing old frames."
            )

        else:
            suggestions.append(
                "Inspect the frame evidence and compare it with nearby normal frames. If this is repeatable, test with a lower resolution and a smaller model."
            )

        return suggestions

    def _is_slow_for_queue(self, ctx: FrameInferenceContext, local_baseline_ms: float) -> bool:
        cfg = self.config

        if ctx.input_interval_ms is not None and ctx.input_interval_ms > 0:
            return ctx.total_ms > ctx.input_interval_ms * cfg.cascade_slow_ratio

        if cfg.expected_input_fps and cfg.expected_input_fps > 0:
            budget_ms = 1000.0 / cfg.expected_input_fps
            return ctx.total_ms > budget_ms * cfg.cascade_slow_ratio

        return ctx.total_ms > local_baseline_ms * cfg.yellow_slowdown_ratio

    def _record_frame(
        self,
        ctx: FrameInferenceContext,
        state: str,
        cause: str,
        local_baseline_ms: Optional[float],
        local_mad_ms: Optional[float],
        slowdown_ratio: float,
        robust_z: float,
    ) -> None:
        self.frame_summaries.append({
            "frame_id": ctx.frame_id,
            "timestamp_sec": ctx.timestamp_sec,
            "state": state,
            "cause": cause,
            "total_ms": ctx.total_ms,
            "local_baseline_ms": local_baseline_ms,
            "local_mad_ms": local_mad_ms,
            "slowdown_ratio": slowdown_ratio,
            "robust_z": robust_z,
            "target_count": ctx.target_count,
            "scene_complexity": ctx.scene_complexity,
            "source_type": ctx.source_type,
            "ros_callback_ms": ctx.ros_callback_ms,
            "ros_executor_delay_ms": ctx.ros_executor_delay_ms,
            "ros_tf_wait_ms": ctx.ros_tf_wait_ms,
            "ros_arrival_delay_ms": ctx.ros_arrival_delay_ms,
            "ros_dropped_frames_estimate": ctx.ros_dropped_frames_estimate,
        })


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _mad(values: List[float], center: float) -> float:
    if not values:
        return 1.0
    deviations = [abs(float(x) - center) for x in values]
    raw_mad = float(median(deviations))
    # 1.4826 makes MAD comparable to std under normal distribution.
    return max(raw_mad * 1.4826, 1e-6)


def _stage_ratio(stage_ms: Dict[str, float], total_ms: float) -> Dict[str, float]:
    total = max(float(total_ms), EPS)
    return {str(k): float(v) / total for k, v in stage_ms.items()}


def _state_rank(state: str) -> int:
    return {"GREEN": 0, "YELLOW": 1, "RED": 2}.get(state, 0)


def _none_to_zero(value: Optional[float]) -> float:
    try:
        if value is None:
            return 0.0
        v = float(value)
        if math.isfinite(v):
            return v
        return 0.0
    except Exception:
        return 0.0
