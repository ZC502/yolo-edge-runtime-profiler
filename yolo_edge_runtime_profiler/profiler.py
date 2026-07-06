from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np

from .metrics import EPS, RollingWindow, entropy_from_labels, entropy_from_values, normalize_speed_dict, safe_float
from .status import RuntimeState, RuntimeStatus, max_state
from .exporters import JsonExporter, CsvExporter


@dataclass
class ProfilerConfig:
    window_size: int = 100
    min_window: int = 10

    # Frames used to initialize the runtime, video decoder, CUDA/PyTorch context, etc.
    # Warmup frames are reported, but are NOT used for the rolling baseline and cannot trigger capture.
    warmup_frames: int = 30

    # Tail latency must be both absolutely large and relatively large.
    # This prevents small 1-2 ms micro-jitter from becoming a false RED/YELLOW event.
    min_tail_latency_ms: float = 30.0
    yellow_tail_coeff: float = 2.0
    red_tail_coeff: float = 4.0

    # Postprocess pressure must also pass an absolute-ms gate.
    yellow_postprocess_ratio: float = 0.30
    red_postprocess_ratio: float = 0.50
    yellow_postprocess_spike: float = 2.0
    red_postprocess_spike: float = 4.0
    low_postprocess_ms: float = 3.0
    min_postprocess_ms_for_spike: float = 3.0
    min_postprocess_ms_for_pressure: float = 3.0

    confidence_entropy_bins: int = 10
    enable_color: bool = True


class YoloEdgeRuntimeProfiler:
    """
    Zero-intrusion runtime profiler for Ultralytics YOLO Results objects.

    Main API:
        status = profiler.update(result)

    It reads:
        result.speed
        result.boxes.conf
        result.boxes.cls

    It does not modify the model or inference pipeline.
    """

    def __init__(
        self,
        window_size: int = 100,
        *,
        config: Optional[ProfilerConfig] = None,
    ) -> None:
        self.config = config or ProfilerConfig(window_size=window_size)
        self.window = RollingWindow(maxlen=self.config.window_size)
        self.frame_index = 0
        self.records: List[RuntimeStatus] = []
        self.low_postprocess_notice_printed = False
        self.last_status: Optional[RuntimeStatus] = None

    def update(self, result: Any, *, wall_time_ms: Optional[float] = None) -> RuntimeStatus:
        speed = normalize_speed_dict(getattr(result, "speed", None))
        if wall_time_ms is not None:
            speed["wall"] = safe_float(wall_time_ms)

        box_count, confidences, classes = self._extract_output_pressure(result)
        return self.update_from_parts(
            speed=speed,
            box_count=box_count,
            confidences=confidences,
            classes=classes,
        )

    def update_from_parts(
        self,
        *,
        speed: Dict[str, float],
        box_count: int = 0,
        confidences: Optional[Sequence[float]] = None,
        classes: Optional[Sequence[int]] = None,
    ) -> RuntimeStatus:
        speed = normalize_speed_dict(speed)
        pre = speed["preprocess"]
        inf = speed["inference"]
        post = speed["postprocess"]
        total = speed["total"]

        self.frame_index += 1

        total_safe = max(total, EPS)
        stage_ratio = {
            "preprocess": pre / total_safe,
            "inference": inf / total_safe,
            "postprocess": post / total_safe,
        }

        confidences = list(confidences or [])
        classes = list(classes or [])
        conf_mean = float(np.mean(confidences)) if confidences else 0.0
        conf_min = float(np.min(confidences)) if confidences else 0.0
        conf_max = float(np.max(confidences)) if confidences else 0.0
        confidence_entropy = entropy_from_values(confidences, bins=self.config.confidence_entropy_bins)
        class_entropy = entropy_from_labels(classes)

        # Warmup frames are intentionally excluded from the rolling baseline.
        # This prevents CUDA/PyTorch/OpenCV/video-decoder cold-start spikes from poisoning p95/p99.
        if self.frame_index <= self.config.warmup_frames:
            rolling = self._current_only_stats(total_ms=total, postprocess_ms=post, box_count=box_count)
            status = self._make_status(
                state=RuntimeState.GREEN,
                cause="WARMUP",
                reason=f"Warming up pipeline: {self.frame_index}/{self.config.warmup_frames}",
                speed=speed,
                stage_ratio=stage_ratio,
                rolling=rolling,
                box_count=box_count,
                box_pressure_coeff=1.0,
                conf_mean=conf_mean,
                conf_min=conf_min,
                conf_max=conf_max,
                confidence_entropy=confidence_entropy,
                classes=classes,
                class_entropy=class_entropy,
                residuals={
                    "current_tail_coeff_p50": 1.0,
                    "tail_latency_coeff_p95_p50": 1.0,
                    "tail_latency_coeff_p99_p50": 1.0,
                    "postprocess_spike_coeff": 1.0,
                    "postprocess_ratio": float(stage_ratio["postprocess"]),
                    "box_pressure_coeff": 1.0,
                },
                low_postprocess_path=False,
                enough=False,
            )
            self._store_status(status)
            return status

        # Only non-warmup frames enter the baseline.
        self.window.append(
            preprocess_ms=pre,
            inference_ms=inf,
            postprocess_ms=post,
            total_ms=total,
            box_count=box_count,
        )

        enough = len(self.window) >= max(1, self.config.min_window)
        rolling = self.window.stats()
        low_postprocess_path = enough and rolling["postprocess_p95"] <= self.config.low_postprocess_ms

        if low_postprocess_path and not self.low_postprocess_notice_printed:
            self.low_postprocess_notice_printed = True
            print(
                "[INFO] Low-postprocess path detected. "
                "Postprocess lag appears minimal; auditing total and stage tail latency instead."
            )

        # Current-frame tail coefficient. This is the main tail trigger.
        current_tail_coeff = total / max(rolling["total_p50"], EPS)

        # Postprocess spike coefficient is gated by current absolute postprocess latency.
        # A 2x jump from 0.5 ms to 1.0 ms is not useful; a jump to 8-20 ms is.
        post_median = rolling["postprocess_median"]
        if post >= self.config.min_postprocess_ms_for_spike:
            postprocess_spike_coeff = post / max(post_median, EPS)
        else:
            postprocess_spike_coeff = 1.0

        box_median = rolling["box_median"]
        if box_median > 0:
            box_pressure_coeff = float(box_count) / max(float(box_median), EPS)
        else:
            box_pressure_coeff = 1.0 if box_count == 0 else float(box_count)

        state, cause, reason = self._classify(
            enough=enough,
            low_postprocess_path=low_postprocess_path,
            rolling=rolling,
            stage_ratio=stage_ratio,
            postprocess_spike_coeff=postprocess_spike_coeff,
            current_total_ms=total,
            current_postprocess_ms=post,
            current_tail_coeff=current_tail_coeff,
        )

        status = self._make_status(
            state=state,
            cause=cause,
            reason=reason,
            speed=speed,
            stage_ratio=stage_ratio,
            rolling=rolling,
            box_count=box_count,
            box_pressure_coeff=box_pressure_coeff,
            conf_mean=conf_mean,
            conf_min=conf_min,
            conf_max=conf_max,
            confidence_entropy=confidence_entropy,
            classes=classes,
            class_entropy=class_entropy,
            residuals={
                "current_tail_coeff_p50": float(current_tail_coeff),
                "tail_latency_coeff_p95_p50": float(rolling["tail_coeff_p95_p50"]),
                "tail_latency_coeff_p99_p50": float(rolling["tail_coeff_p99_p50"]),
                "postprocess_spike_coeff": float(postprocess_spike_coeff),
                "postprocess_ratio": float(stage_ratio["postprocess"]),
                "box_pressure_coeff": float(box_pressure_coeff),
            },
            low_postprocess_path=low_postprocess_path,
            enough=enough,
        )
        self._store_status(status)
        return status

    def _classify(
        self,
        *,
        enough: bool,
        low_postprocess_path: bool,
        rolling: Dict[str, float],
        stage_ratio: Dict[str, float],
        postprocess_spike_coeff: float,
        current_total_ms: float,
        current_postprocess_ms: float,
        current_tail_coeff: float,
    ) -> Tuple[RuntimeState, str, str]:
        if not enough:
            return (
                RuntimeState.GREEN,
                "BASELINE_COLLECTION",
                f"Collecting baseline window: {len(self.window)}/{self.config.min_window}",
            )

        post_ratio = stage_ratio["postprocess"]
        state = RuntimeState.GREEN
        cause = "STABLE_RUNTIME"
        reason = "Runtime appears stable within the current rolling window."

        # Tail latency: current frame must be slow in absolute terms AND inflated relative to p50.
        if current_total_ms >= self.config.min_tail_latency_ms:
            if current_tail_coeff >= self.config.red_tail_coeff:
                state = max_state(state, RuntimeState.RED)
                cause = "TAIL_LATENCY_SPIKE"
                reason = f"Current latency ({current_total_ms:.1f}ms) is {current_tail_coeff:.2f}x rolling p50."
            elif current_tail_coeff >= self.config.yellow_tail_coeff:
                state = max_state(state, RuntimeState.YELLOW)
                cause = "TAIL_LATENCY_RISING"
                reason = f"Current latency ({current_total_ms:.1f}ms) is {current_tail_coeff:.2f}x rolling p50."

        # In end-to-end / low-postprocess paths, postprocess-specific alarms are de-emphasized.
        if not low_postprocess_path:
            # Postprocess dominance: only meaningful above an absolute postprocess floor.
            if current_postprocess_ms >= self.config.min_postprocess_ms_for_pressure:
                if post_ratio >= self.config.red_postprocess_ratio:
                    state = max_state(state, RuntimeState.RED)
                    cause = "POSTPROCESS_DOMINANT"
                    reason = f"Postprocess is {post_ratio:.0%} of current total latency."
                elif post_ratio >= self.config.yellow_postprocess_ratio and state != RuntimeState.RED:
                    state = max_state(state, RuntimeState.YELLOW)
                    cause = "POSTPROCESS_PRESSURE"
                    reason = f"Postprocess is {post_ratio:.0%} of current total latency."

            # Postprocess spike: also requires absolute postprocess latency to exceed the floor.
            if current_postprocess_ms >= self.config.min_postprocess_ms_for_spike:
                if postprocess_spike_coeff >= self.config.red_postprocess_spike:
                    state = max_state(state, RuntimeState.RED)
                    cause = "POSTPROCESS_SPIKE"
                    reason = f"Postprocess spike is {postprocess_spike_coeff:.2f}x rolling median."
                elif postprocess_spike_coeff >= self.config.yellow_postprocess_spike and state != RuntimeState.RED:
                    state = max_state(state, RuntimeState.YELLOW)
                    cause = "POSTPROCESS_SPIKE_RISING"
                    reason = f"Postprocess spike is {postprocess_spike_coeff:.2f}x rolling median."
        else:
            if state == RuntimeState.GREEN:
                cause = "LOW_POSTPROCESS_PATH"
                reason = "Postprocess is near-zero; auditing total/preprocess/inference tail latency."

        return state, cause, reason

    def _make_status(
        self,
        *,
        state: RuntimeState,
        cause: str,
        reason: str,
        speed: Dict[str, float],
        stage_ratio: Dict[str, float],
        rolling: Dict[str, float],
        box_count: int,
        box_pressure_coeff: float,
        conf_mean: float,
        conf_min: float,
        conf_max: float,
        confidence_entropy: float,
        classes: Sequence[int],
        class_entropy: float,
        residuals: Dict[str, float],
        low_postprocess_path: bool,
        enough: bool,
    ) -> RuntimeStatus:
        pre = speed["preprocess"]
        inf = speed["inference"]
        post = speed["postprocess"]
        total = speed["total"]
        return RuntimeStatus(
            frame_index=self.frame_index,
            state=state,
            dominant_cause=cause,
            reason=reason,
            timestamp_sec=time.time(),
            stage_ms={
                "preprocess": pre,
                "inference": inf,
                "postprocess": post,
                "total": total,
                **({"wall": speed["wall"]} if "wall" in speed else {}),
            },
            latency_ms={
                "current": total,
                "p50": rolling["total_p50"],
                "p95": rolling["total_p95"],
                "p99": rolling["total_p99"],
                "max": rolling["total_max"],
                "tail_coeff_p95_p50": rolling["tail_coeff_p95_p50"],
                "tail_coeff_p99_p50": rolling["tail_coeff_p99_p50"],
                "current_tail_coeff_p50": residuals.get("current_tail_coeff_p50", 1.0),
            },
            stage_ratio=stage_ratio,
            output_pressure={
                "box_count": int(box_count),
                "box_pressure_coeff": float(box_pressure_coeff),
                "confidence_mean": conf_mean,
                "confidence_min": conf_min,
                "confidence_max": conf_max,
                "confidence_entropy": confidence_entropy,
                "class_count": int(len(set(classes))) if classes else 0,
                "class_entropy": class_entropy,
            },
            residuals=residuals,
            low_postprocess_path=low_postprocess_path,
            enough_samples=enough,
            window_size=len(self.window),
            raw_speed=dict(speed),
        )

    @staticmethod
    def _current_only_stats(*, total_ms: float, postprocess_ms: float, box_count: int) -> Dict[str, float]:
        return {
            "total_p50": float(total_ms),
            "total_p95": float(total_ms),
            "total_p99": float(total_ms),
            "total_max": float(total_ms),
            "tail_coeff_p95_p50": 1.0,
            "tail_coeff_p99_p50": 1.0,
            "postprocess_median": float(postprocess_ms),
            "postprocess_p95": float(postprocess_ms),
            "box_median": float(box_count),
            "box_p95": float(box_count),
        }

    def _store_status(self, status: RuntimeStatus) -> None:
        self.last_status = status
        self.records.append(status)

    def _extract_output_pressure(self, result: Any) -> Tuple[int, List[float], List[int]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return 0, [], []

        try:
            box_count = int(len(boxes))
        except Exception:
            box_count = 0

        confidences: List[float] = []
        classes: List[int] = []

        conf = getattr(boxes, "conf", None)
        if conf is not None:
            confidences = self._to_float_list(conf)

        cls = getattr(boxes, "cls", None)
        if cls is not None:
            classes = [int(x) for x in self._to_float_list(cls)]

        return box_count, confidences, classes

    @staticmethod
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

    def export_json(self, path: str) -> None:
        JsonExporter.export(path, self.records)

    def export_csv(self, path: str) -> None:
        CsvExporter.export(path, self.records)

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"frames": 0, "state_counts": {}, "dominant_causes": {}}

        state_counts: Dict[str, int] = {}
        cause_counts: Dict[str, int] = {}
        for r in self.records:
            state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1
            cause_counts[r.dominant_cause] = cause_counts.get(r.dominant_cause, 0) + 1

        latest = self.records[-1]
        return {
            "frames": len(self.records),
            "state_counts": state_counts,
            "dominant_causes": cause_counts,
            "latest": latest.to_dict(),
        }
