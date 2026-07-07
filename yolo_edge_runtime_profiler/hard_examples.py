from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import json
import re
import time

import numpy as np

from .status import RuntimeState, RuntimeStatus


DEFAULT_TRIGGER_CAUSES: Tuple[str, ...] = (
    "TAIL_LATENCY_SPIKE",
    "POSTPROCESS_SPIKE",
    "POSTPROCESS_DOMINANT",
)


@dataclass
class CaptureResult:
    saved: bool
    reason: str
    image_path: str = ""
    metadata_path: str = ""
    image_saved: bool = False
    metadata_saved: bool = False


@dataclass
class LocalHardExampleConfig:
    """
    Local-first hard-example capture policy.

    Defaults are intentionally conservative:
      - RED only
      - meaningful causes only
      - minimum box count
      - cooldown
      - item limit

    This prevents sparse-scene micro-jitter from creating low-value captures.
    """

    output_dir: str = "hard_examples"
    enabled: bool = True

    selection_mode: str = "pressure_or_confidence"

    trigger_states: Tuple[str, ...] = ("RED",)
    trigger_causes: Tuple[str, ...] = DEFAULT_TRIGGER_CAUSES

    min_confidence_entropy: Optional[float] = None
    max_confidence_mean: Optional[float] = None

    min_box_count: int = 5
    min_box_count_for_pressure: int = 5

    cooldown_sec: float = 2.0
    max_items: int = 50
    require_enough_samples: bool = True

    image_ext: str = "jpg"
    save_image: bool = True
    save_metadata: bool = True


class LocalHardExampleRecorder:
    """
    Local-first hard-example recorder.

    It saves:
      - current frame image, when available
      - sidecar JSON metadata with full RuntimeStatus and capture policy

    It never uploads data.
    """

    def __init__(
        self,
        output_dir: str = "hard_examples",
        *,
        config: Optional[LocalHardExampleConfig] = None,
        enabled: bool = True,
        selection_mode: str = "pressure_or_confidence",
        trigger_states: Sequence[str] = ("RED",),
        trigger_causes: Optional[Sequence[str]] = None,
        min_confidence_entropy: Optional[float] = None,
        max_confidence_mean: Optional[float] = None,
        min_box_count: int = 5,
        min_box_count_for_pressure: Optional[int] = None,
        cooldown_sec: float = 2.0,
        max_items: int = 50,
    ) -> None:
        if config is None:
            config = LocalHardExampleConfig(
                output_dir=output_dir,
                enabled=enabled,
                selection_mode=selection_mode,
                trigger_states=tuple(trigger_states),
                trigger_causes=tuple(trigger_causes) if trigger_causes is not None else DEFAULT_TRIGGER_CAUSES,
                min_confidence_entropy=min_confidence_entropy,
                max_confidence_mean=max_confidence_mean,
                min_box_count=int(min_box_count),
                min_box_count_for_pressure=int(
                    min_box_count_for_pressure if min_box_count_for_pressure is not None else min_box_count
                ),
                cooldown_sec=float(cooldown_sec),
                max_items=int(max_items),
            )

        self.config = config
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0
        self.last_capture_time = 0.0

    def maybe_save(
        self,
        *,
        frame: Any,
        status: RuntimeStatus,
        source_info: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> CaptureResult:
        if not self.config.enabled:
            return CaptureResult(saved=False, reason="recorder_disabled")

        if self.saved_count >= self.config.max_items:
            return CaptureResult(saved=False, reason="max_items_reached")

        if self.config.require_enough_samples and not status.enough_samples:
            return CaptureResult(saved=False, reason="waiting_for_baseline_window")

        now = time.time()
        if (now - self.last_capture_time) < self.config.cooldown_sec:
            return CaptureResult(saved=False, reason="cooldown")

        selection = self._selection_decision(status)
        if not selection["selected"]:
            return CaptureResult(saved=False, reason=selection["reason"])

        ts_ms = int(now * 1000)
        state = _state_value(status.state)
        cause = _sanitize(status.dominant_cause or "UNKNOWN")
        stem = f"frame_{status.frame_index:06d}_{ts_ms}_{state}_{cause}"

        image_path = self.output_dir / f"{stem}.{self.config.image_ext.lstrip('.')}"
        meta_path = self.output_dir / f"{stem}.json"

        image_saved = False
        image_error = ""

        if self.config.save_image and frame is not None:
            try:
                image_saved = _write_image(image_path, frame)
                if not image_saved:
                    image_error = "no_available_image_writer"
            except Exception as e:
                image_error = f"{type(e).__name__}: {e}"
                image_saved = False

        metadata_saved = False
        if self.config.save_metadata:
            payload = self._metadata_payload(
                status=status,
                selection=selection,
                image_path=str(image_path) if image_saved else "",
                image_saved=image_saved,
                image_error=image_error,
                source_info=source_info or {},
                extra_metadata=extra_metadata or {},
            )
            meta_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            metadata_saved = True

        self.saved_count += 1
        self.last_capture_time = now

        return CaptureResult(
            saved=True,
            reason=selection["reason"],
            image_path=str(image_path) if image_saved else "",
            metadata_path=str(meta_path) if metadata_saved else "",
            image_saved=image_saved,
            metadata_saved=metadata_saved,
        )

    def _selection_decision(self, status: RuntimeStatus) -> Dict[str, Any]:
        pressure = self._pressure_signal(status)
        confidence = self._confidence_signal(status)
        mode = self.config.selection_mode.strip().lower()

        if mode == "pressure_only":
            selected = pressure["active"]
        elif mode == "confidence_only":
            selected = confidence["active"]
        elif mode == "pressure_and_confidence":
            selected = pressure["active"] and confidence["active"]
        else:
            selected = pressure["active"] or confidence["active"]

        if selected:
            if pressure["active"] and confidence["active"]:
                reason = "runtime_pressure_and_confidence_uncertainty"
            elif pressure["active"]:
                reason = "runtime_pressure"
            else:
                reason = "confidence_uncertainty"
        else:
            reason = "no_capture_trigger"

        return {
            "selected": bool(selected),
            "reason": reason,
            "selection_mode": self.config.selection_mode,
            "pressure_signal": pressure,
            "confidence_signal": confidence,
        }

    def _pressure_signal(self, status: RuntimeStatus) -> Dict[str, Any]:
        state = _state_value(status.state)
        cause = str(status.dominant_cause or "")
        box_count = int(status.output_pressure.get("box_count", 0) or 0)

        state_match = state in set(self.config.trigger_states)
        cause_match = cause in set(self.config.trigger_causes)
        box_count_match = box_count >= int(self.config.min_box_count_for_pressure)

        active = state_match and cause_match and box_count_match

        return {
            "active": bool(active),
            "state": state,
            "dominant_cause": cause,
            "box_count": box_count,
            "state_match": bool(state_match),
            "cause_match": bool(cause_match),
            "box_count_match": bool(box_count_match),
            "min_box_count_for_pressure": int(self.config.min_box_count_for_pressure),
            "trigger_states": list(self.config.trigger_states),
            "trigger_causes": list(self.config.trigger_causes),
        }

    def _confidence_signal(self, status: RuntimeStatus) -> Dict[str, Any]:
        box_count = int(status.output_pressure.get("box_count", 0) or 0)
        conf_entropy = float(status.output_pressure.get("confidence_entropy", 0.0) or 0.0)
        conf_mean = float(status.output_pressure.get("confidence_mean", 0.0) or 0.0)

        box_count_match = box_count >= int(self.config.min_box_count)

        entropy_active = False
        mean_active = False

        if box_count_match:
            if self.config.min_confidence_entropy is not None:
                entropy_active = conf_entropy >= float(self.config.min_confidence_entropy)
            if self.config.max_confidence_mean is not None:
                mean_active = conf_mean <= float(self.config.max_confidence_mean)

        return {
            "active": bool(entropy_active or mean_active),
            "box_count": box_count,
            "confidence_entropy": conf_entropy,
            "confidence_mean": conf_mean,
            "min_confidence_entropy": self.config.min_confidence_entropy,
            "max_confidence_mean": self.config.max_confidence_mean,
            "min_box_count": int(self.config.min_box_count),
            "box_count_match": bool(box_count_match),
            "entropy_active": bool(entropy_active),
            "mean_active": bool(mean_active),
        }

    def _metadata_payload(
        self,
        *,
        status: RuntimeStatus,
        selection: Dict[str, Any],
        image_path: str,
        image_saved: bool,
        image_error: str,
        source_info: Dict[str, Any],
        extra_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema": "yolo-edge-runtime-profiler.hard-example.v0.1",
            "created_timestamp_sec": time.time(),
            "capture": {
                "image_saved": image_saved,
                "image_path": image_path,
                "image_error": image_error,
                "metadata_only": not image_saved,
            },
            "selection": selection,
            "status": status.to_dict(),
            "capture_policy": asdict(self.config),
            "source_info": source_info,
            "extra_metadata": extra_metadata,
            "privacy_note": "Local-first capture. No upload was performed by YERP.",
        }


def _state_value(state: Any) -> str:
    if isinstance(state, RuntimeState):
        return state.value
    return str(state)


def _sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value[:80] or "UNKNOWN"


def _write_image(path: Path, frame: Any) -> bool:
    arr = np.asarray(frame)
    if arr.size == 0:
        return False

    try:
        import cv2  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(cv2.imwrite(str(path), arr))
        if ok:
            return True
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        arr2 = np.asarray(arr)
        if arr2.dtype != np.uint8:
            arr2 = np.clip(arr2, 0, 255).astype(np.uint8)

        if arr2.ndim == 2:
            img = Image.fromarray(arr2)
        elif arr2.ndim == 3 and arr2.shape[2] in (3, 4):
            img = Image.fromarray(arr2)
        else:
            return False

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        return True
    except Exception:
        return False
