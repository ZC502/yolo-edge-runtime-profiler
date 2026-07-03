from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict
import json


class RuntimeState(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


_STATE_ORDER = {
    RuntimeState.GREEN: 0,
    RuntimeState.YELLOW: 1,
    RuntimeState.RED: 2,
}


def max_state(a: RuntimeState, b: RuntimeState) -> RuntimeState:
    return a if _STATE_ORDER[a] >= _STATE_ORDER[b] else b


@dataclass
class RuntimeStatus:
    frame_index: int
    state: RuntimeState
    dominant_cause: str
    reason: str
    timestamp_sec: float

    stage_ms: Dict[str, float] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    stage_ratio: Dict[str, float] = field(default_factory=dict)
    output_pressure: Dict[str, Any] = field(default_factory=dict)
    residuals: Dict[str, float] = field(default_factory=dict)

    low_postprocess_path: bool = False
    enough_samples: bool = False
    window_size: int = 0
    raw_speed: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def to_compact_string(self, color: bool = False) -> str:
        state = self.state.value
        if color:
            if self.state == RuntimeState.GREEN:
                state = f"\033[92m{state}\033[0m"
            elif self.state == RuntimeState.YELLOW:
                state = f"\033[93m{state}\033[0m"
            else:
                state = f"\033[91m{state}\033[0m"

        cur = self.latency_ms.get("current", 0.0)
        p50 = self.latency_ms.get("p50", 0.0)
        p95 = self.latency_ms.get("p95", 0.0)
        tail = self.latency_ms.get("tail_coeff_p95_p50", 1.0)
        post = self.stage_ms.get("postprocess", 0.0)
        boxes = self.output_pressure.get("box_count", 0)

        return (
            f"[{state}] frame={self.frame_index} "
            f"current={cur:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms "
            f"tail={tail:.2f}x post={post:.2f}ms boxes={boxes} "
            f"cause={self.dominant_cause}"
        )
