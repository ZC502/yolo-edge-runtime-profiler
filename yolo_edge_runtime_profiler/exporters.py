from __future__ import annotations

from typing import Iterable
import csv
import json
from pathlib import Path

from .status import RuntimeStatus


class JsonExporter:
    @staticmethod
    def export(path: str, records: Iterable[RuntimeStatus]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        rows = [r.to_dict() for r in records]
        payload = {
            "schema": "yolo-edge-runtime-profiler.v0.1",
            "frame_count": len(rows),
            "records": rows,
        }

        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class CsvExporter:
    FIELDNAMES = [
        "frame_index",
        "timestamp_sec",
        "state",
        "dominant_cause",
        "reason",
        "total_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "tail_coeff_p95_p50",
        "tail_coeff_p99_p50",
        "postprocess_ratio",
        "postprocess_spike_coeff",
        "box_count",
        "box_pressure_coeff",
        "confidence_mean",
        "confidence_entropy",
        "class_count",
        "class_entropy",
        "low_postprocess_path",
        "enough_samples",
        "window_size",
    ]

    @staticmethod
    def export(path: str, records: Iterable[RuntimeStatus]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CsvExporter.FIELDNAMES)
            writer.writeheader()
            for r in records:
                writer.writerow(CsvExporter._row(r))

    @staticmethod
    def _row(r: RuntimeStatus) -> dict:
        return {
            "frame_index": r.frame_index,
            "timestamp_sec": f"{r.timestamp_sec:.6f}",
            "state": r.state.value,
            "dominant_cause": r.dominant_cause,
            "reason": r.reason,
            "total_ms": f"{r.stage_ms.get('total', 0.0):.6f}",
            "preprocess_ms": f"{r.stage_ms.get('preprocess', 0.0):.6f}",
            "inference_ms": f"{r.stage_ms.get('inference', 0.0):.6f}",
            "postprocess_ms": f"{r.stage_ms.get('postprocess', 0.0):.6f}",
            "p50_ms": f"{r.latency_ms.get('p50', 0.0):.6f}",
            "p95_ms": f"{r.latency_ms.get('p95', 0.0):.6f}",
            "p99_ms": f"{r.latency_ms.get('p99', 0.0):.6f}",
            "max_ms": f"{r.latency_ms.get('max', 0.0):.6f}",
            "tail_coeff_p95_p50": f"{r.latency_ms.get('tail_coeff_p95_p50', 0.0):.6f}",
            "tail_coeff_p99_p50": f"{r.latency_ms.get('tail_coeff_p99_p50', 0.0):.6f}",
            "postprocess_ratio": f"{r.stage_ratio.get('postprocess', 0.0):.6f}",
            "postprocess_spike_coeff": f"{r.residuals.get('postprocess_spike_coeff', 0.0):.6f}",
            "box_count": r.output_pressure.get("box_count", 0),
            "box_pressure_coeff": f"{r.output_pressure.get('box_pressure_coeff', 0.0):.6f}",
            "confidence_mean": f"{r.output_pressure.get('confidence_mean', 0.0):.6f}",
            "confidence_entropy": f"{r.output_pressure.get('confidence_entropy', 0.0):.6f}",
            "class_count": r.output_pressure.get("class_count", 0),
            "class_entropy": f"{r.output_pressure.get('class_entropy', 0.0):.6f}",
            "low_postprocess_path": r.low_postprocess_path,
            "enough_samples": r.enough_samples,
            "window_size": r.window_size,
        }
