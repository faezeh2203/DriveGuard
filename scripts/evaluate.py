"""ارزیابی سیستم روی ویدیوی برچسب‌خورده (Ground Truth).

ویدیوی ورودی باید با یک فایل برچسب CSV همراه باشد؛ هر سطر::

    start_sec,end_sec,state

که ``state`` یکی از FORWARD / LEFT / RIGHT / DOWN / UP / NO_FACE است.

خروجی: متریک‌های Accuracy، F1-score و False Positive Rate (طبق
مشخصات پروژه) به همراه Latency میانگین هر فریم.

Usage:
    python scripts/evaluate.py --video clip.mp4 --labels labels.csv \
        [--report telemetry/eval_report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dms.config import DMSConfig  # noqa: E402
from dms.pipeline import DMSPipeline  # noqa: E402
from dms.telemetry import evaluate  # noqa: E402
from dms.video import VideoSource  # noqa: E402


def load_labels(path: Path) -> list[tuple[float, float, str]]:
    """خواندن فایل برچسب CSV."""
    labels = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append((float(row["start_sec"]),
                           float(row["end_sec"]),
                           row["state"].strip().upper()))
    return labels


def ground_truth_at(labels, t: float) -> str:
    """برچسب واقعی در لحظه t (برچسب اولین بازه پوشش‌دهنده)."""
    for start, end, state in labels:
        if start <= t < end:
            return state
    return "FORWARD"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="مسیر ویدیوی آزمون")
    parser.add_argument("--labels", required=True, help="فایل CSV برچسب‌ها")
    parser.add_argument("--report", default="telemetry/eval_report.json",
                        help="مسیر خروجی گزارش JSON")
    parser.add_argument("--config", default=None, help="پیکربندی JSON اختیاری")
    args = parser.parse_args(argv)

    config = DMSConfig.from_json(args.config) if args.config else DMSConfig()
    labels = load_labels(Path(args.labels))

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []

    with DMSPipeline(config=config, enable_sound=False) as pipeline:
        with VideoSource(args.video) as source:
            for frame in source:
                result = pipeline.process(frame.image,
                                          int(frame.frame_id * 1000 / source.fps))
                y_pred.append(result.decision.stable_state.value)
                y_true.append(ground_truth_at(labels, frame.frame_id / source.fps))
                latencies.append(result.latency_ms)

    metrics = evaluate(y_true, y_pred)
    metrics["latency_ms_mean"] = round(statistics.mean(latencies), 2)
    metrics["latency_ms_p95"] = round(
        sorted(latencies)[int(0.95 * len(latencies))], 2)
    metrics["processed_frames"] = len(y_true)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nreport saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
