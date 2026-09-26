"""ارزیابی مصنوعی (بدون ویدیو) — سنجش دقت تخمین زاویه و طبقه‌بندی.

صدها چهره مصنوعی با زاویه‌های معلوم (Ground Truth) ساخته و از
کل زنجیره (SolvePnP → هموارسازی → طبقه‌بندی) عبور داده می‌شوند؛
خروجی، متریک‌های Accuracy/F1/FPR و خطای زاویه‌ای است که در
گزارش پروژه استفاده می‌شود.

Usage:
    python scripts/synthetic_eval.py [--n 300] [--report telemetry/synthetic_report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import make_synthetic_face  # noqa: E402

from dms.attention import AttentionClassifier  # noqa: E402
from dms.config import DMSConfig, default_camera_matrix  # noqa: E402
from dms.head_pose import HeadPoseEstimator  # noqa: E402
from dms.telemetry import evaluate  # noqa: E402


def true_state(pitch: float, yaw: float, cfg) -> str:
    """برچسب واقعی مطابق قواعد آستانه‌ای."""
    if abs(yaw) > cfg.head_pose.yaw_threshold:
        return "RIGHT" if yaw > 0 else "LEFT"
    if pitch < cfg.head_pose.pitch_down_threshold:
        return "DOWN"
    if pitch > cfg.head_pose.pitch_up_threshold:
        return "UP"
    return "FORWARD"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="تعداد نمونه")
    parser.add_argument("--report", default="telemetry/synthetic_report.json")
    args = parser.parse_args(argv)

    cfg = DMSConfig()
    rng = np.random.default_rng(2024)
    camera = default_camera_matrix((640, 480))
    estimator = HeadPoseEstimator(camera_matrix=camera,
                                  use_extrinsic_guess=False)
    classifier = AttentionClassifier(cfg.head_pose)

    y_true, y_pred = [], []
    errors = {"pitch": [], "yaw": [], "roll": []}

    for _ in range(args.n):
        pitch = float(rng.uniform(-35, 35))
        yaw = float(rng.uniform(-45, 45))
        roll = float(rng.uniform(-25, 25))
        face = make_synthetic_face(pitch, yaw, roll, ear=0.30)

        pose = estimator.estimate(face[list((1, 199, 33, 263, 61, 291)), :2],
                                  (640, 480))
        if not pose.success:
            continue
        errors["pitch"].append(abs(pose.angles.pitch - pitch))
        errors["yaw"].append(abs(pose.angles.yaw - yaw))
        errors["roll"].append(abs(pose.angles.roll - roll))

        attention = classifier.classify(pose.angles, face)
        y_pred.append(attention.state.value)
        y_true.append(true_state(pitch, yaw, cfg))

    metrics = evaluate(y_true, y_pred)
    metrics["angle_error_deg_mean"] = {
        axis: round(statistics.mean(vals), 3)
        for axis, vals in errors.items()
    }
    metrics["angle_error_deg_max"] = {
        axis: round(max(vals), 3) for axis, vals in errors.items()
    }
    metrics["samples"] = len(y_true)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nreport saved: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
