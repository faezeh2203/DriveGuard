"""تنظیمات سیستم (Configuration)"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Tuple


class ConfigError(ValueError):
    """مقدار نامعتبر در پیکربندی."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class HeadPoseConfig:
    """آستانه‌های زاویه‌ای برای طبقه‌بندی وضعیت سر."""

    yaw_threshold: float = 25.0
    pitch_down_threshold: float = -15.0
    pitch_up_threshold: float = 20.0
    roll_threshold: float = 30.0


@dataclass(frozen=True)
class DetectionConfig:
    """پارامترهای تشخیص چهره با MediaPipe Face Landmarker."""

    model_path: str = "assets/face_landmarker.task"
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_lost_frames: int = 10
    process_width: int = 480


@dataclass(frozen=True)
class DrowsinessConfig:
    """پارامترهای تشخیص خواب‌آلودگی."""

    ear_closed_threshold: float = 0.20
    ear_blink_frames: int = 3
    closed_seconds: float = 2.0
    nod_pitch_drop: float = 12.0
    nod_count: int = 3
    nod_window_seconds: float = 10.0
    nodding_alert: bool = True
    perclos_window_seconds: float = 60.0
    perclos_min_seconds: float = 20.0
    perclos_threshold: float = 0.30
    recovery_seconds: float = 1.5

    def __post_init__(self) -> None:
        _check(0.0 < self.ear_closed_threshold < 1.0, "ear_closed_threshold must be in (0, 1)")
        _check(self.closed_seconds > 0, "closed_seconds must be > 0")
        _check(self.nod_count >= 1, "nod_count must be >= 1")
        _check(0.0 < self.perclos_threshold <= 1.0, "perclos_threshold must be in (0, 1]")
        _check(self.recovery_seconds >= 0, "recovery_seconds must be >= 0")


@dataclass(frozen=True)
class DecisionConfig:
    """پارامترهای موتور تصمیم و هشدار."""

    state_confirm_frames: int = 5
    distraction_alert_seconds: float = 2.0
    phone_alert_seconds: float = 1.0
    lateral_alert_seconds: float = 5.0
    lateral_alert_enabled: bool = True
    drowsiness_alert: bool = True
    recovery_seconds: float = 1.0
    escalation_seconds: float = 6.0
    alert_cooldown_seconds: float = 3.0
    assumed_fps: float = 30.0

    def __post_init__(self) -> None:
        _check(self.state_confirm_frames >= 1, "state_confirm_frames must be >= 1")
        for name in ("distraction_alert_seconds", "phone_alert_seconds",
                     "lateral_alert_seconds", "recovery_seconds",
                     "alert_cooldown_seconds"):
            _check(getattr(self, name) >= 0, f"{name} must be >= 0")
        _check(self.escalation_seconds > 0, "escalation_seconds must be > 0")


@dataclass(frozen=True)
class SmoothingConfig:
    """پارامترهای هموارسازی زمانی زوایای سر."""

    use_kalman: bool = True
    process_noise: float = 1.0
    measurement_noise: float = 4.0
    ma_window: int = 5


@dataclass(frozen=True)
class TelemetryConfig:
    """پارامترهای ثبت داده (Telemetry)."""

    db_path: str = "telemetry/dms_log.db"
    export_dir: str = "telemetry"
    log_every_frame: bool = True
    camera_id: str = "webcam-0"


@dataclass(frozen=True)
class PhoneConfig:
    """پارامترهای تشخیص تلفن همراه (تماس دستی / گوشی در دست)."""

    enabled: bool = True
    model_path: str = "assets/yolo11n.onnx"
    score_threshold: float = 0.4
    nms_threshold: float = 0.45
    input_size: int = 320
    max_results: int = 5
    near_face_expand: float = 0.6
    detect_every: int = 15
    hold_frames: int = 15
    process_width: int = 480

    def __post_init__(self) -> None:
        _check(self.detect_every >= 1, "phone.detect_every must be >= 1")
        _check(self.hold_frames >= 0, "phone.hold_frames must be >= 0")
        _check(self.input_size % 32 == 0, "phone.input_size must be a multiple of 32")


@dataclass(frozen=True)
class AlertConfig:
    """پارامترهای هشدار صوتی (آژیر + هشدار گفتاری فارسی)."""

    sound_enabled: bool = True
    warning_frequency: int = 1200
    critical_frequency: int = 1800
    warning_interval: float = 1.0
    critical_interval: float = 0.5
    beep_ms: int = 350
    volume: float = 1.0
    voice_enabled: bool = True
    voice_dir: str = "assets/voice"
    voice_repeat_seconds: float = 5.0
    voice_on_clear: bool = True

    def __post_init__(self) -> None:
        for name in ("warning_frequency", "critical_frequency"):
            _check(37 <= getattr(self, name) <= 32767, f"alert.{name} must be in 37..32767 Hz")
        _check(self.warning_interval > 0 and self.critical_interval > 0,
               "alert intervals must be > 0")
        _check(0.0 <= self.volume <= 1.0, "alert.volume must be in [0, 1]")
        _check(self.voice_repeat_seconds > 0, "alert.voice_repeat_seconds must be > 0")


@dataclass(frozen=True)
class DMSConfig:
    """پیکربندی سراسری سیستم تشخیص حواس‌پرتی راننده."""

    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    target_fps: int = 30

    @classmethod
    def from_json(cls, path: str | Path) -> "DMSConfig":
        """ساخت پیکربندی از فایل JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "DMSConfig":
        """ساخت پیکربندی از دیکشنری (کلیدهای غایب = مقدار پیش‌فرض)."""
        if not isinstance(raw, dict):
            raise ConfigError("config root must be a JSON object")
        cfg = cls()
        valid = {f.name for f in fields(cls)}
        updates = {}
        for key, value in raw.items():
            if key not in valid:
                raise ConfigError(f"unknown config section: {key!r}")
            sub = getattr(cfg, key)
            if hasattr(sub, "__dataclass_fields__"):
                if not isinstance(value, dict):
                    raise ConfigError(f"section {key!r} must be an object")
                sub_fields = {f.name for f in fields(sub)}
                unknown = set(value) - sub_fields
                if unknown:
                    raise ConfigError(
                        f"unknown keys in {key!r}: {', '.join(sorted(unknown))}")
                updates[key] = type(sub)(**{**asdict(sub), **value})
            else:
                updates[key] = value
        return cls(**updates)

    def to_dict(self) -> dict:
        """تبدیل کامل پیکربندی به دیکشنری (برای ثبت در گزارش‌ها)."""
        return asdict(self)


def default_camera_matrix(image_size: Tuple[int, int]) -> "object":
    """ساخت ماتریس دوربین تقریبی برای SolvePnP."""
    import numpy as np

    width, height = image_size
    focal = width  # تقریب رایج: f ≈ عرض تصویر برای FOV ≈ 53°
    center = (width / 2.0, height / 2.0)
    return np.array(
        [[focal, 0, center[0]],
         [0, focal, center[1]],
         [0, 0, 1]],
        dtype=np.float64,
    )
