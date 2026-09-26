"""
Fixtureهای مشترک تست‌ها
=========================

برای تست کردن زنجیره بدون وابستگی به مدل‌های سنگین (MediaPipe) و
وبکم واقعی، از **داده مصنوعی** استفاده می‌کنیم:

1. ``synthetic_landmarks``: ساخت لندمارک‌های 478 نقطه‌ای مصنوعی
   با پروجکشن مدل سه‌بعدی صورت در زاویه معلوم (Ground Truth دقیق).
2. ``fake_detector``: جایگزین (Stub) تشخیص‌دهنده چهره که
   لندمارک‌های از پیش ساخته‌شده را برمی‌گرداند.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# اطمینان از اینکه پکیج dms از ریشه پروژه import می‌شود
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dms.config import DMSConfig, default_camera_matrix
from dms.geometry import GENERIC_3D_FACE_MODEL, project_face_model
from dms.geometry import LEFT_EYE_EAR_IDS, RIGHT_EYE_EAR_IDS


def make_synthetic_face(pitch: float, yaw: float, roll: float,
                        size: tuple = (640, 480),
                        ear: float = 0.30) -> np.ndarray:
    """ساخت لندمارک‌های مصنوعی 478 نقطه‌ای یک چهره در وضعیت معلوم.

    شش نقطه استاندارد با پروجکشن پرسپکتیو مدل سه‌بعدی ساخته می‌شوند
    (زاویه دقیقاً معلوم = Ground Truth)؛ پروجکشن با همان ماتریس
    دوربین پیش‌فرض تخمین‌گر انجام می‌شود تا زاویه بازیابی‌شده با
    Ground Truth منطبق باشد. نقاط باقی‌مانده به‌گونه‌ای پر می‌شوند
    که EAR دو چشم برابر مقدار درخواستی باشد (برای تست خواب‌آلودگی).

    Returns:
        آرایه (478, 3) مختصات پیکسلی لندمارک‌ها.
    """
    w, h = size
    pts2d = project_face_model(pitch, yaw, roll, translation=(0, 0, 500),
                               camera_matrix=default_camera_matrix(size))
    landmarks = np.zeros((478, 3), dtype=np.float64)

    # نقاط استاندارد: نوک بینی(1)، چانه(199)، چشم(33,263)، دهان(61,291)
    model_ids = (1, 199, 33, 263, 61, 291)
    for i, mid in enumerate(model_ids):
        landmarks[mid, :2] = pts2d[i]
        landmarks[mid, 2] = 0.0

    # میانگین مقیاس برای جایگذاری سایر نقاط نسبت به مرکز صورت
    center = pts2d.mean(axis=0)
    rng = np.random.default_rng(42)
    for idx in range(478):
        if idx in model_ids:
            continue
        landmarks[idx, :2] = center + rng.uniform(-80, 80, 2)
        landmarks[idx, 2] = 0.0

    # چیدمان نقاط چشم با EAR معلوم؛ لندمارک‌های لنگر مدل (33 و 263)
    # دست‌نخورده می‌مانند تا تناظر 2D-3D تخمین‌گر خراب نشود:
    _fill_eye(landmarks, LEFT_EYE_EAR_IDS, ear, eye_w=60, anchor_pos=0, direction=+1)
    _fill_eye(landmarks, RIGHT_EYE_EAR_IDS, ear, eye_w=60, anchor_pos=3, direction=-1)

    # عنبیه در مرکز هر چشم (نگاه رو به جلو) — ۵ نقطه هر عنبیه هم‌مکان
    left_center = (landmarks[33, :2] + landmarks[133, :2]) / 2.0
    right_center = (landmarks[362, :2] + landmarks[263, :2]) / 2.0
    for idx in range(468, 473):
        landmarks[idx, :2] = left_center
    for idx in range(473, 478):
        landmarks[idx, :2] = right_center
    return landmarks


def _fill_eye(landmarks: np.ndarray, ids: tuple, ear: float,
              eye_w: float, anchor_pos: int, direction: float) -> None:
    """چیدمان شش نقطه چشم برای EAR معلوم.

    الگوی استاندارد EAR (Soukupová & Čech): جفت افقی ``ids[0]``/``ids[3]``
    و چهار نقطه پلک بالا/پایین. ``anchor_pos`` جایگاه لندمارک لنگر مدل
    است (۰ برای چشم چپ، ۳ برای چشم راست) که مقدار فعلی آن حفظ می‌شود.
    """
    anchor_idx = ids[anchor_pos]
    base = landmarks[anchor_idx, :2].copy()
    other_idx = ids[3 - anchor_pos]
    other = base + np.array([direction * eye_w, 0.0])

    mid = (base + other) / 2.0
    v = ear * eye_w  # فاصله عمودی هر جفت پلک
    upper = mid + np.array([0.0, -v / 2])
    lower = mid + np.array([0.0, +v / 2])

    landmarks[other_idx, :2] = other
    landmarks[ids[1], :2] = upper
    landmarks[ids[2], :2] = upper
    landmarks[ids[4], :2] = lower
    landmarks[ids[5], :2] = lower


class FakeDetector:
    """جایگزین تستی :class:`dms.face_detector.FaceDetector`.

    زوایای سر و EAR فریم‌های بعدی را از پیش دریافت می‌کند و
    لندمارک مصنوعی متناظر را برمی‌گرداند؛ رفتار «چهره دیده نشدن»
    نیز با ``visible=False`` شبیه‌سازی می‌شود.
    """

    def __init__(self) -> None:
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.ear = 0.30
        self.visible = True
        self._frame_idx = 0

    def set_state(self, pitch=0.0, yaw=0.0, roll=0.0,
                  ear=0.30, visible=True) -> None:
        """تنظیم وضعیت چهره در فریم‌های بعدی."""
        self.pitch, self.yaw, self.roll = pitch, yaw, roll
        self.ear, self.visible = ear, visible

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        from dms.face_detector import FaceObservation

        self._frame_idx += 1
        if not self.visible:
            return None
        landmarks = make_synthetic_face(self.pitch, self.yaw, self.roll,
                                        ear=self.ear)
        x_min, y_min = landmarks[:, 0].min(), landmarks[:, 1].min()
        x_max, y_max = landmarks[:, 0].max(), landmarks[:, 1].max()
        return FaceObservation(
            landmarks=landmarks,
            bbox=(float(x_min), float(y_min),
                  float(x_max - x_min), float(y_max - y_min)),
            landmark_count=478,
        )

    def close(self) -> None:
        pass


@pytest.fixture
def fake_detector() -> FakeDetector:
    """تشخیص‌دهنده مصنوعی قابل تنظیم برای تست‌های پایپ‌لاین."""
    return FakeDetector()


class FakePhoneDetector:
    """جایگزین تستی :class:`dms.phone_detector.PhoneDetector`.

    حالت ``near`` گوشی را دقیقاً روی مرکز چهره (نزدیک) و حالت
    ``far`` آن را دور از چهره (دو برابر عرض چهره به سمت راست)
    قرار می‌دهد تا قاعده «نزدیکی به چهره» مستقل از هندسه چهره
    مصنوعی تست شود.
    """

    def __init__(self) -> None:
        self.mode = "none"  # none | near | far

    def set_mode(self, mode: str) -> None:
        """تنظیم حالت گوشی در فریم‌های بعدی."""
        assert mode in ("none", "near", "far")
        self.mode = mode

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int,
                face_bbox: tuple | None = None) -> list:
        """بازگرداندن جعبه گوشی (مصنوعی) بر اساس حالت فعلی."""
        if self.mode == "none" or face_bbox is None:
            return []
        fx, fy, fw, fh = face_bbox
        w, h = 60.0, 120.0
        if self.mode == "near":
            cx, cy = fx + fw / 2.0, fy + fh / 2.0
        else:  # far — خارج از ناحیه بزرگ‌شده چهره
            cx, cy = fx + fw * 2.0, fy + fh / 2.0
        return [(cx - w / 2.0, cy - h / 2.0, w, h)]

    def close(self) -> None:
        pass


@pytest.fixture
def fake_phone_detector() -> FakePhoneDetector:
    """تشخیص‌دهنده گوشی مصنوعی برای تست‌های پایپ‌لاین."""
    return FakePhoneDetector()


@pytest.fixture
def config(tmp_path) -> DMSConfig:
    """پیکربندی تست: بدون کالمن/با آستانه‌های مرجع و Telemetry موقت."""
    from dataclasses import replace

    cfg = DMSConfig()
    cfg = replace(
        cfg,
        telemetry=replace(cfg.telemetry, db_path=str(tmp_path / "test.db"),
                          export_dir=str(tmp_path)),
    )
    return cfg


@pytest.fixture
def pipeline(config, fake_detector):
    """پایپ‌لاین کامل با تشخیص‌دهنده مصنوعی و Telemetry موقت."""
    from dms.pipeline import DMSPipeline

    p = DMSPipeline(config=config, detector=fake_detector,
                    enable_sound=False)
    yield p
    p.close()
