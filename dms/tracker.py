"""
ردیابی چهره (Face Tracking)
============================

ردیاب تک‌هدف الهام‌گرفته از **SORT**: هر تشخیص با فیلتر کالمنِ
جعبه محصور (مرکز + ابعاد) هموار می‌شود و در فریم‌هایی که تشخیص
ناموفق است (نویز لحظه‌ای، پلک زدن مدل تشخیص)، تا ``max_lost_frames``
فریم آخرین وضعیت «نگه داشته می‌شود». این مکانیزم، سقوط لحظه‌ای
زاویه‌ها و هشدارهای کاذب «No-face» را از بین می‌برد.
"""

from __future__ import annotations

from dataclasses import dataclass

from .smoothing import KalmanFilter1D


@dataclass
class TrackState:
    """وضعیت ردیابی هدف.

    Attributes:
        cx, cy: مرکز هموارشده جعبه محصور.
        w, h: عرض و ارتفاع هموارشده.
        lost_frames: تعداد فریم‌های متوالی بدون تشخیص.
        age: عمر ردیاب (تعداد فریم‌های پردازش‌شده).
    """

    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    lost_frames: int = 0
    age: int = 0

    @property
    def bbox(self) -> tuple:
        """جعبه محصور ``(x, y, w, h)``."""
        return (self.cx - self.w / 2.0, self.cy - self.h / 2.0, self.w, self.h)


class FaceTracker:
    """ردیاب چهره تک‌هدف با فیلتر کالمن و نگهداری هدف گم‌شده.

    Args:
        max_lost_frames: حداکثر فریم‌های نگهداری پس از گم شدن.
        process_noise: نویز فرایند کالمن روی مرکز جعبه.
        measurement_noise: نویز اندازه‌گیری کالمن.
    """

    def __init__(self, max_lost_frames: int = 10,
                 process_noise: float = 4.0,
                 measurement_noise: float = 9.0) -> None:
        self.max_lost_frames = max_lost_frames
        self._kf = {
            "cx": KalmanFilter1D(process_noise, measurement_noise),
            "cy": KalmanFilter1D(process_noise, measurement_noise),
            "w": KalmanFilter1D(process_noise * 0.5, measurement_noise),
            "h": KalmanFilter1D(process_noise * 0.5, measurement_noise),
        }
        self.state = TrackState()

    def reset(self) -> None:
        """حذف کامل هدف و بازنشانی فیلترها."""
        for kf in self._kf.values():
            kf.reset()
        self.state = TrackState()

    @property
    def is_tracking(self) -> bool:
        """True تا زمانی که هدف نگه داشته می‌شود (حتی اگر این فریم دیده نشده)."""
        return self.state.age > 0 and self.state.lost_frames < self.max_lost_frames

    def update(self, bbox: tuple | None, dt: float = 1.0 / 30.0) -> TrackState:
        """به‌روزرسانی ردیاب با جعبه تشخیص جدید (یا None).

        Args:
            bbox: جعبه ``(x, y, w, h)`` یا ``None`` اگر چهره دیده نشد.
            dt: گام زمانی.

        Returns:
            وضعیت جدید :class:`TrackState`.
        """
        if bbox is not None:
            self.state.age += 1
            x, y, w, h = bbox
            self.state.cx = self._kf["cx"].step(x + w / 2.0, dt)
            self.state.cy = self._kf["cy"].step(y + h / 2.0, dt)
            self.state.w = self._kf["w"].step(max(w, 1.0), dt)
            self.state.h = self._kf["h"].step(max(h, 1.0), dt)
            self.state.lost_frames = 0
            return self.state

        # بدون تشخیص:
        if self.state.age == 0:
            return self.state  # هنوز هدفی وجود ندارد
        if self.state.lost_frames + 1 >= self.max_lost_frames:
            self.reset()  # هدف واقعاً گم شده است
            return self.state
        # پیش‌بینی بدون اندازه‌گیری: حرکت با سرعت فعلی ادامه می‌یابد
        self.state.age += 1
        self.state.lost_frames += 1
        for kf in self._kf.values():
            kf.step(kf.value, dt)
        return self.state
