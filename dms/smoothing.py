"""
هموارسازی زمانی (Temporal Smoothing)
=====================================

سه ابزار کاهش نویز زمانی در این ماژول پیاده‌سازی شده است:

1. :class:`KalmanFilter1D` — فیلتر کالمن تک‌متغیره با مدل حرکت
   شبه-ثابت (Constant Velocity) برای هموارسازی هر یک از زوایای
   Pitch/Yaw/Roll (مطابق مشخصات پروژه: Kalman Filter).
2. :class:`MovingAverage` — میانگین متحرک پنجره‌ای به‌عنوان
   مکمل ساده و پشتیبان فیلتر کالمن.
3. :class:`HysteresisFilter` — هیسترزیس وضعیت: تغییر وضعیت
   طبقه‌بندی تنها پس از ``N`` فریم متوالی پذیرفته می‌شود تا
   نویز لحظه‌ای باعث هشدار کاذب (False Alarm) نشود.
"""

from __future__ import annotations

from collections import deque
from typing import Hashable, List, Optional


class KalmanFilter1D:
    """فیلتر کالمن تک‌متغیره با مدل سرعت ثابت.

    بردار حالت ``[position, velocity]`` و مدل پیش‌بینی زیر است::

        x[k+1] = F x[k] + w ,   F = [[1, dt], [0, 1]]

    نویز فرایند و اندازه‌گیری به‌صورت اسکالر (واریانس) داده می‌شوند
    و در ماتریس‌های Q و R به‌کار می‌روند. این پیاده‌سازی عمداً
    کوچک و مستقل از کتابخانه‌های خارجی نگه داشته شده است تا رفتار
    فیلتر در تست‌های واحد به‌طور قطعی (Deterministic) قابل بررسی باشد.

    Args:
        process_noise: واریانس نویز فرایند (نوسان واقعی سیگنال).
            مقدار بزرگ‌تر ⇒ فیلتر سریع‌تر دنبال می‌کند.
        measurement_noise: واریانس نویز اندازه‌گیری. مقدار بزرگ‌تر
            ⇒ خروجی هموارتر اما با تأخیر بیشتر.
        initial_value: مقدار اولیه موقعیت (None ⇒ 0).
    """

    def __init__(self, process_noise: float = 1.0,
                 measurement_noise: float = 4.0,
                 initial_value: float | None = None) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        # حالت: [x, v]
        self.state = (0.0, 0.0) if initial_value is None else (float(initial_value), 0.0)
        # کوواریانس خطای تخمین (2x2)
        self.covariance = ((10.0, 0.0), (0.0, 10.0))
        self._initialized = initial_value is not None

    def reset(self, value: float | None = None) -> None:
        """بازنشانی فیلتر (مثلاً پس از گم شدن چهره)."""
        self.state = (0.0, 0.0) if value is None else (float(value), 0.0)
        self.covariance = ((10.0, 0.0), (0.0, 10.0))
        self._initialized = value is not None

    @property
    def value(self) -> float:
        """آخرین تخمین موقعیت (خروجی فیلتر)."""
        return self.state[0]

    def step(self, measurement: float, dt: float = 1.0 / 30.0) -> float:
        """یک گام کامل پیش‌بینی + به‌روزرسانی کالمن.

        Args:
            measurement: اندازه‌گیری جدید (مثلاً yaw خام فریم فعلی).
            dt: فاصله زمانی بین دو گام (ثانیه).

        Returns:
            تخمین هموارشده موقعیت.
        """
        if not self._initialized:
            self.state = (float(measurement), 0.0)
            self.covariance = ((self.process_noise, 0.0), (0.0, 25.0))
            self._initialized = True
            return self.state[0]

        x, v = self.state
        p00, p01 = self.covariance[0]
        p10, p11 = self.covariance[1]

        # --- Predict ---
        x_pred = x + v * dt
        v_pred = v
        q = self.process_noise
        p00_pred = p00 + dt * (p10 + p01) + dt * dt * p11 + q
        p01_pred = p01 + dt * p11
        p10_pred = p10 + dt * p11
        p11_pred = p11 + q * 0.25

        # --- Update ---
        r = self.measurement_noise
        y = float(measurement) - x_pred          # innovation
        s = p00_pred + r                          # innovation covariance
        if s < 1e-12:
            k0, k1 = 0.0, 0.0
        else:
            k0 = p00_pred / s                     # Kalman gain (position)
            k1 = p10_pred / s                     # Kalman gain (velocity)

        x_new = x_pred + k0 * y
        v_new = v_pred + k1 * y
        self.state = (x_new, v_new)
        self.covariance = (
            ((1 - k0) * p00_pred, (1 - k0) * p01_pred),
            ((1 - k1) * p00_pred, (1 - k1) * p01_pred),
        )
        return x_new


class MovingAverage:
    """میانگین متحرک با پنجره ثابت (Simple Moving Average).

    Args:
        window: طول پنجره (تعداد نمونه). باید >= 1 باشد.

    Raises:
        ValueError: اگر ``window`` کمتر از 1 باشد.
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._buffer: deque = deque(maxlen=window)

    def reset(self) -> None:
        """خالی کردن پنجره."""
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        """True اگر پنجره پر شده باشد."""
        return len(self._buffer) == self._buffer.maxlen

    def update(self, value: float) -> float:
        """افزودن نمونه و بازگرداندن میانگین فعلی پنجره."""
        self._buffer.append(float(value))
        return sum(self._buffer) / len(self._buffer)


class HysteresisFilter:
    """فیلتر هیسترزیس وضعیت (State Debounce).

    یک وضعیت جدید تنها زمانی به‌عنوان وضعیت پایدار پذیرفته می‌شود که
    ``min_frames`` بار متوالی رخ داده باشد؛ در غیر این صورت آخرین
    وضعیت پایدار حفظ می‌شود. این همان «False Alarm با Hysteresis
    Filtering» ذکرشده در مشخصات پروژه است.

    Args:
        min_frames: حداقل تعداد فریم متوالی برای تأیید وضعیت جدید.
        initial: وضعیت اولیه (None ⇒ هنوز وضعیتی پایدار نشده).

    Raises:
        ValueError: اگر ``min_frames`` کمتر از 1 باشد.
    """

    def __init__(self, min_frames: int = 5,
                 initial: Optional[Hashable] = None) -> None:
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        self.min_frames = min_frames
        self.stable_state = initial
        self._candidate: Hashable = initial
        self._count = 0

    def reset(self, state: Optional[Hashable] = None) -> None:
        """بازنشانی به وضعیت پایدار مشخص (یا بدون وضعیت)."""
        self.stable_state = state
        self._candidate = state
        self._count = 0

    def update(self, raw_state: Hashable) -> Hashable:
        """ثبت وضعیت خام فریم جاری و بازگرداندن وضعیت پایدار.

        Args:
            raw_state: وضعیت طبقه‌بندی‌شده فریم جاری (خام، بدون هیسترزیس).

        Returns:
            وضعیت پایدار پس از اعمال هیسترزیس.
        """
        if raw_state == self._candidate:
            self._count += 1
        else:
            self._candidate = raw_state
            self._count = 1

        if self._count >= self.min_frames and raw_state != self.stable_state:
            self.stable_state = raw_state
        return self.stable_state

    def candidate_progress(self) -> int:
        """تعداد فریم‌های متوالی ثبت‌شده برای وضعیت کاندید فعلی."""
        return self._count


def smooth_series(values: List[float], window: int = 5) -> List[float]:
    """هموارسازی یک سری کامل با میانگین متحرک (ابزار کمکی آزمون/گزارش).

    Args:
        values: دنباله مقادیر ورودی.
        window: طول پنجره میانگین متحرک.

    Returns:
        لیست هم‌طول ورودی؛ عنصر i میانگین پنجره‌ای است که به i ختم می‌شود.
    """
    ma = MovingAverage(window)
    return [ma.update(v) for v in values]
