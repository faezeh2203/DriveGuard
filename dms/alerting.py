"""
هشدار به راننده (Driver Alerting)
==================================

:class:`AlertManager` تصمیم موتور تصمیم را به اکشن فیزیکی تبدیل
می‌کند:

- **تصویری**: رسم قاب هشدار رنگی و متن روی فریم (برای نمایش در
  پنجره OpenCV یا داشبورد وب).
- **صوتی**: بوق هشدار از طریق ``winsound`` (ویندوز) یا ``buzzer``
  ساده روی Raspberry Pi؛ اگر صدا در دسترس نباشد، بی‌صدا رد می‌شود.

سیستم باید در محیط‌های Headless (بدون صدا/نمایشگر، مثلاً سرور
آزمون) هم بدون خطا کار کند؛ بنابراین همه اکشن‌ها ایمن پیچیده
شده‌اند.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2
import numpy as np

from .decision import AlertLevel, Decision

# رنگ BGR قاب هشدار برای هر سطح
ALERT_COLORS = {
    AlertLevel.NONE: (180, 180, 180),
    AlertLevel.INFO: (30, 200, 60),
    AlertLevel.WARNING: (30, 120, 255),
    AlertLevel.CRITICAL: (30, 30, 230),
}


@dataclass(frozen=True)
class AlertEvent:
    """یک رخداد هشدار برای ثبت در Telemetry و داشبورد.

    Attributes:
        alert_id: شناسه دوره هشدار.
        level: سطح هشدار.
        message: پیام متنی.
        state: وضعیت توجه پایدار.
    """

    alert_id: int | None
    level: AlertLevel
    message: str
    state: str


class AlertManager:
    """مدیریت هشدار تصویری/صوتی با کنترل نویز صوتی.

    Args:
        enable_sound: فعال‌سازی بوق صوتی (در محیط تست معمولاً خاموش).
        beep_interval_seconds: حداقل فاصله بین دو بوق (ثانیه).
    """

    def __init__(self, enable_sound: bool = True,
                 beep_interval_seconds: float = 1.0) -> None:
        self.enable_sound = enable_sound
        self.beep_interval_seconds = beep_interval_seconds
        self._last_beep_ts = -1e9
        self.last_event: AlertEvent | None = None

    def handle(self, decision: Decision, timestamp: float) -> AlertEvent | None:
        """پردازش تصمیم و صدور اکشن هشدار.

        Args:
            decision: خروجی موتور تصمیم.
            timestamp: زمان فعلی (ثانیه).

        Returns:
            :class:`AlertEvent` اگر هشدار فعال صادر شده باشد، وگرنه None.
        """
        if decision.alert_level is AlertLevel.NONE:
            self.last_event = None
            return None
        if decision.alert_level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
            self._beep(timestamp)
        event = AlertEvent(
            alert_id=decision.alert_id,
            level=decision.alert_level,
            message=decision.message,
            state=decision.stable_state.value,
        )
        self.last_event = event
        return event

    def _beep(self, timestamp: float) -> None:
        """صدای هشدار با رعایت فاصله حداقل بین بوق‌ها."""
        if not self.enable_sound:
            return
        if timestamp - self._last_beep_ts < self.beep_interval_seconds:
            return
        self._last_beep_ts = timestamp
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(frequency=1200, duration=250)
            else:
                # روی Linux/Raspberry Pi: بوق کنسولی جایگزین
                print("\a", end="", flush=True)
        except Exception:
            # نبود سخت‌افزار/درایور صدا هرگز نباید پایپ‌لاین را متوقف کند
            pass

    # ------------------------------------------------------------------
    def annotate(self, frame: np.ndarray, decision: Decision,
                 angles_text: str = "") -> np.ndarray:
        """رسم وضعیت و هشدار روی فریم (برای نمایش زنده).

        Args:
            frame: تصویر BGR ورودی (بدون تغییر باقی می‌ماند).
            decision: خروجی موتور تصمیم.
            angles_text: متن اختیاری زوایا (مثلاً ``"Y:3 P:-2 R:1"``).

        Returns:
            نسخه حاشیه‌نویسی‌شده فریم.
        """
        out = frame.copy()
        color = ALERT_COLORS[decision.alert_level]
        h, w = out.shape[:2]
        thickness = 8 if decision.alert_level.is_upper(AlertLevel.INFO) else 2
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)

        cv2.putText(out, f"State: {decision.stable_state.value}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if angles_text:
            cv2.putText(out, angles_text, (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if decision.message:
            cv2.putText(out, "ALERT", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return out
