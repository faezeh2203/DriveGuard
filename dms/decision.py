"""
موتور تصمیم و هشدار (Decision Engine & Alerting Logic)
=======================================================

:class:`DecisionEngine` خروجی‌های طبقه‌بند توجه و آشکارساز
خواب‌آلودگی را ترکیب می‌کند و سیاست نهایی هشدار را اعمال می‌کند:

1. **هیسترزیس وضعیت**: تغییر وضعیت توجه تنها پس از ``N`` فریم
   متوالی پذیرفته می‌شود (کاهش False Alarm — الزام مشخصات پروژه).
2. **هشدار حواس‌پرتی**: اگر وضعیت پایدار حواس‌پرتی بیش از
   ``distraction_alert_seconds`` ثانیه ادامه یابد، هشدار فعال می‌شود.
3. **هشدار خواب‌آلودگی**: مستقل از وضعیت توجه، در صورت اعلام
   ماژول Drowsiness فعال می‌شود.
4. **Cooldown**: بین دو هشدار متوالی حداقل فاصله زمانی رعایت
   می‌شود تا هشدار آزاردهنده نشود.

خروجی هر فریم یک :class:`Decision` است که کلید «موتور تصمیم» در
پایپ‌لاین (Attention Classification → Decision Engine → Driver
Alerting) است.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .attention import AttentionResult, AttentionState
from .config import DecisionConfig
from .drowsiness import DrowsinessResult


class AlertLevel(Enum):
    """شدت هشدار."""

    NONE = "NONE"
    INFO = "INFO"        # وضعیت حواس‌پرتی پایدار اما هنوز زیر آستانه هشدار
    WARNING = "WARNING"  # هشدار حواس‌پرتی
    CRITICAL = "CRITICAL"  # هشدار خواب‌آلودگی (بالاترین اولویت)

    def is_upper(self, other: "AlertLevel") -> bool:
        """True اگر این سطح شدیدتر از ``other`` باشد."""
        return _LEVEL_ORDER[self] > _LEVEL_ORDER[other]


_LEVEL_ORDER = {
    AlertLevel.NONE: 0,
    AlertLevel.INFO: 1,
    AlertLevel.WARNING: 2,
    AlertLevel.CRITICAL: 3,
}


@dataclass(frozen=True)
class Decision:
    """خروجی موتور تصمیم برای یک فریم.

    Attributes:
        stable_state: وضعیت توجه پایدار پس از هیسترزیس.
        alert_level: سطح هشدار جاری.
        message: پیام متنی هشدار (فارسی) برای نمایش به راننده.
        distracted_seconds: مدت حواس‌پرتی پایدار جاری (ثانیه).
        alert_id: شناسه یکتای دوره هشدار فعال (None اگر هشداری فعال نباشد).
    """

    stable_state: AttentionState
    alert_level: AlertLevel
    message: str = ""
    distracted_seconds: float = 0.0
    alert_id: Optional[int] = None


@dataclass
class _Episode:
    """یک دوره حواس‌پرتی (برای شمارش و متریک‌ها)."""

    start_ts: float
    state: AttentionState
    alerted: bool = False
    end_ts: Optional[float] = None
    alert_id: Optional[int] = None


class DecisionEngine:
    """ترکیب‌کننده توجه و خواب‌آلودگی با سیاست هیسترزیس/Cooldown.

    Args:
        config: پارامترهای موتور تصمیم.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self.reset()

    def reset(self) -> None:
        """بازنشانی کامل حالت موتور (شروع جلسه جدید)."""
        self._stable_state: AttentionState = AttentionState.FORWARD
        self._candidate_state: AttentionState = AttentionState.FORWARD
        self._candidate_count = 0
        self._episode: Optional[_Episode] = None
        self._last_alert_ts: Optional[float] = None
        self._alert_counter = 0
        self.episodes: list[_Episode] = []

    @property
    def stable_state(self) -> AttentionState:
        """وضعیت توجه پایدار جاری."""
        return self._stable_state

    def update(self, attention: AttentionResult,
               drowsiness: DrowsinessResult | None,
               timestamp: float) -> Decision:
        """پردازش یک فریم و تولید تصمیم.

        Args:
            attention: خروجی طبقه‌بند توجه (خام).
            drowsiness: خروجی آشکارساز خواب‌آلودگی (می‌تواند None باشد
                اگر چهره دیده نشده باشد).
            timestamp: زمان فریم بر حسب ثانیه.

        Returns:
            شیء :class:`Decision` برای این فریم.
        """
        self._apply_hysteresis(attention.state)
        self._update_episode(timestamp)

        level = AlertLevel.NONE
        message = ""

        drowsy_flag = bool(drowsiness and drowsiness.drowsy)
        if drowsy_flag:
            level = AlertLevel.CRITICAL
            message = "خطر: خواب‌آلودگی تشخیص داده شد! توقف کنید."
            decision_alert_id = self._emit_alert(timestamp)
        elif self._episode is not None:
            distracted = timestamp - self._episode.start_ts
            if distracted >= self.config.distraction_alert_seconds:
                # چرخش سر به چپ/راست (آینه‌ها، گذرها و پیچ) وظیفه
                # راننده است؛ به‌طور پیش‌فرض هشدار ندارد و فقط برچسب
                # اطلاع‌رسانی می‌گیرد (سیاست سخت‌گیرانه: اختیاری).
                lateral = self._stable_state in (
                    AttentionState.LEFT, AttentionState.RIGHT)
                if lateral and not self.config.lateral_alert_enabled:
                    level = AlertLevel.INFO
                    message = f"توجه: {self._stable_state.label_fa}"
                    decision_alert_id = None
                else:
                    level = AlertLevel.WARNING
                    if self._stable_state is AttentionState.PHONE:
                        message = "استفاده از تلفن همراه در حین رانندگی ممنوع است!"
                    else:
                        message = f"حواس‌پرتی: {self._stable_state.label_fa}"
                    decision_alert_id = self._emit_alert(timestamp)
            else:
                level = AlertLevel.INFO
                message = f"توجه: {self._stable_state.label_fa}"
                decision_alert_id = None
        else:
            decision_alert_id = None

        distracted_seconds = (
            timestamp - self._episode.start_ts if self._episode is not None else 0.0
        )
        return Decision(
            stable_state=self._stable_state,
            alert_level=level,
            message=message,
            distracted_seconds=distracted_seconds,
            alert_id=decision_alert_id,
        )

    # ------------------------------------------------------------------
    def _apply_hysteresis(self, raw_state: AttentionState) -> None:
        """اعمال فیلتر هیسترزیس روی وضعیت خام (الگوریتم شمارش متوالی)."""
        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1
        if (self._candidate_count >= self.config.state_confirm_frames
                and raw_state != self._stable_state):
            self._stable_state = raw_state

    def _update_episode(self, timestamp: float) -> None:
        """مدیریت دوره حواس‌پرتی جاری (شروع/پایان)."""
        if self._stable_state.is_distracted:
            if self._episode is None or self._episode.state != self._stable_state:
                if self._episode is not None and self._episode.end_ts is None:
                    self._episode.end_ts = timestamp
                self._episode = _Episode(start_ts=timestamp, state=self._stable_state)
                self.episodes.append(self._episode)
        else:
            if self._episode is not None:
                self._episode.end_ts = timestamp
                self._episode = None

    def _emit_alert(self, timestamp: float) -> Optional[int]:
        """صدور/ادامه هشدار با رعایت Cooldown.

        معناشناسی شناسه هشدار: تا زمانی که یک دوره هشدار فعال است،
        همان شناسه بازگردانده می‌شود (هشدار «در جریان» است)؛ هشدار
        جدید تنها پس از پایان Cooldown با شناسه تازه صادر می‌شود.

        Returns:
            شناسه هشدار جاری، یا ``None`` اگر هشدار جدید در فاز
            Cooldown مسدود شده باشد.
        """
        episode = self._episode
        if episode is not None and episode.alerted:
            return episode.alert_id  # هشدار جاری ادامه دارد
        if (self._last_alert_ts is not None
                and timestamp - self._last_alert_ts < self.config.alert_cooldown_seconds):
            return None  # هشدار جدید هنوز مجاز نیست
        self._last_alert_ts = timestamp
        self._alert_counter += 1
        if episode is not None:
            episode.alerted = True
            episode.alert_id = self._alert_counter
        return self._alert_counter

    @property
    def alert_count(self) -> int:
        """تعداد کل هشدارهای صادرشده."""
        return self._alert_counter


__all__ = ["AlertLevel", "Decision", "DecisionEngine"]
