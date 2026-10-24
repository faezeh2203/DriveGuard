"""
موتور تصمیم و هشدار (Decision Engine & Alerting Logic)
=======================================================

:class:`DecisionEngine` خروجی‌های طبقه‌بند توجه و آشکارساز
خواب‌آلودگی را ترکیب می‌کند و سیاست نهایی هشدار را اعمال می‌کند:

1. **هیسترزیس وضعیت**: تغییر وضعیت توجه تنها پس از ``N`` فریم
   متوالی پذیرفته می‌شود (کاهش False Alarm).
2. **تایمر پیوسته حواس‌پرتی**: زمان حواس‌پرتی از لحظه ترک «رو به جلو»
   شمرده می‌شود و با جابه‌جایی بین حالت‌ها (مثلاً پایین ← موبایل) صفر
   نمی‌شود؛ بنابراین راننده نمی‌تواند با تغییر حالت هشدار را دور بزند.
3. **آستانه وابسته به رفتار**: موبایل (``phone_alert_seconds``)،
   نگاه پایین/بالا یا ناپیدا بودن چهره (``distraction_alert_seconds``) و
   نگاه طولانی به اطراف (``lateral_alert_seconds``).
4. **هشدار خواب‌آلودگی**: مستقل از وضعیت توجه و با بالاترین اولویت.
5. **هشدار ماندگار (Latched)**: هشدار فعال تا زمانی که راننده رفتارش را
   اصلاح نکند — یعنی به مدت ``recovery_seconds`` پیوسته رو به جلو و
   هوشیار باشد — ادامه می‌یابد.
6. **ارتقا (Escalation)**: هشدار WARNING که بیش از ``escalation_seconds``
   ادامه یابد به CRITICAL ارتقا می‌یابد.
7. **Cooldown بدون سرکوب**: حواس‌پرتی تازه در فاصله کوتاه پس از پایان
   هشدار قبلی، همان شناسه هشدار را دوباره باز می‌کند (برای آمار)، اما
   هشدار هرگز به خاطر Cooldown حذف نمی‌شود.
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
    INFO = "INFO"        # حواس‌پرتی در جریان اما هنوز زیر آستانه هشدار
    WARNING = "WARNING"  # هشدار حواس‌پرتی
    CRITICAL = "CRITICAL"  # خواب‌آلودگی یا هشدار طولانی‌شده (بالاترین اولویت)

    def is_upper(self, other: "AlertLevel") -> bool:
        """True اگر این سطح شدیدتر از ``other`` باشد."""
        return _LEVEL_ORDER[self] > _LEVEL_ORDER[other]


_LEVEL_ORDER = {
    AlertLevel.NONE: 0,
    AlertLevel.INFO: 1,
    AlertLevel.WARNING: 2,
    AlertLevel.CRITICAL: 3,
}


class AlertReason(Enum):
    """علت هشدار فعال."""

    NONE = "NONE"
    DROWSY = "DROWSY"
    PHONE = "PHONE"
    LOOK_DOWN = "LOOK_DOWN"
    LOOK_UP = "LOOK_UP"
    LATERAL = "LATERAL"
    NO_FACE = "NO_FACE"

    @property
    def message_fa(self) -> str:
        """پیام هشدار فارسی برای راننده."""
        return _REASON_MESSAGES[self]


_REASON_MESSAGES = {
    AlertReason.NONE: "",
    AlertReason.DROWSY: "خطر: خواب‌آلودگی تشخیص داده شد! توقف کنید و استراحت کنید.",
    AlertReason.PHONE: "استفاده از تلفن همراه در حین رانندگی ممنوع است!",
    AlertReason.LOOK_DOWN: "حواس‌پرتی: به جاده نگاه کنید! (نگاه به پایین)",
    AlertReason.LOOK_UP: "حواس‌پرتی: به جاده نگاه کنید!",
    AlertReason.LATERAL: "حواس‌پرتی: نگاه طولانی به اطراف — به جلو نگاه کنید!",
    AlertReason.NO_FACE: "حواس‌پرتی: چهره راننده دیده نمی‌شود — به جلو نگاه کنید!",
}

_STATE_REASON = {
    AttentionState.PHONE: AlertReason.PHONE,
    AttentionState.DOWN: AlertReason.LOOK_DOWN,
    AttentionState.UP: AlertReason.LOOK_UP,
    AttentionState.LEFT: AlertReason.LATERAL,
    AttentionState.RIGHT: AlertReason.LATERAL,
    AttentionState.NO_FACE: AlertReason.NO_FACE,
}


class Transition(Enum):
    """تغییر وضعیت هشدار در این فریم (برای ثبت رخداد)."""

    NONE = ""
    STARTED = "STARTED"
    ESCALATED = "ESCALATED"
    REASON_CHANGED = "REASON_CHANGED"
    CLEARED = "CLEARED"


@dataclass(frozen=True)
class Decision:
    """خروجی موتور تصمیم برای یک فریم.

    Attributes:
        stable_state: وضعیت توجه پایدار پس از هیسترزیس.
        alert_level: سطح هشدار جاری.
        message: پیام متنی هشدار (فارسی) برای نمایش به راننده.
        distracted_seconds: مدت حواس‌پرتی پیوسته جاری (ثانیه).
        alert_id: شناسه یکتای دوره هشدار فعال (None اگر هشداری فعال نباشد).
        reason: علت هشدار فعال.
        alert_active: True تا زمانی که هشدار ماندگار فعال است.
        alert_seconds: مدت فعال بودن هشدار جاری (ثانیه).
        threshold_seconds: آستانه هشدار برای وضعیت جاری (برای نوار پیشرفت).
        recovery_progress: پیشرفت بازیابی راننده (0..1) — هشدار در 1 قطع می‌شود.
        transition: تغییر وضعیت هشدار در همین فریم.
    """

    stable_state: AttentionState
    alert_level: AlertLevel
    message: str = ""
    distracted_seconds: float = 0.0
    alert_id: Optional[int] = None
    reason: AlertReason = AlertReason.NONE
    alert_active: bool = False
    alert_seconds: float = 0.0
    threshold_seconds: float = 0.0
    recovery_progress: float = 0.0
    transition: Transition = Transition.NONE


@dataclass
class _Episode:
    """یک دوره حواس‌پرتی در یک وضعیت مشخص (برای شمارش و متریک‌ها)."""

    start_ts: float
    state: AttentionState
    alerted: bool = False
    end_ts: Optional[float] = None
    alert_id: Optional[int] = None


@dataclass
class _ActiveAlert:
    """هشدار ماندگار جاری."""

    alert_id: int
    reason: AlertReason
    start_ts: float
    level: AlertLevel
    recovery_start: Optional[float] = None


class DecisionEngine:
    """ترکیب‌کننده توجه و خواب‌آلودگی با هشدار ماندگار و هیسترزیس.

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
        self._distraction_start: Optional[float] = None
        self._active: Optional[_ActiveAlert] = None
        self._last_clear_ts: Optional[float] = None
        self._last_alert_id: Optional[int] = None
        self._alert_counter = 0
        self.episodes: list[_Episode] = []

    @property
    def stable_state(self) -> AttentionState:
        """وضعیت توجه پایدار جاری."""
        return self._stable_state

    @property
    def alert_active(self) -> bool:
        """True اگر هشدار ماندگار فعال باشد."""
        return self._active is not None

    @property
    def alert_count(self) -> int:
        """تعداد کل هشدارهای صادرشده (دوره‌های هشدار مجزا)."""
        return self._alert_counter

    # ------------------------------------------------------------------
    def threshold_for(self, state: AttentionState) -> Optional[float]:
        """آستانه زمانی هشدار برای یک وضعیت (None = بدون هشدار)."""
        cfg = self.config
        if state is AttentionState.PHONE:
            return cfg.phone_alert_seconds
        if state in (AttentionState.LEFT, AttentionState.RIGHT):
            return cfg.lateral_alert_seconds if cfg.lateral_alert_enabled else None
        if state.is_distracted:
            return cfg.distraction_alert_seconds
        return None

    def update(self, attention: AttentionResult,
               drowsiness: DrowsinessResult | None,
               timestamp: float) -> Decision:
        """پردازش یک فریم و تولید تصمیم.

        Args:
            attention: خروجی طبقه‌بند توجه (خام).
            drowsiness: خروجی آشکارساز خواب‌آلودگی (None اگر چهره دیده نشده).
            timestamp: زمان فریم بر حسب ثانیه.

        Returns:
            شیء :class:`Decision` برای این فریم.
        """
        cfg = self.config
        self._apply_hysteresis(attention.state)
        self._update_episode(timestamp)

        state = self._stable_state
        distracted = state.is_distracted
        if distracted:
            if self._distraction_start is None:
                self._distraction_start = timestamp
            distracted_seconds = timestamp - self._distraction_start
        else:
            self._distraction_start = None
            distracted_seconds = 0.0

        drowsy = bool(cfg.drowsiness_alert and drowsiness and drowsiness.drowsy)
        threshold = self.threshold_for(state)

        trigger = AlertReason.NONE
        if drowsy:
            trigger = AlertReason.DROWSY
        elif threshold is not None and distracted_seconds >= threshold:
            trigger = _STATE_REASON[state]

        transition = Transition.NONE
        active = self._active
        if trigger is not AlertReason.NONE:
            if active is None:
                active = self._open_alert(trigger, timestamp)
                transition = Transition.STARTED
            else:
                active.recovery_start = None
                if trigger is not active.reason:
                    active.reason = trigger
                    transition = Transition.REASON_CHANGED
        elif active is not None:
            # راننده رفتارش را اصلاح کرده؟ (رو به جلو و هوشیار)
            if not distracted and not drowsy:
                if active.recovery_start is None:
                    active.recovery_start = timestamp
                if timestamp - active.recovery_start >= cfg.recovery_seconds:
                    return self._close_alert(timestamp, state)
            else:
                active.recovery_start = None

        if active is None:
            return self._idle_decision(state, distracted_seconds, threshold)

        # --- هشدار فعال ---
        alert_seconds = timestamp - active.start_ts
        level = AlertLevel.WARNING
        if active.reason is AlertReason.DROWSY or alert_seconds >= cfg.escalation_seconds:
            level = AlertLevel.CRITICAL
        if level is not active.level:
            if level.is_upper(active.level) and transition is Transition.NONE:
                transition = Transition.ESCALATED
            active.level = level

        recovery = 0.0
        message = active.reason.message_fa
        if active.recovery_start is not None:
            span = max(cfg.recovery_seconds, 1e-6)
            recovery = min((timestamp - active.recovery_start) / span, 1.0)
            message = f"{message} (در حال بازگشت به حالت عادی...)"
        elif level is AlertLevel.CRITICAL and active.reason is not AlertReason.DROWSY:
            message = f"هشدار جدی — {message}"

        if self._episode is not None and not self._episode.alerted:
            self._episode.alerted = True
            self._episode.alert_id = active.alert_id

        return Decision(
            stable_state=state,
            alert_level=level,
            message=message,
            distracted_seconds=distracted_seconds,
            alert_id=active.alert_id,
            reason=active.reason,
            alert_active=True,
            alert_seconds=alert_seconds,
            threshold_seconds=threshold or 0.0,
            recovery_progress=recovery,
            transition=transition,
        )

    # ------------------------------------------------------------------
    def _idle_decision(self, state: AttentionState, distracted_seconds: float,
                       threshold: Optional[float]) -> Decision:
        """تصمیم وقتی هشداری فعال نیست (NONE یا INFO)."""
        if not state.is_distracted:
            return Decision(stable_state=state, alert_level=AlertLevel.NONE)
        if threshold is not None:
            remaining = max(threshold - distracted_seconds, 0.0)
            message = f"توجه: {state.label_fa} — هشدار تا {remaining:.1f} ثانیه دیگر"
        else:
            message = f"توجه: {state.label_fa}"
        return Decision(
            stable_state=state,
            alert_level=AlertLevel.INFO,
            message=message,
            distracted_seconds=distracted_seconds,
            threshold_seconds=threshold or 0.0,
        )

    def _open_alert(self, reason: AlertReason, timestamp: float) -> _ActiveAlert:
        """شروع دوره هشدار جدید (یا بازگشایی هشدار اخیر در بازه Cooldown)."""
        reopen = (self._last_clear_ts is not None
                  and self._last_alert_id is not None
                  and timestamp - self._last_clear_ts < self.config.alert_cooldown_seconds)
        if reopen:
            alert_id = self._last_alert_id
        else:
            self._alert_counter += 1
            alert_id = self._alert_counter
        self._active = _ActiveAlert(alert_id=alert_id, reason=reason,
                                    start_ts=timestamp, level=AlertLevel.WARNING)
        return self._active

    def _close_alert(self, timestamp: float, state: AttentionState) -> Decision:
        """پایان هشدار پس از بازیابی کامل راننده."""
        active = self._active
        assert active is not None
        self._active = None
        self._last_clear_ts = timestamp
        self._last_alert_id = active.alert_id
        return Decision(
            stable_state=state,
            alert_level=AlertLevel.NONE,
            message="",
            alert_id=active.alert_id,
            reason=active.reason,
            alert_seconds=timestamp - active.start_ts,
            recovery_progress=1.0,
            transition=Transition.CLEARED,
        )

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
        """مدیریت دوره حواس‌پرتی جاری به تفکیک وضعیت (برای متریک‌ها)."""
        if self._stable_state.is_distracted:
            if self._episode is None or self._episode.state != self._stable_state:
                if self._episode is not None and self._episode.end_ts is None:
                    self._episode.end_ts = timestamp
                self._episode = _Episode(start_ts=timestamp, state=self._stable_state)
                self.episodes.append(self._episode)
        elif self._episode is not None:
            self._episode.end_ts = timestamp
            self._episode = None


__all__ = ["AlertLevel", "AlertReason", "Decision", "DecisionEngine", "Transition"]
