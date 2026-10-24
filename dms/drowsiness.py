"""
تشخیص خواب‌آلودگی (Drowsiness Detection)
=========================================

سه نشانه مستقل پایش می‌شود و هر کدام می‌تواند خواب‌آلودگی را اعلام کند:

1. **Microsleep** — بسته ماندن پیوسته چشم‌ها بیش از ``closed_seconds``
   ثانیه (معیار Eye Aspect Ratio؛ Soukupová & Čech, 2016). پلک زدن طبیعی
   (کمتر از ``ear_blink_frames`` فریم) فقط شمارش می‌شود.
2. **Head Nodding** — افت و برگشت مکرر pitch (سر خم شدن به پایین) در
   یک پنجره زمانی.
3. **PERCLOS** — درصد زمان بسته بودن چشم‌ها در یک پنجره لغزان (معیار
   استاندارد NHTSA برای خستگی تدریجی که پلک‌های کند و نیمه‌بسته را
   هم می‌گیرد).

**ماندگاری (Latch)**: پس از اعلام خواب‌آلودگی، وضعیت تا زمانی که چشم‌ها
به مدت ``recovery_seconds`` پیوسته باز نمانند (و nodding/PERCLOS برطرف
نشده باشد) پابرجا می‌ماند؛ باز شدن لحظه‌ای چشم هشدار را قطع نمی‌کند.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .config import DrowsinessConfig


@dataclass(frozen=True)
class DrowsinessResult:
    """خروجی ماژول خواب‌آلودگی در یک فریم.

    Attributes:
        eyes_closed: بسته بودن چشم‌ها در این فریم (EAR < آستانه).
        drowsy: اعلام خواب‌آلودگی (ماندگار تا بازیابی راننده).
        nodding: الگوی سر تکان دادن (افت و برگشت مکرر pitch).
        closed_duration: مدت بسته بودن پیوسته چشم‌ها (ثانیه).
        blink_count: تعداد پلک زدن‌های ثبت‌شده تا کنون.
        perclos: نسبت زمان بسته بودن چشم در پنجره لغزان (0..1).
        reason: علت اعلام خواب‌آلودگی: ``"eyes_closed"``، ``"nodding"``،
            ``"perclos"`` یا رشته خالی.
    """

    eyes_closed: bool
    drowsy: bool
    nodding: bool
    closed_duration: float
    blink_count: int
    perclos: float = 0.0
    reason: str = ""


class DrowsinessDetector:
    """آشکارساز خواب‌آلودگی مبتنی بر EAR، nodding و PERCLOS.

    Args:
        config: پارامترهای آستانه (EAR، مدت بسته بودن و ...).
    """

    def __init__(self, config: DrowsinessConfig | None = None) -> None:
        self.config = config or DrowsinessConfig()
        self.reset()

    # سازگاری با نسخه قبل (ثابت‌ها اکنون در پیکربندی هستند)
    @property
    def NOD_PITCH_DROP(self) -> float:  # noqa: N802
        return self.config.nod_pitch_drop

    @property
    def NOD_COUNT(self) -> int:  # noqa: N802
        return self.config.nod_count

    @property
    def NOD_WINDOW_SECONDS(self) -> float:  # noqa: N802
        return self.config.nod_window_seconds

    def reset(self) -> None:
        """بازنشانی تمام حالت‌های زمانی آشکارساز."""
        self._closed_start: Optional[float] = None
        self._closed_run_frames = 0
        self._closed_duration = 0.0
        self._open_start: Optional[float] = None
        self._drowsy = False
        self._reason = ""
        self._blink_count = 0
        self._nod_times: deque[float] = deque()
        self._in_nod = False
        # نمونه‌های PERCLOS: (timestamp, closed)
        self._samples: deque[tuple[float, bool]] = deque()
        self._closed_time = 0.0  # مجموع زمان بسته در پنجره
        self._last_ts: Optional[float] = None
        self._last_result: Optional[DrowsinessResult] = None

    @property
    def closed_duration(self) -> float:
        """مدت بسته بودن پیوسته جاری چشم‌ها (ثانیه؛ 0 اگر باز باشند)."""
        return self._closed_duration

    @property
    def last_result(self) -> Optional[DrowsinessResult]:
        """آخرین نتیجه محاسبه‌شده (برای فریم‌های بدون چهره)."""
        return self._last_result

    def update(self, ear: float, pitch: float, timestamp: float,
               fps: float | None = None) -> DrowsinessResult:
        """به‌روزرسانی وضعیت با EAR و pitch فریم جاری.

        Args:
            ear: میانگین Eye Aspect Ratio فریم جاری.
            pitch: زاویه pitch هموارشده (درجه).
            timestamp: زمان فریم بر حسب ثانیه (صعودی).
            fps: (منسوخ؛ نادیده گرفته می‌شود — زمان از timestamp محاسبه می‌شود).

        Returns:
            شیء :class:`DrowsinessResult`.
        """
        cfg = self.config
        closed = ear < cfg.ear_closed_threshold
        if closed:
            if self._closed_start is None:
                self._closed_start = timestamp
            self._closed_run_frames += 1
            self._open_start = None
        else:
            if self._closed_start is not None and self._closed_run_frames <= cfg.ear_blink_frames:
                self._blink_count += 1
            self._closed_start = None
            self._closed_run_frames = 0
            if self._open_start is None:
                self._open_start = timestamp

        closed_duration = (timestamp - self._closed_start) if self._closed_start is not None else 0.0
        self._closed_duration = closed_duration

        perclos = self._update_perclos(closed, timestamp)
        nodding = self._update_nodding(pitch, timestamp)

        # --- تشخیص علت‌ها ---
        reason = ""
        if closed and closed_duration >= cfg.closed_seconds:
            reason = "eyes_closed"
        elif cfg.nodding_alert and nodding:
            reason = "nodding"
        elif perclos >= cfg.perclos_threshold and self._window_span() >= cfg.perclos_min_seconds:
            reason = "perclos"

        if reason:
            self._drowsy = True
            self._reason = reason
        elif self._drowsy:
            # خروج تنها پس از باز ماندن پیوسته چشم‌ها (بازیابی واقعی)
            open_for = (timestamp - self._open_start) if self._open_start is not None else 0.0
            if not closed and open_for >= cfg.recovery_seconds:
                self._drowsy = False
                self._reason = ""

        result = DrowsinessResult(
            eyes_closed=closed,
            drowsy=self._drowsy,
            nodding=nodding,
            closed_duration=closed_duration,
            blink_count=self._blink_count,
            perclos=perclos,
            reason=self._reason if self._drowsy else "",
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    def _update_perclos(self, closed: bool, timestamp: float) -> float:
        """PERCLOS وزن‌دار با زمان در پنجره لغزان."""
        if self._last_ts is not None and self._samples:
            dt = max(0.0, min(timestamp - self._last_ts, 1.0))
            # زمان بین نمونه قبلی و فعلی به وضعیت نمونه قبلی نسبت داده می‌شود
            if self._samples[-1][1]:
                self._closed_time += dt
        self._last_ts = timestamp
        self._samples.append((timestamp, closed))

        horizon = timestamp - self.config.perclos_window_seconds
        while len(self._samples) > 1 and self._samples[1][0] <= horizon:
            t0, c0 = self._samples.popleft()
            if c0:
                self._closed_time -= min(self._samples[0][0] - t0, 1.0)
        self._closed_time = max(self._closed_time, 0.0)
        span = self._window_span()
        return min(self._closed_time / span, 1.0) if span > 0 else 0.0

    def _window_span(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def _update_nodding(self, pitch: float, timestamp: float) -> bool:
        """پایش الگوی «افت و برگشت مکرر سر» (Head Nodding).

        هر بار که pitch به‌اندازه ``nod_pitch_drop`` درجه سقوط کند و سپس
        بازیابی شود، یک nod ثبت می‌شود؛ اگر در پنجره ``nod_window_seconds``
        ثانیه حداقل ``nod_count`` nod رخ دهد، nodding برقرار است.
        """
        drop = self.config.nod_pitch_drop
        if pitch < -drop:
            if not self._in_nod:
                self._in_nod = True
                self._nod_times.append(timestamp)
        elif pitch > -drop * 0.5:
            self._in_nod = False

        horizon = timestamp - self.config.nod_window_seconds
        while self._nod_times and self._nod_times[0] < horizon:
            self._nod_times.popleft()
        return len(self._nod_times) >= self.config.nod_count


__all__ = ["DrowsinessDetector", "DrowsinessResult"]
