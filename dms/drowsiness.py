"""
تشخیص خواب‌آلودگی (Drowsiness Detection)
=========================================

معیار اصلی، **Eye Aspect Ratio (EAR)** است (Soukupová & Čech, 2016):
وقتی چشم‌ها بسته می‌شوند، EAR به‌شدت افت می‌کند. دو حالت تفکیک می‌شود:

- **پلک زدن طبیعی**: بسته بودن کوتاه (کمتر از ``ear_blink_frames`` فریم).
- **Microsleep / خواب‌آلودگی**: بسته ماندن چشم‌ها بیش از
  ``closed_seconds`` ثانیه.

علاوه بر این، علامت هشدار «سر تکان دادن» (Head Nodding) نیز پایش
می‌شود: افت مکرر و شدید pitch (سر خم شدن به پایین) در حالی که سر
دوباره بالا می‌آید، نشانه شروع خواب‌آلودگی است.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import DrowsinessConfig


@dataclass(frozen=True)
class DrowsinessResult:
    """خروجی ماژول خواب‌آلودگی در یک فریم.

    Attributes:
        eyes_closed: بسته بودن چشم‌ها در این فریم (EAR < آستانه).
        drowsy: اعلام خواب‌آلودگی (بسته بودن طولانی یا سر تکان مکرر).
        nodding: الگوی سر تکان دادن (افت و برگشت مکرر pitch).
        closed_duration: مدت بسته بودن پیوسته چشم‌ها (ثانیه).
        blink_count: تعداد پلک زدن‌های ثبت‌شده تا کنون.
    """

    eyes_closed: bool
    drowsy: bool
    nodding: bool
    closed_duration: float
    blink_count: int


class DrowsinessDetector:
    """آشکارساز خواب‌آلودگی مبتنی بر EAR و الگوی سر تکان دادن.

    Args:
        config: پارامترهای آستانه (EAR، مدت بسته بودن و ...).
    """

    #: حداقل افت pitch (درجه) برای شمارش یک «سر تکان»
    NOD_PITCH_DROP = 12.0
    #: تعداد سر تکان در پنجره زمانی برای اعلام nodding
    NOD_COUNT = 3
    #: طول پنجره پایش سر تکان (ثانیه)
    NOD_WINDOW_SECONDS = 10.0

    def __init__(self, config: DrowsinessConfig | None = None) -> None:
        self.config = config or DrowsinessConfig()
        self.reset()

    def reset(self) -> None:
        """بازنشانی تمام حالت‌های زمانی آشکارساز."""
        self._closed_start: Optional[float] = None
        self._closed_run_frames = 0
        self._closed_duration = 0.0
        self._drowsy = False
        self._blink_count = 0
        self._nod_times: list[float] = []
        self._in_nod = False

    @property
    def closed_duration(self) -> float:
        """مدت بسته بودن پیوسته جاری چشم‌ها (ثانیه؛ 0 اگر باز باشند)."""
        return self._closed_duration

    def update(self, ear: float, pitch: float, timestamp: float,
               fps: float = 30.0) -> DrowsinessResult:
        """به‌روزرسانی وضعیت با EAR و pitch فریم جاری.

        Args:
            ear: میانگین Eye Aspect Ratio فریم جاری.
            pitch: زاویه pitch هموارشده (درجه).
            timestamp: زمان فریم بر حسب ثانیه (صعودی).
            fps: نرخ فریم برای تبدیل فریم به ثانیه.

        Returns:
            شیء :class:`DrowsinessResult`.
        """
        closed = ear < self.config.ear_closed_threshold
        if closed:
            if self._closed_start is None:
                self._closed_start = timestamp
            self._closed_run_frames += 1
        else:
            if self._closed_start is not None:
                if self._closed_run_frames <= self.config.ear_blink_frames:
                    self._blink_count += 1
            self._closed_start = None
            self._closed_run_frames = 0
            self._closed_duration = 0.0
            self._drowsy = False

        closed_duration = (timestamp - self._closed_start) if self._closed_start is not None else 0.0
        self._closed_duration = closed_duration
        if closed and closed_duration >= self.config.closed_seconds:
            self._drowsy = True

        nodding = self._update_nodding(pitch, timestamp)
        return DrowsinessResult(
            eyes_closed=closed,
            drowsy=self._drowsy,
            nodding=nodding,
            closed_duration=closed_duration,
            blink_count=self._blink_count,
        )

    def _update_nodding(self, pitch: float, timestamp: float) -> bool:
        """پایش الگوی «افت و برگشت مکرر سر» (Head Nodding).

        هر بار که pitch به‌اندازه ``NOD_PITCH_DROP`` درجه سقوط کند و
        سپس بازیابی شود، یک nod ثبت می‌شود؛ اگر در پنجره
        ``NOD_WINDOW_SECONDS`` ثانیه بیش از ``NOD_COUNT`` nod رخ دهد،
        nodding برقرار است.
        """
        if pitch < -self.NOD_PITCH_DROP:
            if not self._in_nod:
                self._in_nod = True
                self._nod_times.append(timestamp)
        elif pitch > -self.NOD_PITCH_DROP * 0.5:
            self._in_nod = False

        # حذف nodهای قدیمی خارج از پنجره
        horizon = timestamp - self.NOD_WINDOW_SECONDS
        self._nod_times = [t for t in self._nod_times if t >= horizon]
        return len(self._nod_times) > self.NOD_COUNT
