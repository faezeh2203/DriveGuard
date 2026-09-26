"""
مدیریت ورودی ویدیو (Video Capture)
===================================

کلاس :class:`VideoSource` یک لایه یکپارچه روی ``cv2.VideoCapture``
است که سه نوع منبع را پشتیبانی می‌کند:

- دوربین وبکم (شماره‌گذاری صحیح یا 0)
- فایل ویدیویی (مسیر فایل)
- جریان شبکه (آدرس RTSP/HTTP)

این کلاس همچنین FPS واقعی منبع را در اختیار می‌گذارد و در صورت
در دسترس نبودن FPS (رایج در برخی وبکم‌ها) مقدار پیش‌فرض 30 را
برمی‌گرداند.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class Frame:
    """یک فریم ویدیو همراه با متادیتا.

    Attributes:
        image: تصویر BGR به صورت آرایه numpy.
        timestamp: زمان فریم بر حسب ثانیه (از شروع منبع).
        frame_id: شماره فریم (از صفر).
    """

    image: np.ndarray
    timestamp: float
    frame_id: int

    @property
    def size(self) -> tuple:
        """اندازه تصویر به صورت ``(width, height)``."""
        h, w = self.image.shape[:2]
        return (w, h)


class VideoSource:
    """پوشش امن ``cv2.VideoCapture`` با مدیریت خطا و FPS.

    Args:
        source: منبع ویدیو — عدد صحیح (وبکم)، مسیر فایل یا URL.
        target_fps: FPS جایگزین هنگامی که منبع FPS گزارش نمی‌کند.

    Raises:
        IOError: اگر باز کردن منبع ناموفق باشد.
    """

    def __init__(self, source, target_fps: float = DEFAULT_FPS) -> None:
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video source: {source!r}")
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported if reported and reported > 1.0 else target_fps

    @property
    def frame_size(self) -> tuple:
        """اندازه ``(width, height)`` فریم‌های منبع."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read(self) -> Optional[Frame]:
        """خواندن فریم بعدی.

        Returns:
            شیء :class:`Frame` یا ``None`` در پایان جریان/خطا.
        """
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame_id = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        # برای وبکم شمارنده POS_FRAMES معتبر نیست؛ از ساعت سیستم استفاده می‌کنیم
        timestamp = time.monotonic() if isinstance(self.source, int) else frame_id / self.fps
        return Frame(image=image, timestamp=timestamp, frame_id=frame_id)

    def release(self) -> None:
        """آزادسازی منبع ویدیو."""
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __iter__(self):
        """تکرار روی فریم‌ها تا پایان جریان."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
