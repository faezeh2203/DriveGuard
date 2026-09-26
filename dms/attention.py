"""
طبقه‌بندی وضعیت توجه (Attention State Classification)
======================================================

بر پایه زوایای هموارشده سر (Yaw/Pitch/Roll) وضعیت توجه راننده
در یکی از حالت‌های زیر طبقه‌بندی می‌شود (مطابق مشخصات پروژه):

- ``FORWARD``: نگاه به جلو (درست)
- ``LEFT`` / ``RIGHT``: برگرداندن سر به چپ/راست (|yaw| > آستانه)
- ``DOWN``: نگاه به پایین — سناریوی رایج استفاده از موبایل (pitch < آستانه)
- ``UP``: نگاه به بالا (فراتر از آستانه)
- ``PHONE``: تلفن همراه نزدیک چهره راننده دیده شده (تماس دستی / گوشی در دست)
- ``NO_FACE``: چهره‌ای تشخیص داده نشده (خارج از کادر / پوشش صورت)

وضعیت ``PHONE`` در پایپ‌لاین (نه در این کلاس) از خروجی
:class:`dms.phone_detector.PhoneDetector` تزریق می‌شود: هرگاه جعبه
گوشی با ناحیه بزرگ‌شده جعبه چهره هم‌پوشانی داشته باشد، وضعیت نهایی
به ``PHONE`` ارتقا می‌یابد.

اگر عنبیه‌ها (Iris، لندمارک‌های ۴۶۸..۴۷۷) موجود باشند، جهت نگاه
چشم‌ها (Gaze) نیز تخمین زده می‌شود و به تصمیم کمک می‌کند:
در حالت مرزی (سر رو به جلو اما چشم‌ها منحرف)، وضعیت ``FORWARD`` حفظ
نمی‌شود و به حالت چپ/راست ارتقا می‌یابد.

آستانه‌ها مطابق مشخصات پروژه: yaw > 25° و pitch < -15° (قابل تنظیم
در :class:`dms.config.HeadPoseConfig`).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .config import HeadPoseConfig
from .geometry import EulerAngle, LEFT_EYE_EAR_IDS, RIGHT_EYE_EAR_IDS, eye_aspect_ratio


class AttentionState(enum.Enum):
    """حالت‌های توجه راننده."""

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"        # نگاه به پایین / استفاده از موبایل
    UP = "UP"
    PHONE = "PHONE"      # تلفن همراه نزدیک چهره (تماس دستی)
    NO_FACE = "NO_FACE"

    @property
    def is_distracted(self) -> bool:
        """True برای همه حالت‌های حواس‌پرتی (غیر از FORWARD)."""
        return self is not AttentionState.FORWARD

    @property
    def label_fa(self) -> str:
        """برچسب فارسی حالت برای نمایش در رابط کاربری."""
        return _FA_LABELS[self]


_FA_LABELS = {
    AttentionState.FORWARD: "رو به جلو",
    AttentionState.LEFT: "سر به چپ",
    AttentionState.RIGHT: "سر به راست",
    AttentionState.DOWN: "نگاه به پایین (موبایل)",
    AttentionState.UP: "نگاه به بالا",
    AttentionState.PHONE: "تلفن همراه در دست",
    AttentionState.NO_FACE: "چهره دیده نشد",
}


@dataclass(frozen=True)
class AttentionResult:
    """خروجی طبقه‌بندی توجه.

    Attributes:
        raw_state: وضعیت خام (بدون در نظر گرفتن جهت عنبیه).
        state: وضعیت نهایی پس از ترکیب جهت نگاه چشم‌ها.
        gaze_offset: انحراف افقی عنبیه نسبت به مرکز چشم (بدون بعد،
            مثبت = نگاه به راست تصویر)؛ اگر عنبیه موجود نباشد 0.
        ear: میانگین Eye Aspect Ratio دو چشم (برای ماژول خواب‌آلودگی).
    """

    raw_state: AttentionState
    state: AttentionState
    gaze_offset: float = 0.0
    ear: float = 0.0


class AttentionClassifier:
    """طبقه‌بند قاعده‌مند (Threshold-based) وضعیت توجه.

    طبقه‌بندی مبتنی بر آستانه همان رویکردی است که در پیاده‌سازی‌های
    مرجع DMS استفاده می‌شود؛ گزینه جایگزین (کلاسیفایر Softmax CNN یا
    LSTM زمانی) در گزارش پروژه به‌عنوان کار آینده بحث شده است.

    Args:
        config: آستانه‌های زاویه‌ای.
        gaze_gain: ضریب ارتقای انحراف عنبیه به وضعیت چپ/راست؛
            انحراف مطلق عنبیه بیشتر از این مقدار (بدون بعد) زمانی که
            سر رو به جلو است، وضعیت را به LEFT/RIGHT تغییر می‌دهد.
    """

    GAZE_OFFSET_LIMIT = 0.12

    def __init__(self, config: HeadPoseConfig | None = None,
                 gaze_gain: float = 0.06) -> None:
        self.config = config or HeadPoseConfig()
        self.gaze_gain = gaze_gain

    def classify(self, angles: EulerAngle,
                 landmarks=None) -> AttentionResult:
        """طبقه‌بندی وضعیت توجه از زوایای سر (و لندمارک‌های اختیاری).

        Args:
            angles: زوایای هموارشده Pitch/Yaw/Roll (درجه).
            landmarks: آرایه (N, 3) لندمارک‌ها؛ اگر 478 نقطه‌ای باشد
                جهت نگاه چشم‌ها نیز استفاده می‌شود.

        Returns:
            شیء :class:`AttentionResult`.
        """
        ear = self._compute_ear(landmarks)
        gaze = self._gaze_offset(landmarks)

        cfg = self.config
        if abs(angles.yaw) > cfg.yaw_threshold:
            raw = AttentionState.RIGHT if angles.yaw > 0 else AttentionState.LEFT
        elif angles.pitch < cfg.pitch_down_threshold:
            raw = AttentionState.DOWN
        elif angles.pitch > cfg.pitch_up_threshold:
            raw = AttentionState.UP
        else:
            raw = AttentionState.FORWARD

        state = raw
        # ترکیب جهت نگاه چشم‌ها در حالت مرزی
        if raw is AttentionState.FORWARD and abs(gaze) > self.gaze_gain:
            state = AttentionState.RIGHT if gaze > 0 else AttentionState.LEFT

        return AttentionResult(raw_state=raw, state=state,
                               gaze_offset=gaze, ear=ear)

    def _compute_ear(self, landmarks) -> float:
        """میانگین EAR دو چشم (0 اگر لندمارک در دسترس نباشد)."""
        if landmarks is None:
            return 0.0
        landmarks = np.asarray(landmarks)
        if landmarks.shape[0] < max(LEFT_EYE_EAR_IDS):
            return 0.0
        left = eye_aspect_ratio(landmarks[list(LEFT_EYE_EAR_IDS), :2])
        right = eye_aspect_ratio(landmarks[list(RIGHT_EYE_EAR_IDS), :2])
        return (left + right) / 2.0

    def _gaze_offset(self, landmarks) -> float:
        """انحراف افقی نرمال‌شده عنبیه نسبت به مرکز چشم.

        عنبیه در مدل 478 لندمارکی با اندیس‌های 468..477 موجود است.
        خروجی بر حسب «نسبت به عرض چشم» نرمال می‌شود تا به مقیاس
        تصویر وابسته نباشد.
        """
        if landmarks is None:
            return 0.0
        landmarks = np.asarray(landmarks)
        if landmarks.shape[0] < 478:
            return 0.0
        # مرکز عنبیه = میانگین ۵ نقطه هر عنبیه؛ گوشه‌های افقی چشم: 33 و 263
        left_iris = landmarks[468:473, :2].mean(axis=0)
        right_iris = landmarks[473:478, :2].mean(axis=0)
        left_corner = landmarks[33, :2]
        right_corner = landmarks[263, :2]
        eye_width = float(np.linalg.norm(right_corner - left_corner))
        if eye_width < 1e-6:
            return 0.0
        # جابه‌جایی نسبی عنبیه‌ها نسبت به وسط فاصله دو گوشه چشم
        mid = (left_corner + right_corner) / 2.0
        irises_mid = (left_iris + right_iris) / 2.0
        offset = float((irises_mid[0] - mid[0]) / eye_width)
        return float(np.clip(offset, -self.GAZE_OFFSET_LIMIT, self.GAZE_OFFSET_LIMIT))


__all__ = ["AttentionState", "AttentionResult", "AttentionClassifier"]
