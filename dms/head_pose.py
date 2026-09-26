"""
تخمین وضعیت سر (Head Pose Estimation)
=======================================

رویارد اصلی مطابق مشخصات پروژه، روش **کلاسیک PnP + نگاشت 2D-3D**
است: شش نقطه استاندارد مدل سه‌بعدی صورت به شش لندمارک تصویری متناظر
جفت می‌شود و ``cv2.solvePnP`` بردار چرخش سر را محاسبه می‌کند؛ سپس
بردار چرخش به زوایای اولر (Yaw/Pitch/Roll) تبدیل می‌شود.

مسیر جایگزین (روش‌های CNN مانند HopeNet/FSA-Net/6DRepNet) در
گزارش پروژه (docs/literature_review.md) بررسی شده است؛ برای
اجرای Real-Time روی سخت‌افزار ضعیف (وبکم/Laptop/Raspberry Pi)
روش PnP دقت کافی و هزینه محاسباتی بسیار کمتری دارد و به‌عنوان
روش پیش‌فرض سیستم انتخاب شده است.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import default_camera_matrix
from .geometry import EulerAngle, GENERIC_3D_FACE_MODEL, rotation_vector_to_euler


@dataclass(frozen=True)
class HeadPose:
    """وضعیت کامل سر در یک فریم.

    Attributes:
        angles: زوایای اولر (Pitch/Yaw/Roll بر حسب درجه).
        reprojection_error: میانگین خطای بازتاب (پیکسل) — معیار کیفیت
            حل PnP؛ مقادیر بزرگ (> ~15) یعنی نتیجه نامطمئن است.
        success: True اگر SolvePnP همگرا شده باشد.
    """

    angles: EulerAngle
    reprojection_error: float
    success: bool


class HeadPoseEstimator:
    """تخمین‌گر وضعیت سر مبتنی بر ``cv2.solvePnP``.

    Args:
        use_extrinsic_guess: استفاده از نتیجه فریم قبلی به‌عنوان حدس
            اولیه (``SOLVEPNP_ITERATIVE`` با پارامتر useExtrinsicGuess)
            که پایداری زمانی را بهبود می‌دهد.
        camera_matrix: ماتریس داخلی دوربین در صورت کالیبراسیون واقعی؛
            اگر ``None`` باشد از تقریب ``default_camera_matrix`` بر اساس
            اندازه تصویر استفاده می‌شود.
    """

    def __init__(self, use_extrinsic_guess: bool = True,
                 camera_matrix: np.ndarray | None = None) -> None:
        self.use_extrinsic_guess = use_extrinsic_guess
        self._custom_camera = (
            np.asarray(camera_matrix, dtype=np.float64) if camera_matrix is not None
            else None
        )
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None
        self._size_cached: tuple | None = None
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def reset(self) -> None:
        """حذف حدس اولیه (مثلاً پس از گم شدن چهره)."""
        self._rvec = None
        self._tvec = None

    def _camera_for_size(self, image_size: tuple) -> np.ndarray:
        """ماتریس دوربین: مقدار کالیبره‌شده یا تقریب بر اساس اندازه تصویر."""
        if self._custom_camera is not None:
            return self._custom_camera
        if image_size != self._size_cached:
            self._camera_matrix = default_camera_matrix(image_size)
            self._size_cached = image_size
        return self._camera_matrix

    def estimate(self, model_points_2d: np.ndarray, image_size: tuple) -> HeadPose:
        """تخمین وضعیت سر از شش نقطه تصویری.

        Args:
            model_points_2d: آرایه (6, 2) مختصات تصویری نقاط مدل
                (خروجی ``FaceObservation.model_points_2d``).
            image_size: ``(width, height)`` تصویر.

        Returns:
            شیء :class:`HeadPose`. در صورت عدم همگرایی،
            ``success=False`` و زوایا صفر برمی‌گردد.
        """
        model_points_2d = np.asarray(model_points_2d, dtype=np.float64).reshape(-1, 2)
        if model_points_2d.shape[0] != GENERIC_3D_FACE_MODEL.shape[0]:
            raise ValueError(
                f"expected {GENERIC_3D_FACE_MODEL.shape[0]} points, "
                f"got {model_points_2d.shape[0]}"
            )

        camera_matrix = self._camera_for_size(tuple(image_size))
        flags = cv2.SOLVEPNP_ITERATIVE
        kwargs = {}
        if self.use_extrinsic_guess and self._rvec is not None:
            kwargs = {"rvec": self._rvec.copy(), "tvec": self._tvec.copy(),
                      "useExtrinsicGuess": True}

        try:
            ok, rvec, tvec = cv2.solvePnP(
                GENERIC_3D_FACE_MODEL, model_points_2d,
                camera_matrix, self._dist_coeffs,
                flags=flags, **kwargs,
            )
        except cv2.error:
            ok = False

        if not ok:
            self.reset()
            return HeadPose(
                angles=EulerAngle(pitch=0.0, yaw=0.0, roll=0.0),
                reprojection_error=float("inf"),
                success=False,
            )

        self._rvec, self._tvec = rvec, tvec
        projected, _ = cv2.projectPoints(
            GENERIC_3D_FACE_MODEL, rvec, tvec, camera_matrix, self._dist_coeffs
        )
        error = float(np.mean(np.linalg.norm(
            projected.reshape(-1, 2) - model_points_2d, axis=1)))
        return HeadPose(
            angles=rotation_vector_to_euler(rvec),
            reprojection_error=error,
            success=True,
        )
