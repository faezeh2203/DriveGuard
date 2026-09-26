"""
تشخیص چهره و استخراج لندمارک (Face Detection & Landmark Extraction)
====================================================================

از **MediaPipe Face Landmarker** (مدل Face Mesh با ۴۶۸ نقطه + ۱۰
نقطه عنبیه) استفاده می‌کنیم؛ مطابق مشخصات پروژه که MediaPipe Face
Mesh (468 Landmark) را به‌عنوان ماژول Landmark Extraction معرفی
کرده است. برای مرحله Face Detection نیز مدل مذکور به‌صورت درونی
از BlazeFace استفاده می‌کند (RetinaFace گزینه جایگزین است).

خروجی ماژول، لندمارک‌ها در **مختصات پیکسلی تصویر** است (تا مستقیماً
به SolvePnP داده شود) به همراه جعبه محصورکننده چهره.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import FACE_MODEL_LANDMARK_IDS

# تعداد لندمارک مدل Face Mesh بدون عنبیه = 468، با عنبیه = 478
BASE_LANDMARK_COUNT = 468


@dataclass(frozen=True)
class FaceObservation:
    """نتیجه تشخیص چهره در یک فریم.

    Attributes:
        landmarks: آرایه (N, 3) مختصات پیکسلی لندمارک‌ها (x, y, z).
        bbox: جعبه محصور به صورت ``(x, y, w, h)`` در مختصات تصویر.
        landmark_count: تعداد کل لندمارک‌ها (468 یا 478).
    """

    landmarks: np.ndarray
    bbox: tuple
    landmark_count: int

    @property
    def has_iris(self) -> bool:
        """True اگر لندمارک‌های عنبیه (478 نقطه‌ای) موجود باشد."""
        return self.landmark_count > BASE_LANDMARK_COUNT

    def model_points_2d(self) -> np.ndarray:
        """مختصات تصویری شش نقطه استاندارد مدل سه‌بعدی صورت.

        Returns:
            آرایه (6, 2) مناسب ورودی ``cv2.solvePnP``.
        """
        return self.landmarks[list(FACE_MODEL_LANDMARK_IDS), :2]

    def eye_points(self, left_ids: tuple, right_ids: tuple) -> tuple:
        """مختصات لندمارک‌های دو چشم برای محاسبه EAR.

        Returns:
            تاپل ``(left_eye (6,2), right_eye (6,2))``.
        """
        return self.landmarks[list(left_ids), :2], self.landmarks[list(right_ids), :2]


class FaceDetector:
    """پوشش MediaPipe Face Landmarker در حالت VIDEO.

    MediaPipe در حالت VIDEO با بهره‌گیری از شماره فریم (timestamp)
    ردیابی زمانی انجام می‌دهد؛ بنابراین فراخوانی‌ها باید با
    timestamp صعودی انجام شوند.

    Args:
        model_path: مسیر فایل ``face_landmarker.task``.
        num_faces: حداکثر تعداد چهره.
        min_face_detection_confidence: آستانه اطمینان تشخیص.
        min_tracking_confidence: آستانه اطمینان ردیابی.
        process_width: عرض پردازش (کوچک‌سازی فریم برای سرعت؛ 0 = خاموش).

    Raises:
        FileNotFoundError: اگر فایل مدل موجود نباشد.
    """

    def __init__(
        self,
        model_path: str = "assets/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        process_width: int = 480,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"model file not found: {model_path} — "
                "download it via scripts/download_model.py"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp_image_class = mp.Image
        self._mp_format = mp.ImageFormat.SRGB
        self._last_ts_ms: int = -1
        self._process_width = int(process_width)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceObservation | None:
        """پردازش یک فریم BGR و استخراج چهره اصلی.

        Args:
            frame_bgr: تصویر ورودی BGR.
            timestamp_ms: زمان فریم بر حسب میلی‌ثانیه (صعودی).

        Returns:
            شیء :class:`FaceObservation` یا ``None`` اگر چهره‌ای دیده نشد.

        Raises:
            ValueError: اگر timestamp غیر صعودی باشد.
        """
        if timestamp_ms <= self._last_ts_ms:
            raise ValueError(
                f"timestamp must be strictly increasing: {timestamp_ms} <= {self._last_ts_ms}"
            )
        self._last_ts_ms = timestamp_ms

        height, width = frame_bgr.shape[:2]
        # کوچک‌سازی فریم برای سرعت پردازش (لندمارک‌ها نرمال هستند و
        # مستقیماً به مختصات تصویر اصلی برمی‌گردند)
        if 0 < self._process_width < width:
            scale = self._process_width / width
            small = cv2.resize(
                frame_bgr,
                (self._process_width, max(int(height * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_class(image_format=self._mp_format, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        if not result.face_landmarks:
            return None

        raw = result.face_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in raw], dtype=np.float64)
        pts[:, 0] *= width   # مختصات نرمال → تصویر اصلی
        pts[:, 1] *= height

        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        bbox = (float(x_min), float(y_min),
                float(x_max - x_min), float(y_max - y_min))
        return FaceObservation(
            landmarks=pts,
            bbox=bbox,
            landmark_count=len(pts),
        )

    def close(self) -> None:
        """بستن landmarker و آزادسازی منابع."""
        self._landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
