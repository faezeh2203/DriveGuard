"""
شناسایی تلفن همراه (Phone Detection)
=====================================

ماژول شناسایی گوشی در دست راننده بر پایه **YOLO11n در قالب ONNX**
اجرا‌شده با **ONNX Runtime** (کوچک‌ترین نسخه YOLO Ultralytics آموزش
دیده روی COCO که کلاس ``cell phone`` دارد). این مسیر با فناوری‌های
الزامی پروژه (ONNX Runtime) هم‌خوان است و بدون GPU روی CPU اجرا
می‌شود.

سیاست تشخیص «تماس دستی / گوشی در دست»:

- مرکز جعبه گوشی داخل **ناحیه بزرگ‌شده جعبه چهره** باشد
  (تابع :func:`phone_near_face`) ⇒ گوشی نزدیک صورت/گوش ⇒ وضعیت PHONE؛
- گوشی در سایر نقاط کادر (مثلاً روی پایه داشبورد برای مسیریابی)
  «نزدیک چهره» نیست و هشدار ایجاد نمی‌کند؛
- نگاه به گوشی روی پا بدون دید کامل گوشی، از مسیر pitch (وضعیت
  DOWN) پوشش داده می‌شود.

برای حفظ نرخ Real-Time، تشخیص هر ``detect_every`` فریم یک‌بار اجرا
و نتیجه تا ``hold_frames`` فریم نگه داشته می‌شود (در پایپ‌لاین) و
ورودی مدل به ``input_size`` کوچک می‌شود (Letterbox).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PHONE_LABEL_KEYWORD = "phone"

# برچسب‌های استاندارد COCO (۸۰ کلاس؛ همان ترتیب خروجی YOLO)
COCO80_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# اندیس کلاس «cell phone» در فهرست ۸۰ کلاسی COCO
PHONE_CLASS_ID = COCO80_CLASS_NAMES.index("cell phone")


def phone_near_face(phone_bbox: tuple, face_bbox: tuple,
                    expand: float = 0.6) -> bool:
    """بررسی «نزدیک بودن گوشی به چهره».

    مرکز جعبه گوشی باید داخل جعبه چهره باشد که از هر طرف به نسبت
    ``expand`` از ابعاد چهره بزرگ شده است.

    Args:
        phone_bbox: جعبه گوشی ``(x, y, w, h)``.
        face_bbox: جعبه چهره ``(x, y, w, h)``.
        expand: نسبت بزرگ‌نمایی جعبه چهره (نسبت به طول/عرض آن).

    Returns:
        True اگر گوشی نزدیک چهره باشد.
    """
    fx, fy, fw, fh = face_bbox
    px, py, pw, ph = phone_bbox
    margin_x = fw * expand / 2.0
    margin_y = fh * expand / 2.0
    cx = px + pw / 2.0
    cy = py + ph / 2.0
    return (fx - margin_x <= cx <= fx + fw + margin_x
            and fy - margin_y <= cy <= fy + fh + margin_y)


class PhoneDetector:
    """تشخیص‌گر تلفن همراه با YOLO11n ONNX و ONNX Runtime.

    Args:
        model_path: مسیر فایل ``yolo11n.onnx``.
        score_threshold: حداقل اطمینان تشخیص.
        nms_threshold: آستانه IoU برای حذف جعبه‌های تکراری.
        input_size: اندازه ورودی مدل (باید مضرب 32 باشد).
        max_results: حداکثر تعداد جعبه گوشی برگشتی.
        process_width: عرض پردازش (کوچک‌سازی فریم برای سرعت؛ 0 = خاموش).

    Raises:
        FileNotFoundError: اگر فایل مدل موجود نباشد.
    """

    def __init__(self, model_path: str = "assets/yolo11n.onnx",
                 score_threshold: float = 0.4,
                 nms_threshold: float = 0.45,
                 input_size: int = 480,
                 max_results: int = 5,
                 process_width: int = 480) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"model file not found: {model_path} — "
                "download it via scripts/download_model.py"
            )
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._input_size = int(input_size)
        self._max_results = int(max_results)
        self._process_width = int(process_width)
        self._last_ts_ms: int = -1

        # پیش‌گرم‌سازی جلسه (تخصیص حافظه/Threadها) تا اولین فریم واقعی کند نباشد
        dummy = np.zeros((1, 3, self._input_size, self._input_size),
                         dtype=np.float32)
        self._session.run([self._output_name], {self._input_name: dummy})

    # ------------------------------------------------------------------
    def _letterbox(self, frame_bgr: np.ndarray) -> tuple:
        """تغییر اندازه با حفظ نسبت (Letterbox) به ورودی مربعی مدل.

        Returns:
            تاپل ``(padded_image, scale, pad_x, pad_y)``.
        """
        height, width = frame_bgr.shape[:2]
        size = self._input_size
        scale = min(size / width, size / height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return padded, scale, pad_x, pad_y

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int,
                face_bbox: tuple | None = None) -> list:
        """تشخیص تلفن همراه در فریم.

        Args:
            frame_bgr: تصویر BGR فریم.
            timestamp_ms: زمان فریم بر حسب میلی‌ثانیه (صعودی).
            face_bbox: جعبه چهره ``(x, y, w, h)`` — در تشخیص‌دهنده
                واقعی نادیده گرفته می‌شود و صرفاً برای هم‌امضایی با
                جایگزین‌های تستی (Stub) پذیرفته می‌شود.

        Returns:
            لیست جعبه‌های گوشی ``[(x, y, w, h), ...]`` در مختصات
            تصویر اصلی (خالی اگر گوشی‌ای دیده نشد).

        Raises:
            ValueError: اگر timestamp غیر صعودی باشد.
        """
        if timestamp_ms <= self._last_ts_ms:
            raise ValueError(
                f"timestamp must be strictly increasing: {timestamp_ms} <= {self._last_ts_ms}"
            )
        self._last_ts_ms = timestamp_ms

        height, width = frame_bgr.shape[:2]
        # کوچک‌سازی فریم بزرگ وبکم پیش از Letterbox (صرفه‌جویی CPU)
        if 0 < self._process_width < width:
            scale_in = self._process_width / width
            frame_bgr = cv2.resize(
                frame_bgr,
                (self._process_width, max(int(height * scale_in), 1)),
                interpolation=cv2.INTER_AREA,
            )

        padded, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, swapRB=True, crop=False)

        preds = self._session.run(
            [self._output_name], {self._input_name: blob})[0]  # (1,84,N)
        preds = np.asarray(preds[0]).T  # (N, 84)

        # وکتوری‌زه: فیلتر کلاس «cell phone» قبل از حلقه پایتونی
        class_scores = preds[:, 4:]
        best_ids = class_scores.argmax(axis=1)
        best_scores = class_scores[np.arange(len(preds)), best_ids]
        mask = (best_ids == PHONE_CLASS_ID) & (best_scores >= self._score_threshold)
        if not mask.any():
            return []

        boxes_px, scores = [], []
        for row, score in zip(preds[mask], best_scores[mask]):
            cx, cy, w, h = row[:4]
            # از مختصات Letterbox به مختصات فریم (کوچک‌شده) برمی‌گردیم
            x = (cx - w / 2.0 - pad_x) / scale
            y = (cy - h / 2.0 - pad_y) / scale
            boxes_px.append((x, y, w / scale, h / scale))
            scores.append(float(score))

        if not boxes_px:
            return []

        keep = cv2.dnn.NMSBoxes(
            [list(b) for b in boxes_px], scores,
            self._score_threshold, self._nms_threshold)
        keep = list(np.asarray(keep).reshape(-1))[:self._max_results]

        # جبران کوچک‌سازی اولیه به مختصات تصویر اصلی
        height_s, width_s = frame_bgr.shape[:2]
        fx = width / width_s
        fy = height / height_s
        return [
            (boxes_px[i][0] * fx, boxes_px[i][1] * fy,
             boxes_px[i][2] * fx, boxes_px[i][3] * fy)
            for i in keep
        ]

    def close(self) -> None:
        """آزادسازی جلسه ONNX Runtime."""
        self._session = None

    def __enter__(self) -> "PhoneDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["PhoneDetector", "phone_near_face", "PHONE_LABEL_KEYWORD",
           "PHONE_CLASS_ID", "COCO80_CLASS_NAMES"]
