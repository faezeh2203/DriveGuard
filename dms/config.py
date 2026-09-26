"""
تنظیمات سیستم (Configuration)
==============================

تمام پارامترهای قابل تنظیم سیستم در کلاس‌های dataclass این ماژول تعریف شده‌اند.
مقادیر پیش‌فرض بر اساس مقادیر مرجع در مقالات و مشخصات پروژه انتخاب شده‌اند
(آستانه yaw > 25° و pitch < -15° بر اساس شرح پروژه).

یک فایل پیکربندی JSON اختیاری نیز پشتیبانی می‌شود؛ کلیدهای فایل باید
هم‌نام با فیلدهای کلاس ``DMSConfig`` باشند.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class HeadPoseConfig:
    """آستانه‌های زاویه‌ای برای طبقه‌بندی وضعیت سر.

    Attributes:
        yaw_threshold: زاویه انحراف چپ/راست (درجه). |yaw| بزرگ‌تر از این
            مقدار به معنی «برگرداندن سر» است (مقدار مرجع پروژه: 25).
        pitch_down_threshold: آستانه نگاه به پایین (درجه). pitch کوچک‌تر از
            این مقدار یعنی سر به پایین خم شده (مثلاً استفاده از موبایل)
            (مقدار مرجع پروژه: -15).
        pitch_up_threshold: آستانه نگاه به بالا (درجه).
        roll_threshold: آستانه خم کردن سر به پهلو (درجه).
    """

    yaw_threshold: float = 25.0
    pitch_down_threshold: float = -15.0
    pitch_up_threshold: float = 20.0
    roll_threshold: float = 30.0


@dataclass(frozen=True)
class DetectionConfig:
    """پارامترهای تشخیص چهره با MediaPipe Face Landmarker.

    Attributes:
        model_path: مسیر فایل مدل ``face_landmarker.task``.
        num_faces: حداکثر تعداد چهره‌های شناسایی‌شده در هر فریم.
        min_face_detection_confidence: حداقل اطمینان رؤیت چهره.
        min_tracking_confidence: حداقل اطمینان ردیابی بین فریم‌ها.
        max_lost_frames: حداکثر تعداد فریمی که ردیاب چهره گم‌شده را
            با آخرین موقعیت «نگه می‌دارد» (دورریز نویز کوتاه).
        process_width: عرض پردازش؛ فریم‌های بزرگ‌تر قبل از تشخیص به
            این عرض کوچک می‌شوند (تسریع Real-Time روی CPU). لندمارک‌های
            خروجی همیشه به مختصات تصویر اصلی برمی‌گردند. 0 = بدون تغییر اندازه.
    """

    model_path: str = "assets/face_landmarker.task"
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_lost_frames: int = 10
    process_width: int = 480


@dataclass(frozen=True)
class DrowsinessConfig:
    """پارامترهای تشخیص خواب‌آلودگی.

    EAR پایین‌تر از ``ear_closed_threshold`` نشانه بسته بودن چشم است؛
    اگر این وضعیت بیش از ``closed_seconds`` ثانیه باقی بماند،
    راننده خواب‌آلود تلقی می‌شود.

    Attributes:
        ear_closed_threshold: آستانه بسته بودن چشم بر اساس Eye Aspect Ratio.
        ear_blink_frames: حداکثر طول پلک زدن طبیعی (فریم) که هشدار ایجاد نمی‌کند.
        closed_seconds: مدت بسته بودن پیوسته چشم‌ها برای اعلام خواب‌آلودگی (ثانیه).
    """

    ear_closed_threshold: float = 0.20
    ear_blink_frames: int = 3
    closed_seconds: float = 2.0


@dataclass(frozen=True)
class DecisionConfig:
    """پارامترهای موتور تصمیم و هشدار.

    فیلتر هیسترزیس: یک وضعیت جدید تنها زمانی پذیرفته می‌شود که
    ``state_confirm_frames`` فریم متوالی تکرار شود؛ این کار از
    هشدارهای کاذب نویز لحظه‌ای جلوگیری می‌کند.

    Attributes:
        state_confirm_frames: تعداد فریم تأیید برای تغییر وضعیت (هیسترزیس).
        distraction_alert_seconds: مدت حواس‌پرتی پیوسته برای فعال‌شدن هشدار (ثانیه).
        drowsiness_alert: فعال‌سازی هشدار خواب‌آلودگی.
        alert_cooldown_seconds: حداقل فاصله بین دو هشدار متوالی (ثانیه).
        assumed_fps: نرخ فریم فرضی برای تبدیل فریم به ثانیه هنگامی که
            timestamp واقعی در دسترس نیست.
        lateral_alert_enabled: هشدار برای چرخش سر به چپ/راست.
            پیش‌فرض **خاموش**: نگاه به آینه‌ها، گذرها و چپ/راست هنگام
            پیچ، وظیفه قانونی راننده است و حواس‌پرتی محسوب نمی‌شود؛
            این وضعیت‌ها تنها برچسب اطلاع‌رسانی (INFO) می‌گیرند و هرگز
            هشدار WARNING صادر نمی‌کنند. برای ناوگان‌هایی با سیاست
            سخت‌گیرانه (پایش دقیق راننده) قابل فعال‌سازی است.
    """

    state_confirm_frames: int = 5
    distraction_alert_seconds: float = 2.0
    drowsiness_alert: bool = True
    alert_cooldown_seconds: float = 3.0
    assumed_fps: float = 30.0
    lateral_alert_enabled: bool = False


@dataclass(frozen=True)
class SmoothingConfig:
    """پارامترهای هموارسازی زمانی زوایای سر.

    Attributes:
        use_kalman: فعال‌سازی فیلتر کالمن روی Yaw/Pitch/Roll.
        process_noise: واریانس نویز فرایند (نوسان واقعی سر).
        measurement_noise: واریانس نویز اندازه‌گیری (خطای تخمین PnP).
        ma_window: طول پنجره میانگین متحرک پشتیبان.
    """

    use_kalman: bool = True
    process_noise: float = 1.0
    measurement_noise: float = 4.0
    ma_window: int = 5


@dataclass(frozen=True)
class TelemetryConfig:
    """پارامترهای ثبت داده (Telemetry).

    Attributes:
        db_path: مسیر فایل پایگاه‌داده SQLite.
        export_dir: پوشه خروجی JSON/CSV.
        log_every_frame: ثبت وضعیت هر فریم (True) یا فقط رخدادها (False).
        camera_id: شناسه دوربین/منبع برای درج در رکوردها.
    """

    db_path: str = "telemetry/dms_log.db"
    export_dir: str = "telemetry"
    log_every_frame: bool = True
    camera_id: str = "webcam-0"


@dataclass(frozen=True)
class PhoneConfig:
    """پارامترهای تشخیص تلفن همراه (تماس دستی / گوشی در دست).

    گوشی با مدل **YOLO11n ONNX** (ONNX Runtime، کلاس ``cell phone``
    از COCO) شناسایی می‌شود؛ هرگاه مرکز جعبه گوشی داخل ناحیه
    بزرگ‌شده جعبه چهره باشد («نزدیک چهره»)، وضعیت PHONE اعلام
    می‌شود. گوشی روی پایه داشبورد (دور از چهره) هشدار ایجاد نمی‌کند.

    Attributes:
        enabled: فعال/غیرفعال بودن ماژول تلفن همراه.
        model_path: مسیر فایل مدل ``yolo11n.onnx``.
        score_threshold: حداقل اطمینان تشخیص گوشی.
        nms_threshold: آستانه IoU برای حذف جعبه‌های تکراری.
        input_size: اندازه ورودی مدل (مضرب 32؛ کوچک‌تر = سریع‌تر).
        max_results: حداکثر تعداد جعبه گوشی برگشتی.
        near_face_expand: نسبت بزرگ‌نمایی جعبه چهره برای تعریف
            «نزدیک چهره» (۰.۶ یعنی نصف طول/عرض چهره از هر طرف).
        detect_every: اجرای تشخیص گوشی هر چند فریم (صرفه‌جویی CPU).
        hold_frames: نگهداری نتیجه آخرین تشخیص بین اجراها (فریم).
        process_width: عرض پردازش فریم برای این مدل (0 = بدون تغییر).
    """

    enabled: bool = True
    model_path: str = "assets/yolo11n.onnx"
    score_threshold: float = 0.4
    nms_threshold: float = 0.45
    input_size: int = 320
    max_results: int = 5
    near_face_expand: float = 0.6
    detect_every: int = 15
    hold_frames: int = 15
    process_width: int = 480


@dataclass(frozen=True)
class DMSConfig:
    """پیکربندی سراسری سیستم تشخیص حواس‌پرتی راننده."""

    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    target_fps: int = 30

    @classmethod
    def from_json(cls, path: str | Path) -> "DMSConfig":
        """ساخت پیکربندی از فایل JSON.

        کلیدهای ناشناخته نادیده گرفته می‌شوند و کلیدهای غایب به
        مقدار پیش‌فرض برمی‌گردند؛ ساختار فایل به شکل زیر است::

            {
              "head_pose": {"yaw_threshold": 25.0},
              "decision": {"state_confirm_frames": 5}
            }

        Args:
            path: مسیر فایل JSON.

        Returns:
            نمونه ``DMSConfig`` ساخته‌شده از فایل.

        Raises:
            FileNotFoundError: اگر فایل وجود نداشته باشد.
            json.JSONDecodeError: اگر فایل JSON معتبر نباشد.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        valid = {f.name: f.type for f in fields(cls)}
        updates = {}
        for key, value in raw.items():
            if key not in valid:
                continue
            sub = getattr(cfg, key)
            if hasattr(sub, "__dataclass_fields__"):
                updates[key] = type(sub)(**{**asdict(sub), **value})
            else:
                updates[key] = value
        return cls(**updates)

    def to_dict(self) -> dict:
        """تبدیل کامل پیکربندی به دیکشنری (برای ثبت در گزارش‌ها)."""
        return asdict(self)


def default_camera_matrix(image_size: Tuple[int, int]) -> "object":
    """ساخت ماتریس دوربین تقریبی برای SolvePnP.

    چون در کاربرد واقعی کالیبراسیون دوربین در دسترس نیست، از یک
    ماتریس تقریبی با فرض FOV ≈ 60° استفاده می‌کنیم؛ این رویکرد در
    تمام مقالات مرجع DMS مبتنی بر SolvePnP رایج است.

    Args:
        image_size: اندازه تصویر به صورت ``(width, height)``.

    Returns:
        ماتریس داخلی دوربین 3×3 (numpy array).
    """
    import numpy as np

    width, height = image_size
    focal = width  # تقریب رایج: f ≈ عرض تصویر برای FOV ≈ 53°
    center = (width / 2.0, height / 2.0)
    return np.array(
        [[focal, 0, center[0]],
         [0, focal, center[1]],
         [0, 0, 1]],
        dtype=np.float64,
    )
