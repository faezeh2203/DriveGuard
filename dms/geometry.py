"""
هندسه صورت و تبدیلات چرخش (Geometry)
=====================================

این ماژول شامل موارد زیر است:

1. مدل سه‌بعدی عمومی صورت (Generic 3D Face Model) شامل شش نقطه شاخص
   که با نقاط متناظر آن‌ها در مدل ۴۶۸ لندمارکی MediaPipe جفت می‌شود.
2. تبدیل بردار چرخش (Rotation Vector خروجی ``cv2.solvePnP``) به
   زوایای اولر (Pitch/Yaw/Roll) بر حسب درجه.

قرارداد مختصات:
    از دستگاه مختصات دوربین OpenCV استفاده می‌کنیم:
    ``x`` به سمت راست تصویر، ``y`` به سمت پایین تصویر و ``z``
    به سمت داخل صحنه. مدل سه‌بعدی صورت طوری تعریف شده که در حالت
    «نگاه مستقیم» محور +z آن رو به دوربین باشد.

قرارداد زوایا (خروجی سیستم):
    - **Pitch**: بالا/پایین بردن سر. مثبت = نگاه به بالا، منفی = نگاه به پایین.
    - **Yaw**: چپ/راست گرداندن سر. مثبت = گرداندن سر به راست تصویر.
    - **Roll**: خم کردن سر به پهلو (کج کردن). مثبت = کج به راست تصویر.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# مدل سه‌بعدی عمومی صورت (بر حسب واحد دلخواه ≈ میلی‌متر)
# ---------------------------------------------------------------------------
# این شش نقطه، نقاط استاندارد مرجع برای تخمین وضعیت سر با SolvePnP هستند
# و با اندیس‌های لندمارک MediaPipe Face Mesh در زیر جفت شده‌اند.
# هر ردیف: (x, y, z) در دستگاه مختصات سر؛ مبدأ روی نوک بینی.
# قرارداد محورها با تصویر سازگار است: +x راست تصویر، +y پایین تصویر
# (چانه زیر بینی یعنی y بزرگ‌تر) و +z به سمت دوربین در نگاه روبه‌جلو.
GENERIC_3D_FACE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),          #  نوک بینی (مبدأ چرخش سر)
        (0.0, 330.0, -65.0),      #  نوک چانه (پایین‌تر از بینی)
        (-225.0, -170.0, -135.0), #  گوشه خارجی چشم چپ (بالا-چپ)
        (225.0, -170.0, -135.0),  #  گوشه خارجی چشم راست (بالا-راست)
        (-150.0, 150.0, -125.0),  #  گوشه دهان چپ
        (150.0, 150.0, -125.0),   #  گوشه دهان راست
    ],
    dtype=np.float64,
)

# اندیس نقاط متناظر در مدل ۴۶۸ لندمارکی MediaPipe Face Mesh
# (نقاط استاندارد: نوک بینی 1، چانه 199، گوشه چشم‌ها 33/263، گوشه دهان‌ها 61/291)
FACE_MODEL_LANDMARK_IDS = (1, 199, 33, 263, 61, 291)

# اندیس لندمارک‌های چشم برای محاسبه Eye Aspect Ratio (ترتیب پادساعتگرد)
LEFT_EYE_EAR_IDS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_IDS = (362, 385, 387, 263, 373, 380)

# اندیس لندمارک‌های عنبیه (در مدل 478 لندمارکی با iris)
LEFT_IRIS_IDS = (468, 469, 470, 471, 472)
RIGHT_IRIS_IDS = (473, 474, 475, 476, 477)


@dataclass(frozen=True)
class EulerAngle:
    """زوایای اولر سر بر حسب درجه.

    Attributes:
        pitch: بالا/پایین (x).
        yaw: چپ/راست (y).
        roll: کج به پهلو (z).
    """

    pitch: float
    yaw: float
    roll: float

    def as_tuple(self) -> tuple:
        """بازگرداندن زوایا به صورت ``(pitch, yaw, roll)``."""
        return (self.pitch, self.yaw, self.roll)


def rotation_vector_to_euler(rvec: np.ndarray) -> EulerAngle:
    """تبدیل بردار چرخش به زوایای اولر Pitch/Yaw/Roll.

    بردار چرخش با فرمول رودریگز به ماتریس چرخش تبدیل و سپس با
    تجزیه استاندارد ``xyz`` به زوایای اولر تفکیک می‌شود
    (پیاده‌سازی معادل ``scipy.Rotation.as_euler('xyz')`` بدون وابستگی).

    Args:
        rvec: بردار چرخش 3×1 خروجی ``cv2.solvePnP``.

    Returns:
        شیء :class:`EulerAngle` بر حسب درجه.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    rot_matrix, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_euler(rot_matrix)


def rotation_matrix_to_euler(rot_matrix: np.ndarray) -> EulerAngle:
    """تجزیه ماتریس چرخش به زوایای اولر.

    ماتریس چرخش به صورت ``R = Rz(roll) · Ry(yaw) · Rx(pitch)``
    (چرخش به ترتیب حول محورهای ثابت x، y، z) تجزیه می‌شود؛ فرمول
    استاندارد این قرارداد::

        pitch = atan2(R[2,1], R[2,2])                       # محور x
        yaw   = atan2(-R[2,0], sqrt(R[2,1]² + R[2,2]²))     # محور y
        roll  = atan2(R[1,0], R[0,0])                       # محور z

    Args:
        rot_matrix: ماتریس چرخش 3×3.

    Returns:
        زوایای اولر بر حسب درجه.
    """
    rot_matrix = np.asarray(rot_matrix, dtype=np.float64)
    sin_yaw = -rot_matrix[2, 0]
    sin_yaw = float(np.clip(sin_yaw, -1.0, 1.0))
    pitch = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    yaw = np.arctan2(sin_yaw, np.sqrt(rot_matrix[2, 1] ** 2 + rot_matrix[2, 2] ** 2))
    roll = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return EulerAngle(
        pitch=float(np.degrees(pitch)),
        yaw=float(np.degrees(yaw)),
        roll=float(np.degrees(roll)),
    )


def euler_to_rotation_vector(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """ساخت بردار چرخش از زوایای اولر (عکس تابع بالا؛ برای تست و شبیه‌سازی).

    Args:
        pitch, yaw, roll: زوایا بر حسب درجه.

    Returns:
        بردار چرخش 3×1 numpy.
    """
    rx, ry, rz = np.radians([pitch, yaw, roll])
    # R = Rz(roll) @ Ry(yaw) @ Rx(pitch)  (معکوس تجزیه xyz)
    rx_m = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    ry_m = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rz_m = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rot_matrix = rz_m @ ry_m @ rx_m
    rvec, _ = cv2.Rodrigues(rot_matrix)
    return rvec.reshape(3, 1)


def project_face_model(
    pitch: float,
    yaw: float,
    roll: float,
    translation: tuple = (0.0, 0.0, 500.0),
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """پروجکشن مدل سه‌بعدی صورت با چرخش معلوم به مختصات تصویر.

    این تابع برای ساخت داده‌های آزمون (Ground Truth مصنوعی) استفاده
    می‌شود: مدل صورت را با زوایای معلوم می‌چرخانیم، با ماتریس دوربین
    به صفحه تصویر می‌تابانیم و خروجی را به‌عنوان لندمارک‌های ورودی
    به تخمین‌گر وضعیت سر می‌دهیم.

    Args:
        pitch, yaw, roll: چرخش معلوم سر بر حسب درجه.
        translation: انتقال ``(tx, ty, tz)`` سر نسبت به دوربین.
        camera_matrix: ماتریس داخلی دوربین (در صورت None ساخته می‌شود).

    Returns:
        آرایه (6, 2) مختصات تصویری شش نقطه مدل.
    """
    rvec = euler_to_rotation_vector(pitch, yaw, roll)
    rot_matrix, _ = cv2.Rodrigues(rvec)
    tvec = np.array(translation, dtype=np.float64).reshape(3, 1)
    if camera_matrix is None:
        camera_matrix = np.array(
            [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]], dtype=np.float64
        )
    pts_cam = (rot_matrix @ GENERIC_3D_FACE_MODEL.T) + tvec
    pts_img = (camera_matrix @ pts_cam).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    return pts_img


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """محاسبه Eye Aspect Ratio (EAR) برای تشخیص باز/بسته بودن چشم.

    EAR نسبت فاصله عمودی پلک‌ها به فاصله افقی گوشه‌های چشم است
    (Soukupová & Čech, 2016)::

        EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)

    چشم باز EAR ≈ 0.25..0.35 و چشم بسته EAR ≈ < 0.2 دارد.

    Args:
        eye_points: آرایه (6, 2) مختصات شش لندمارک چشم
            (به ترتیب ``LEFT_EYE_EAR_IDS`` / ``RIGHT_EYE_EAR_IDS``).

    Returns:
        نسبت EAR (بدون بعد).
    """
    pts = np.asarray(eye_points, dtype=np.float64)
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-9:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))
