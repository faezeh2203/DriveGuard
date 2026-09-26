"""تست‌های تخمین وضعیت سر (SolvePnP) با داده مصنوعی Ground Truth دار."""

import numpy as np
import pytest

from dms.config import default_camera_matrix
from dms.head_pose import HeadPoseEstimator
from dms.geometry import FACE_MODEL_LANDMARK_IDS, project_face_model

SIZE = (640, 480)
CAMERA = default_camera_matrix(SIZE)


def build_points(pitch, yaw, roll) -> np.ndarray:
    """ساخت شش نقطه تصویری از مدل با چرخش معلوم (Ground Truth).

    پروجکشن با همان ماتریس دوربینی انجام می‌شود که تخمین‌گر استفاده
    می‌کند تا خطای مدل دوربین در تست دخالت نکند.
    """
    pts2d = project_face_model(pitch, yaw, roll, translation=(0, 0, 500),
                               camera_matrix=CAMERA)
    full = np.zeros((478, 3))
    for i, mid in enumerate(FACE_MODEL_LANDMARK_IDS):
        full[mid, :2] = pts2d[i]
    return full[list(FACE_MODEL_LANDMARK_IDS), :2]


def make_estimator(**kwargs) -> HeadPoseEstimator:
    """تخمین‌گر با ماتریس دوربین معلوم (هم‌سان با داده آزمون)."""
    return HeadPoseEstimator(camera_matrix=CAMERA, **kwargs)


class TestRecovery:
    """زاویه معلوم باید از نقطه‌های تصویری بازیابی شود."""

    @pytest.mark.parametrize("pitch,yaw,roll", [
        (0, 0, 0),
        (10, 20, 5),
        (-15, -25, -10),
        (20, -30, 15),
        (-25, 35, 20),
    ])
    def test_angle_recovery(self, pitch, yaw, roll):
        est = make_estimator(use_extrinsic_guess=False)
        pose = est.estimate(build_points(pitch, yaw, roll), SIZE)
        assert pose.success
        assert pose.angles.pitch == pytest.approx(pitch, abs=2.0)
        assert pose.angles.yaw == pytest.approx(yaw, abs=2.0)
        assert pose.angles.roll == pytest.approx(roll, abs=2.0)

    def test_reprojection_error_small_for_exact_points(self):
        est = make_estimator(use_extrinsic_guess=False)
        pose = est.estimate(build_points(5, 10, 0), SIZE)
        assert pose.reprojection_error < 1.0

    def test_frontal_pose_near_zero(self):
        est = make_estimator(use_extrinsic_guess=False)
        pose = est.estimate(build_points(0, 0, 0), SIZE)
        assert abs(pose.angles.pitch) < 1
        assert abs(pose.angles.yaw) < 1
        assert abs(pose.angles.roll) < 1

    def test_pitch_down_is_negative(self):
        """قرارداد پروژه: نگاه به پایین = pitch منفی."""
        est = make_estimator(use_extrinsic_guess=False)
        pose = est.estimate(build_points(-20, 0, 0), SIZE)
        assert pose.angles.pitch < -10

    def test_yaw_positive_is_right(self):
        """قرارداد پروژه: yaw مثبت = چرخش به راست تصویر."""
        est = make_estimator(use_extrinsic_guess=False)
        pose = est.estimate(build_points(0, 30, 0), SIZE)
        assert pose.angles.yaw > 10


class TestRobustness:
    """رفتار تخمین‌گر در ورودی‌های خراب/نویزی."""

    def test_wrong_point_count_raises(self):
        est = HeadPoseEstimator()
        with pytest.raises(ValueError):
            est.estimate(np.zeros((5, 2)), SIZE)

    def test_noisy_points_still_close(self):
        est = make_estimator(use_extrinsic_guess=False)
        rng = np.random.default_rng(0)
        pts = build_points(8, -12, 3) + rng.normal(0, 1.0, (6, 2))
        pose = est.estimate(pts, SIZE)
        assert pose.success
        assert pose.angles.pitch == pytest.approx(8, abs=5)
        assert pose.angles.yaw == pytest.approx(-12, abs=5)

    def test_reprojection_error_reflects_noise(self):
        est = make_estimator(use_extrinsic_guess=False)
        clean = est.estimate(build_points(0, 0, 0), SIZE)
        noisy = est.estimate(build_points(0, 0, 0) + np.full((6, 2), 5.0), SIZE)
        assert noisy.reprojection_error > clean.reprojection_error

    def test_default_camera_matrix_fallback(self):
        """بدون ماتریس کالیبره، تقریب پیش‌فرض استفاده می‌شود (خطای کمی بیشتر)."""
        est = HeadPoseEstimator(use_extrinsic_guess=False)
        pose = est.estimate(build_points(0, 0, 0), SIZE)
        assert pose.success
        assert abs(pose.angles.pitch) < 5


class TestExtrinsicGuess:
    """حالت حدس اولیه باید پایداری بین فریم‌ها را حفظ کند."""

    def test_sequential_frames_consistent(self):
        est = make_estimator(use_extrinsic_guess=True)
        first = est.estimate(build_points(0, 0, 0), SIZE)
        second = est.estimate(build_points(5, 10, 0), SIZE)
        assert first.success and second.success
        assert second.angles.yaw == pytest.approx(10, abs=2.0)

    def test_reset_clears_guess(self):
        est = make_estimator()
        est.estimate(build_points(0, 0, 0), SIZE)
        est.reset()
        pose = est.estimate(build_points(0, 0, 0), SIZE)
        assert pose.success
