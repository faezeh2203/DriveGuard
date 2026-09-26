"""تست‌های هندسه: مدل سه‌بعدی صورت، تبدیل چرخش و EAR."""

import math

import numpy as np
import pytest

from dms.geometry import (
    EulerAngle,
    GENERIC_3D_FACE_MODEL,
    eye_aspect_ratio,
    euler_to_rotation_vector,
    project_face_model,
    rotation_matrix_to_euler,
    rotation_vector_to_euler,
)


class TestEulerRoundtrip:
    """تبدیل زوایا به بردار چرخش و بازگشت باید بازتاب دقیق باشد."""

    @pytest.mark.parametrize("pitch,yaw,roll", [
        (0, 0, 0),
        (10, -20, 5),
        (-15, 30, -10),
        (45, -45, 45),
        (89, 0, 0),
        (0, 89, 0),
        (-30, 15, -25),
    ])
    def test_roundtrip(self, pitch, yaw, roll):
        rvec = euler_to_rotation_vector(pitch, yaw, roll)
        out = rotation_vector_to_euler(rvec)
        assert out.pitch == pytest.approx(pitch, abs=1e-6)
        assert out.yaw == pytest.approx(yaw, abs=1e-6)
        assert out.roll == pytest.approx(roll, abs=1e-6)

    def test_zero_rotation_is_zero(self):
        out = rotation_vector_to_euler(np.zeros(3))
        assert (out.pitch, out.yaw, out.roll) == (0, 0, 0)


class TestRotationMatrixDecomposition:
    """تجزیه ماتریس چرخش باید با چرخش‌های محوری استاندارد سازگار باشد."""

    def test_identity(self):
        out = rotation_matrix_to_euler(np.eye(3))
        assert (out.pitch, out.yaw, out.roll) == (0, 0, 0)

    def test_pure_yaw_is_second_component(self):
        # چرخش 90 درجه حول y: atan2(R[2,1], R[2,2]) = atan2(0, 0) مرزی است؛
        # زاویه کمتر انتخاب می‌کنیم تا ناپایداری عددی رخ ندهد
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        out = rotation_matrix_to_euler(ry)
        assert out.yaw == pytest.approx(30, abs=1e-6)
        assert out.pitch == pytest.approx(0, abs=1e-6)

    def test_matrix_input_validated(self):
        with pytest.raises((ValueError, TypeError, IndexError)):
            rotation_matrix_to_euler(np.zeros((2, 2)))


class TestFaceModel:
    """مدل سه‌بعدی صورت باید متقارن و کامل باشد."""

    def test_shape(self):
        assert GENERIC_3D_FACE_MODEL.shape == (6, 3)

    def test_bilateral_symmetry(self):
        # نقاط چپ/راست (چشم و دهان) باید قرینه در x باشند
        nose, chin, le, re, lm, rm = GENERIC_3D_FACE_MODEL
        assert le[0] == pytest.approx(-re[0])
        assert lm[0] == pytest.approx(-rm[0])
        assert le[1] == pytest.approx(re[1])

    def test_nose_at_origin(self):
        assert np.allclose(GENERIC_3D_FACE_MODEL[0], 0)


class TestProjection:
    """پروجکشن مدل باید رفتار پرسپکتیو صحیح داشته باشد."""

    def test_frontal_projection_symmetric_about_center(self):
        pts = project_face_model(0, 0, 0)
        nose, chin, le, re, lm, rm = pts
        cx = nose[0]
        assert le[0] < cx < re[0]
        assert abs(le[0] - cx) == pytest.approx(abs(re[0] - cx), rel=1e-6)
        # در نگاه روبه‌جلو، چانه زیر بینی است (محور y تصویر به پایین مثبت)
        assert chin[1] > nose[1]

    def test_yaw_moves_chin_horizontally(self):
        frontal = project_face_model(0, 0, 0)
        turned = project_face_model(0, 25, 0)
        # چرخش سر باید چانه (که دور از محور چرخش است) را جابه‌جا کند
        assert turned[1][0] != pytest.approx(frontal[1][0], abs=1.0)

    def test_yaw_moves_face_right_for_positive_yaw(self):
        frontal = project_face_model(0, 0, 0)
        turned = project_face_model(0, 25, 0)
        # قرارداد: yaw مثبت = گرداندن چهره به راست تصویر
        assert turned[1][0] < frontal[1][0]  # چانه به سمت چپ تصویر می‌رود

    def test_pitch_moves_chin_vertically(self):
        frontal = project_face_model(0, 0, 0)
        down = project_face_model(-20, 0, 0)
        assert down[1][1] != pytest.approx(frontal[1][1], abs=1.0)

    def test_points_finite(self):
        pts = project_face_model(10, -10, 5)
        assert np.all(np.isfinite(pts))


class TestEAR:
    """Eye Aspect Ratio باید با هندسه چشم رفتار صحیح داشته باشد."""

    def _eye(self, vertical: float, horizontal: float) -> np.ndarray:
        return np.array([
            [0, 0], [-horizontal / 4, -vertical / 2],
            [horizontal / 4, -vertical / 2], [horizontal, 0],
            [horizontal / 4, vertical / 2], [-horizontal / 4, vertical / 2],
        ], dtype=float)

    def test_open_eye_typical_value(self):
        # چشم باز: EAR حدود 0.25 تا 0.35
        ear = eye_aspect_ratio(self._eye(vertical=16, horizontal=60))
        assert 0.2 < ear < 0.4

    def test_closed_eye_smaller_than_open(self):
        assert eye_aspect_ratio(self._eye(2, 60)) < eye_aspect_ratio(self._eye(16, 60))

    def test_degenerate_eye_returns_zero(self):
        pts = np.zeros((6, 2))
        assert eye_aspect_ratio(pts) == 0.0

    def test_scale_invariance(self):
        # EAR بدون بعد است؛ بزرگ‌نمایی یکنواخت نباید تغییرش دهد
        e1 = eye_aspect_ratio(self._eye(10, 40))
        e2 = eye_aspect_ratio(self._eye(20, 80))
        assert e1 == pytest.approx(e2)

    def test_euler_angle_dataclass(self):
        a = EulerAngle(pitch=1.5, yaw=2.5, roll=3.5)
        assert a.as_tuple() == (1.5, 2.5, 3.5)
