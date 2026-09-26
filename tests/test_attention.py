"""تست‌های طبقه‌بندی وضعیت توجه با آستانه‌های مرجع پروژه (yaw>25, pitch<-15)."""

import pytest

from dms.attention import AttentionClassifier, AttentionResult, AttentionState
from dms.config import HeadPoseConfig
from dms.geometry import EulerAngle
from tests.conftest import make_synthetic_face


@pytest.fixture
def clf() -> AttentionClassifier:
    return AttentionClassifier(HeadPoseConfig())


class TestThresholdRules:
    """قواعد آستانه‌ای مطابق مشخصات پروژه."""

    def test_forward(self, clf):
        assert clf.classify(EulerAngle(0, 0, 0)).state is AttentionState.FORWARD

    def test_yaw_right(self, clf):
        # طبق قرارداد پروژه: yaw مثبت = راست تصویر
        assert clf.classify(EulerAngle(0, 30, 0)).state is AttentionState.RIGHT

    def test_yaw_left(self, clf):
        assert clf.classify(EulerAngle(0, -30, 0)).state is AttentionState.LEFT

    def test_yaw_boundary_25(self, clf):
        # دقیقاً روی آستانه 25° هنوز FORWARD است
        assert clf.classify(EulerAngle(0, 25, 0)).state is AttentionState.FORWARD
        assert clf.classify(EulerAngle(0, 25.1, 0)).state is AttentionState.RIGHT

    def test_down_phone_usage(self, clf):
        assert clf.classify(EulerAngle(-20, 0, 0)).state is AttentionState.DOWN

    def test_pitch_boundary_minus15(self, clf):
        assert clf.classify(EulerAngle(-15, 0, 0)).state is AttentionState.FORWARD
        assert clf.classify(EulerAngle(-15.5, 0, 0)).state is AttentionState.DOWN

    def test_up(self, clf):
        assert clf.classify(EulerAngle(25, 0, 0)).state is AttentionState.UP

    def test_yaw_dominates_pitch(self, clf):
        # هم yaw و هم pitch خارج آستانه: قاعده yaw اولویت دارد
        assert clf.classify(EulerAngle(-30, 40, 0)).state is AttentionState.RIGHT

    def test_is_distracted_property(self):
        assert AttentionState.FORWARD.is_distracted is False
        for s in (AttentionState.LEFT, AttentionState.RIGHT,
                  AttentionState.DOWN, AttentionState.UP,
                  AttentionState.NO_FACE):
            assert s.is_distracted is True

    def test_fa_labels_exist(self):
        for s in AttentionState:
            assert s.label_fa


class TestGazeIris:
    """ترکیب جهت نگاه عنبیه در حالت مرزی سر."""

    def test_gaze_zero_without_iris_landmarks(self, clf):
        res = clf.classify(EulerAngle(0, 0, 0), landmarks=None)
        assert res.gaze_offset == 0.0

    def test_468_landmarks_ignored_for_gaze(self, clf):
        face = make_synthetic_face(0, 0, 0)[:468]
        res = clf.classify(EulerAngle(0, 0, 0), landmarks=face)
        assert res.gaze_offset == 0.0
        assert res.state is AttentionState.FORWARD


class TestEARComputation:
    """EAR باید از لندمارک‌های مصنوعی درست خوانده شود."""

    def test_ear_from_synthetic_face(self, clf):
        face = make_synthetic_face(0, 0, 0, ear=0.28)
        res = clf.classify(EulerAngle(0, 0, 0), landmarks=face)
        assert res.ear == pytest.approx(0.28, abs=0.02)

    def test_closed_ear(self, clf):
        face = make_synthetic_face(0, 0, 0, ear=0.10)
        res = clf.classify(EulerAngle(0, 0, 0), landmarks=face)
        assert res.ear == pytest.approx(0.10, abs=0.02)

    def test_no_landmarks_ear_zero(self, clf):
        assert clf.classify(EulerAngle(0, 0, 0), landmarks=None).ear == 0.0


class TestResultContract:
    """قرارداد خروجی."""

    def test_result_type(self, clf):
        res = clf.classify(EulerAngle(0, 0, 0))
        assert isinstance(res, AttentionResult)
        assert res.raw_state is AttentionState.FORWARD
        assert res.state is AttentionState.FORWARD
