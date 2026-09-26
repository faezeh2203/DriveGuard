"""تست‌های ردیاب چهره (FaceTracker)."""

import pytest

from dms.tracker import FaceTracker

BBOX = (100.0, 80.0, 200.0, 240.0)


@pytest.fixture
def tracker() -> FaceTracker:
    return FaceTracker(max_lost_frames=5)


class TestTracking:
    def test_new_track_centers_bbox(self, tracker):
        s = tracker.update(BBOX)
        assert s.cx == pytest.approx(200.0)
        assert s.cy == pytest.approx(200.0)
        assert s.w == pytest.approx(200.0)

    def test_smoothing_reduces_jitter(self, tracker):
        tracker.update(BBOX)
        s = tracker.update((105.0, 80.0, 200.0, 240.0))
        # فیلتر کالمن فقط بخشی از پرش را دنبال می‌کند
        assert 200.0 < s.cx < 205.0

    def test_bbox_property(self, tracker):
        s = tracker.update(BBOX)
        x, y, w, h = s.bbox
        assert x == pytest.approx(s.cx - w / 2)


class TestLostHandling:
    def test_short_loss_holds_target(self, tracker):
        tracker.update(BBOX)
        for _ in range(3):
            s = tracker.update(None)
        assert tracker.is_tracking
        assert s.cx == pytest.approx(200.0, abs=2.0)

    def test_long_loss_resets(self, tracker):
        tracker.update(BBOX)
        for _ in range(6):  # بیشتر از max_lost_frames=5
            tracker.update(None)
        assert not tracker.is_tracking
        assert tracker.state.age == 0

    def test_recovery_after_short_loss(self, tracker):
        tracker.update(BBOX)
        tracker.update(None)
        s = tracker.update(BBOX)
        assert s.lost_frames == 0
        assert tracker.is_tracking


class TestInvalidInput:
    def test_zero_size_bbox_clamped(self, tracker):
        s = tracker.update((0.0, 0.0, 0.0, 0.0))
        assert s.w >= 1.0
        assert s.h >= 1.0
