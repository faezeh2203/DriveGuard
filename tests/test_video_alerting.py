"""تست‌های منبع ویدیو و انوتیشن هشدار."""

import numpy as np
import pytest

from dms.alerting import ALERT_COLORS, AlertManager
from dms.attention import AttentionState
from dms.decision import AlertLevel, Decision
from dms.video import VideoSource


class TestAlertAnnotation:
    """رسم وضعیت روی فریم."""

    def _decision(self, level, state=AttentionState.FORWARD, message=""):
        return Decision(stable_state=state, alert_level=level, message=message)

    def test_none_level_draws_frame(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        out = AlertManager(enable_sound=False).annotate(
            img, self._decision(AlertLevel.NONE))
        assert out.shape == img.shape
        assert not np.array_equal(out, img)  # چیزی رسم شده

    def test_critical_uses_red(self):
        img = np.full((240, 320, 3), 128, dtype=np.uint8)
        out = AlertManager(enable_sound=False).annotate(
            img, self._decision(AlertLevel.CRITICAL, message="خطر"))
        # پیکسل گوشه بالا-چپ باید رنگ قرمز قاب باشد
        assert tuple(out[3, 3]) == ALERT_COLORS[AlertLevel.CRITICAL]

    def test_original_frame_untouched(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        copy = img.copy()
        AlertManager(enable_sound=False).annotate(
            img, self._decision(AlertLevel.WARNING, message="x"))
        assert np.array_equal(img, copy)

    def test_alert_event_emitted(self):
        mgr = AlertManager(enable_sound=False)
        d = self._decision(AlertLevel.WARNING, AttentionState.LEFT, "متن")
        evt = mgr.handle(d, timestamp=1.0)
        assert evt is not None
        assert evt.level is AlertLevel.WARNING
        assert evt.state == "LEFT"

    def test_none_level_no_event(self):
        mgr = AlertManager(enable_sound=False)
        d = self._decision(AlertLevel.NONE)
        assert mgr.handle(d, timestamp=0.0) is None

    def test_sound_disabled_is_silent(self):
        mgr = AlertManager(enable_sound=False)
        d = self._decision(AlertLevel.CRITICAL, message="!")
        assert mgr.handle(d, timestamp=0.0) is not None  # بدون خطا


class TestVideoSource:
    """منبع ویدیو با فایل ساخته‌شده از فریم‌های مصنوعی."""

    @pytest.fixture
    def video_file(self, tmp_path):
        import cv2

        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
        for i in range(15):
            frame = np.full((240, 320, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_open_file(self, video_file):
        src = VideoSource(str(video_file))
        assert src.frame_size == (320, 240)
        src.release()

    def test_read_frames(self, video_file):
        with VideoSource(str(video_file)) as src:
            frames = list(src)
        assert len(frames) == 15
        assert frames[0].image.shape == (240, 320, 3)

    def test_frame_metadata(self, video_file):
        with VideoSource(str(video_file)) as src:
            f0 = src.read()
            f1 = src.read()
        assert f0.frame_id < f1.frame_id
        assert f1.timestamp > f0.timestamp

    def test_invalid_source_raises(self):
        with pytest.raises(IOError):
            VideoSource(str("Z:/definitely/not/a/file.mp4"))

    def test_fps_default_for_missing(self, video_file):
        src = VideoSource(str(video_file))
        assert src.fps > 1.0
        src.release()
