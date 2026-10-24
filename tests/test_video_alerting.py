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


class TestSoundWorker:
    def test_handle_never_blocks(self, monkeypatch):
        import time
        from dms.alerting import SoundWorker
        beeps = []
        monkeypatch.setattr(SoundWorker, "_play",
                            lambda self, wav: (beeps.append(wav), time.sleep(0.25)))
        mgr = AlertManager(enable_sound=True)
        try:
            d = Decision(stable_state=AttentionState.PHONE,
                         alert_level=AlertLevel.CRITICAL, message="!")
            t0 = time.perf_counter()
            for i in range(30):
                mgr.handle(d, timestamp=i / 30)
            assert time.perf_counter() - t0 < 0.1  # بوق ۲۵۰ms حلقه را مسدود نمی‌کند
            time.sleep(0.6)
            assert beeps  # بوق در پس‌زمینه تکرار می‌شود
            n = len(beeps)
            mgr.handle(Decision(stable_state=AttentionState.FORWARD,
                                alert_level=AlertLevel.NONE), timestamp=2.0)
            time.sleep(0.6)
            assert len(beeps) <= n + 1  # پس از پایان هشدار، بوق قطع می‌شود
        finally:
            mgr.close()

    def test_annotate_with_hud(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        d = Decision(stable_state=AttentionState.DOWN, alert_level=AlertLevel.INFO,
                     distracted_seconds=1.0, threshold_seconds=2.0)
        out = AlertManager(enable_sound=False).annotate(
            img, d, angles_text="Y 0", info={"FPS": "30"}, bbox=(50, 50, 80, 80))
        assert out.shape == img.shape


class TestVoiceAlerts:
    """آژیر بلند + هشدار گفتاری فارسی."""

    @pytest.fixture
    def worker(self, monkeypatch, tmp_path):
        import time
        from dms.alerting import SoundWorker, make_siren, to_wav_bytes
        from dms.config import AlertConfig
        for key in ("LOOK_DOWN", "PHONE", "CLEARED", "ESCALATED"):
            (tmp_path / f"{key}.wav").write_bytes(to_wav_bytes(make_siren(400, 50)))
        played = []
        monkeypatch.setattr(SoundWorker, "_play",
                            lambda self, wav: (played.append(wav), time.sleep(0.02)))
        w = SoundWorker(AlertConfig(voice_dir=str(tmp_path), warning_interval=0.1,
                                    voice_repeat_seconds=10))
        w.start()
        yield w, played
        w.stop()

    def _decision(self, level, reason, transition=None, active=True):
        from dms.decision import AlertReason, Transition
        return Decision(stable_state=AttentionState.DOWN, alert_level=level,
                        reason=getattr(AlertReason, reason), alert_active=active,
                        transition=transition or Transition.NONE)

    def test_siren_is_loud(self):
        from dms.alerting import make_siren
        s = make_siren(1200, 300)
        rms = float(np.sqrt(np.mean(s ** 2)))
        assert np.abs(s).max() <= 1.0 and rms > 0.6  # موج تقریباً مربعی و رسا

    def test_voices_loaded(self, worker):
        w, _ = worker
        assert {"LOOK_DOWN", "PHONE", "CLEARED"} <= set(w.voices)

    def test_voice_played_on_alert(self, worker):
        import time
        w, played = worker
        w.update(self._decision(AlertLevel.WARNING, "LOOK_DOWN"))
        time.sleep(0.3)
        assert w.voices["LOOK_DOWN"] in played
        # جمله تا voice_repeat_seconds تکرار نمی‌شود، آژیر ادامه دارد
        assert played.count(w.voices["LOOK_DOWN"]) == 1
        assert played.count(w._sirens[AlertLevel.WARNING]) >= 2

    def test_reason_change_speaks_immediately(self, worker):
        import time
        w, played = worker
        w.update(self._decision(AlertLevel.WARNING, "LOOK_DOWN"))
        time.sleep(0.15)
        w.update(self._decision(AlertLevel.WARNING, "PHONE"))
        time.sleep(0.2)
        assert w.voices["PHONE"] in played

    def test_clear_says_thanks_and_stops(self, worker):
        import time
        from dms.decision import Transition
        w, played = worker
        w.update(self._decision(AlertLevel.WARNING, "LOOK_DOWN"))
        time.sleep(0.15)
        w.update(self._decision(AlertLevel.NONE, "LOOK_DOWN", Transition.CLEARED, active=False))
        time.sleep(0.2)
        assert played[-1] == w.voices["CLEARED"]
        n = len(played)
        time.sleep(0.3)
        assert len(played) == n

    def test_missing_voice_dir_falls_back_to_siren(self, tmp_path):
        from dms.alerting import SoundWorker
        from dms.config import AlertConfig
        w = SoundWorker(AlertConfig(voice_dir=str(tmp_path / "none")))
        assert w.voices == {}

    def test_volume_validated(self):
        from dms.config import AlertConfig, ConfigError
        with pytest.raises(ConfigError):
            AlertConfig(volume=1.5)
