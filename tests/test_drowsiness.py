"""تست‌های تشخیص خواب‌آلودگی (EAR، مدت بسته بودن، سر تکان دادن)."""

import pytest

from dms.config import DrowsinessConfig
from dms.drowsiness import DrowsinessDetector


@pytest.fixture
def det() -> DrowsinessDetector:
    return DrowsinessDetector(DrowsinessConfig(
        ear_closed_threshold=0.2, ear_blink_frames=3, closed_seconds=2.0))


class TestEARStates:
    """باز/بسته بودن چشم."""

    def test_open_eyes(self, det):
        r = det.update(ear=0.30, pitch=0, timestamp=0.0)
        assert not r.eyes_closed and not r.drowsy

    def test_closed_eyes_flag(self, det):
        r = det.update(ear=0.15, pitch=0, timestamp=0.0)
        assert r.eyes_closed


class TestBlinkVsMicrosleep:
    """تفکیک پلک زدن طبیعی از خواب کوتاه."""

    def test_short_blink_is_not_drowsy(self, det):
        # 3 فریم بسته در 30fps = 0.1 ثانیه << 2 ثانیه
        t = 0.0
        for _ in range(3):
            r = det.update(0.15, 0, t); t += 1 / 30
        assert not r.drowsy
        for _ in range(30):
            r = det.update(0.30, 0, t); t += 1 / 30
        assert not r.drowsy

    def test_blink_counted(self, det):
        t = 0.0
        for _ in range(5):  # پنج چرخه پلک زدن
            for _ in range(2):
                det.update(0.15, 0, t); t += 1 / 30
            for _ in range(10):
                det.update(0.30, 0, t); t += 1 / 30
        r = det.update(0.30, 0, t)
        assert r.blink_count == 5

    def test_long_closure_triggers_drowsiness(self, det):
        t = 0.0
        r = None
        # بیش از 2 ثانیه چشم بسته
        for _ in range(70):
            r = det.update(0.15, 0, t); t += 1 / 30
        assert r.drowsy
        assert r.closed_duration >= 2.0

    def test_drowsiness_latched_until_recovery(self, det):
        # باز شدن لحظه‌ای چشم هشدار را قطع نمی‌کند (ماندگاری)
        t = 0.0
        for _ in range(70):
            r = det.update(0.15, 0, t); t += 1 / 30
        assert r.drowsy and r.reason == "eyes_closed"
        r = det.update(0.30, 0, t); t += 1 / 30
        assert r.drowsy
        # باز ماندن پیوسته بیش از recovery_seconds (۱.۵s) → پایان
        for _ in range(50):
            r = det.update(0.30, 0, t); t += 1 / 30
        assert not r.drowsy

    def test_reclosing_resets_recovery(self, det):
        t = 0.0
        for _ in range(70):
            det.update(0.15, 0, t); t += 1 / 30
        for _ in range(30):  # ۱ ثانیه باز
            det.update(0.30, 0, t); t += 1 / 30
        det.update(0.15, 0, t); t += 1 / 30  # دوباره بسته
        for _ in range(30):  # ۱ ثانیه باز — هنوز کافی نیست
            r = det.update(0.30, 0, t); t += 1 / 30
        assert r.drowsy

    def test_closed_duration_accumulates(self, det):
        t = 0.0
        det.update(0.15, 0, t)
        t += 1.0
        r = det.update(0.15, 0, t)
        assert r.closed_duration == pytest.approx(1.0, abs=0.01)


class TestHeadNodding:
    """الگوی سر تکان دادن (nodding)."""

    def _nod(self, det, t0, fps=30):
        """یک چرخه سر تکان: افت pitch و بازیابی."""
        t = t0
        for _ in range(6):
            det.update(0.3, -14, t); t += 1 / fps
        for _ in range(12):
            det.update(0.3, -20, t); t += 1 / fps
        for _ in range(8):
            det.update(0.3, -4, t); t += 1 / fps
        return t

    def test_single_nod_not_alarming(self, det):
        t = self._nod(det, 0.0)
        r = det.update(0.3, 0, t)
        assert not r.nodding

    def test_repeated_nods_trigger(self, det):
        t = 0.0
        for i in range(4):
            t = self._nod(det, t)
        r = det.update(0.3, 0, t)
        assert r.nodding

    def test_nodding_triggers_drowsiness(self, det):
        t = 0.0
        for i in range(3):
            t = self._nod(det, t)
        r = det.update(0.3, 0, t)
        assert r.nodding
        assert r.drowsy and r.reason == "nodding"

    def test_nodding_alert_can_be_disabled(self):
        det = DrowsinessDetector(DrowsinessConfig(nodding_alert=False))
        t = 0.0
        for i in range(4):
            t = self._nod(det, t)
        r = det.update(0.3, 0, t)
        assert r.nodding and not r.drowsy

    def test_nods_expire_from_window(self, det):
        t = 0.0
        for i in range(4):
            t = self._nod(det, t)
        # عبور از پنجره 10 ثانیه‌ای بدون nod جدید
        t += 11.0
        r = det.update(0.3, 0, t)
        assert not r.nodding


class TestPerclos:
    """PERCLOS: درصد زمان بسته بودن چشم در پنجره لغزان."""

    def test_perclos_measured(self, det):
        # الگوی ۲۰٪ بسته (۶ از ۳۰ فریم، کوتاه‌تر از microsleep)
        t = 0.0
        for _ in range(30):
            for i in range(30):
                r = det.update(0.15 if i < 6 else 0.30, 0, t); t += 1 / 30
        assert r.perclos == pytest.approx(0.2, abs=0.03)
        assert not r.drowsy

    def test_high_perclos_triggers(self, det):
        # ۴۰٪ زمان بسته (پلک‌های کند و طولانی) به مدت ۳۰ ثانیه
        t = 0.0
        r = None
        for _ in range(30):
            for i in range(30):
                r = det.update(0.15 if i < 12 else 0.30, 0, t); t += 1 / 30
        assert r.perclos > 0.3
        assert r.drowsy

    def test_perclos_needs_min_data(self, det):
        t = 0.0
        for _ in range(5):
            for i in range(30):
                r = det.update(0.15 if i < 15 else 0.30, 0, t); t += 1 / 30
        assert not r.drowsy  # کمتر از perclos_min_seconds داده


class TestReset:
    def test_reset_clears_counters(self, det):
        det.update(0.15, 0, 0.0)
        det.reset()
        r = det.update(0.30, 0, 0.0)
        assert r.blink_count == 0
        assert r.closed_duration == 0.0
