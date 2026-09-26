"""تست‌های هموارسازی: کالمن، میانگین متحرک و هیسترزیس."""

import pytest

from dms.smoothing import HysteresisFilter, KalmanFilter1D, MovingAverage, smooth_series


class TestKalman1D:
    """فیلتر کالمن تک‌متغیره."""

    def test_first_measurement_is_adopted(self):
        kf = KalmanFilter1D()
        assert kf.step(12.0) == pytest.approx(12.0)

    def test_converges_to_constant_signal(self):
        kf = KalmanFilter1D(process_noise=1.0, measurement_noise=4.0)
        out = [kf.step(20.0) for _ in range(50)]
        assert out[-1] == pytest.approx(20.0, abs=0.5)

    def test_reduces_noise(self):
        # نویز سینوسی حول مقدار ثابت: خروجی باید کم‌نوسان‌تر از ورودی باشد
        import math
        kf = KalmanFilter1D(process_noise=0.5, measurement_noise=10.0)
        measurements = [25.0 + 5.0 * math.sin(i * 0.9) for i in range(60)]
        outputs = [kf.step(m) for m in measurements]
        def std(seq):
            mean = sum(seq) / len(seq)
            return (sum((x - mean) ** 2 for x in seq) / len(seq)) ** 0.5
        assert std(outputs[10:]) < std(measurements[10:])

    def test_tracks_ramp_with_lag(self):
        kf = KalmanFilter1D(process_noise=2.0, measurement_noise=2.0)
        outputs = []
        for i in range(200):
            outputs.append(kf.step(float(i)))  # شیب 1 در فریم
        # تخمین باید به شیب واقعی نزدیک شود (خطای همگرا کوچک اما صفر نیست)
        error = abs(outputs[-1] - 199.0)
        assert error < 5.0

    def test_reset(self):
        kf = KalmanFilter1D()
        kf.step(10.0)
        kf.reset(5.0)
        assert kf.value == pytest.approx(5.0)

    def test_deterministic(self):
        def run():
            kf = KalmanFilter1D()
            return [kf.step(10.0 + 0.1 * i) for i in range(20)]
        assert run() == run()


class TestMovingAverage:
    """میانگین متحرک پنجره‌ای."""

    def test_basic_average(self):
        ma = MovingAverage(window=3)
        ma.update(1); ma.update(2)
        assert ma.update(3) == pytest.approx(2.0)

    def test_window_slides(self):
        ma = MovingAverage(window=2)
        ma.update(1)
        assert ma.update(3) == pytest.approx(2.0)
        assert ma.update(5) == pytest.approx(4.0)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            MovingAverage(window=0)

    def test_is_ready(self):
        ma = MovingAverage(window=2)
        assert not ma.is_ready
        ma.update(1)
        assert not ma.is_ready
        ma.update(2)
        assert ma.is_ready

    def test_smooth_series_length(self):
        out = smooth_series([1, 2, 3, 4, 5], window=3)
        assert len(out) == 5
        assert out[-1] == pytest.approx(4.0)


class TestHysteresis:
    """هیسترزیس وضعیت — کاهش False Alarm."""

    def test_initial_state_none(self):
        h = HysteresisFilter(min_frames=3)
        assert h.stable_state is None

    def test_requires_consecutive_frames(self):
        h = HysteresisFilter(min_frames=3, initial="F")
        assert h.update("L") == "F"   # یک فریم کافی نیست
        assert h.update("L") == "F"
        assert h.update("L") == "L"   # سومین فریم متوالی

    def test_break_in_sequence_resets_counter(self):
        h = HysteresisFilter(min_frames=3, initial="F")
        h.update("L"); h.update("L")
        h.update("F")                  # زنجیره می‌شکند
        assert h.update("L") == "F"    # شمارنده از نو

    def test_return_to_previous_state_immediate_series(self):
        h = HysteresisFilter(min_frames=2, initial="F")
        h.update("L")
        assert h.update("L") == "L"
        h.update("F")
        assert h.update("F") == "F"

    def test_invalid_min_frames(self):
        with pytest.raises(ValueError):
            HysteresisFilter(min_frames=0)

    def test_candidate_progress(self):
        h = HysteresisFilter(min_frames=5, initial="F")
        h.update("L")
        h.update("L")
        assert h.candidate_progress() == 2

    def test_reset(self):
        h = HysteresisFilter(min_frames=2, initial="F")
        h.update("L"); h.update("L")
        h.reset("F")
        assert h.stable_state == "F"
        assert h.candidate_progress() == 0

    def test_non_hashable_safe_with_enum_like(self):
        # وضعیت‌ها هر Hashable (مثلاً Enum) هستند
        class S:
            A = "A"
            B = "B"
        h = HysteresisFilter(min_frames=1, initial=S.A)
        assert h.update(S.B) == S.B
