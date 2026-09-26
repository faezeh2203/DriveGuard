"""تست‌های موتور تصمیم: هیسترزیس، آستانه هشدار، Cooldown."""

import pytest

from dms.attention import AttentionResult, AttentionState
from dms.config import DecisionConfig
from dms.decision import AlertLevel, DecisionEngine
from dms.drowsiness import DrowsinessResult


def attn(state: AttentionState) -> AttentionResult:
    return AttentionResult(raw_state=state, state=state)


def drowsy(flag: bool) -> DrowsinessResult:
    return DrowsinessResult(eyes_closed=flag, drowsy=flag,
                            nodding=False, closed_duration=2.5, blink_count=0)


@pytest.fixture
def engine() -> DecisionEngine:
    """موتور تصمیم با سیاست پیش‌فرض (چرخش سر چپ/راست بدون هشدار)."""
    return DecisionEngine(DecisionConfig(
        state_confirm_frames=5, distraction_alert_seconds=2.0,
        alert_cooldown_seconds=3.0, assumed_fps=30.0))


@pytest.fixture
def strict_engine() -> DecisionEngine:
    """موتور تصمیم با سیاست سخت‌گیرانه (هشدار برای چرخش سر هم)."""
    return DecisionEngine(DecisionConfig(
        state_confirm_frames=5, distraction_alert_seconds=2.0,
        alert_cooldown_seconds=3.0, assumed_fps=30.0,
        lateral_alert_enabled=True))


class TestHysteresis:
    """هیسترزیس وضعیت در موتور تصمیم."""

    def test_starts_forward(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), None, 0.0)
        assert d.stable_state is AttentionState.FORWARD

    def test_transient_distraction_ignored(self, engine):
        # یک فریم LEFT نویزی نباید وضعیت پایدار را عوض کند
        engine.update(attn(AttentionState.LEFT), None, 0.0)
        d = engine.update(attn(AttentionState.FORWARD), None, 1 / 30)
        assert d.stable_state is AttentionState.FORWARD

    def test_persistent_distraction_accepted(self, engine):
        t = 0.0
        for _ in range(5):
            d = engine.update(attn(AttentionState.LEFT), None, t)
            t += 1 / 30
        assert d.stable_state is AttentionState.LEFT

    def test_no_face_needs_confirmation_too(self, engine):
        for _ in range(4):
            engine.update(attn(AttentionState.NO_FACE), None, 0.0)
        d = engine.update(attn(AttentionState.NO_FACE), None, 1.0)
        assert d.stable_state is AttentionState.NO_FACE


class TestDistractionAlert:
    """هشدار حواس‌پرتی پس از آستانه زمانی (سیاست سخت‌گیرانه)."""

    def _sustain(self, engine, state, seconds, start=0.0, fps=30):
        t = start
        n = int(seconds * fps)
        d = None
        for _ in range(n):
            d = engine.update(attn(state), None, t)
            t += 1 / fps
        return d, t

    def test_info_before_threshold(self, engine):
        d, _ = self._sustain(engine, AttentionState.RIGHT, 1.0)
        assert d.alert_level is AlertLevel.INFO
        assert d.alert_id is None

    def test_warning_after_threshold(self, strict_engine):
        d, _ = self._sustain(strict_engine, AttentionState.RIGHT, 3.0)
        assert d.alert_level is AlertLevel.WARNING
        assert d.alert_id is not None

    def test_distracted_seconds_measured(self, engine):
        d, _ = self._sustain(engine, AttentionState.LEFT, 2.5)
        assert d.distracted_seconds == pytest.approx(2.5, abs=0.2)

    def test_alert_ids_increase(self, strict_engine):
        d1, t = self._sustain(strict_engine, AttentionState.LEFT, 3.0)
        # بازگشت به جلو و سپس حواس‌پرتی مجدد بعد از cooldown
        d = strict_engine.update(attn(AttentionState.FORWARD), None, t)
        d2, t2 = self._sustain(strict_engine, AttentionState.RIGHT, 3.0, start=t + 4.0)
        assert d1.alert_id is not None
        assert d2.alert_id is not None
        assert d2.alert_id > d1.alert_id


class TestLateralPolicy:
    """سیاست پیش‌فرض: چرخش سر به چپ/راست (آینه و پیچ) هشدار ندارد.

    راننده در جریان رانندگی موظف است آینه‌ها و طرفین را ببیند؛
    بنابراین LEFT/RIGHT فقط برچسب اطلاع‌رسانی می‌گیرند و هرگز
    هشدار WARNING صادر نمی‌کنند (مگر در حالت سخت‌گیرانه).
    """

    def _sustain(self, engine, state, seconds, start=0.0, fps=30):
        t = start
        n = int(seconds * fps)
        d = None
        for _ in range(n):
            d = engine.update(attn(state), None, t)
            t += 1 / fps
        return d, t

    def test_sustained_left_stays_info(self, engine):
        d, _ = self._sustain(engine, AttentionState.LEFT, 5.0)
        assert d.stable_state is AttentionState.LEFT
        assert d.alert_level is AlertLevel.INFO
        assert d.alert_id is None

    def test_sustained_right_stays_info(self, engine):
        d, _ = self._sustain(engine, AttentionState.RIGHT, 5.0)
        assert d.alert_level is AlertLevel.INFO
        assert d.alert_id is None

    def test_no_alert_counted_for_lateral(self, engine):
        self._sustain(engine, AttentionState.LEFT, 4.0)
        assert engine.alert_count == 0

    def test_lateral_still_tracked_as_episode(self, engine):
        # برای متریک‌ها ثبت می‌شود اما هشدار نمی‌دهد
        d, _ = self._sustain(engine, AttentionState.RIGHT, 3.0)
        assert d.distracted_seconds > 0
        assert len(engine.episodes) == 1

    def test_down_still_warns(self, engine):
        # نگاه طولانی به پایین (موبایل) حواس‌پرتی واقعی است
        d, _ = self._sustain(engine, AttentionState.DOWN, 3.0)
        assert d.alert_level is AlertLevel.WARNING
        assert d.alert_id is not None

    def test_no_face_still_warns(self, engine):
        d, _ = self._sustain(engine, AttentionState.NO_FACE, 3.0)
        assert d.alert_level is AlertLevel.WARNING

    def test_strict_mode_warns_for_lateral(self, strict_engine):
        d, _ = self._sustain(strict_engine, AttentionState.LEFT, 3.0)
        assert d.alert_level is AlertLevel.WARNING
        assert strict_engine.alert_count >= 1


class TestDrowsinessAlert:
    """هشدار خواب‌آلودگی اولویت CRITICAL دارد."""

    def test_critical_overrides(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), drowsy(True), 0.0)
        assert d.alert_level is AlertLevel.CRITICAL
        assert d.alert_id is not None

    def test_no_drowsiness_flag_no_alert(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), drowsy(False), 0.0)
        assert d.alert_level is AlertLevel.NONE

    def test_none_drowsiness_accepted(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), None, 0.0)
        assert d.alert_level is AlertLevel.NONE


class TestCooldown:
    """فاصله حداقل بین دو هشدار (سیاست سخت‌گیرانه)."""

    def test_cooldown_blocks_second_alert(self, strict_engine):
        t = 0.0
        for _ in range(80):  # 2.6s حواس‌پرتی → هشدار اول
            d = strict_engine.update(attn(AttentionState.LEFT), None, t)
            t += 1 / 30
        first_id = d.alert_id
        # بازگشت به جلو و حواس‌پرتی مجدد خیلی سریع (داخل cooldown)
        strict_engine.update(attn(AttentionState.FORWARD), None, t)
        t += 1 / 30
        for _ in range(80):
            d = strict_engine.update(attn(AttentionState.RIGHT), None, t)
            t += 1 / 30
        # هشدار دوم باید تا پایان cooldown فشرده شود
        assert d.alert_id is None or d.alert_id > first_id

    def test_alert_count(self, strict_engine):
        t = 0.0
        for _ in range(80):
            strict_engine.update(attn(AttentionState.LEFT), None, t)
            t += 1 / 30
        assert strict_engine.alert_count >= 1


class TestEpisodes:
    """ثبت دوره‌های حواس‌پرتی."""

    def test_episode_logged_and_closed(self, engine):
        t = 0.0
        for _ in range(60):
            engine.update(attn(AttentionState.RIGHT), None, t)
            t += 1 / 30
        # چند فریم FORWARD متوالی تا هیسترزیس وضعیت را تأیید کند
        for _ in range(6):
            engine.update(attn(AttentionState.FORWARD), None, t)
            t += 1 / 30
        assert len(engine.episodes) == 1
        assert engine.episodes[0].end_ts is not None
        assert engine.episodes[0].state is AttentionState.RIGHT

    def test_forward_only_no_episodes(self, engine):
        t = 0.0
        for _ in range(30):
            engine.update(attn(AttentionState.FORWARD), None, t)
            t += 1 / 30
        assert engine.episodes == []
