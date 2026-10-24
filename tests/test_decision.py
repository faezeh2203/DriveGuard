"""تست‌های موتور تصمیم: هیسترزیس، آستانه‌ها، هشدار ماندگار، ارتقا و Cooldown."""

import pytest

from dms.attention import AttentionResult, AttentionState
from dms.config import DecisionConfig
from dms.decision import AlertLevel, AlertReason, DecisionEngine, Transition
from dms.drowsiness import DrowsinessResult

FPS = 30


def attn(state: AttentionState) -> AttentionResult:
    return AttentionResult(raw_state=state, state=state)


def drowsy(flag: bool) -> DrowsinessResult:
    return DrowsinessResult(eyes_closed=flag, drowsy=flag,
                            nodding=False, closed_duration=2.5, blink_count=0)


def sustain(engine, state, seconds, start=0.0, fps=FPS, drowsiness=None):
    """اجرای یک وضعیت ثابت به مدت مشخص؛ خروجی: (آخرین تصمیم، زمان بعدی، همه تصمیم‌ها)."""
    t = start
    decisions = []
    for _ in range(int(round(seconds * fps))):
        decisions.append(engine.update(attn(state), drowsiness, t))
        t += 1 / fps
    return decisions[-1], t, decisions


@pytest.fixture
def engine() -> DecisionEngine:
    """موتور تصمیم با سیاست پیش‌فرض."""
    return DecisionEngine(DecisionConfig())


@pytest.fixture
def strict_engine() -> DecisionEngine:
    """سیاست سخت‌گیرانه: نگاه به اطراف هم پس از ۲ ثانیه هشدار دارد."""
    return DecisionEngine(DecisionConfig(lateral_alert_seconds=2.0))


class TestHysteresis:
    def test_starts_forward(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), None, 0.0)
        assert d.stable_state is AttentionState.FORWARD
        assert d.alert_level is AlertLevel.NONE

    def test_transient_distraction_ignored(self, engine):
        engine.update(attn(AttentionState.LEFT), None, 0.0)
        d = engine.update(attn(AttentionState.FORWARD), None, 1 / 30)
        assert d.stable_state is AttentionState.FORWARD

    def test_persistent_distraction_accepted(self, engine):
        d, _, _ = sustain(engine, AttentionState.LEFT, 5 / FPS)
        assert d.stable_state is AttentionState.LEFT

    def test_no_face_needs_confirmation_too(self, engine):
        for _ in range(4):
            engine.update(attn(AttentionState.NO_FACE), None, 0.0)
        d = engine.update(attn(AttentionState.NO_FACE), None, 1.0)
        assert d.stable_state is AttentionState.NO_FACE


class TestThresholds:
    """آستانه هشدار وابسته به نوع حواس‌پرتی."""

    def test_info_before_threshold(self, engine):
        d, _, _ = sustain(engine, AttentionState.DOWN, 1.0)
        assert d.alert_level is AlertLevel.INFO
        assert d.alert_id is None
        assert not d.alert_active
        assert d.threshold_seconds == pytest.approx(2.0)

    def test_down_warns_after_threshold(self, engine):
        d, _, _ = sustain(engine, AttentionState.DOWN, 3.0)
        assert d.alert_level is AlertLevel.WARNING
        assert d.reason is AlertReason.LOOK_DOWN
        assert d.alert_id is not None and d.alert_active

    def test_phone_warns_fast(self, engine):
        d, _, _ = sustain(engine, AttentionState.PHONE, 1.5)
        assert d.alert_level is AlertLevel.WARNING
        assert d.reason is AlertReason.PHONE
        assert "تلفن همراه" in d.message

    def test_no_face_warns(self, engine):
        d, _, _ = sustain(engine, AttentionState.NO_FACE, 3.0)
        assert d.alert_level is AlertLevel.WARNING
        assert d.reason is AlertReason.NO_FACE

    def test_distracted_seconds_measured(self, engine):
        d, _, _ = sustain(engine, AttentionState.LEFT, 2.5)
        # منهای ۵ فریم تأیید هیسترزیس
        assert d.distracted_seconds == pytest.approx(2.5 - 5 / FPS, abs=0.1)


class TestLateralPolicy:
    """نگاه کوتاه به آینه/اطراف مجاز است؛ نگاه طولانی هشدار دارد."""

    def test_mirror_check_no_alert(self, engine):
        d, _, _ = sustain(engine, AttentionState.LEFT, 2.0)
        assert d.alert_level is AlertLevel.INFO
        assert engine.alert_count == 0

    def test_long_lateral_look_warns(self, engine):
        d, _, _ = sustain(engine, AttentionState.RIGHT, 6.0)
        assert d.alert_level is AlertLevel.WARNING
        assert d.reason is AlertReason.LATERAL

    def test_lateral_can_be_disabled(self):
        eng = DecisionEngine(DecisionConfig(lateral_alert_enabled=False))
        d, _, _ = sustain(eng, AttentionState.LEFT, 10.0)
        assert d.alert_level is AlertLevel.INFO
        assert eng.alert_count == 0

    def test_strict_mode_warns_for_lateral(self, strict_engine):
        d, _, _ = sustain(strict_engine, AttentionState.LEFT, 3.0)
        assert d.alert_level is AlertLevel.WARNING
        assert strict_engine.alert_count == 1

    def test_lateral_still_tracked_as_episode(self, engine):
        d, _, _ = sustain(engine, AttentionState.RIGHT, 3.0)
        assert d.distracted_seconds > 0
        assert len(engine.episodes) == 1


class TestContinuousTimer:
    """جابه‌جایی بین حالت‌های حواس‌پرتی تایمر را صفر نمی‌کند."""

    def test_switching_states_does_not_reset_timer(self, engine):
        _, t, _ = sustain(engine, AttentionState.DOWN, 1.2)
        d, _, _ = sustain(engine, AttentionState.PHONE, 0.5, start=t)
        # مجموع > ۱ ثانیه (آستانه موبایل) → هشدار
        assert d.alert_level is AlertLevel.WARNING
        assert d.reason is AlertReason.PHONE

    def test_alternating_down_and_up_still_alerts(self, engine):
        t = 0.0
        d = None
        for _ in range(4):
            d, t, _ = sustain(engine, AttentionState.DOWN, 0.6, start=t)
            d, t, _ = sustain(engine, AttentionState.UP, 0.6, start=t)
        assert d.alert_active


class TestLatchedAlert:
    """هشدار تا اصلاح رفتار راننده ادامه دارد."""

    def test_brief_glance_forward_does_not_clear(self, engine):
        _, t, _ = sustain(engine, AttentionState.PHONE, 2.0)
        # ۰.۵ ثانیه نگاه به جلو (کمتر از recovery_seconds = ۱)
        d, t, _ = sustain(engine, AttentionState.FORWARD, 0.5, start=t)
        assert d.alert_active
        assert d.alert_level is AlertLevel.WARNING
        assert 0.0 < d.recovery_progress < 1.0
        # برگشت به موبایل: همان هشدار ادامه دارد (شناسه ثابت)
        d2, _, _ = sustain(engine, AttentionState.PHONE, 0.2, start=t)
        assert d2.alert_id == d.alert_id
        assert d2.recovery_progress == 0.0
        assert engine.alert_count == 1

    def test_sustained_forward_clears(self, engine):
        _, t, _ = sustain(engine, AttentionState.DOWN, 3.0)
        d, _, ds = sustain(engine, AttentionState.FORWARD, 2.0, start=t)
        assert d.alert_level is AlertLevel.NONE
        assert not d.alert_active
        cleared = [x for x in ds if x.transition is Transition.CLEARED]
        assert len(cleared) == 1
        assert cleared[0].reason is AlertReason.LOOK_DOWN

    def test_alert_persists_while_distracted(self, engine):
        _, _, ds = sustain(engine, AttentionState.PHONE, 10.0)
        active = [d for d in ds if d.alert_active]
        # پس از شروع، هیچ فریمی بدون هشدار نیست
        first = next(i for i, d in enumerate(ds) if d.alert_active)
        assert all(d.alert_active for d in ds[first:])
        assert len({d.alert_id for d in active}) == 1

    def test_transitions_reported_once(self, engine):
        _, _, ds = sustain(engine, AttentionState.DOWN, 3.0)
        started = [d for d in ds if d.transition is Transition.STARTED]
        assert len(started) == 1


class TestEscalation:
    def test_long_warning_escalates_to_critical(self, engine):
        _, _, ds = sustain(engine, AttentionState.PHONE, 9.0)
        assert ds[-1].alert_level is AlertLevel.CRITICAL
        escalated = [d for d in ds if d.transition is Transition.ESCALATED]
        assert len(escalated) == 1
        assert "هشدار جدی" in ds[-1].message


class TestDrowsinessAlert:
    def test_critical_overrides(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), drowsy(True), 0.0)
        assert d.alert_level is AlertLevel.CRITICAL
        assert d.reason is AlertReason.DROWSY
        assert d.alert_id is not None

    def test_drowsy_beats_phone(self, engine):
        _, t, _ = sustain(engine, AttentionState.PHONE, 2.0)
        d = engine.update(attn(AttentionState.PHONE), drowsy(True), t)
        assert d.reason is AlertReason.DROWSY
        assert d.transition is Transition.REASON_CHANGED

    def test_no_drowsiness_flag_no_alert(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), drowsy(False), 0.0)
        assert d.alert_level is AlertLevel.NONE

    def test_none_drowsiness_accepted(self, engine):
        d = engine.update(attn(AttentionState.FORWARD), None, 0.0)
        assert d.alert_level is AlertLevel.NONE

    def test_drowsiness_alert_can_be_disabled(self):
        eng = DecisionEngine(DecisionConfig(drowsiness_alert=False))
        d = eng.update(attn(AttentionState.FORWARD), drowsy(True), 0.0)
        assert d.alert_level is AlertLevel.NONE


class TestCooldown:
    """Cooldown هرگز هشدار را سرکوب نمی‌کند؛ فقط شناسه را بازاستفاده می‌کند."""

    def test_quick_redistraction_reopens_same_alert(self, engine):
        _, t, _ = sustain(engine, AttentionState.DOWN, 3.0)
        first_id = engine.alert_count
        _, t, _ = sustain(engine, AttentionState.FORWARD, 1.5, start=t)
        assert not engine.alert_active
        d, _, _ = sustain(engine, AttentionState.DOWN, 2.5, start=t)
        assert d.alert_level is AlertLevel.WARNING  # سرکوب نشده
        assert d.alert_id == first_id
        assert engine.alert_count == 1

    def test_alert_ids_increase_after_cooldown(self, engine):
        d1, t, _ = sustain(engine, AttentionState.DOWN, 3.0)
        _, t, _ = sustain(engine, AttentionState.FORWARD, 5.0, start=t)
        d2, _, _ = sustain(engine, AttentionState.PHONE, 2.0, start=t)
        assert d2.alert_id > d1.alert_id
        assert engine.alert_count == 2


class TestEpisodes:
    def test_episode_logged_and_closed(self, engine):
        _, t, _ = sustain(engine, AttentionState.RIGHT, 2.0)
        sustain(engine, AttentionState.FORWARD, 6 / FPS, start=t)
        assert len(engine.episodes) == 1
        assert engine.episodes[0].end_ts is not None
        assert engine.episodes[0].state is AttentionState.RIGHT

    def test_forward_only_no_episodes(self, engine):
        sustain(engine, AttentionState.FORWARD, 1.0)
        assert engine.episodes == []

    def test_reset(self, engine):
        sustain(engine, AttentionState.PHONE, 3.0)
        engine.reset()
        assert not engine.alert_active and engine.alert_count == 0
