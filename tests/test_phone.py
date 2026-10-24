"""تست‌های شناسایی تلفن همراه: قاعده نزدیکی، پایپ‌لاین و پیام هشدار."""

from pathlib import Path

import numpy as np
import pytest

from dms.attention import AttentionState
from dms.config import DMSConfig, PhoneConfig
from dms.decision import AlertLevel
from dms.phone_detector import PhoneDetector, phone_near_face
from tests.conftest import FakePhoneDetector

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FPS = 30


def run_frames(pipeline, detector, n, pitch=0.0, yaw=0.0, ear=0.30,
               visible=True, start_ms=0):
    """اجرای n فریم با وضعیت ثابت؛ خروجی: آخرین FrameResult و ts."""
    detector.set_state(pitch=pitch, yaw=yaw, ear=ear, visible=visible)
    result = None
    for i in range(n):
        result = pipeline.process(FRAME, start_ms + i * 1000 // FPS)
    return result, start_ms + n * 1000 // FPS


@pytest.fixture
def phone_pipeline(config, fake_detector, fake_phone_detector):
    """پایپ‌لاین با تشخیص‌دهنده گوشی مصنوعی (فعال)."""
    from dms.pipeline import DMSPipeline

    p = DMSPipeline(config=config, detector=fake_detector,
                    phone_detector=fake_phone_detector, enable_sound=False)
    yield p
    p.close()


class TestNearFaceRule:
    """قاعده هندسی «نزدیک بودن گوشی به چهره»."""

    FACE = (200.0, 150.0, 200.0, 240.0)  # x,y,w,h

    def test_center_inside_face(self):
        phone = (270.0, 220.0, 60, 120)  # مرکز دقیقاً مرکز چهره
        assert phone_near_face(phone, self.FACE)

    def test_center_inside_expanded_margin(self):
        # مرکز گوشی کمی بیرون جعبه چهره اما داخل ناحیه بزرگ‌شده (تماس با گوش)
        phone = (410.0, 150.0, 40, 100)
        assert phone_near_face(phone, self.FACE)

    def test_center_far_away(self):
        phone = (600.0, 420.0, 30, 60)  # گوشه پایین کادر
        assert not phone_near_face(phone, self.FACE)

    def test_expand_zero_strict(self):
        # بدون بزرگ‌نمایی، گوشی بیرون جعبه دقیق چهره رد می‌شود
        phone = (410.0, 150.0, 40, 100)
        assert not phone_near_face(phone, self.FACE, expand=0.0)


class TestPhoneState:
    """تزریق وضعیت PHONE در پایپ‌لاین."""

    def test_phone_near_face_confirmed(self, phone_pipeline, fake_detector,
                                       fake_phone_detector):
        fake_phone_detector.set_mode("near")
        result, _ = run_frames(phone_pipeline, fake_detector, 12)
        assert result.decision.stable_state is AttentionState.PHONE
        assert result.attention.state is AttentionState.PHONE

    def test_phone_far_is_not_phone(self, phone_pipeline, fake_detector,
                                    fake_phone_detector):
        fake_phone_detector.set_mode("far")  # گوشی روی پایه داشبورد
        result, _ = run_frames(phone_pipeline, fake_detector, 12)
        assert result.decision.stable_state is AttentionState.FORWARD

    def test_phone_none_is_forward(self, phone_pipeline, fake_detector,
                                   fake_phone_detector):
        fake_phone_detector.set_mode("none")
        result, _ = run_frames(phone_pipeline, fake_detector, 12)
        assert result.decision.stable_state is AttentionState.FORWARD

    def test_phone_gone_returns_forward(self, phone_pipeline, fake_detector,
                                        fake_phone_detector):
        fake_phone_detector.set_mode("near")
        result, ts = run_frames(phone_pipeline, fake_detector, 12)
        assert result.decision.stable_state is AttentionState.PHONE
        fake_phone_detector.set_mode("none")
        result, _ = run_frames(phone_pipeline, fake_detector, 10, start_ms=ts)
        assert result.decision.stable_state is AttentionState.FORWARD

    def test_phone_requires_face(self, phone_pipeline, fake_detector,
                                 fake_phone_detector, config):
        fake_phone_detector.set_mode("near")
        result, ts = run_frames(phone_pipeline, fake_detector, 5)
        # چهره گم می‌شود (طولانی) → NO_FACE بر PHONE اولویت دارد
        result, _ = run_frames(phone_pipeline, fake_detector, 15,
                               visible=False, start_ms=ts)
        assert result.decision.stable_state is AttentionState.NO_FACE


class TestPhoneAlert:
    """هشدار استفاده از تلفن همراه."""

    def test_warning_after_threshold(self, phone_pipeline, fake_detector,
                                     fake_phone_detector):
        fake_phone_detector.set_mode("near")
        result, _ = run_frames(phone_pipeline, fake_detector, 80)
        assert result.decision.alert_level is AlertLevel.WARNING
        assert result.decision.alert_id is not None

    def test_message_mentions_phone(self, phone_pipeline, fake_detector,
                                    fake_phone_detector):
        fake_phone_detector.set_mode("near")
        result, _ = run_frames(phone_pipeline, fake_detector, 80)
        assert "تلفن همراه" in result.decision.message

    def test_info_before_threshold(self, phone_pipeline, fake_detector,
                                   fake_phone_detector):
        fake_phone_detector.set_mode("near")
        # ۱۲ فریم = 0.4s < آستانه 2s ⇒ هنوز فقط INFO
        result, _ = run_frames(phone_pipeline, fake_detector, 12)
        assert result.decision.alert_level is AlertLevel.INFO


class TestPhonePriority:
    """اولویت PHONE نسبت به سایر وضعیت‌ها."""

    def test_phone_overrides_down(self, phone_pipeline, fake_detector,
                                  fake_phone_detector):
        # سر پایین (نگاه به گوشی روی پا) + گوشی نزدیک چهره ⇒ PHONE
        fake_phone_detector.set_mode("near")
        result, _ = run_frames(phone_pipeline, fake_detector, 12, pitch=-20)
        assert result.decision.stable_state is AttentionState.PHONE


class TestPhoneDisabled:
    """رفتار پایپ‌لاین بدون ماژول گوشی."""

    def test_no_phone_detector_no_crash(self, config, fake_detector):
        from dms.pipeline import DMSPipeline

        config = _disable_phone(config)
        p = DMSPipeline(config=config, detector=fake_detector,
                        enable_sound=False)
        try:
            result, _ = run_frames(p, fake_detector, 8)
            assert result.decision.stable_state is AttentionState.FORWARD
        finally:
            p.close()

    def test_enabled_but_model_missing_warns_only(self, config, fake_detector,
                                                  caplog):
        from dms.pipeline import DMSPipeline

        # مدل واقعی وجود ندارد؛ باید فقط هشدار لاگ شود و پایپ‌لاین بالا بیاید
        import logging
        caplog.set_level(logging.WARNING, logger="dms.pipeline")
        config = _use_missing_model(config)
        p = DMSPipeline(config=config, detector=fake_detector,
                        enable_sound=False)
        try:
            assert "phone detection disabled" in caplog.text
            result, _ = run_frames(p, fake_detector, 8)
            assert result.decision.stable_state is AttentionState.FORWARD
        finally:
            p.close()


def _disable_phone(cfg: DMSConfig) -> DMSConfig:
    from dataclasses import replace
    return replace(cfg, phone=replace(cfg.phone, enabled=False))


def _use_missing_model(cfg: DMSConfig) -> DMSConfig:
    from dataclasses import replace
    return replace(cfg, phone=replace(cfg.phone,
                                      model_path="no/such/model.tflite"))


class TestPhoneConfig:
    """پیکربندی ماژول گوشی."""

    def test_defaults(self):
        cfg = PhoneConfig()
        assert cfg.enabled
        assert cfg.score_threshold == pytest.approx(0.4)
        assert cfg.detect_every == 15
        assert cfg.near_face_expand == pytest.approx(0.6)

    def test_from_json(self, tmp_path):
        import json

        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({
            "phone": {"score_threshold": 0.6, "detect_every": 1},
        }), encoding="utf-8")
        cfg = DMSConfig.from_json(cfg_file)
        assert cfg.phone.score_threshold == pytest.approx(0.6)
        assert cfg.phone.detect_every == 1
        # سایر مقادیر دست‌نخورده
        assert cfg.phone.enabled
        assert cfg.head_pose.yaw_threshold == 25.0


MODEL = Path(__file__).resolve().parents[1] / "assets" / "yolo11n.onnx"

@pytest.mark.skipif(not MODEL.exists(),
                    reason="مدل yolo11n.onnx دانلود نشده است")
class TestPhoneDetectorReal:
    """تست دود تشخیص‌دهنده واقعی (YOLO11n ONNX) روی فریم خالی."""

    def test_no_phone_in_blank_frame(self):
        detector = PhoneDetector(model_path=str(MODEL))
        try:
            phones = detector.process(np.zeros((480, 640, 3), np.uint8), 33)
            assert phones == []
        finally:
            detector.close()

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PhoneDetector(model_path=str(tmp_path / "missing.onnx"))

    def test_timestamp_monotonicity(self):
        detector = PhoneDetector(model_path=str(MODEL))
        try:
            detector.process(np.zeros((480, 640, 3), np.uint8), 100)
            with pytest.raises(ValueError):
                detector.process(np.zeros((480, 640, 3), np.uint8), 50)
        finally:
            detector.close()

    def test_phone_detection_latency(self):
        """زمان استنتاج ONNX باید برای اجرای هر ۱۵ فریم قابل قبول باشد."""
        import time
        detector = PhoneDetector(model_path=str(MODEL))
        try:
            frame = np.zeros((480, 640, 3), np.uint8)
            for i in range(3):  # گرم کردن جلسه
                detector.process(frame, 33 + i * 1000)
            t0 = time.perf_counter()
            for i in range(5):
                detector.process(frame, 5000 + (i + 1) * 1000)
            avg_ms = (time.perf_counter() - t0) / 5 * 1000
            print(f"\nphone detect latency: {avg_ms:.0f} ms")
            # سقف سخاوتمندانه: بار CPU سیستم‌های اشتراکی نوسان زیادی دارد؛
            # هدف تست، رد کردن رگرسیون‌های شدید (مثل حلقه پایتونی روی
            # ۸۴۰۰ anchor) است نه اندازه‌گیری دقیق کارایی.
            assert avg_ms < 1500.0
        finally:
            detector.close()
