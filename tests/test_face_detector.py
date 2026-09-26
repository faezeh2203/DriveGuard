"""تست دود تشخیص‌دهنده واقعی MediaPipe (در نبود مدل، رد می‌شود)."""

from pathlib import Path

import numpy as np
import pytest

from dms.face_detector import FaceDetector

MODEL = Path(__file__).resolve().parents[1] / "assets" / "face_landmarker.task"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="مدل face_landmarker.task دانلود نشده است")


@pytest.fixture
def detector():
    d = FaceDetector(model_path=str(MODEL))
    yield d
    d.close()


BLANK = np.zeros((480, 640, 3), dtype=np.uint8)


class TestFaceDetectorSmoke:
    def test_no_face_in_blank_frame(self, detector):
        assert detector.process(BLANK, timestamp_ms=33) is None

    def test_timestamp_monotonicity_enforced(self, detector):
        detector.process(BLANK, timestamp_ms=100)
        with pytest.raises(ValueError):
            detector.process(BLANK, timestamp_ms=100)  # تکراری
        with pytest.raises(ValueError):
            detector.process(BLANK, timestamp_ms=50)   # نزولی

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FaceDetector(model_path=str(tmp_path / "missing.task"))
