"""تست‌های داشبورد Flask و CLI."""

import pytest

from dms.attention import AttentionResult, AttentionState
from dms.config import DMSConfig
from dms.dashboard import Dashboard, SharedState
from dms.decision import AlertLevel, Decision
from dms.geometry import EulerAngle
from dms.main import _source_to_capture, parse_args
from dms.pipeline import FrameResult


def make_result(frame_id=1, ts=0.033):
    """ساخت FrameResult نمونه برای تغذیه داشبورد."""
    return FrameResult(
        frame_id=frame_id, timestamp=ts,
        angles=EulerAngle(pitch=-2.0, yaw=8.0, roll=1.0),
        attention=AttentionResult(raw_state=AttentionState.FORWARD,
                                  state=AttentionState.FORWARD, ear=0.30),
        drowsiness=None,
        decision=Decision(stable_state=AttentionState.FORWARD,
                          alert_level=AlertLevel.NONE),
        bbox=(10, 10, 100, 100), latency_ms=1.5, fps=29.8,
    )


class TestSharedState:
    def test_publish_and_latest(self):
        shared = SharedState()
        assert shared.latest is None
        r = make_result()
        shared.publish(r, jpeg=b"jpg")
        assert shared.latest is r
        assert shared.jpeg == b"jpg"

    def test_history_bounded(self):
        shared = SharedState(history=5)
        for i in range(10):
            shared.publish(make_result(frame_id=i))
        hist = shared.history
        assert len(hist) == 5
        assert hist[-1].frame_id == 9

    def test_thread_safety_smoke(self):
        import threading
        shared = SharedState()
        def worker():
            for i in range(50):
                shared.publish(make_result(frame_id=i))
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert shared.latest is not None


class TestDashboardAPI:
    @pytest.fixture
    def client(self):
        shared = SharedState()
        dash = Dashboard(shared)
        dash.app.config["TESTING"] = True
        return dash.app.test_client(), shared

    def test_index_page(self, client):
        http, _ = client
        resp = http.get("/")
        assert resp.status_code == 200
        assert "حواس‌پرتی".encode() in resp.data

    def test_status_before_first_frame(self, client):
        http, _ = client
        data = http.get("/api/status").get_json()
        assert data["state"] == "NO_FACE"
        assert data["fps"] == 0.0

    def test_status_after_publish(self, client):
        http, shared = client
        shared.publish(make_result())
        data = http.get("/api/status").get_json()
        assert data["state"] == "FORWARD"
        assert data["yaw"] == pytest.approx(8.0)
        assert data["fps"] == pytest.approx(29.8)

    def test_history_endpoint(self, client):
        http, shared = client
        for i in range(3):
            shared.publish(make_result(frame_id=i))
        data = http.get("/api/history").get_json()
        assert len(data) == 3
        assert data[0]["state"] == "FORWARD"


class TestCLI:
    def test_parse_defaults(self):
        args = parse_args([])
        assert args.source == "0"
        assert not args.dashboard
        assert not args.no_display

    def test_parse_custom(self):
        args = parse_args(["--source", "video.mp4", "--dashboard",
                           "--port", "8080", "--no-sound"])
        assert args.source == "video.mp4"
        assert args.dashboard
        assert args.port == 8080
        assert args.no_sound

    def test_source_conversion(self):
        assert _source_to_capture("0") == 0
        assert _source_to_capture("1") == 1
        assert _source_to_capture("video.mp4") == "video.mp4"


class TestConfigJSON:
    def test_roundtrip_config_dict(self):
        cfg = DMSConfig()
        d = cfg.to_dict()
        assert d["head_pose"]["yaw_threshold"] == 25.0
        assert "decision" in d and "telemetry" in d
