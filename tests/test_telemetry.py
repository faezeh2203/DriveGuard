"""تست‌های Telemetry: ثبت SQLite، خروجی JSON/CSV و متریک‌های ارزیابی."""

import csv
import json
import sqlite3

import pytest

from dms.config import DMSConfig
from dms.decision import AlertLevel, Decision
from dms.attention import AttentionState
from dms.geometry import EulerAngle
from dms.telemetry import (
    AlertRecord,
    FrameRecord,
    TelemetryStore,
    accuracy,
    confusion_counts,
    evaluate,
    f1_score,
    false_positive_rate,
)


def make_frame_record(session="s1", state="FORWARD", frame_id=1):
    return FrameRecord(
        session_id=session, camera_id="cam", frame_id=frame_id, ts=frame_id / 30.0,
        fps=30.0, angles=EulerAngle(1, 2, 3), state=state, ear=0.3,
        eyes_closed=False, drowsy=False,
        decision=Decision(stable_state=AttentionState.FORWARD,
                          alert_level=AlertLevel.NONE),
        latency_ms=5.0,
    )


@pytest.fixture
def store(tmp_path):
    s = TelemetryStore(db_path=str(tmp_path / "t.db"), session_id="s1")
    yield s
    s.close()


class TestStorage:
    def test_frame_roundtrip(self, store):
        store.log_frame(make_frame_record())
        assert store.frame_count() == 1

    def test_log_every_frame_disabled(self, tmp_path):
        s = TelemetryStore(db_path=str(tmp_path / "x.db"), log_every_frame=False)
        s.log_frame(make_frame_record())
        assert s.frame_count() == 0
        s.close()

    def test_alert_log(self, store):
        store.log_alert(AlertRecord("s1", 1, AlertLevel.WARNING,
                                    "LEFT", "حواس‌پرتی", 1.0))
        assert store.alert_count() == 1

    def test_state_summary(self, store):
        for i in range(3):
            store.log_frame(make_frame_record(frame_id=i, state="FORWARD"))
        store.log_frame(make_frame_record(frame_id=10, state="LEFT"))
        assert store.state_summary() == {"FORWARD": 3, "LEFT": 1}

    def test_parent_dirs_created(self, tmp_path):
        s = TelemetryStore(db_path=str(tmp_path / "a/b/c/t.db"))
        assert (tmp_path / "a/b/c/t.db").exists()
        s.close()


class TestExport:
    def _fill(self, store):
        for i in range(4):
            store.log_frame(make_frame_record(frame_id=i))
        store.log_alert(AlertRecord("s1", 1, AlertLevel.WARNING,
                                    "LEFT", "پیام", 1.0))

    def test_export_json(self, store, tmp_path):
        self._fill(store)
        out = store.export_json(tmp_path / "out.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["frame_count"] == 4
        assert data["alert_count"] == 1
        assert data["frames"][0]["state"] == "FORWARD"

    def test_export_csv(self, store, tmp_path):
        self._fill(store)
        out = store.export_csv(tmp_path / "out.csv")
        rows = list(csv.reader(out.open(encoding="utf-8")))
        assert rows[0][:2] == ["frame_id", "ts"]
        assert len(rows) == 5  # header + 4 فریم


class TestMetrics:
    """متریک‌های Accuracy / F1 / FPR با داده‌های محاسبه‌دستی."""

    Y_TRUE = ["FORWARD", "FORWARD", "LEFT", "LEFT", "DOWN"]
    Y_PRED = ["FORWARD", "LEFT", "LEFT", "FORWARD", "DOWN"]

    def test_counts_manual(self):
        c = confusion_counts(self.Y_TRUE, self.Y_PRED)
        # FORWARD: (T,T)=tp؛ (T,L)=fn؛ (L,F)=fp؛ (L,L)=tn؛ (D,D)=tn
        assert c == {"tp": 1, "tn": 2, "fp": 1, "fn": 1}

    def test_accuracy_manual(self):
        assert accuracy(self.Y_TRUE, self.Y_PRED) == pytest.approx(3 / 5)

    def test_f1_manual(self):
        # P = 1/2، R = 1/2 ⇒ F1 = 0.5
        assert f1_score(self.Y_TRUE, self.Y_PRED) == pytest.approx(0.5)

    def test_fpr_manual(self):
        # منفی‌های واقعی: 3 مورد (L,L,D)؛ fp = 1 ⇒ FPR = 1/3
        assert false_positive_rate(self.Y_TRUE, self.Y_PRED) == pytest.approx(1 / 3)

    def test_perfect_classifier(self):
        y = ["FORWARD", "LEFT", "DOWN"]
        e = evaluate(y, y)
        assert e["accuracy"] == 1.0
        assert e["f1_score"] == 1.0
        assert e["false_positive_rate"] == 0.0

    def test_all_negative_predictions(self):
        y = ["LEFT", "DOWN"]
        p = ["FORWARD", "FORWARD"]
        assert f1_score(y, p) == 0.0
        assert false_positive_rate(y, p) == 1.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            accuracy(["FORWARD"], [])

    def test_empty_accuracy_raises(self):
        with pytest.raises(ValueError):
            accuracy([], [])

    def test_evaluate_keys(self):
        e = evaluate(self.Y_TRUE, self.Y_PRED)
        assert set(e) == {"accuracy", "f1_score", "false_positive_rate",
                          "false_alarm_rate", "counts"}


class TestConfig:
    """پیکربندی JSON."""

    def test_from_json(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({
            "head_pose": {"yaw_threshold": 30.0},
            "decision": {"state_confirm_frames": 10},
        }), encoding="utf-8")
        cfg = DMSConfig.from_json(cfg_file)
        assert cfg.head_pose.yaw_threshold == 30.0
        assert cfg.decision.state_confirm_frames == 10
        # مقدار پیش‌فرض دست‌نخورده
        assert cfg.head_pose.pitch_down_threshold == -15.0

    def test_from_json_rejects_unknown_section(self, tmp_path):
        # غلط املایی در پیکربندی ایمنی نباید بی‌صدا نادیده گرفته شود
        from dms.config import ConfigError
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"nonexistent": 1}), encoding="utf-8")
        with pytest.raises(ConfigError):
            DMSConfig.from_json(cfg_file)

    def test_from_json_rejects_unknown_key(self, tmp_path):
        from dms.config import ConfigError
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"decision": {"distraction_secs": 1}}),
                            encoding="utf-8")
        with pytest.raises(ConfigError, match="distraction_secs"):
            DMSConfig.from_json(cfg_file)

    def test_invalid_value_rejected(self):
        from dms.config import ConfigError, DecisionConfig
        with pytest.raises(ConfigError):
            DecisionConfig(state_confirm_frames=0)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            DMSConfig.from_json("nope.json")

    def test_defaults_documented_values(self):
        cfg = DMSConfig()
        assert cfg.head_pose.yaw_threshold == 25.0
        assert cfg.head_pose.pitch_down_threshold == -15.0
        assert cfg.target_fps == 30


class TestMigration:
    def test_old_alerts_schema_upgraded(self, tmp_path):
        import sqlite3
        db = tmp_path / "old.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " session_id TEXT NOT NULL, alert_id INTEGER, level TEXT NOT NULL,"
                    " state TEXT NOT NULL, message TEXT NOT NULL, ts REAL NOT NULL)")
        con.execute("INSERT INTO alerts (session_id, alert_id, level, state, message, ts)"
                    " VALUES ('s1', 1, 'WARNING', 'LEFT', 'm', 1.0)")
        con.commit(); con.close()
        with TelemetryStore(str(db), session_id="s1") as store:
            rows = store.recent_alerts()
            assert rows[0]["kind"] == "STARTED"
            store.log_alert(AlertRecord("s1", 1, AlertLevel.NONE, "FORWARD", "ok", 2.0,
                                        kind="CLEARED", reason="LATERAL", duration=1.0))
            assert store.alert_count(kind="CLEARED") == 1

    def test_cross_thread_access(self, tmp_path):
        import threading
        store = TelemetryStore(str(tmp_path / "t.db"), session_id="s1")
        errors = []
        def reader():
            try:
                for _ in range(20):
                    store.frame_count(); store.recent_alerts()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        th = threading.Thread(target=reader)
        th.start()
        for i in range(50):
            store.log_frame(make_frame_record(frame_id=i))
        th.join()
        store.close()
        assert errors == []


class TestFalseAlarmRate:
    def test_false_alarm_rate(self):
        from dms.telemetry import false_alarm_rate
        y_true = ["FORWARD", "FORWARD", "FORWARD", "FORWARD", "DOWN"]
        y_pred = ["FORWARD", "LEFT", "FORWARD", "FORWARD", "DOWN"]
        assert false_alarm_rate(y_true, y_pred) == pytest.approx(0.25)
