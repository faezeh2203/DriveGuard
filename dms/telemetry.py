"""
ثبت داده و متریک‌ها (Telemetry & Metrics)
==========================================

سه وظیفه در این ماژول انجام می‌شود (مطابق مشخصات پروژه):

1. **Telemetry**: ثبت وضعیت هر فریم و رخدادهای هشدار در پایگاه‌داده
   **SQLite** (جدول‌های ``frames`` و ``alerts``).
2. **خروجی JSON/CSV**: صادرکردن گزارش جلسه برای تحلیل بعدی.
3. **Metrics**: محاسبه Accuracy، F1-score و False Positive Rate
   با مقایسه وضعیت سیستم و Ground Truth (برای ارزیابی روی ویدیوی
   برچسب‌خورده — تابع‌های خالص و بدون وابستگی به پایگاه‌داده).
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .decision import AlertLevel, Decision
from .geometry import EulerAngle

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    camera_id    TEXT NOT NULL,
    frame_id     INTEGER NOT NULL,
    ts           REAL NOT NULL,
    fps          REAL NOT NULL,
    pitch        REAL NOT NULL,
    yaw          REAL NOT NULL,
    roll         REAL NOT NULL,
    state        TEXT NOT NULL,
    ear          REAL NOT NULL,
    eyes_closed  INTEGER NOT NULL,
    drowsy       INTEGER NOT NULL,
    alert_level  TEXT NOT NULL,
    alert_id     INTEGER,
    latency_ms   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    alert_id     INTEGER,
    level        TEXT NOT NULL,
    state        TEXT NOT NULL,
    message      TEXT NOT NULL,
    ts           REAL NOT NULL
);
"""


@dataclass(frozen=True)
class FrameRecord:
    """رکورد وضعیت یک فریم برای ثبت در پایگاه‌داده."""

    session_id: str
    camera_id: str
    frame_id: int
    ts: float
    fps: float
    angles: EulerAngle
    state: str
    ear: float
    eyes_closed: bool
    drowsy: bool
    decision: Decision
    latency_ms: float


@dataclass(frozen=True)
class AlertRecord:
    """رکورد یک رخداد هشدار."""

    session_id: str
    alert_id: int | None
    level: AlertLevel
    state: str
    message: str
    ts: float


class TelemetryStore:
    """ذخیره‌ساز Telemetry روی SQLite با خروجی JSON/CSV.

    Args:
        db_path: مسیر فایل پایگاه‌داده (پوشه‌ها در صورت نیاز ساخته می‌شوند).
        session_id: شناسه جلسه (پیش‌فرض: زمان ساخت).
        log_every_frame: ثبت وضعیت هر فریم یا فقط رخدادها.
    """

    def __init__(self, db_path: str = "telemetry/dms_log.db",
                 session_id: str | None = None,
                 log_every_frame: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or time.strftime("session-%Y%m%d-%H%M%S")
        self.log_every_frame = log_every_frame
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._pending = 0  # تعداد رکورد درج‌شده بدون commit (commit دسته‌ای)

    def close(self) -> None:
        """ثبت نهایی رکوردهای باقی‌مانده و بستن اتصال."""
        self.flush()
        self._conn.close()

    def flush(self) -> None:
        """commit رکوردهای درج‌شده اما ثبت‌نشده."""
        if self._pending:
            self._conn.commit()
            self._pending = 0

    def __enter__(self) -> "TelemetryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def log_frame(self, record: FrameRecord) -> None:
        """ثبت وضعیت یک فریم (اگر ``log_every_frame`` فعال باشد)."""
        if not self.log_every_frame:
            return
        self._conn.execute(
            "INSERT INTO frames (session_id, camera_id, frame_id, ts, fps,"
            " pitch, yaw, roll, state, ear, eyes_closed, drowsy,"
            " alert_level, alert_id, latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.session_id, record.camera_id, record.frame_id,
                record.ts, record.fps,
                record.angles.pitch, record.angles.yaw, record.angles.roll,
                record.state, record.ear, int(record.eyes_closed),
                int(record.drowsy), record.decision.alert_level.value,
                record.decision.alert_id, record.latency_ms,
            ),
        )
        # commit دسته‌ای (هر ۱۰ فریم) برای حفظ نرخ Real-Time
        self._pending += 1
        if self._pending >= 10:
            self.flush()

    def log_alert(self, record: AlertRecord) -> None:
        """ثبت یک رخداد هشدار."""
        self._conn.execute(
            "INSERT INTO alerts (session_id, alert_id, level, state,"
            " message, ts) VALUES (?,?,?,?,?,?)",
            (
                record.session_id, record.alert_id, record.level.value,
                record.state, record.message, record.ts,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    def frame_count(self) -> int:
        self.flush()
        """تعداد فریم‌های ثبت‌شده در جلسه جاری."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id = ?",
            (self.session_id,),
        )
        return int(cur.fetchone()[0])

    def alert_count(self) -> int:
        self.flush()
        """تعداد هشدارهای ثبت‌شده در جلسه جاری."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE session_id = ?",
            (self.session_id,),
        )
        return int(cur.fetchone()[0])

    def state_summary(self) -> dict:
        self.flush()
        """توزیع وضعیت‌های توجه (شمارش بر حسب وضعیت)."""
        cur = self._conn.execute(
            "SELECT state, COUNT(*) FROM frames WHERE session_id = ?"
            " GROUP BY state ORDER BY COUNT(*) DESC",
            (self.session_id,),
        )
        return {state: int(count) for state, count in cur.fetchall()}

    def export_json(self, path: str | Path) -> Path:
        """خروجی جلسه به صورت JSON (فریم‌ها + هشدارها + خلاصه).

        Args:
            path: مسیر فایل خروجی.

        Returns:
            مسیر فایل ساخته‌شده.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()
        frames = [
            {
                "frame_id": r[0], "ts": r[1], "fps": r[2],
                "pitch": r[3], "yaw": r[4], "roll": r[5],
                "state": r[6], "ear": r[7], "drowsy": bool(r[8]),
                "alert_level": r[9], "latency_ms": r[10],
            }
            for r in self._conn.execute(
                "SELECT frame_id, ts, fps, pitch, yaw, roll, state, ear,"
                " drowsy, alert_level, latency_ms FROM frames"
                " WHERE session_id = ? ORDER BY frame_id",
                (self.session_id,),
            )
        ]
        alerts = [
            {"alert_id": r[0], "level": r[1], "state": r[2],
             "message": r[3], "ts": r[4]}
            for r in self._conn.execute(
                "SELECT alert_id, level, state, message, ts FROM alerts"
                " WHERE session_id = ? ORDER BY id",
                (self.session_id,),
            )
        ]
        payload = {
            "session_id": self.session_id,
            "frame_count": len(frames),
            "alert_count": len(alerts),
            "state_summary": self.state_summary(),
            "frames": frames,
            "alerts": alerts,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def export_csv(self, path: str | Path) -> Path:
        """خروجی فریم‌های جلسه به صورت CSV.

        Args:
            path: مسیر فایل خروجی.

        Returns:
            مسیر فایل ساخته‌شده.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()
        rows = self._conn.execute(
            "SELECT frame_id, ts, fps, pitch, yaw, roll, state, ear,"
            " eyes_closed, drowsy, alert_level, alert_id, latency_ms"
            " FROM frames WHERE session_id = ? ORDER BY frame_id",
            (self.session_id,),
        )
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "frame_id", "ts", "fps", "pitch", "yaw", "roll", "state",
                "ear", "eyes_closed", "drowsy", "alert_level", "alert_id",
                "latency_ms",
            ])
            writer.writerows(rows)
        return path


# ---------------------------------------------------------------------------
# متریک‌های ارزیابی (توابع خالص؛ قابل استفاده در اسکریپت ارزیابی)
# ---------------------------------------------------------------------------

def confusion_counts(y_true: Sequence[str], y_pred: Sequence[str],
                     positive: str = "FORWARD") -> dict:
    """شمارش‌های ماتریس درهم‌ریختگی برای سناریوی دوبخشی.

    «مثبت» = وضعیت درست (``FORWARD``)؛ هر چیز دیگر «حواس‌پرتی» است.

    Args:
        y_true: وضعیت‌های واقعی.
        y_pred: وضعیت‌های پیش‌بینی سیستم.
        positive: برچسب کلاس مثبت.

    Returns:
        دیکشنری ``tp, tn, fp, fn``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch between y_true and y_pred")
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive
        p_pos = p == positive
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and not p_pos:
            tn += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            fp += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """دقت کلی (درصدر پیش‌بینی‌های صحیح) — 0..1."""
    if not y_true:
        raise ValueError("empty sequences")
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch between y_true and y_pred")
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def f1_score(y_true: Sequence[str], y_pred: Sequence[str],
             positive: str = "FORWARD") -> float:
    """F1-score کلاس مثبت (``FORWARD``) — 0..1.

    F1 = 2·P·R / (P+R) که در آن P = TP/(TP+FP) و R = TP/(TP+FN).
    """
    c = confusion_counts(y_true, y_pred, positive)
    if c["tp"] == 0:
        return 0.0
    precision = c["tp"] / (c["tp"] + c["fp"])
    recall = c["tp"] / (c["tp"] + c["fn"])
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def false_positive_rate(y_true: Sequence[str], y_pred: Sequence[str],
                        positive: str = "FORWARD") -> float:
    """نرخ هشدار کاذب — نسبت نمونه‌های منفیِ واقعی که مثبت پیش‌بینی شدند."""
    c = confusion_counts(y_true, y_pred, positive)
    negatives = c["tn"] + c["fp"]
    if negatives == 0:
        return 0.0
    return c["fp"] / negatives


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """محاسبه همه متریک‌های الزامی مشخصات پروژه یک‌جا.

    Returns:
        دیکشنری ``accuracy, f1_score, false_positive_rate, counts``.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "counts": confusion_counts(y_true, y_pred),
    }
