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
import threading
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
    ts           REAL NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'STARTED',
    reason       TEXT NOT NULL DEFAULT '',
    duration     REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id, id);
"""

# ستون‌هایی که در نسخه ۲ به جدول alerts اضافه شدند (مهاجرت پایگاه‌داده قدیمی)
_ALERT_MIGRATIONS = {
    "kind": "TEXT NOT NULL DEFAULT 'STARTED'",
    "reason": "TEXT NOT NULL DEFAULT ''",
    "duration": "REAL NOT NULL DEFAULT 0",
}


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
    kind: str = "STARTED"
    reason: str = ""
    duration: float = 0.0


class TelemetryStore:
    """ذخیره‌ساز Telemetry روی SQLite با خروجی JSON/CSV.

    ایمن برای استفاده هم‌زمان از چند Thread (حلقه پردازش + داشبورد):
    همه دسترسی‌ها با یک قفل سریال می‌شوند و حالت WAL فعال است تا
    خواندن، نوشتن را مسدود نکند.

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
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass  # برخی فایل‌سیستم‌ها WAL را پشتیبانی نمی‌کنند
        self._migrate()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._pending = 0  # تعداد رکورد درج‌شده بدون commit (commit دسته‌ای)
        self._closed = False

    def _migrate(self) -> None:
        """افزودن ستون‌های جدید به جدول alerts در پایگاه‌داده‌های نسخه قبل."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(alerts)")}
        if not cols:
            return  # جدول هنوز ساخته نشده
        for name, decl in _ALERT_MIGRATIONS.items():
            if name not in cols:
                self._conn.execute(f"ALTER TABLE alerts ADD COLUMN {name} {decl}")
        self._conn.commit()

    def close(self) -> None:
        """ثبت نهایی رکوردهای باقی‌مانده و بستن اتصال."""
        with self._lock:
            if self._closed:
                return
            self.flush()
            self._conn.close()
            self._closed = True

    def flush(self) -> None:
        """commit رکوردهای درج‌شده اما ثبت‌نشده."""
        with self._lock:
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
        with self._lock:
            self._insert_frame(record)
            # commit دسته‌ای (هر ۳۰ فریم) برای حفظ نرخ Real-Time
            self._pending += 1
            if self._pending >= 30:
                self.flush()

    def _insert_frame(self, record: FrameRecord) -> None:
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

    def log_alert(self, record: AlertRecord) -> None:
        """ثبت یک رخداد هشدار (شروع، ارتقا، تغییر علت یا پایان)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO alerts (session_id, alert_id, level, state,"
                " message, ts, kind, reason, duration)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.session_id, record.alert_id, record.level.value,
                    record.state, record.message, record.ts,
                    record.kind, record.reason, float(record.duration),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    def _query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            self.flush()
            return self._conn.execute(sql, params).fetchall()

    def frame_count(self) -> int:
        """تعداد فریم‌های ثبت‌شده در جلسه جاری."""
        rows = self._query("SELECT COUNT(*) FROM frames WHERE session_id = ?",
                           (self.session_id,))
        return int(rows[0][0])

    def alert_count(self, kind: str | None = None) -> int:
        """تعداد رخدادهای هشدار ثبت‌شده در جلسه جاری.

        Args:
            kind: فیلتر نوع رخداد (مثلاً ``"STARTED"`` برای تعداد هشدارهای مجزا).
        """
        if kind is None:
            rows = self._query("SELECT COUNT(*) FROM alerts WHERE session_id = ?",
                               (self.session_id,))
        else:
            rows = self._query(
                "SELECT COUNT(*) FROM alerts WHERE session_id = ? AND kind = ?",
                (self.session_id, kind))
        return int(rows[0][0])

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        """آخرین رخدادهای هشدار جلسه جاری (جدیدترین اول)."""
        rows = self._query(
            "SELECT alert_id, level, state, message, ts, kind, reason, duration"
            " FROM alerts WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, int(limit)))
        return [
            {"alert_id": r[0], "level": r[1], "state": r[2], "message": r[3],
             "ts": r[4], "kind": r[5], "reason": r[6], "duration": r[7]}
            for r in rows
        ]

    def state_summary(self) -> dict:
        """توزیع وضعیت‌های توجه (شمارش بر حسب وضعیت)."""
        rows = self._query(
            "SELECT state, COUNT(*) FROM frames WHERE session_id = ?"
            " GROUP BY state ORDER BY COUNT(*) DESC",
            (self.session_id,))
        return {state: int(count) for state, count in rows}

    def export_json(self, path: str | Path) -> Path:
        """خروجی جلسه به صورت JSON (فریم‌ها + هشدارها + خلاصه).

        Args:
            path: مسیر فایل خروجی.

        Returns:
            مسیر فایل ساخته‌شده.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = [
            {
                "frame_id": r[0], "ts": r[1], "fps": r[2],
                "pitch": r[3], "yaw": r[4], "roll": r[5],
                "state": r[6], "ear": r[7], "drowsy": bool(r[8]),
                "alert_level": r[9], "latency_ms": r[10],
            }
            for r in self._query(
                "SELECT frame_id, ts, fps, pitch, yaw, roll, state, ear,"
                " drowsy, alert_level, latency_ms FROM frames"
                " WHERE session_id = ? ORDER BY frame_id",
                (self.session_id,),
            )
        ]
        alerts = [
            {"alert_id": r[0], "level": r[1], "state": r[2],
             "message": r[3], "ts": r[4], "kind": r[5], "reason": r[6],
             "duration": r[7]}
            for r in self._query(
                "SELECT alert_id, level, state, message, ts, kind, reason,"
                " duration FROM alerts WHERE session_id = ? ORDER BY id",
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
        rows = self._query(
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
    """FPR کلاس مثبت — نسبت نمونه‌های منفیِ واقعی که مثبت پیش‌بینی شدند.

    توجه: با ``positive="FORWARD"``، این مقدار نسبت حواس‌پرتی‌های واقعی
    است که سیستم «رو به جلو» تشخیص داده — یعنی **نرخ از دست دادن
    حواس‌پرتی** (Miss Rate)، نه هشدار کاذب. برای نرخ هشدار کاذب از
    :func:`false_alarm_rate` استفاده کنید.
    """
    c = confusion_counts(y_true, y_pred, positive)
    negatives = c["tn"] + c["fp"]
    if negatives == 0:
        return 0.0
    return c["fp"] / negatives


def false_alarm_rate(y_true: Sequence[str], y_pred: Sequence[str],
                     attentive: str = "FORWARD") -> float:
    """نرخ هشدار کاذب — نسبت فریم‌هایی که راننده واقعاً رو به جلو بوده
    اما سیستم حواس‌پرتی تشخیص داده است (معیار آزاردهندگی سیستم)."""
    c = confusion_counts(y_true, y_pred, attentive)
    actual_attentive = c["tp"] + c["fn"]
    if actual_attentive == 0:
        return 0.0
    return c["fn"] / actual_attentive


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """محاسبه همه متریک‌های الزامی مشخصات پروژه یک‌جا.

    Returns:
        دیکشنری ``accuracy, f1_score, false_positive_rate,
        false_alarm_rate, counts``.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "counts": confusion_counts(y_true, y_pred),
    }
