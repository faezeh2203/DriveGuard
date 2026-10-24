"""
داشبورد وب (Flask Dashboard)
=============================

رابط وب برای پایش زنده وضعیت سیستم از طریق مرورگر:

- ``/``                 صفحه داشبورد (قالب ``templates/dashboard.html``)
- ``/api/status``       وضعیت لحظه‌ای به صورت JSON
- ``/api/stream``       جریان Server-Sent Events وضعیت (~۱۰ بار در ثانیه)
- ``/api/history``      نمونه‌های اخیر برای رسم نمودار
- ``/api/alerts``       آخرین رخدادهای هشدار (از Telemetry)
- ``/api/summary``      خلاصه جلسه (تعداد هشدار، توزیع وضعیت‌ها)
- ``/api/export/<fmt>`` دانلود گزارش جلسه (``json`` یا ``csv``)
- ``/video_feed``       جریان MJPEG زنده با حاشیه‌نویسی

داشبورد با ``threading`` از حلقه اصلی پردازش تغذیه می‌شود؛ کلاس
:class:`SharedState` نقطه اشتراک ایمن Thread را فراهم می‌کند.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from flask import Flask, Response, abort, jsonify, render_template, send_file

from .pipeline import FrameResult

if TYPE_CHECKING:
    from .telemetry import TelemetryStore

_PKG_DIR = Path(__file__).resolve().parent


class SharedState:
    """حالت اشتراکی ایمن بین حلقه پردازش و Threadهای Flask."""

    def __init__(self, history: int = 300) -> None:
        self._cond = threading.Condition()
        self._latest: Optional[FrameResult] = None
        self._history: deque = deque(maxlen=history)
        self._jpeg: Optional[bytes] = None
        self._version = 0
        self.started_at = time.time()

    def publish(self, result: FrameResult, jpeg: Optional[bytes] = None) -> None:
        """انتشار خروجی جدید از حلقه پردازش."""
        with self._cond:
            self._latest = result
            self._history.append(result)
            if jpeg is not None:
                self._jpeg = jpeg
            self._version += 1
            self._cond.notify_all()

    def wait_for_update(self, version: int, timeout: float = 1.0) -> int:
        """انتظار تا انتشار فریم جدیدتر از ``version``؛ خروجی: نسخه جاری."""
        with self._cond:
            if self._version == version:
                self._cond.wait(timeout=timeout)
            return self._version

    @property
    def version(self) -> int:
        with self._cond:
            return self._version

    @property
    def latest(self) -> Optional[FrameResult]:
        with self._cond:
            return self._latest

    @property
    def history(self) -> list:
        with self._cond:
            return list(self._history)

    @property
    def jpeg(self) -> Optional[bytes]:
        with self._cond:
            return self._jpeg


def status_payload(result: Optional[FrameResult], started_at: float | None = None) -> dict:
    """تبدیل :class:`FrameResult` به دیکشنری JSON داشبورد."""
    uptime = time.time() - started_at if started_at else 0.0
    if result is None:
        return {"state": "NO_FACE", "state_fa": "در انتظار فریم...",
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "fps": 0.0, "alert_level": "NONE", "message": "",
                "alert_active": False, "waiting": True, "uptime": uptime}
    d = result.decision
    dr = result.drowsiness
    return {
        "frame_id": result.frame_id,
        "ts": result.timestamp,
        "state": d.stable_state.value,
        "state_fa": d.stable_state.label_fa,
        "raw_state": result.attention.raw_state.value,
        "yaw": result.angles.yaw,
        "pitch": result.angles.pitch,
        "roll": result.angles.roll,
        "fps": result.fps,
        "latency_ms": result.latency_ms,
        "alert_level": d.alert_level.value,
        "alert_active": d.alert_active,
        "alert_id": d.alert_id,
        "reason": d.reason.value,
        "message": d.message,
        "distracted_seconds": d.distracted_seconds,
        "threshold_seconds": d.threshold_seconds,
        "alert_seconds": d.alert_seconds,
        "recovery_progress": d.recovery_progress,
        "face_seen": result.face_seen,
        "phone_detected": result.phone_detected,
        "ear": result.attention.ear,
        "eyes_closed": bool(dr and dr.eyes_closed),
        "drowsy": bool(dr and dr.drowsy),
        "perclos": dr.perclos if dr else 0.0,
        "blink_count": dr.blink_count if dr else 0,
        "alert_count": result.alert_count,
        "uptime": uptime,
        "waiting": False,
    }


class Dashboard:
    """سرور داشبورد Flask.

    Args:
        shared: حالت اشتراکی تغذیه‌شده توسط حلقه پردازش.
        store: ذخیره‌ساز Telemetry (اختیاری؛ برای لاگ هشدار و خروجی گزارش).
    """

    def __init__(self, shared: SharedState, store: "TelemetryStore | None" = None) -> None:
        self.shared = shared
        self.store = store
        self.app = Flask("dms-dashboard",
                         template_folder=str(_PKG_DIR / "templates"),
                         static_folder=str(_PKG_DIR / "static"))
        self.app.json.ensure_ascii = False
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/api/status", "status", self._status)
        self.app.add_url_rule("/api/stream", "stream", self._stream)
        self.app.add_url_rule("/api/history", "history", self._history)
        self.app.add_url_rule("/api/alerts", "alerts", self._alerts)
        self.app.add_url_rule("/api/summary", "summary", self._summary)
        self.app.add_url_rule("/api/export/<fmt>", "export", self._export)
        self.app.add_url_rule("/video_feed", "feed", self._video_feed)

    # ------------------------------------------------------------------
    def _index(self) -> str:
        return render_template("dashboard.html",
                               session_id=getattr(self.store, "session_id", ""))

    def _status(self) -> Response:
        return jsonify(status_payload(self.shared.latest, self.shared.started_at))

    def _stream(self) -> Response:
        """Server-Sent Events: ارسال وضعیت با هر فریم جدید (حداکثر ~۱۰Hz)."""
        def generate():
            version = -1
            while True:
                version = self.shared.wait_for_update(version, timeout=1.0)
                data = json.dumps(status_payload(self.shared.latest, self.shared.started_at),
                                  ensure_ascii=False)
                yield f"data: {data}\n\n"
                time.sleep(0.1)
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _history(self) -> Response:
        return jsonify([
            {"t": r.timestamp, "yaw": r.angles.yaw, "pitch": r.angles.pitch,
             "roll": r.angles.roll, "ear": r.attention.ear,
             "state": r.decision.stable_state.value,
             "level": r.decision.alert_level.value}
            for r in self.shared.history
        ])

    def _alerts(self) -> Response:
        if self.store is None:
            return jsonify([])
        try:
            return jsonify(self.store.recent_alerts(limit=30))
        except Exception:  # noqa: BLE001
            return jsonify([])

    def _summary(self) -> Response:
        latest = self.shared.latest
        out = {"session_id": getattr(self.store, "session_id", ""),
               "alert_count": latest.alert_count if latest else 0,
               "uptime": time.time() - self.shared.started_at}
        if self.store is not None:
            try:
                out["state_summary"] = self.store.state_summary()
                out["frame_count"] = self.store.frame_count()
            except Exception:  # noqa: BLE001
                pass
        return jsonify(out)

    def _export(self, fmt: str) -> Response:
        if self.store is None or fmt not in ("json", "csv"):
            abort(404)
        name = f"{self.store.session_id}.{fmt}"
        path = Path(tempfile.gettempdir()) / f"driveguard-{name}"
        if fmt == "json":
            self.store.export_json(path)
        else:
            self.store.export_csv(path)
        return send_file(path, as_attachment=True, download_name=name)

    def _video_feed(self) -> Response:
        """جریان MJPEG زنده (فقط فریم‌های جدید ارسال می‌شوند)."""
        def generate():
            version = -1
            while True:
                new_version = self.shared.wait_for_update(version, timeout=1.0)
                jpeg = self.shared.jpeg
                if jpeg is not None and new_version != version:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                           + jpeg + b"\r\n")
                version = new_version
                time.sleep(0.03)
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------
    def run(self, host: str = "127.0.0.1", port: int = 5000,
            debug: bool = False) -> None:
        """اجرای سرور (بلوکه‌کننده — معمولاً در Thread جدا فراخوانی شود)."""
        self.app.run(host=host, port=port, debug=debug,
                     use_reloader=False, threaded=True)


__all__ = ["SharedState", "Dashboard", "status_payload"]
