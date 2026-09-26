"""
داشبورد وب (Flask Dashboard)
=============================

یک رابط وب سبک برای پایش زنده وضعیت سیستم از طریق مرورگر:

- ``/``             صفحه داشبورد (HTML/JS ساده)
- ``/api/status``   وضعیت لحظه‌ای به صورت JSON (وضعیت توجه، زوایا، FPS و ...)
- ``/api/history``  ۱۰۰ نمونه آخر وضعیت‌ها برای رسم نمودار
- ``/video_feed``   جریان تصویر MJPEG زنده با حاشیه‌نویسی

داشبورد با ``threading`` از یک Producer جدا (حلقه اصلی پردازش)
تغذیه می‌شود؛ کلاس :class:`SharedState` نقطه اشتراک ایمن Thread
را فراهم می‌کند.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict
from typing import Optional

from flask import Flask, Response, jsonify, render_template_string

from .decision import AlertLevel
from .pipeline import DMSPipeline, FrameResult

_PAGE = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<title>DriveGuard — سیستم تشخیص حواس‌پرتی راننده</title>
<style>
  body { font-family: Tahoma, sans-serif; background:#10141c; color:#e8e8e8;
         margin:0; padding:24px; text-align:center; }
  h1 { color:#7ec8ff; font-size:1.4rem; }
  .grid { display:flex; gap:16px; justify-content:center; flex-wrap:wrap; }
  .card { background:#1a2230; border-radius:12px; padding:16px 28px; min-width:150px; }
  .card .v { font-size:2rem; font-weight:bold; }
  .video img { max-width:640px; border-radius:12px; margin-top:16px; }
  #state.FORWARD { color:#5dd879; } #state.LEFT, #state.RIGHT { color:#ffb84d; }
  #state.DOWN, #state.UP, #state.NO_FACE { color:#ff5c5c; }
  #alert { margin-top:12px; font-size:1.2rem; min-height:1.5em; color:#ff5c5c; }
</style>
</head>
<body>
<h1>🚗 DriveGuard — سیستم تشخیص حواس‌پرتی راننده</h1>
<div class="grid">
  <div class="card"><div>وضعیت توجه</div><div class="v" id="state">-</div></div>
  <div class="card"><div>Yaw</div><div class="v" id="yaw">-</div></div>
  <div class="card"><div>Pitch</div><div class="v" id="pitch">-</div></div>
  <div class="card"><div>Roll</div><div class="v" id="roll">-</div></div>
  <div class="card"><div>FPS</div><div class="v" id="fps">-</div></div>
</div>
<div id="alert"></div>
<div class="video"><img src="/video_feed" alt="live"></div>
<script>
async function poll() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    document.getElementById('state').textContent = s.state_fa || s.state;
    document.getElementById('state').className = 'v';
    document.getElementById('state').id = 'state';
    const el = document.getElementById('state');
    el.textContent = s.state_fa || s.state;
    el.style.color = s.alert_level === 'CRITICAL' ? '#ff5c5c'
                   : s.alert_level === 'WARNING' ? '#ffb84d' : '#5dd879';
    document.getElementById('yaw').textContent = s.yaw.toFixed(1) + '°';
    document.getElementById('pitch').textContent = s.pitch.toFixed(1) + '°';
    document.getElementById('roll').textContent = s.roll.toFixed(1) + '°';
    document.getElementById('fps').textContent = s.fps.toFixed(1);
    document.getElementById('alert').textContent = s.message || '';
  } catch (e) { /* اتصال هنوز برقرار نشده */ }
}
setInterval(poll, 500); poll();
</script>
</body>
</html>"""


class SharedState:
    """حالت اشتراکی ایمن بین حلقه پردازش و Threadهای Flask."""

    def __init__(self, history: int = 100) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[FrameResult] = None
        self._history: deque = deque(maxlen=history)
        self._jpeg: Optional[bytes] = None

    def publish(self, result: FrameResult, jpeg: Optional[bytes] = None) -> None:
        """انتشار خروجی جدید از حلقه پردازش."""
        with self._lock:
            self._latest = result
            self._history.append(result)
            if jpeg is not None:
                self._jpeg = jpeg

    @property
    def latest(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest

    @property
    def history(self) -> list:
        with self._lock:
            return list(self._history)

    @property
    def jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg


class Dashboard:
    """سرور داشبورد Flask.

    Args:
        shared: حالت اشتراکی تغذیه‌شده توسط حلقه پردازش.
    """

    def __init__(self, shared: SharedState) -> None:
        self.shared = shared
        self.app = Flask("dms-dashboard")
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/api/status", "status", self._status)
        self.app.add_url_rule("/api/history", "history", self._history)
        self.app.add_url_rule("/video_feed", "feed", self._video_feed)

    # ------------------------------------------------------------------
    def _index(self) -> str:
        """صفحه اصلی داشبورد."""
        return render_template_string(_PAGE)

    def _status(self) -> Response:
        """وضعیت لحظه‌ای به صورت JSON."""
        result = self.shared.latest
        if result is None:
            return jsonify({"state": "NO_FACE", "state_fa": "در انتظار فریم...",
                            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                            "fps": 0.0, "alert_level": "NONE", "message": ""})
        return jsonify({
            "state": result.decision.stable_state.value,
            "state_fa": result.decision.stable_state.label_fa,
            "raw_state": result.attention.raw_state.value,
            "yaw": result.angles.yaw,
            "pitch": result.angles.pitch,
            "roll": result.angles.roll,
            "fps": result.fps,
            "latency_ms": result.latency_ms,
            "alert_level": result.decision.alert_level.value,
            "message": result.decision.message,
            "face_seen": result.face_seen,
            "ear": result.attention.ear,
        })

    def _history(self) -> Response:
        """تاریخچه ۱۰۰ نمونه آخر برای رسم نمودار."""
        return jsonify([
            {"t": r.timestamp, "yaw": r.angles.yaw, "pitch": r.angles.pitch,
             "roll": r.angles.roll, "state": r.decision.stable_state.value}
            for r in self.shared.history
        ])

    def _video_feed(self) -> Response:
        """جریان MJPEG زنده."""
        def generate():
            while True:
                jpeg = self.shared.jpeg
                if jpeg is None:
                    yield b"--frame\r\n\r\n"
                else:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpeg + b"\r\n")
                import time
                time.sleep(0.033)
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # ------------------------------------------------------------------
    def run(self, host: str = "127.0.0.1", port: int = 5000,
            debug: bool = False) -> None:
        """اجرای سرور (بلوکه‌کننده — معمولاً در Thread جدا فراخوانی شود)."""
        self.app.run(host=host, port=port, debug=debug,
                     use_reloader=False, threaded=True)


__all__ = ["SharedState", "Dashboard"]
