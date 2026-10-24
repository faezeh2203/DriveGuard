"""
هشدار به راننده (Driver Alerting)
==================================

:class:`AlertManager` تصمیم موتور تصمیم را به اکشن فیزیکی تبدیل
می‌کند:

- **صوتی**: بوق تکرارشونده در یک Thread جداگانه (:class:`SoundWorker`)
  تا زمانی که هشدار فعال است؛ الگوی بوق به سطح هشدار بستگی دارد
  (WARNING: بوق منظم، CRITICAL: بوق تند و زیر). پخش صدا هرگز حلقه
  پردازش تصویر را مسدود نمی‌کند (``winsound.Beep`` همگام است و در نسخه
  قبل هر بار ~۲۵۰ms فریم‌ها را متوقف می‌کرد).
- **تصویری**: HUD روی فریم — نوار وضعیت، نوار پیشرفت تا هشدار،
  بنر چشمک‌زن هشدار و قاب رنگی.
- **رخداد**: تبدیل تغییر وضعیت هشدار (شروع/ارتقا/پایان) به
  :class:`AlertEvent` برای ثبت در Telemetry.

سیستم در محیط‌های Headless (بدون صدا/نمایشگر) هم بدون خطا کار می‌کند.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import AlertConfig
from .decision import AlertLevel, AlertReason, Decision, Transition

log = logging.getLogger(__name__)

# رنگ BGR قاب هشدار برای هر سطح
ALERT_COLORS = {
    AlertLevel.NONE: (180, 180, 180),
    AlertLevel.INFO: (30, 200, 60),
    AlertLevel.WARNING: (30, 120, 255),
    AlertLevel.CRITICAL: (30, 30, 230),
}

# متن انگلیسی بنر روی تصویر (putText در OpenCV از فارسی پشتیبانی نمی‌کند)
_REASON_TEXT = {
    AlertReason.NONE: "",
    AlertReason.DROWSY: "WAKE UP! DROWSINESS DETECTED",
    AlertReason.PHONE: "PUT THE PHONE DOWN!",
    AlertReason.LOOK_DOWN: "EYES ON THE ROAD!",
    AlertReason.LOOK_UP: "EYES ON THE ROAD!",
    AlertReason.LATERAL: "LOOK AHEAD!",
    AlertReason.NO_FACE: "DRIVER NOT VISIBLE!",
}


@dataclass(frozen=True)
class AlertEvent:
    """یک رخداد هشدار برای ثبت در Telemetry و داشبورد.

    Attributes:
        alert_id: شناسه دوره هشدار.
        level: سطح هشدار.
        message: پیام متنی.
        state: وضعیت توجه پایدار.
        kind: نوع رخداد: ``STARTED``، ``ESCALATED``، ``REASON_CHANGED``،
            ``CLEARED`` یا ``ONGOING`` (ادامه هشدار بدون تغییر).
        reason: علت هشدار.
        duration: مدت هشدار تا این لحظه (ثانیه).
    """

    alert_id: int | None
    level: AlertLevel
    message: str
    state: str
    kind: str = "ONGOING"
    reason: str = ""
    duration: float = 0.0

    @property
    def is_transition(self) -> bool:
        """True اگر رخداد تغییر وضعیت باشد (قابل ثبت در پایگاه‌داده)."""
        return self.kind != "ONGOING"


_SAMPLE_RATE = 24000


def make_siren(frequency: float, duration_ms: int, volume: float = 1.0,
               sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """ساخت آژیر بلند دوتُنه (warble) غنی از هارمونیک — نمونه‌های float32.

    موج سینوسی با هارمونیک‌های فرد و برش نرم، در بلندی یکسان بسیار
    رساتر از ``winsound.Beep`` (بلندگوی سیستمی) شنیده می‌شود.
    """
    n = max(int(sample_rate * duration_ms / 1000), 1)
    t = np.arange(n) / sample_rate
    # تغییر فرکانس بین f و 1.35f با نرخ ۸ هرتز (آژیر پلیسی)
    sweep = frequency * (1.0 + 0.35 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8 * t))))
    phase = 2 * np.pi * np.cumsum(sweep) / sample_rate
    wave_ = np.sin(phase) + 0.45 * np.sin(3 * phase) + 0.25 * np.sin(5 * phase)
    wave_ = np.tanh(2.5 * wave_) / np.tanh(2.5)
    fade = min(int(0.01 * sample_rate), n // 2)
    if fade:
        env = np.ones(n)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave_ *= env
    return (wave_ * 0.98 * volume).astype(np.float32)


def to_wav_bytes(samples: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """تبدیل نمونه‌های float به فایل WAV (۱۶ بیتی، مونو) در حافظه."""
    import io
    import wave

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _load_wav(path: Path, volume: float) -> bytes | None:
    """خواندن WAV و اعمال ضریب بلندی (None اگر خواندن ناموفق باشد)."""
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getsampwidth() != 2:
                return None
            rate, channels = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    except (OSError, EOFError, wave.Error):
        return None
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return to_wav_bytes(pcm.astype(np.float32) / 32767 * volume, rate)


class SoundWorker:
    """پخش آژیر و هشدار گفتاری فارسی در Thread پس‌زمینه.

    حلقه پردازش فقط وضعیت هشدار را با :meth:`update` اعلام می‌کند؛ این
    Thread تا وقتی هشدار فعال است آژیر را تکرار می‌کند و هر
    ``voice_repeat_seconds`` ثانیه (و فوراً هنگام شروع یا تغییر علت هشدار)
    جمله گفتاری متناسب را پخش می‌کند — مثلاً «لطفاً به جلو و جاده نگاه کنید».
    پس از اصلاح رفتار، پیام «متشکرم» پخش می‌شود.

    فایل‌های گفتاری با ``scripts/generate_voice.py`` ساخته می‌شوند؛ اگر
    موجود نباشند فقط آژیر پخش می‌شود.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig()
        cfg = self.config
        self._level = AlertLevel.NONE
        self._reason = AlertReason.NONE
        self._queue: list[str] = []  # جمله‌های یک‌باره (ESCALATED، CLEARED)
        self._cond = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._is_windows = platform.system() == "Windows"
        self._sirens = {
            AlertLevel.WARNING: to_wav_bytes(make_siren(cfg.warning_frequency, cfg.beep_ms, cfg.volume)),
            AlertLevel.CRITICAL: to_wav_bytes(make_siren(cfg.critical_frequency, cfg.beep_ms, cfg.volume)),
        }
        self.voices: dict[str, bytes] = self._load_voices() if cfg.voice_enabled else {}
        self._last_voice_ts = -1e9
        self._last_voice_reason = AlertReason.NONE
        self._tmp_files: dict[int, str] = {}

    def _load_voices(self) -> dict[str, bytes]:
        voice_dir = Path(self.config.voice_dir)
        voices = {}
        for path in sorted(voice_dir.glob("*.wav")) if voice_dir.is_dir() else []:
            data = _load_wav(path, self.config.volume)
            if data is not None:
                voices[path.stem.upper()] = data
        if not voices:
            log.info("voice alerts unavailable (no WAV files in %s) — "
                     "run scripts/generate_voice.py", voice_dir)
        return voices

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="dms-sound", daemon=True)
        self._thread.start()

    def set_level(self, level: AlertLevel) -> None:
        """تنظیم سطح هشدار (سازگاری با نسخه قبل)."""
        with self._cond:
            if level is not self._level:
                self._level = level
                self._cond.notify()

    def update(self, decision: Decision) -> None:
        """اعلام وضعیت هشدار فریم جاری به Thread صدا."""
        with self._cond:
            level = decision.alert_level
            reason = decision.reason if decision.alert_active else AlertReason.NONE
            if decision.transition is Transition.CLEARED:
                self._last_voice_reason = AlertReason.NONE
                if self.config.voice_on_clear:
                    self._queue = ["CLEARED"]
            elif (decision.transition is Transition.ESCALATED
                  and decision.reason is not AlertReason.DROWSY):
                self._queue.append("ESCALATED")
            if level is not self._level or reason is not self._reason or self._queue:
                self._level, self._reason = level, reason
                self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None
        for path in self._tmp_files.values():
            try:
                Path(path).unlink()
            except OSError:
                pass

    @property
    def _active(self) -> bool:
        return self._level in (AlertLevel.WARNING, AlertLevel.CRITICAL)

    def _run(self) -> None:
        cfg = self.config
        while True:
            with self._cond:
                while not self._stop and not self._active and not self._queue:
                    self._cond.wait()
                if self._stop:
                    return
                level, reason = self._level, self._reason
                once = self._queue.pop(0) if self._queue else None

            if once is not None:
                if once in self.voices:
                    self._play(self.voices[once])
                continue

            self._play(self._sirens[level])
            now = time.monotonic()
            voice = self.voices.get(reason.value)
            if voice is not None and (reason is not self._last_voice_reason
                                      or now - self._last_voice_ts >= cfg.voice_repeat_seconds):
                self._play(voice)
                self._last_voice_ts = time.monotonic()
                self._last_voice_reason = reason
                continue

            interval = cfg.critical_interval if level is AlertLevel.CRITICAL else cfg.warning_interval
            with self._cond:
                # انتظار تا آژیر بعدی؛ تغییر وضعیت یا توقف زودتر بیدار می‌کند
                self._cond.wait(timeout=max(interval - cfg.beep_ms / 1000.0, 0.05))
                if self._stop:
                    return

    def _play(self, wav: bytes) -> None:
        """پخش همگام یک WAV (فقط این Thread مسدود می‌شود)."""
        try:
            if self._is_windows:
                import winsound
                winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                return
            import shutil
            import subprocess
            import tempfile

            player = next((p for p in ("paplay", "aplay", "afplay") if shutil.which(p)), None)
            if player is None:
                print("\a", end="", flush=True)
                time.sleep(0.3)
                return
            key = id(wav)
            if key not in self._tmp_files:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                    fh.write(wav)
                    self._tmp_files[key] = fh.name
            subprocess.run([player, self._tmp_files[key]], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except Exception:  # noqa: BLE001
            # نبود سخت‌افزار/درایور صدا هرگز نباید سیستم را متوقف کند
            time.sleep(0.2)


class AlertManager:
    """مدیریت هشدار تصویری/صوتی و تولید رخدادهای هشدار.

    Args:
        enable_sound: فعال‌سازی بوق صوتی (در محیط تست معمولاً خاموش).
        beep_interval_seconds: (سازگاری با نسخه قبل) فاصله بوق WARNING.
        config: پیکربندی صدا.
    """

    def __init__(self, enable_sound: bool = True,
                 beep_interval_seconds: float | None = None,
                 config: AlertConfig | None = None) -> None:
        cfg = config or AlertConfig()
        if beep_interval_seconds is not None:
            from dataclasses import replace
            cfg = replace(cfg, warning_interval=beep_interval_seconds)
        self.config = cfg
        self.enable_sound = enable_sound and cfg.sound_enabled
        self.last_event: AlertEvent | None = None
        self._sound: SoundWorker | None = None
        if self.enable_sound:
            self._sound = SoundWorker(cfg)
            self._sound.start()

    def handle(self, decision: Decision, timestamp: float) -> AlertEvent | None:
        """پردازش تصمیم و صدور اکشن هشدار.

        Args:
            decision: خروجی موتور تصمیم.
            timestamp: زمان فعلی (ثانیه).

        Returns:
            :class:`AlertEvent` برای فریم‌هایی که هشدار فعال دارند یا هشدار
            در آن‌ها پایان یافته؛ وگرنه None.
        """
        level = decision.alert_level
        if self._sound is not None:
            self._sound.update(decision)

        if decision.transition is Transition.CLEARED:
            event = AlertEvent(
                alert_id=decision.alert_id, level=AlertLevel.NONE,
                message="هشدار برطرف شد", state=decision.stable_state.value,
                kind=Transition.CLEARED.value, reason=decision.reason.value,
                duration=decision.alert_seconds,
            )
            self.last_event = None
            return event
        if level not in (AlertLevel.WARNING, AlertLevel.CRITICAL):
            self.last_event = None
            return None
        kind = decision.transition.value or "ONGOING"
        event = AlertEvent(
            alert_id=decision.alert_id,
            level=level,
            message=decision.message,
            state=decision.stable_state.value,
            kind=kind,
            reason=decision.reason.value,
            duration=decision.alert_seconds,
        )
        self.last_event = event
        return event

    def close(self) -> None:
        """توقف Thread صدا."""
        if self._sound is not None:
            self._sound.stop()
            self._sound = None

    # ------------------------------------------------------------------
    def annotate(self, frame: np.ndarray, decision: Decision,
                 angles_text: str = "", info: dict | None = None,
                 bbox: tuple | None = None, timestamp: float = 0.0) -> np.ndarray:
        """رسم HUD وضعیت و هشدار روی فریم (برای نمایش زنده).

        Args:
            frame: تصویر BGR ورودی (بدون تغییر باقی می‌ماند).
            decision: خروجی موتور تصمیم.
            angles_text: متن اختیاری زوایا (مثلاً ``"Y:3 P:-2 R:1"``).
            info: مقادیر اضافی برای نوار پایین (مثلاً ``{"FPS": "29.8"}``).
            bbox: جعبه چهره ``(x, y, w, h)`` برای رسم.
            timestamp: زمان فریم (برای چشمک زدن بنر).

        Returns:
            نسخه حاشیه‌نویسی‌شده فریم.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        level = decision.alert_level
        color = ALERT_COLORS[level]
        alerting = level.is_upper(AlertLevel.INFO)

        if bbox is not None:
            x, y, bw, bh = (int(v) for v in bbox)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)

        # نوار بالا (نیمه‌شفاف)
        bar_h = max(int(h * 0.12), 40)
        _blend_rect(out, (0, 0), (w, bar_h), (20, 20, 20), 0.55)
        scale = max(min(w / 900.0, 1.0), 0.45)
        cv2.putText(out, f"State: {decision.stable_state.value}",
                    (12, int(bar_h * 0.45)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75 * scale + 0.1, color, 2, cv2.LINE_AA)
        if angles_text:
            cv2.putText(out, angles_text, (12, int(bar_h * 0.85)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale + 0.05,
                        (235, 235, 235), 1, cv2.LINE_AA)

        # نوار پیشرفت حواس‌پرتی تا آستانه هشدار
        if decision.threshold_seconds > 0 and decision.distracted_seconds > 0:
            frac = min(decision.distracted_seconds / decision.threshold_seconds, 1.0)
            x0, x1 = int(w * 0.55), w - 12
            y0, y1 = int(bar_h * 0.35), int(bar_h * 0.65)
            cv2.rectangle(out, (x0, y0), (x1, y1), (90, 90, 90), 1)
            cv2.rectangle(out, (x0, y0), (x0 + int((x1 - x0) * frac), y1),
                          color, -1)

        # نوار پایین: FPS/EAR/...
        if info:
            text = "   ".join(f"{k}: {v}" for k, v in info.items())
            _blend_rect(out, (0, h - 28), (w, h), (20, 20, 20), 0.55)
            cv2.putText(out, text, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale + 0.05, (235, 235, 235), 1, cv2.LINE_AA)

        # بنر هشدار و قاب (CRITICAL چشمک می‌زند)
        blink_on = level is not AlertLevel.CRITICAL or int(timestamp * 4) % 2 == 0
        thickness = 8 if alerting else 2
        if alerting and blink_on:
            thickness = 14 if level is AlertLevel.CRITICAL else 8
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)
        if alerting:
            text = _REASON_TEXT.get(decision.reason) or "ALERT"
            if decision.recovery_progress > 0:
                text = f"{text}  ({int(decision.recovery_progress * 100)}%)"
            fs = 1.1 * scale + 0.2
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
            by = int(h * 0.62)
            if blink_on:
                _blend_rect(out, (0, by - th - 18), (w, by + 14), color, 0.65)
            cv2.putText(out, text, (max((w - tw) // 2, 8), by),
                        cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, "ALERT", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 3, cv2.LINE_AA)
        return out


def _blend_rect(img: np.ndarray, p0: tuple, p1: tuple, color: tuple, alpha: float) -> None:
    """رسم مستطیل نیمه‌شفاف درجا."""
    x0, y0 = max(p0[0], 0), max(p0[1], 0)
    x1, y1 = min(p1[0], img.shape[1]), min(p1[1], img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


__all__ = ["ALERT_COLORS", "AlertEvent", "AlertManager", "SoundWorker"]
