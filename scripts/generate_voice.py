"""
ساخت فایل‌های صوتی هشدار گفتاری (فارسی)
=========================================

با موتور Microsoft Edge TTS (صدای عصبی فارسی) جمله‌های هشدار ساخته،
بلندی‌شان نرمال‌سازی/تقویت می‌شود و به صورت WAV در ``assets/voice/``
ذخیره می‌شوند. فقط یک بار (با اینترنت) اجرا می‌شود؛ پس از آن سیستم
بدون اینترنت کار می‌کند.

    python scripts/generate_voice.py                   # صدای زن (Dilara)
    python scripts/generate_voice.py --voice fa-IR-FaridNeural   # صدای مرد
    python scripts/generate_voice.py --force           # بازسازی همه فایل‌ها

پیش‌نیاز: ``pip install edge-tts miniaudio``
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "voice"
SAMPLE_RATE = 24000

#: جمله هر علت هشدار (کلید = نام فایل = AlertReason / رخداد)
PHRASES = {
    "LOOK_DOWN": "لطفاً به جلو و جاده نگاه کنید!",
    "LOOK_UP": "لطفاً به جلو و جاده نگاه کنید!",
    "LATERAL": "حواستان به جلو باشد. به جاده نگاه کنید!",
    "NO_FACE": "راننده دیده نمی‌شود. لطفاً به جلو نگاه کنید!",
    "PHONE": "تلفن همراه را کنار بگذارید و به جاده نگاه کنید!",
    "DROWSY": "خطر! شما خواب‌آلود هستید. لطفاً توقف کنید و استراحت کنید!",
    "ESCALATED": "هشدار جدی! همین حالا به جاده نگاه کنید!",
    "CLEARED": "متشکرم. با احتیاط رانندگی کنید.",
}


async def _synthesize(text: str, voice: str, rate: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume="+0%")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _decode_mp3(data: bytes) -> np.ndarray:
    import miniaudio

    decoded = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32,
                               nchannels=1, sample_rate=SAMPLE_RATE)
    return np.frombuffer(decoded.samples, dtype=np.float32).copy()


def make_loud(samples: np.ndarray, drive: float = 2.2) -> np.ndarray:
    """نرمال‌سازی و فشرده‌سازی نرم (soft-limit) برای بلندی بیشتر بدون اعوجاج شدید."""
    peak = float(np.max(np.abs(samples))) or 1.0
    x = samples / peak
    # حذف سکوت ابتدا و انتها (هشدار باید بی‌درنگ شنیده شود)
    loud = np.flatnonzero(np.abs(x) > 0.02)
    if loud.size:
        margin = int(0.05 * SAMPLE_RATE)
        x = x[max(loud[0] - margin, 0):loud[-1] + margin]
    x = np.tanh(x * drive) / np.tanh(drive)
    return (x * 0.97).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ساخت هشدارهای گفتاری فارسی")
    parser.add_argument("--voice", default="fa-IR-DilaraNeural",
                        help="نام صدا (fa-IR-DilaraNeural یا fa-IR-FaridNeural)")
    parser.add_argument("--rate", default="+5%", help="سرعت گفتار (مثلاً +10%%)")
    parser.add_argument("--force", action="store_true", help="بازسازی فایل‌های موجود")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for key, text in PHRASES.items():
        out = OUT_DIR / f"{key}.wav"
        if out.exists() and not args.force:
            print(f"exists: {out.name}")
            continue
        try:
            mp3 = asyncio.run(_synthesize(text, args.voice, args.rate))
            samples = make_loud(_decode_mp3(mp3))
            write_wav(out, samples)
            print(f"saved: {out.name} ({len(samples) / SAMPLE_RATE:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{key}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
