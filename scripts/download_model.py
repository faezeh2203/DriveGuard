"""دانلود مدل‌های سیستم (Face Landmarker + YOLO11n ONNX).

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --only face   # فقط مدل چهره
    python scripts/download_model.py --only phone  # فقط مدل گوشی
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# مدل‌ها به همراه منابع جایگزین (به ترتیب تلاش؛ GCS در برخی کشورها
# مسدود است و مسیر جایگزین GitHub/HF استفاده می‌شود)
MODELS = {
    "face": {
        "out": "assets/face_landmarker.task",
        "urls": [
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
        ],
    },
    "phone": {
        "out": "assets/yolo11n.onnx",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/"
            "v8.4.0/yolo11n.onnx",
            "https://huggingface.co/ultralytics/assets/resolve/main/"
            "v8.4.0/yolo11n.onnx",
        ],
    },
}

MIN_VALID_SIZE = 1_000_000  # کمتر از این حجم = پاسخ خطای سرور


def download(urls: list[str], out_path: Path) -> Path:
    """دانلود مدل از اولین آدرس موفق.

    Args:
        urls: فهرست آدرس‌های جایگزین به ترتیب تلاش.
        out_path: مسیر مقصد فایل مدل.

    Returns:
        مسیر فایل دانلودشده.

    Raises:
        RuntimeError: اگر همه آدرس‌ها ناموفق باشند.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = "unknown"
    for url in urls:
        try:
            print(f"downloading -> {out_path}\n  from {url}")
            urllib.request.urlretrieve(url, out_path)
            if out_path.stat().st_size >= MIN_VALID_SIZE:
                size_mb = out_path.stat().st_size / 1e6
                print(f"done: {out_path} ({size_mb:.1f} MB)")
                return out_path
            last_error = "invalid response (too small)"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    raise RuntimeError(
        f"download failed for all mirrors ({last_error}).\n"
        "مدل را دستی از یکی از آدرس‌های زیر دانلود و در مسیر بالا قرار دهید:\n"
        + "\n".join(f"  {u}" for u in urls)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["face", "phone"], default=None,
                        help="دانلود فقط یکی از مدل‌ها (پیش‌فرض: هر دو)")
    args = parser.parse_args(argv)

    targets = [args.only] if args.only else list(MODELS)
    failures = []
    for name in targets:
        spec = MODELS[name]
        out = Path(spec["out"])
        if out.exists() and out.stat().st_size >= MIN_VALID_SIZE:
            print(f"model already exists: {out}")
            continue
        try:
            download(spec["urls"], out)
        except RuntimeError as exc:
            failures.append(f"{name}: {exc}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
