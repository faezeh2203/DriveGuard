"""
نقطه ورود برنامه (CLI Entry Point)
===================================

اجرای سیستم تشخیص حواس‌پرتی از خط فرمان::

    python -m dms.main --source 0                # وبکم با پنجره OpenCV
    python -m dms.main --source video.mp4        # فایل ویدیو
    python -m dms.main --source 0 --dashboard    # داشبورد وب Flask
    python -m dms.main --source 0 --config cfg.json

کلید خروج: ``q`` (در پنجره OpenCV) یا ``Ctrl+C``.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2

from .alerting import ALERT_COLORS
from .config import DMSConfig
from .dashboard import Dashboard, SharedState
from .pipeline import DMSPipeline
from .video import VideoSource


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """پارس آرگومان‌های خط فرمان."""
    parser = argparse.ArgumentParser(
        prog="dms",
        description="سیستم تشخیص حواس‌پرتی راننده (Driver Monitoring System)")
    parser.add_argument("--source", default="0",
                        help="منبع ویدیو: شماره وبکم، مسیر فایل یا URL (پیش‌فرض: 0)")
    parser.add_argument("--config", default=None,
                        help="مسیر فایل پیکربندی JSON")
    parser.add_argument("--dashboard", action="store_true",
                        help="فعال‌سازی داشبورد وب Flask")
    parser.add_argument("--port", type=int, default=5000,
                        help="پورت داشبورد (پیش‌فرض: 5000)")
    parser.add_argument("--no-display", action="store_true",
                        help="غیرفعال‌سازی پنجره OpenCV (سرور/headless)")
    parser.add_argument("--no-sound", action="store_true",
                        help="غیرفعال‌سازی هشدار صوتی")
    return parser.parse_args(argv)


def _source_to_capture(source: str):
    """تبدیل رشته منبع به شماره وبکم یا مسیر."""
    if source.isdigit():
        return int(source)
    return source


def main(argv: list[str] | None = None) -> int:
    """اجرای اصلی برنامه؛ کد خروج 0 در پایان موفق."""
    args = parse_args(argv)
    config = DMSConfig.from_json(args.config) if args.config else DMSConfig()
    config = _override_camera_id(config, args.source)

    pipeline = DMSPipeline(config=config, enable_sound=not args.no_sound)
    shared = SharedState()
    stop = threading.Event()

    if args.dashboard:
        dashboard = Dashboard(shared)
        thread = threading.Thread(
            target=dashboard.run, kwargs={"port": args.port}, daemon=True)
        thread.start()
        print(f"[DMS] dashboard: http://127.0.0.1:{args.port}")

    print(f"[DMS] session: {pipeline.session_id} — press 'q' to quit")
    try:
        with VideoSource(_source_to_capture(args.source)) as source:
            for frame in source:
                if stop.is_set():
                    break
                ts_ms = int(frame.timestamp * 1000) if isinstance(
                    args.source, str) else int(frame.frame_id * 1000 / source.fps)
                result = pipeline.process(frame.image, ts_ms)

                angles = result.angles
                annotated = pipeline.alert_manager.annotate(
                    frame.image, result.decision,
                    angles_text=f"Y:{angles.yaw:+.0f} P:{angles.pitch:+.0f} "
                                f"R:{angles.roll:+.0f}")

                ok, jpeg = cv2.imencode(".jpg", annotated)
                shared.publish(result, jpeg.tobytes() if ok else None)

                if not args.no_display:
                    cv2.imshow("DriveGuard - Driver Monitoring System", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("\n[DMS] interrupted")
    finally:
        stop.set()
        if not args.no_display:
            cv2.destroyAllWindows()
        logged = pipeline.store.frame_count()
        pipeline.close()
        print(f"[DMS] frames logged: {logged}")
    return 0


def _override_camera_id(config: DMSConfig, source: str) -> DMSConfig:
    """ثبت شناسه منبع در پیکربندی Telemetry."""
    from dataclasses import replace
    telemetry = replace(config.telemetry, camera_id=f"src-{source}")
    return replace(config, telemetry=telemetry)


if __name__ == "__main__":
    sys.exit(main())
