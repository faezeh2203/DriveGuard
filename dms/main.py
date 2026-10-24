"""
نقطه ورود برنامه (CLI Entry Point)
===================================

اجرای سیستم تشخیص حواس‌پرتی از خط فرمان::

    python -m dms.main --source 0                # وبکم با پنجره OpenCV
    python -m dms.main --source video.mp4        # فایل ویدیو
    python -m dms.main --source 0 --dashboard    # داشبورد وب Flask
    python -m dms.main --source 0 --config cfg.json --export

کلید خروج: ``q`` یا ``Esc`` (در پنجره OpenCV) یا ``Ctrl+C``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from dataclasses import replace
from pathlib import Path

import cv2

from . import __version__
from .config import ConfigError, DMSConfig
from .dashboard import Dashboard, SharedState
from .pipeline import DMSPipeline
from .video import VideoSource

log = logging.getLogger("dms")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """پارس آرگومان‌های خط فرمان."""
    parser = argparse.ArgumentParser(
        prog="dms",
        description="سیستم تشخیص حواس‌پرتی و خواب‌آلودگی راننده (Driver Monitoring System)")
    parser.add_argument("--source", default="0",
                        help="منبع ویدیو: شماره وبکم، مسیر فایل یا URL (پیش‌فرض: 0)")
    parser.add_argument("--config", default=None,
                        help="مسیر فایل پیکربندی JSON")
    parser.add_argument("--dashboard", action="store_true",
                        help="فعال‌سازی داشبورد وب Flask")
    parser.add_argument("--host", default="127.0.0.1",
                        help="آدرس داشبورد (0.0.0.0 برای دسترسی از شبکه؛ پیش‌فرض: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="پورت داشبورد (پیش‌فرض: 5000)")
    parser.add_argument("--no-display", action="store_true",
                        help="غیرفعال‌سازی پنجره OpenCV (سرور/headless)")
    parser.add_argument("--no-sound", action="store_true",
                        help="غیرفعال‌سازی هشدار صوتی")
    parser.add_argument("--export", action="store_true",
                        help="ذخیره گزارش JSON/CSV جلسه در پایان")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="سطح لاگ (پیش‌فرض: INFO)")
    parser.add_argument("--version", action="version", version=f"DriveGuard {__version__}")
    return parser.parse_args(argv)


def _source_to_capture(source: str):
    """تبدیل رشته منبع به شماره وبکم یا مسیر."""
    if source.isdigit():
        return int(source)
    return source


def main(argv: list[str] | None = None) -> int:
    """اجرای اصلی برنامه؛ کد خروج 0 در پایان موفق."""
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        config = DMSConfig.from_json(args.config) if args.config else DMSConfig()
    except (OSError, ValueError, ConfigError) as exc:
        log.error("invalid config: %s", exc)
        return 2
    config = _override_camera_id(config, args.source)

    try:
        source = VideoSource(_source_to_capture(args.source), target_fps=config.target_fps)
    except IOError as exc:
        log.error("%s", exc)
        return 2

    pipeline = DMSPipeline(config=config, enable_sound=not args.no_sound)
    shared = SharedState()
    window = "DriveGuard - Driver Monitoring System"

    if args.dashboard:
        dashboard = Dashboard(shared, store=pipeline.store)
        threading.Thread(target=dashboard.run, name="dms-dashboard",
                         kwargs={"host": args.host, "port": args.port},
                         daemon=True).start()
        log.info("dashboard: http://%s:%d", args.host, args.port)

    log.info("session: %s - press 'q' to quit", pipeline.session_id)
    last_ts_ms = -1
    try:
        with source:
            for frame in source:
                # MediaPipe در حالت VIDEO به timestamp اکیداً صعودی نیاز دارد
                ts_ms = max(int(frame.timestamp * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms
                result = pipeline.process(frame.image, ts_ms)

                a = result.angles
                dr = result.drowsiness
                info = {"FPS": f"{result.fps:.1f}",
                        "EAR": f"{result.attention.ear:.2f}",
                        "PERCLOS": f"{(dr.perclos if dr else 0.0) * 100:.0f}%",
                        "Alerts": str(result.alert_count)}
                annotated = pipeline.alert_manager.annotate(
                    frame.image, result.decision,
                    angles_text=f"Yaw {a.yaw:+.0f}  Pitch {a.pitch:+.0f}  Roll {a.roll:+.0f}",
                    info=info, bbox=result.bbox, timestamp=result.timestamp)

                if args.dashboard:
                    ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    shared.publish(result, jpeg.tobytes() if ok else None)

                if not args.no_display:
                    cv2.imshow(window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        try:
            logged = pipeline.store.frame_count()
            alerts = pipeline.decision_engine.alert_count
            if args.export:
                out_dir = Path(config.telemetry.export_dir)
                j = pipeline.store.export_json(out_dir / f"{pipeline.session_id}.json")
                c = pipeline.store.export_csv(out_dir / f"{pipeline.session_id}.csv")
                log.info("report saved: %s, %s", j, c)
            log.info("frames logged: %d - alerts: %d", logged, alerts)
        finally:
            pipeline.close()
    return 0


def _override_camera_id(config: DMSConfig, source: str) -> DMSConfig:
    """ثبت شناسه منبع در پیکربندی Telemetry."""
    telemetry = replace(config.telemetry, camera_id=f"src-{source}")
    return replace(config, telemetry=telemetry)


if __name__ == "__main__":
    sys.exit(main())
