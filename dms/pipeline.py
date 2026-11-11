"""ارکستراتور پایپ‌لاین (Pipeline)"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .alerting import AlertEvent, AlertManager
from .attention import AttentionClassifier, AttentionResult, AttentionState
from .config import DMSConfig
from .decision import Decision, DecisionEngine
from .drowsiness import DrowsinessDetector, DrowsinessResult
from .face_detector import FaceDetector, FaceObservation
from .geometry import EulerAngle
from .head_pose import HeadPose, HeadPoseEstimator
from .phone_detector import PhoneDetector, phone_near_face
from .smoothing import KalmanFilter1D
from .telemetry import AlertRecord, FrameRecord, TelemetryStore
from .tracker import FaceTracker

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """خروجی کامل پردازش یک فریم، رابط داده‌ای داشبورد و UI."""

    frame_id: int
    timestamp: float
    angles: EulerAngle
    attention: AttentionResult
    drowsiness: Optional[DrowsinessResult]
    decision: Decision
    bbox: Optional[tuple]
    latency_ms: float
    fps: float
    alert_event: Optional[AlertEvent] = None
    face_seen: bool = False
    phone_detected: bool = False
    alert_count: int = 0


class DMSPipeline:
    """پایپ‌لاین کامل تشخیص حواس‌پرتی راننده."""

    def __init__(self, config: DMSConfig | None = None,
                 detector: FaceDetector | None = None,
                 store: TelemetryStore | None = None,
                 enable_sound: bool = False,
                 phone_detector: PhoneDetector | None = None) -> None:
        self.config = config or DMSConfig()
        self.detector = detector
        if self.detector is None:
            self.detector = FaceDetector(
                model_path=self.config.detection.model_path,
                num_faces=self.config.detection.num_faces,
                min_face_detection_confidence=self.config.detection.min_face_detection_confidence,
                min_tracking_confidence=self.config.detection.min_tracking_confidence,
                process_width=self.config.detection.process_width,
            )
        self.store = store or TelemetryStore(
            db_path=self.config.telemetry.db_path,
            log_every_frame=self.config.telemetry.log_every_frame,
        )
        self.tracker = FaceTracker(
            max_lost_frames=self.config.detection.max_lost_frames)
        self.pose_estimator = HeadPoseEstimator()
        self.classifier = AttentionClassifier(self.config.head_pose)
        self.drowsiness_detector = DrowsinessDetector(self.config.drowsiness)
        self.decision_engine = DecisionEngine(self.config.decision)
        self.alert_manager = AlertManager(enable_sound=enable_sound,
                                          config=self.config.alert)

        # --- تشخیص تلفن همراه (اختیاری؛ در نبود مدل غیرفعال می‌شود) ---
        self.phone_detector = phone_detector
        if self.phone_detector is None and self.config.phone.enabled:
            if Path(self.config.phone.model_path).exists():
                try:
                    self.phone_detector = PhoneDetector(
                        model_path=self.config.phone.model_path,
                        score_threshold=self.config.phone.score_threshold,
                        nms_threshold=self.config.phone.nms_threshold,
                        input_size=self.config.phone.input_size,
                        max_results=self.config.phone.max_results,
                        process_width=self.config.phone.process_width,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("phone detection disabled: %s", exc)
            else:
                log.warning("phone detection disabled: model not found (%s); "
                            "run scripts/download_model.py",
                            self.config.phone.model_path)
        self._phones_last: list = []
        self._phone_hold = 0  # فریم‌های باقی‌مانده نگهداری آخرین نتیجه

        # فیلترهای کالمن جداگانه برای سه زاویه (Temporal Smoothing)
        sm = self.config.smoothing
        self._kfs = {
            "pitch": KalmanFilter1D(sm.process_noise, sm.measurement_noise),
            "yaw": KalmanFilter1D(sm.process_noise, sm.measurement_noise),
            "roll": KalmanFilter1D(sm.process_noise, sm.measurement_noise),
        }

        self._frame_counter = 0
        self._last_ts: Optional[float] = None
        self._fps = float(self.config.target_fps)
        self._lost_run = 0  # تعداد فریم‌های متوالی بدون چهره
        self._last_ear: Optional[float] = None
        self._closed = False
        self.session_id = self.store.session_id

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameResult:
        """پردازش کامل یک فریم."""
        t0 = time.perf_counter()
        self._frame_counter += 1
        ts = timestamp_ms / 1000.0
        dt = (ts - self._last_ts) if self._last_ts is not None else 1.0 / self.config.target_fps
        dt = max(min(dt, 1.0), 1e-3)
        self._last_ts = ts
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        observation: FaceObservation | None = None
        if self.detector is not None:
            observation = self.detector.process(frame_bgr, timestamp_ms)
        if observation is not None:
            self._lost_run = 0
        else:
            self._lost_run += 1
        track = self.tracker.update(
            observation.bbox if observation is not None else None, dt)
        # گوشی کنار گوش معمولاً بخشی از صورت را می‌پوشاند، پس از bbox ردیاب هم استفاده می‌کنیم
        face_bbox = None
        if observation is not None:
            face_bbox = observation.bbox
        elif self.tracker.is_tracking:
            face_bbox = track.bbox

        result = self._process_observation(
            observation, self._lost_run, ts, dt, frame_bgr.shape, t0,
            frame_bgr, timestamp_ms, face_bbox)
        return result

    def _detect_phone(self, frame_bgr: np.ndarray | None, timestamp_ms: int,
                      face_bbox: tuple | None) -> bool:
        if self.phone_detector is None or face_bbox is None or frame_bgr is None:
            return False
        # فریم اول بلافاصله بررسی می‌شود و سپس هر detect_every فریم
        if (self._frame_counter - 1) % self.config.phone.detect_every == 0:
            try:
                phones = self.phone_detector.process(frame_bgr, timestamp_ms,
                                                     face_bbox)
            except Exception:  # noqa: BLE001
                phones = self._phones_last  # خطای لحظه‌ای: نتیجه قبلی
            self._phones_last = phones
            self._phone_hold = self.config.phone.hold_frames
            return any(phone_near_face(p, face_bbox,
                                       self.config.phone.near_face_expand)
                       for p in phones)
        if self._phone_hold > 0:
            self._phone_hold -= 1
            return any(phone_near_face(p, face_bbox,
                                       self.config.phone.near_face_expand)
                       for p in self._phones_last)
        return False

    def _process_observation(self, observation, lost_frames: int,
                             ts: float, dt: float, shape,
                             t0: float,
                             frame_bgr: np.ndarray | None = None,
                             timestamp_ms: int = 0,
                             face_bbox: tuple | None = None) -> FrameResult:
        if observation is None and lost_frames >= self.config.detection.max_lost_frames:
            # اگر راننده خوابش برده و سرش از کادر افتاده، هشدار خواب‌آلودگی نباید قطع شود
            angles = EulerAngle(0.0, 0.0, 0.0)
            for kf in self._kfs.values():
                kf.reset()
            self.pose_estimator.reset()
            attention = AttentionResult(
                raw_state=AttentionState.NO_FACE, state=AttentionState.NO_FACE)
            drowsy = self.drowsiness_detector.last_result
        elif observation is None:
            # بدون لندمارک EAR صفر می‌شود و چشم بسته حساب می‌شد؛ نتیجه قبلی را نگه می‌داریم
            angles = self._smoothed_angles()
            attention = self.classifier.classify(angles, None)
            drowsy = self.drowsiness_detector.last_result
            if self._last_ear is not None:
                attention = AttentionResult(
                    raw_state=attention.raw_state, state=attention.state,
                    gaze_offset=attention.gaze_offset, ear=self._last_ear)
        else:
            pose: HeadPose = self.pose_estimator.estimate(
                observation.model_points_2d(), (shape[1], shape[0]))
            if pose.success:
                angles = self._smooth_angles(pose.angles, dt)
            else:
                angles = self._smoothed_angles()
            attention = self.classifier.classify(angles, observation.landmarks)
            drowsy = self.drowsiness_detector.update(
                attention.ear, angles.pitch, ts)
            self._last_ear = attention.ear

        # --- تزریق وضعیت تلفن همراه (گوشی نزدیک چهره) ---
        phone = self._detect_phone(frame_bgr, timestamp_ms, face_bbox)
        if phone:
            attention = AttentionResult(
                raw_state=attention.raw_state,
                state=AttentionState.PHONE,
                gaze_offset=attention.gaze_offset,
                ear=attention.ear,
            )

        decision = self.decision_engine.update(attention, drowsy, ts)
        alert_event = self.alert_manager.handle(decision, ts)
        latency = (time.perf_counter() - t0) * 1000.0

        self._log(observation, angles, attention, drowsy, decision,
                  alert_event, ts, latency)

        return FrameResult(
            frame_id=self._frame_counter,
            timestamp=ts,
            angles=angles,
            attention=attention,
            drowsiness=drowsy,
            decision=decision,
            bbox=observation.bbox if observation is not None else None,
            latency_ms=latency,
            fps=self._fps,
            alert_event=alert_event,
            face_seen=observation is not None,
            phone_detected=phone,
            alert_count=self.decision_engine.alert_count,
        )

    def _smooth_angles(self, raw: EulerAngle, dt: float) -> EulerAngle:
        if not self.config.smoothing.use_kalman:
            return raw
        return EulerAngle(
            pitch=self._kfs["pitch"].step(raw.pitch, dt),
            yaw=self._kfs["yaw"].step(raw.yaw, dt),
            roll=self._kfs["roll"].step(raw.roll, dt),
        )

    def _smoothed_angles(self) -> EulerAngle:
        return EulerAngle(
            pitch=self._kfs["pitch"].value,
            yaw=self._kfs["yaw"].value,
            roll=self._kfs["roll"].value,
        )

    def _log(self, observation, angles, attention, drowsy, decision,
             alert_event, ts: float, latency: float) -> None:
        record = FrameRecord(
            session_id=self.session_id,
            camera_id=self.config.telemetry.camera_id,
            frame_id=self._frame_counter,
            ts=ts,
            fps=self._fps,
            angles=angles,
            state=decision.stable_state.value,
            ear=attention.ear,
            eyes_closed=bool(drowsy and drowsy.eyes_closed),
            drowsy=bool(drowsy and drowsy.drowsy),
            decision=decision,
            latency_ms=latency,
        )
        try:
            self.store.log_frame(record)
        except Exception:  # noqa: BLE001
            # خطای پایگاه‌داده هرگز نباید پردازش زنده را متوقف کند
            log.exception("telemetry: failed to log frame")
        # فقط شروع/ارتقا/پایان هشدار ثبت می‌شود، نه هر فریم
        if alert_event is not None and alert_event.is_transition:
            try:
                self.store.log_alert(AlertRecord(
                    session_id=self.session_id,
                    alert_id=alert_event.alert_id,
                    level=alert_event.level,
                    state=alert_event.state,
                    message=alert_event.message,
                    ts=ts,
                    kind=alert_event.kind,
                    reason=alert_event.reason,
                    duration=alert_event.duration,
                ))
            except Exception:  # noqa: BLE001
                log.exception("telemetry: failed to log alert")

    def close(self) -> None:
        """بستن منابع (تشخیص‌دهنده‌ها، صدا و پایگاه‌داده)؛ تکرار آن بی‌خطر است."""
        if self._closed:
            return
        self._closed = True
        self.alert_manager.close()
        if self.detector is not None:
            self.detector.close()
        if self.phone_detector is not None:
            self.phone_detector.close()
        self.store.close()

    def __enter__(self) -> "DMSPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["DMSPipeline", "FrameResult"]
