# راهنمای استفاده و مرجع API

## ۱. اجرای برنامه

### وبکم با پنجره OpenCV
```bash
python -m dms.main --source 0
```
پنجره‌ای با قاب رنگی وضعیت نمایش داده می‌شود:
- خاکستری: عادی (رو به جلو)
- سبز: وضعیت حواس‌پرتی پایدار اما زیر آستانه هشدار (INFO)
- نارنجی: هشدار حواس‌پرتی (WARNING)
- قرمز: هشدار خواب‌آلودگی (CRITICAL)

### داشبورد وب
```bash
python -m dms.main --source 0 --dashboard --port 5000
```
- صفحه اصلی: وضعیت توجه، زوایا، FPS و جریان زنده MJPEG
- `GET /api/status` — وضعیت لحظه‌ای JSON
- `GET /api/history` — ۱۰۰ نمونه آخر (برای نمودار)
- `GET /video_feed` — جریان تصویر حاشیه‌نویسی‌شده

### فایل ویدیو / سرور headless
```bash
python -m dms.main --source video.mp4 --no-display --no-sound
python -m dms.main --source rtsp://camera/stream --no-display
```

### پارامترهای CLI

| پارامتر | شرح |
|---|---|
| `--source` | شماره وبکم / مسیر فایل / URL (پیش‌فرض `0`) |
| `--config` | فایل پیکربندی JSON |
| `--dashboard` | فعال‌سازی داشبورد Flask |
| `--port` | پورت داشبورد (پیش‌فرض 5000) |
| `--no-display` | بدون پنجره OpenCV |
| `--no-sound` | بدون هشدار صوتی |

## ۲. استفاده به‌عنوان کتابخانه

```python
import cv2
from dms.pipeline import DMSPipeline
from dms.config import DMSConfig

config = DMSConfig()  # یا DMSConfig.from_json("cfg.json")
pipeline = DMSPipeline(config=config, enable_sound=False)

cap = cv2.VideoCapture(0)
ts_ms = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ts_ms += 33  # timestamp صعودی اجباری MediaPipe
    result = pipeline.process(frame, ts_ms)

    print(result.decision.stable_state.value)   # FORWARD / LEFT / ...
    print(result.angles.yaw, result.angles.pitch, result.angles.roll)
    print(result.decision.alert_level.value)    # NONE / INFO / WARNING / CRITICAL
    print(result.fps, result.latency_ms)

pipeline.close()
```

### خروجی JSON/CSV جلسه

```python
from dms.telemetry import TelemetryStore

store = TelemetryStore("telemetry/dms_log.db", session_id="run-1")
# ... اتصال به pipeline (پارامتر store در DMSPipeline)
store.export_json("telemetry/session.json")
store.export_csv("telemetry/session.csv")
print(store.state_summary())   # {"FORWARD": 812, "LEFT": 40, ...}
```

### متریک‌های ارزیابی

```python
from dms.telemetry import evaluate

metrics = evaluate(y_true=["FORWARD", "LEFT", "FORWARD"],
                   y_pred=["FORWARD", "LEFT", "DOWN"])
# {"accuracy": 0.67, "f1_score": 0.8, "false_positive_rate": 0.0, ...}
```

## ۳. مرجع ماژول‌ها

### `dms.config`
- `DMSConfig` — پیکربندی سراسری (dataclass تو در تو).
- `DMSConfig.from_json(path)` — بارگذاری با بازنویسی جزئی کلیدها.
- `default_camera_matrix((w, h))` — ماتریس دوربین تقریبی (FOV≈60°).
- آستانه‌های مرجع: `yaw_threshold=25°`، `pitch_down_threshold=-15°`.
- `PhoneConfig` — پیکربندی ماژول گوشی (مدل، آستانه‌ها، `detect_every`،
  `near_face_expand`)؛ نمونه JSON: `{"phone": {"enabled": true,
  "score_threshold": 0.4, "detect_every": 15}}`.

### `dms.geometry`
- `GENERIC_3D_FACE_MODEL` — مدل سه‌بعدی شش‌نقطه‌ای صورت.
- `FACE_MODEL_LANDMARK_IDS = (1, 199, 33, 263, 61, 291)` — تناظر لندمارک.
- `rotation_vector_to_euler(rvec) → EulerAngle` — تبدیل بردار چرخش.
- `euler_to_rotation_vector(pitch, yaw, roll)` — عکس تبدیل (تست/شبیه‌سازی).
- `project_face_model(pitch, yaw, roll, ...)` — ساخت داده آزمون.
- `eye_aspect_ratio(eye_points)` — EAR شش نقطه چشم.

### `dms.video`
- `VideoSource(source)` — پوشش WebCam/File/RTSP با `Frame` متادیتادار؛
  پشتیبانی از `with` و iteration.

### `dms.face_detector`
- `FaceDetector(model_path, num_faces, ...)` — MediaPipe FaceLandmarker
  در حالت VIDEO (timestamp صعودی اجباری).
- `process(frame_bgr, timestamp_ms) → FaceObservation | None`.
- `FaceObservation.model_points_2d()` — شش نقطه ورودی SolvePnP.

### `dms.head_pose`
- `HeadPoseEstimator(camera_matrix=None, use_extrinsic_guess=True)`.
- `estimate(model_points_2d, image_size) → HeadPose` شامل
  زوایا، خطای بازتاب و پرچم موفقیت.

### `dms.smoothing`
- `KalmanFilter1D(process_noise, measurement_noise)` — مدل
  `[position, velocity]`، روش پیش‌بینی/به‌روزرسانی استاندارد.
- `MovingAverage(window)` — میانگین متحرک پنجره‌ای.
- `HysteresisFilter(min_frames, initial)` — تأیید وضعیت پس از N فریم.

### `dms.tracker`
- `FaceTracker(max_lost_frames)` — ردیاب تک‌هدف؛ نگهداری هدف در
  گم شدن کوتاه، ریست در گم شدن طولانی.

### `dms.attention`
- `AttentionState` — هفت حالت توجه (شامل `PHONE`) + `is_distracted` + برچسب فارسی.
- `AttentionClassifier(config, gaze_gain)` — طبقه‌بندی آستانه‌ای
  با ترکیب Gaze عنبیه و محاسبه EAR.

### `dms.phone_detector`
- `PhoneDetector(model_path, score_threshold, nms_threshold, input_size, ...)` —
  تشخیص تلفن همراه با YOLO11n ONNX (ONNX Runtime، CPU)؛
  `process(frame_bgr, timestamp_ms, face_bbox=None)` → لیست جعبه‌های
  گوشی `[(x, y, w, h), ...]` در مختصات تصویر اصلی.
- `phone_near_face(phone_bbox, face_bbox, expand)` — قاعده
  «گوشی نزدیک چهره» (تماس دستی)؛ گوشی روی پایه دور از چهره رد می‌شود.
- اگر فایل مدل موجود نباشد، پایپ‌لاین با پیام هشدار و بدون ماژول
  گوشی بالا می‌آید (بقیه قابلیت‌ها سالم می‌مانند).

### `dms.drowsiness`
- `DrowsinessDetector(config)` — EAR + پنجره زمانی + Head Nodding؛
  خروجی `DrowsinessResult(eyes_closed, drowsy, nodding, ...)`.

### `dms.decision`
- `AlertLevel` — NONE / INFO / WARNING / CRITICAL با مقایسه شدت.
- `DecisionEngine(config)` — هیسترزیس + آستانه زمانی حواس‌پرتی +
  Cooldown + ثبت دوره‌های حواس‌پرتی (`episodes`)؛ **سیاست پیش‌فرض:**
  چرخش سر چپ/راست (آینه/پیچ) فقط INFO — هشدار برای DOWN، PHONE و
  NO_FACE؛ گزینه `lateral_alert_enabled=True` حالت سخت‌گیرانه.

### `dms.alerting`
- `AlertManager(enable_sound)` — Beep (winsound/کنسول) و
  `annotate(frame, decision)` برای رسم قاب وضعیت.

### `dms.telemetry`
- `TelemetryStore(db_path, session_id, log_every_frame)` — ثبت
  فریم‌ها و هشدارها؛ `export_json/export_csv/state_summary`؛
  commit دسته‌ای برای حفظ Real-Time.
- توابع متریک: `accuracy`, `f1_score`, `false_positive_rate`,
  `evaluate`, `confusion_counts`.

### `dms.pipeline`
- `DMSPipeline(config, detector, store, enable_sound)` — ارکستراتور؛
  `process(frame_bgr, timestamp_ms) → FrameResult` شامل تمام
  خروجی‌های زنجیره؛ پشتیبانی از `with`.

### `dms.dashboard`
- `SharedState` — بافر اشتراکی امن بین Threadها.
- `Dashboard(shared)` — Flask: `/`, `/api/status`, `/api/history`,
  `/video_feed`.

## ۴. پوشش تست

| فایل تست | حوزه | تعداد |
|---|---|---|
| `test_geometry.py` | تبدیل چرخش، مدل صورت، EAR | 18 |
| `test_head_pose.py` | بازیابی زاویه، نویز، حدس اولیه | 13 |
| `test_smoothing.py` | کالمن، میانگین متحرک، هیسترزیس | 22 |
| `test_attention.py` | آستانه‌ها، Gaze، EAR | 15 |
| `test_drowsiness.py` | EAR، پلک زدن، سر تکان | 13 |
| `test_decision.py` | هیسترزیس، هشدار، Cooldown، دوره‌ها | 15 |
| `test_tracker.py` | ردیابی، گم شدن | 7 |
| `test_telemetry.py` | SQLite، JSON/CSV، متریک‌ها، پیکربندی | 20 |
| `test_pipeline.py` | انتگرال، Telemetry، کارایی، نویز | 11 |
| `test_phone.py` | قاعده نزدیکی، وضعیت PHONE، هشدار، پیکربندی، دود واقعی | 17 |
| `test_video_alerting.py` | منبع ویدیو، انوتیشن | 12 |
| `test_face_detector.py` | دود MediaPipe واقعی | 4 |
| `test_dashboard_cli.py` | داشبورد Flask، SharedState، CLI | 11 |
| **جمع** | | **184** |

اجرای سری کامل (بدون وبکم، بدون GPU) با پوشش کد ≈ ۹۱٪:

```bash
python -m pytest tests/ -v --cov=dms
```
