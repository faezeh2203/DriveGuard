# DriveGuard 🚗

**سیستم هوشمند تشخیص حواس‌پرتی و خواب‌آلودگی راننده بر پایه بینایی ماشین (Real-Time DMS)**

پیاده‌سازی کامل و Real-Time سیستم تشخیص حواس‌پرتی و خواب‌آلودگی راننده
بر پایه بینایی ماشین — پروژه درس بینایی ماشین، نیم‌سال ۱۴۰۴.

سیستم از تصویر دوربین، وضعیت سر راننده را تخمین می‌زند (Yaw/Pitch/Roll)،
وضعیت توجه او را طبقه‌بندی می‌کند (رو به جلو، چپ، راست، پایین/موبایل،
بدون چهره)، خواب‌آلودگی را از روی بسته بودن چشم‌ها تشخیص می‌دهد و در
صورت حواس‌پرتی پایدار **هشدار** صادر می‌کند؛ هشدار تا زمانی که راننده رفتارش را اصلاح نکند ادامه دارد. تمام رخدادها در SQLite ثبت
و به صورت JSON/CSV قابل خروجی‌گیری هستند.

---

## ویژگی‌ها

| قابلیت | شرح |
|---|---|
| تشخیص چهره و لندمارک | MediaPipe Face Mesh با ۴۶۸ لندمارک + ۱۰ نقطه عنبیه (BlazeFace داخلی) |
| **شناسایی تلفن همراه** | YOLO11n ONNX (ONNX Runtime): گوشی نزدیک چهره = تماس دستی → وضعیت **PHONE** و هشدار |
| تخمین وضعیت سر | روش کلاسیک **SolvePnP** با نگاشت 2D-3D (شش نقطه استاندارد) |
| خروجی زاویه‌ای | Pitch / Yaw / Roll بر حسب درجه، هموارشده با **فیلتر کالمن** |
| طبقه‌بندی توجه | Forward / Left / Right / Down (موبایل) / Phone / Up / No-face |
| ترکیب جهت نگاه | تخمین Gaze از عنبیه در حالت‌های مرزی |
| هیسترزیس | تغییر وضعیت تنها پس از N فریم متوالی (کاهش False Alarm) |
| تشخیص خواب‌آلودگی | Eye Aspect Ratio (EAR) + الگوی سر تکان دادن (Head Nodding) |
| موتور تصمیم | تایمر پیوسته حواس‌پرتی، آستانه جدا برای موبایل (۱s)، نگاه پایین/چهره ناپیدا (۲s) و نگاه طولانی به اطراف (۵s) |
| **هشدار ماندگار** | هشدار تا اصلاح رفتار راننده (۱ ثانیه نگاه پیوسته به جلو / ۱.۵ ثانیه چشم باز) ادامه دارد و پس از ۶ ثانیه به CRITICAL ارتقا می‌یابد |
| هشدار | HUD تصویری (بنر چشمک‌زن، نوار پیشرفت) و بوق تکرارشونده غیرمسدودکننده در Thread جدا |
| خواب‌آلودگی پیشرفته | EAR + Head Nodding + **PERCLOS** (درصد زمان بسته بودن چشم) |
| Telemetry | ثبت هر فریم و رخدادها در SQLite، خروجی JSON/CSV |
| داشبورد وب | Flask: وضعیت زنده، نمودار تاریخچه، جریان تصویر MJPEG |
| متریک‌های ارزیابی | Accuracy، F1-score، False Positive Rate، Latency |

## معماری پایپ‌لاین

```
Video Capture → Face Detection → Landmark Extraction (468)
    → Head Pose Estimation (SolvePnP) → Temporal Smoothing (Kalman)
    → Attention Classification ⇄ Phone Detection (YOLO11n ONNX)
    → Drowsiness Detection
    → Decision Engine (Hysteresis) → Driver Alerting → Telemetry (SQLite)
```

نمودار کامل جریان و نمودار کلاس‌ها: [docs/architecture.md](docs/architecture.md)

## نصب و راه‌اندازی

پیش‌نیاز: Python 3.10 یا بالاتر.

```bash
# ۱) نصب وابستگی‌ها
pip install -r requirements.txt

# ۲) دانلود مدل‌ها (Face Landmarker + YOLO11n برای تشخیص گوشی)
python scripts/download_model.py

# ۳) اجرا با وبکم (پنجره OpenCV)
python -m dms.main --source 0

# اجرا با داشبورد وب
python -m dms.main --source 0 --dashboard --port 5000

# اجرا روی فایل ویدیو بدون نمایشگر (سرور/headless)
python -m dms.main --source video.mp4 --no-display --no-sound
```

سپس داشبورد در آدرس <http://127.0.0.1:5000> در دسترس است.
برای خروج از برنامه کلید `q` یا `Ctrl+C`.

## تست‌ها

```bash
# اجرای کل مجموعه (۲۲۵ تست)
python -m pytest tests/ -v

# با گزارش پوشش
python -m pytest tests/ --cov=dms --cov-report=term-missing
```

تست‌ها بدون وابستگی به وبکم یا مدل سنگین اجرا می‌شوند؛ چهره‌های آزمون
با پروجکشن پرسپکتیو مدل سه‌بعدی و زاویه معلوم (Ground Truth دقیق)
ساخته می‌شوند.

## ارزیابی

```bash
# ارزیابی مصنوعی: دقت بازیابی زاویه + دقت طبقه‌بندی روی ۳۰۰ نمونه
python scripts/synthetic_eval.py --n 300

# ارزیابی روی ویدیوی برچسب‌خورده (نیاز به labels.csv)
python scripts/evaluate.py --video clip.mp4 --labels labels.csv
```

نتیجه ارزیابی مصنوعی مرجع (300 نمونه): **Accuracy = 1.00،
F1 = 1.00، FPR = 0.00، میانگین خطای زاویه = 0.000°**
(گزارش کامل: `telemetry/synthetic_report.json` و [docs/report.md](docs/report.md)).

## ساختار پروژه

```
├── dms/                    پکیج اصلی سیستم
│   ├── config.py           پیکربندی و آستانه‌ها (JSON قابل تنظیم)
│   ├── geometry.py         مدل سه‌بعدی صورت، تبدیل چرخش، EAR
│   ├── video.py            منبع ویدیو (وبکم/فایل/RTSP)
│   ├── face_detector.py    تشخیص چهره + ۴۶۸ لندمارک (MediaPipe)
│   ├── head_pose.py        تخمین وضعیت سر (SolvePnP)
│   ├── smoothing.py        کالمن، میانگین متحرک، هیسترزیس
│   ├── tracker.py          ردیابی تک‌هدف چهره (الهام از SORT)
│   ├── attention.py        طبقه‌بندی وضعیت توجه
│   ├── phone_detector.py   شناسایی تلفن همراه (YOLO11n ONNX)
│   ├── drowsiness.py       تشخیص خواب‌آلودگی (EAR + Nodding)
│   ├── decision.py         موتور تصمیم (هیسترزیس + هشدار ماندگار + ارتقا)
│   ├── alerting.py         هشدار تصویری (HUD) / صوتی (Thread پس‌زمینه)
│   ├── templates/, static/ رابط وب داشبورد (HTML/CSS/JS، بدون نیاز به اینترنت)
│   ├── telemetry.py        SQLite + خروجی JSON/CSV + متریک‌ها
│   ├── pipeline.py         ارکستراتور پایپ‌لاین
│   ├── dashboard.py        داشبورد وب Flask
│   └── main.py             نقطه ورود CLI
├── tests/                  ۲۲۵ تست pytest
├── scripts/                دانلود مدل، ارزیابی مصنوعی و ویدیویی
├── docs/                   مستندات فنی و گزارش پروژه
├── assets/                 مدل face_landmarker.task
└── telemetry/              خروجی‌های ثبت‌شده (ساخته می‌شود)
```

## پیکربندی

آستانه‌ها در `dms/config.py` مستند شده‌اند و می‌توان آن‌ها را با فایل
JSON بازنویسی کرد (مقادیر مرجع مشخصات پروژه: `yaw > 25°`،
`pitch < -15°`):

```json
{
  "head_pose": {"yaw_threshold": 25.0, "pitch_down_threshold": -15.0},
  "decision": {"state_confirm_frames": 5, "distraction_alert_seconds": 2.0}
}
```

```bash
python -m dms.main --source 0 --config my_config.json
```

## مستندات

- [معماری و نمودارها (Flowchart + UML)](docs/architecture.md)
- [مرور پیشینه پژوهشی](docs/literature_review.md)
- [راهنمای استفاده و API](docs/usage.md)
- [گزارش کامل پروژه](docs/report.md)

## منابع اصلی

- Ruiz, Chong & Rehg, *"HopeNet: A CNN for Head Pose Estimation"*, CVPRW 2018.
- Yang et al., *"FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation"*, CVPR 2019.
- Yu et al., *"6DRepNet: Latent Space Nonlinear Regression for Head Pose Estimation"*, CVPR 2023.
- Soukupová & Čech, *"Real-Time Eye Blink Detection using Facial Landmarks"*, 2016.
- Zhang et al., *"Facial Landmark Detection via Deep Multi-task Learning"*, 300W-LP / AFLW2000.


## تغییرات نسخه ۲

جزئیات کامل در [CHANGELOG.md](CHANGELOG.md).
