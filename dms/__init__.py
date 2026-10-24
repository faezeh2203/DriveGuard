"""
dms - Driver Monitoring System
==============================

سیستم تشخیص حواس‌پرتی و خواب‌آلودگی راننده بر پایه بینایی ماشین Real-Time.

پایپ‌لاین اصلی:
    Video Capture → Face Detection → Facial Landmark Extraction
    → Head Pose Estimation → Attention Classification → Decision Engine
    → Driver Alerting → Telemetry

Modules
-------
config      تنظیمات سیستم و آستانه‌های تشخیص
geometry    مدل سه‌بعدی صورت و تبدیلات چرخش
video       مدیریت ورودی ویدیو (وبکم، فایل، RTSP)
face_detector تشخیص چهره و استخراج ۴۶۸ لندمارک (MediaPipe Face Mesh)
head_pose   تخمین وضعیت سر (Yaw/Pitch/Roll) با SolvePnP
smoothing   فیلتر کالمن، میانگین متحرک و هیسترزیس
tracker     ردیابی چهره (تک‌هدف، الهام‌گرفته از SORT)
attention   طبقه‌بندی وضعیت توجه راننده
phone_detector شناسایی تلفن همراه نزدیک چهره (تماس دستی)
drowsiness  تشخیص خواب‌آلودگی با EAR
decision    موتور تصمیم و تولید هشدار
alerting    مدیریت هشدار (تصویری/صوتی)
telemetry   ثبت داده در SQLite و خروجی JSON/CSV و محاسبه متریک‌ها
pipeline    ارکستراتور پایپ‌لاین (پردازش هر فریم)
dashboard   داشبورد وب Flask
main        نقطه ورود برنامه (CLI)
"""

__version__ = "2.1.0"
__author__ = "Rooufi - Esmaeili (4042)"
