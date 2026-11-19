# -*- coding: utf-8 -*-
"""
شمارش انگشت با نمایش سوپراسپرت‌ها  |  Supercar Hand Counter  (v1.1.0)
----------------------------------------------------------------------
دستت را جلوی وبکم نشان بده؛ به تعداد انگشت‌های بالا آمده (حتی با دو دست
تا ۱۰)، یک سوپراسپرت واقعی در پنل کنار تصویر دوربین ظاهر می‌شود.

امکانات:
- دو دست همزمان تا ۱۰ انگشت و ۱۰ ماشین
- ماشین مخفی برای حالت مشت (ایستر اگ)
- ذخیره عکس لحظه با کلید S
- افکت محو شدن هنگام تعویض ماشین‌ها
- اسکلت دست، FPS زنده و رابط فارسی

اجرا:  python supercar_counter.py
راهنما: python supercar_counter.py --help
"""

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ------------------------------------------------------------------
# مسیرها و تنظیمات
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
MODEL_PATH = ASSETS / "models" / "hand_landmarker.task"
FONT_DIR = ASSETS / "fonts"
CARS_DIR = ASSETS / "cars"
CAPTURES_DIR = BASE_DIR / "captures"

WINDOW_W, WINDOW_H = 1280, 720
VIDEO_W, VIDEO_H = 960, 720
PANEL_W = WINDOW_W - VIDEO_W          # 320

PH_W, PH_H = 280, 186                 # اندازه عکس ماشین داخل کارت
CARD_MARGIN = 10                      # حاشیه سایه دور کارت ماشین
FADE_SECONDS = 0.22                   # مدت افکت محو شدن تعویض ماشین
TOAST_SECONDS = 1.5                   # مدت نمایش پیام «ذخیره شد»
MAX_FINGERS = 10                      # سقف شمارش (دو دست)

# رنگ‌ها (RGB)
COLOR_TEXT_DARK = (17, 24, 39)        # #111827
COLOR_TEXT_GRAY = (107, 114, 128)     # #6B7280
COLOR_TEXT_LIGHT = (156, 163, 175)    # #9CA3AF
COLOR_BORDER = (229, 231, 235)        # #E5E7EB
COLOR_PH_BG = (243, 244, 246)         # #F3F4F6
COLOR_BADGE_BG = (249, 250, 251)      # #F9FAFB

# نوک انگشت‌ها روی تصویر (BGR: نارنجی)
COLOR_ACCENT_BGR = (11, 158, 245)

# اتصالات استاندارد اسکلت دست (۲۱ نقطه)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # شست
    (0, 5), (5, 6), (6, 7), (7, 8),           # اشاره
    (5, 9), (9, 10), (10, 11), (11, 12),      # میانی
    (9, 13), (13, 14), (14, 15), (15, 16),    # حلقه
    (13, 17), (17, 18), (18, 19), (19, 20),   # کوچک
    (0, 17),                                   # مچ تا پایه انگشت کوچک
]

# ماشین هر تعداد انگشت (۱ تا ۱۰)
CARS = {
    1: dict(file="car_1.jpg", name="لامبورگینی آونتادور", sub="رنگ: زرد"),
    2: dict(file="car_2.jpg", name="فراری", sub="رنگ: قرمز"),
    3: dict(file="car_3.jpg", name="بوگاتی شیرون", sub="رنگ: آبی"),
    4: dict(file="car_4.jpg", name="مک‌لارن 720S", sub="رنگ: نارنجی"),
    5: dict(file="car_5.jpg", name="پورشه 911 کررا", sub="رنگ: سفید"),
    6: dict(file="car_6.jpg", name="آستون مارتین", sub="رنگ: سبز"),
    7: dict(file="car_7.jpg", name="آئودی R8", sub="رنگ: خاکستری"),
    8: dict(file="car_8.jpg", name="مرسدس AMG GT", sub="رنگ: نقره‌ای"),
    9: dict(file="car_9.jpg", name="نیسان GT-R", sub="رنگ: آبی"),
    10: dict(file="car_10.jpg", name="کونینگ‌سیگ", sub="رنگ: مشکی"),
}

# ماشین مخفی حالت مشت (ایستر اگ)
SECRET_CAR = dict(file="secret.jpg", name="لامبورگینی وننو", sub="ایستر اگ مخفی")

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_RTL = True
except ImportError:
    HAS_RTL = False

try:
    import mediapipe as mp
except ImportError:
    mp = None

# اگر موتور raqm موجود باشد، شکل‌دهی و راست‌به‌چپ خودِ Pillow آن را انجام میدهد
try:
    from PIL import features
    USE_RAQM = features.check("raqm")
except Exception:
    USE_RAQM = False

FONT_LAYOUT = (ImageFont.Layout.RAQM if USE_RAQM else ImageFont.Layout.BASIC)


def shape_text(text):
    """آماده‌سازی متن فارسی برای نمایش درست (حروف چسبیده و راست‌به‌چپ).

    با raqm متن خام داده می‌شود (شکل‌دهی داخلی)؛ در غیر این صورت از
    arabic-reshaper + python-bidi استفاده می‌شود.
    """
    if USE_RAQM:
        return text
    if HAS_RTL:
        return get_display(arabic_reshaper.reshape(text))
    return text


def fa_digits(value):
    """تبدیل ارقام لاتین به ارقام فارسی."""
    return str(value).translate(FA_DIGITS)


# ------------------------------------------------------------------
# الگوریتم شمارش انگشت
# ------------------------------------------------------------------
def _angle(a, b, c):
    """زاویه بین بردارهای b->a و b->c بر حسب درجه."""
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    dot = v1x * v2x + v1y * v2y
    norm = math.hypot(v1x, v1y) * math.hypot(v2x, v2y)
    if norm == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / norm))
    return math.degrees(math.acos(cosang))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def count_fingers(lm):
    """تعداد انگشت‌های بالا آمده را از ۲۱ نقطه دست حساب می‌کند (۰ تا ۵).

    چهار انگشت اشاره تا کوچک: اگر زاویه مفصل میانی نزدیک ۱۸۰ درجه باشد
    یعنی انگشت صاف و بالا است.
    شست: اگر نوک شست از مفصل‌های پشت آن دورتر باشد، یعنی باز شده است.
    هر دو روش نسبت به چرخش دست مقاوم هستند.
    """
    count = 0
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        if _angle(lm[mcp], lm[pip], lm[tip]) > 160.0:
            count += 1
    if _dist(lm[4], lm[17]) > max(_dist(lm[3], lm[17]), _dist(lm[2], lm[17])):
        count += 1
    return count


class StableState:
    """پایدارساز وضعیت (وجود دست، تعداد انگشت).

    فقط حالتی که چند فریم پشت سر هم تکرار شود پذیرفته می‌شود تا
    ماشین‌ها و عدد بدون لرزش تغییر کنند.
    """

    def __init__(self, need=3):
        self.need = need
        self.last_raw = None
        self.streak = 0
        self.value = (False, 0)

    def update(self, raw):
        if raw == self.last_raw:
            self.streak += 1
        else:
            self.last_raw = raw
            self.streak = 1
        if self.streak >= self.need and raw != self.value:
            self.value = raw
        return self.value


# ------------------------------------------------------------------
# تشخیص دست با MediaPipe Tasks API
# ------------------------------------------------------------------
class HandDetector:
    def __init__(self, model_path, running_mode="VIDEO", num_hands=2):
        if mp is None:
            raise ImportError("mediapipe نصب نیست؛ دستور «pip install -r requirements.txt» را اجرا کنید.")
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        mode = vision.RunningMode.VIDEO if running_mode == "VIDEO" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._mode = running_mode

    def detect(self, frame_bgr, ts_ms=0):
        """فریم BGR را می‌گیرد و فهرست دست‌ها را برمی‌گرداند.
        هر دست: فهرست ۲۱ نقطه با مختصات پیکسلی."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        if self._mode == "VIDEO":
            result = self._detector.detect_for_video(image, int(ts_ms))
        else:
            result = self._detector.detect(image)

        h, w = frame_bgr.shape[:2]
        hands = []
        if result.hand_landmarks:
            for lms in result.hand_landmarks:
                hands.append([(p.x * w, p.y * h) for p in lms])
        return hands


# ------------------------------------------------------------------
# ابزارهای تصویری
# ------------------------------------------------------------------
def cover_crop_bgr(img, tw, th):
    """تغییر اندازه با حفظ نسبت و برش از مرکز تا دقیقا tw در th شود."""
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = max(tw, int(round(w * scale))), max(th, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    x = (nw - tw) // 2
    y = (nh - th) // 2
    return img[y:y + th, x:x + tw].copy()


class Fonts:
    def __init__(self):
        self._cache = {}
        self.regular = FONT_DIR / "Vazirmatn-Regular.ttf"
        self.bold = FONT_DIR / "Vazirmatn-Bold.ttf"

    def get(self, size, bold=False):
        key = (size, bold)
        if key not in self._cache:
            path = self.bold if bold else self.regular
            self._cache[key] = ImageFont.truetype(str(path), size, layout_engine=FONT_LAYOUT)
        return self._cache[key]


def rounded_mask(size, radius):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255
    )
    return mask


def make_car_card(photo_bgr, w, h, margin):
    """کارت ماشین با گوشه گرد، حاشیه روشن و سایه ملایم (سبک مینیمال)."""
    tw, th = w + 2 * margin, h + 2 * margin
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    # سایه ملایم
    shadow = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [margin - 2, margin + 3, margin + w + 2, margin + h + 5],
        radius=13, fill=(0, 0, 0, 55),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    canvas.alpha_composite(shadow)

    # عکس با گوشه گرد
    photo = Image.fromarray(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB))
    canvas.paste(photo, (margin, margin), rounded_mask((w, h), 12))

    # حاشیه ظریف
    ImageDraw.Draw(canvas).rounded_rectangle(
        [margin, margin, margin + w - 1, margin + h - 1],
        radius=12, outline=COLOR_BORDER + (255,), width=1,
    )
    return canvas


def make_placeholder_card(w, h, margin, fonts):
    """کارت خالی وقتی دستی جلوی دوربین نیست."""
    tw, th = w + 2 * margin, h + 2 * margin
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle(
        [margin, margin, margin + w - 1, margin + h - 1],
        radius=12, fill=COLOR_PH_BG + (255,), outline=COLOR_BORDER + (255,), width=1,
    )
    t1 = shape_text("دستت را نشان بده")
    t2 = shape_text("تا ماشین‌ها ظاهر شوند")
    d.text((tw // 2, th // 2 - 12), t1, font=fonts.get(17), fill=COLOR_TEXT_LIGHT + (255,), anchor="mm")
    d.text((tw // 2, th // 2 + 16), t2, font=fonts.get(13), fill=COLOR_TEXT_LIGHT + (255,), anchor="mm")
    return canvas


# ------------------------------------------------------------------
# پنل گالری (سمت چپ تصویر دوربین) با افکت محو شدن
# ------------------------------------------------------------------
class GalleryPanel:
    def __init__(self, fonts, width, height):
        self.fonts = fonts
        self.w, self.h = width, height

        self.cards = {}
        for k, v in CARS.items():
            self.cards[(True, k)] = (make_car_card(v["photo"], PH_W, PH_H, CARD_MARGIN),
                                     v["name"], v["sub"], COLOR_TEXT_DARK)
        self.cards[(True, 0)] = (make_car_card(SECRET_CAR["photo"], PH_W, PH_H, CARD_MARGIN),
                                 SECRET_CAR["name"], SECRET_CAR["sub"], COLOR_TEXT_DARK)
        self.cards[(False, 0)] = (make_placeholder_card(PH_W, PH_H, CARD_MARGIN, fonts),
                                  "منتظر دست شما", "", COLOR_TEXT_LIGHT)

        self._bg = self._build_static_bg()
        self._last_key = (False, 0)
        self._fade = None   # (کلید قبلی، کلید جدید، زمان شروع)

    def _build_static_bg(self):
        img = Image.new("RGB", (self.w, self.h), (255, 255, 255))
        d = ImageDraw.Draw(img)

        # سربرگ پنل
        d.text((self.w // 2, 36), shape_text("گالری سوپراسپرت"),
               font=self.fonts.get(20, bold=True), fill=COLOR_TEXT_DARK, anchor="mm")

        # خط جداکننده زیر سربرگ و بالای نوار پایین
        d.line([(20, 64), (self.w - 20, 64)], fill=COLOR_BORDER, width=1)
        d.line([(20, 668), (self.w - 20, 668)], fill=COLOR_BORDER, width=1)

        # دایره شمارنده انگشت
        cx, cy, r = self.w // 2, 470, 46
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=COLOR_BADGE_BG, outline=COLOR_BORDER, width=2)

        # برچسب زیر دایره
        d.text((cx, 544), shape_text("انگشت"), font=self.fonts.get(14),
               fill=COLOR_TEXT_GRAY, anchor="mm")

        # راهنمای کلیدها (نوار پایین)
        d.text((self.w - 24, 700), shape_text("خروج: Q | عکس: S"),
               font=self.fonts.get(13), fill=COLOR_TEXT_GRAY, anchor="rm")
        return img

    def render(self, count, fps, hand_present):
        key = (bool(hand_present), max(0, min(int(count), MAX_FINGERS)))
        now = time.time()

        if key != self._last_key:
            self._fade = (self._last_key, key, now)
            self._last_key = key

        card_new, name_new, sub_new, color_new = self.cards[key]
        card = card_new
        name, sub, color = name_new, sub_new, color_new

        # افکت محو شدن بین وضعیت قبلی و جدید
        if self._fade is not None:
            from_key, to_key, t0 = self._fade
            p = (now - t0) / FADE_SECONDS
            if p >= 1.0 or from_key == to_key:
                self._fade = None
            else:
                card_old, name_old, sub_old, color_old = self.cards[from_key]
                card = Image.blend(card_old, card_new, p)
                if p >= 0.5:
                    name, sub, color = name_new, sub_new, color_new
                else:
                    name, sub, color = name_old, sub_old, color_old

        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        img.paste(card, (CARD_MARGIN, 74), card)

        if name:
            d.text((self.w // 2, 316), shape_text(name),
                   font=self.fonts.get(19, bold=True), fill=color, anchor="mm")
        if sub:
            d.text((self.w // 2, 346), shape_text(sub),
                   font=self.fonts.get(14), fill=COLOR_TEXT_GRAY, anchor="mm")

        # عدد داخل دایره
        d.text((self.w // 2, 470), fa_digits(key[1]), font=self.fonts.get(40, bold=True),
               fill=COLOR_TEXT_DARK, anchor="mm")

        # FPS زنده (سمت چپ نوار پایین)
        d.text((24, 700), f"FPS: {fa_digits(int(round(fps)))}", font=self.fonts.get(13),
               fill=COLOR_TEXT_GRAY, anchor="lm")

        return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR


# ------------------------------------------------------------------
# پوشش‌های روی تصویر دوربین (شمارنده، FPS و پیام ذخیره)
# ------------------------------------------------------------------
def _chip(pil, d, text, font, x, y, fill, text_fill):
    bb = d.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.rounded_rectangle([x, y, x + w + 26, y + h + 18], radius=16, fill=fill)
    d.text((x + (w + 26) // 2, y + (h + 18) // 2), text, font=font,
           fill=text_fill, anchor="mm")
    return y + h + 34


def draw_video_chips(frame_bgr, count, fps, fonts, toast=None):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    d = ImageDraw.Draw(pil)

    y = _chip(pil, d, shape_text(fa_digits(count) + " انگشت"),
              fonts.get(18, bold=True), 14, 14, (17, 24, 39, 215), (255, 255, 255, 255))
    y = _chip(pil, d, f"FPS: {fa_digits(int(round(fps)))}",
              fonts.get(13), 14, y, (17, 24, 39, 170), (229, 231, 235, 255))
    if toast:
        _chip(pil, d, shape_text(toast), fonts.get(14, bold=True), 14, y,
              (5, 122, 85, 215), (255, 255, 255, 255))   # BGR->RGB سبز تیره

    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame_bgr, hands):
    """اسکلت دست‌ها: خطوط سفید + نوک انگشت‌های نارنجی."""
    for pts in hands:
        p = [(int(round(x)), int(round(y))) for x, y in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, p[a], p[b], (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(21):
            cv2.circle(frame_bgr, p[i], 3, (255, 255, 255), -1, cv2.LINE_AA)
        for tip in (4, 8, 12, 16, 20):
            cv2.circle(frame_bgr, p[tip], 6, COLOR_ACCENT_BGR, -1, cv2.LINE_AA)
    return frame_bgr


# ------------------------------------------------------------------
# برنامه اصلی
# ------------------------------------------------------------------
class SupercarApp:
    def __init__(self, running_mode="VIDEO"):
        if not MODEL_PATH.exists():
            sys.exit("خطا: فایل مدل یافت نشد: " + str(MODEL_PATH))
        if not FONT_DIR.exists() or not CARS_DIR.exists():
            sys.exit("خطا: پوشه assets کنار برنامه کامل نیست.")

        self.fonts = Fonts()
        self._load_photos()

        self.panel = GalleryPanel(self.fonts, PANEL_W, WINDOW_H)
        self.detector = HandDetector(MODEL_PATH, running_mode=running_mode)
        self.stab = StableState(need=1 if running_mode == "IMAGE" else 3)
        self.fps = 30.0
        self._t_prev = None
        self._toast = None          # (متن پیام، زمان پایان نمایش)
        CAPTURES_DIR.mkdir(exist_ok=True)

    def _load_photos(self):
        entries = list(CARS.values()) + [SECRET_CAR]
        for v in entries:
            path = CARS_DIR / v["file"]
            img = cv2.imread(str(path))
            if img is None:
                sys.exit("خطا: عکس ماشین بارگذاری نشد: " + str(path))
            v["photo"] = cover_crop_bgr(img, PH_W, PH_H)

    def _update_fps(self):
        now = time.time()
        if self._t_prev is not None:
            dt = now - self._t_prev
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._t_prev = now

    def _toast_text(self):
        if self._toast and time.time() < self._toast[1]:
            return self._toast[0]
        self._toast = None
        return None

    def save_snapshot(self, out_frame):
        """ذخیره عکس لحظه در پوشه captures و نمایش پیام."""
        name = "snapshot_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        path = CAPTURES_DIR / name
        cv2.imwrite(str(path), out_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self._toast = ("ذخیره شد!", time.time() + TOAST_SECONDS)
        return path

    def process_frame(self, frame_bgr, mirror=True):
        """یک فریم وبکم را پردازش و خروجی نهایی 1280x720 را برمی‌گرداند."""
        self._update_fps()
        frame = cover_crop_bgr(frame_bgr, VIDEO_W, VIDEO_H)
        if mirror:
            frame = cv2.flip(frame, 1)

        hands = self.detector.detect(frame, time.time() * 1000)
        raw = min(sum(count_fingers(h) for h in hands), MAX_FINGERS)
        hand_present = len(hands) > 0
        present, count = self.stab.update((hand_present, raw))

        frame = draw_skeleton(frame, hands)
        frame = draw_video_chips(frame, count, self.fps, self.fonts,
                                 toast=self._toast_text())
        panel_bgr = self.panel.render(count, self.fps, hand_present=present)
        return cv2.hconcat([frame, panel_bgr]), count

    def run(self, source=0):
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            sys.exit("خطا: دوربین باز نشد. مطمئن شوید وبکم به سیستم وصل است و برنامه دیگری آن را اشغال نکرده.")

        print("دستت را جلوی دوربین نشان بده؛ ۱ تا ۱۰ انگشت = ۱۰ سوپراسپرت!")
        print("مشت که بزنی، ماشین مخفی ظاهر می‌شود. عکس: کلید S | خروج: Q یا ESC")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("خطا: دریافت فریم از دوربین قطع شد.")
                break
            out, count = self.process_frame(frame, mirror=True)
            cv2.imshow("Supercar Hand Counter", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = self.save_snapshot(out)
                print("عکس ذخیره شد:", path)
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="شمارش انگشت با نمایش سوپراسپرت‌ها (Supercar Hand Counter)")
    parser.add_argument("--source", default="0",
                        help="شماره وبکم (پیش‌فرض 0) یا مسیر یک فایل ویدیو")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    SupercarApp(running_mode="VIDEO").run(source)


if __name__ == "__main__":
    main()
