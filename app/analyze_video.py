# -*- coding: utf-8 -*-
"""
Анализ титров из видео по схеме LunaTranslator:
  1) СКАН области -> детект появления/пропадания текста (white_ratio, гистерезис)
  2) OCR выбранных кадров в каждом сегменте (EasyOCR, GPU)
  3) Перевод через интернет (Google web API, без ключа — как в LunaTranslator)
  4) Результат в txt
  5) Озвучка русских фраз (TTS) поверх видео, каждая в свой тайминг

Запуск:  python analyze_video.py [путь_к_видео]
"""
import os
import re
import sys
import time
import json
import math
import wave
import html
import difflib
import shutil
import asyncio
import tempfile
import subprocess
import warnings
import cv2
import numpy as np
import requests

# мультипроцессинг для параллельного скана/OCR (Windows: spawn)
import multiprocessing as _mp


def _open_capture(video_path, hw=True):
    """VideoCapture с аппаратным декодированием (NVDEC/D3D11) и фолбэком."""
    if hw:
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION,
                                    cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(video_path)


# переопределяется переменными окружения:
#   DSH_HW=1          -- включить аппаратный декод (ВНИМАНИЕ: NVDEC иначе
#                        конвертирует цвета, пороги чуть уезжают -- по
#                        умолчанию выключен ради точности)
#   DSH_PARALLEL=0    -- старые последовательные пути (отладка)
#   DSH_SCAN_WORKERS  -- число воркеров скана (по умолчанию: все ядра, до 12)
#   DSH_OCR_BATCH     -- размер OCR-очереди батча (по умолчанию 6)
#   DSH_OCR_WORKERS   -- число GPU-воркеров OCR (по умолчанию: по числу GPU)
_HW_OK = os.environ.get("DSH_HW", "0") == "1"
_PARALLEL = os.environ.get("DSH_PARALLEL", "1") != "0"

# локальные библиотеки (edge-tts, imageio-ffmpeg) лежат в Vid\pylibs
APP_DIR = os.path.dirname(os.path.abspath(__file__))   # Vid\app
ROOT = os.path.dirname(APP_DIR)                        # Vid
_LIBS = os.path.join(ROOT, "pylibs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# ============================ НАСТРОЙКИ ============================
BASE = ROOT

VIDEO_PATH = os.path.join(BASE, "Chapter 1.mp4")   # видео по умолчанию
OUT_DIR = os.path.join(BASE, "Output")             # куда писать результат

# Область титров в координатах 1920x1080 (масштабируется под размер видео)
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 317, 827, 1602, 1010

# Вторая зона (центр экрана): титры на ЧЁРНОМ экране. Правило активации:
# экран явно чёрный (медиана яркости кадра ниже BLACK_FRAME_THR -- медиана,
# а не средняя, чтобы сам текст не портил оценку фона) И в зоне 1 в этот
# момент нет текста. Тёмные, но не чёрные кадры (заставки, сцены) зону 2
# не включают: в них текст ловит зона 1.
CENTER_ROI_X1, CENTER_ROI_Y1, CENTER_ROI_X2, CENTER_ROI_Y2 = 250, 430, 1550, 660

# Третья зона: имя говорящего над титрами. Работает всегда, когда работает
# зона 1, и по тем же таймингам (OCR тех же кадров). Область только
# ДЕТЕКТИТСЯ: текст не переводится и не озвучивается, пишется в файл
# отдельной строкой перед "EN:". Нет плашки с именем (повествование,
# а не реплика) -- пустая строка. На сегментах зоны 2 (чёрный экран)
# зона 3 не активна.
SPEAKER_ROI_X1, SPEAKER_ROI_Y1, SPEAKER_ROI_X2, SPEAKER_ROI_Y2 = 320, 760, 761, 811
SPEAKER_OCR_MAG = 3.0    # мелкий текст имени лучше читается в 3x
SPEAKER_MIN_CONF = 0.5   # мин. уверенность OCR имени
# GUI-надписи, проскальзывающие в зону говорящего (не имена):
SPEAKER_NOISE = {"online"}

# Канал тусклого текста (13.80_3): часть реплик показывается приглушённо-
# белым (min RGB ниже 200), зона 1 их "не видит", но при этом в зоне 3
# горит яркая плашка имени. Правило активации канала: ярких субтитров нет
# (wrA_s < WR_OFF) И плашка имени есть (wrS_s > SPK_ON) И в зоне 1 есть
# тусклый текст (wrC > WR_ON_C). OCR зоны 1 с пониженными порогами.
DIM_TEXT_THR = 120       # "тусклый текст": min(RGB) выше этого
WR_ON_C = 0.0013         # порог появления тусклого текста
WR_OFF_C = 0.0006        # порог исчезновения
SPK_ON = 0.015           # доля ярких пикселей в зоне 3 = "плашка имени есть"

BLACK_FRAME_THR = 8      # медиана яркости кадра, ниже которой экран "явно чёрный"
BRIGHT_THR = 60          # зона 2: пиксель "яркий", если max(R,G,B) выше этого
                         # (на чёрном фоне виден и белый, и цветной текст,
                         # например красный "Error!!")
CZ_WR_ON = 0.004         # порог появления текста в зоне 2 (фон там 0.000,
                         # короткое цветное "Error!!" даёт ~0.007)
CZ_WR_OFF = 0.0015       # порог исчезновения в зоне 2

# --- детект текста ---
WHITE_THR = 200          # порог "белого" пикселя (min из RGB)
WR_ON = 0.011            # доля белых пикселей => текст появился (плато реплики)
WR_OFF = 0.004           # ниже => текст исчез
OFF_HOLD_SEC = 0.09      # сколько секунд держится ниже WR_OFF, чтобы закрыть сегмент
MIN_SEGMENT_SEC = 0.40   # сегменты короче = вспышки, отбрасываем
MERGE_GAP_SEC = 0.25     # слияние сегментов с паузой меньше этой
MEDIAN_WIN = 7           # медианное сглаживание метрики (кадров)
BACKTRACK_SEC = 1.5      # откат старта сегмента назад для точного времени начала
GAP_WR_THR = 0.0025      # порог добора сегментов из промежутков (мелкие фразы
                         # вроде "15." дают плато ~0.003); мусор отсеется на OCR
GAP_MIN_SEC = 0.30       # мин. длительность кандидата из щели
GAP_MIN_CONF = 0.85      # фраза из щели принимается, если она числовая или
                         # уверенность OCR не ниже этого
# Короткие легитимные фразы/карточки, которые фильтр мусора не выбрасывает
# (остальные короткие не-слова вроде "Jk" -- выбрасывает)
SHORT_WORDS_OK = {"good", "end", "yes", "ok", "okay", "hey", "wow",
                  "you", "stop", "run", "sorry", "beak", "gray"}

# --- OCR ---
OCR_POINT_STEP = 2.0     # точка OCR внутри сегмента каждые N секунд
OCR_POINT_WINDOW = 0.20  # окно поиска кадра с максимальной метрикой вокруг точки, сек
OCR_END_OFFSETS = (0.08, 0.35, 0.70)  # конечные точки: столько секунд до конца сегмента
OCR_END_WINDOW = 0.12    # окно вокруг конечной точки, сек
OCR_EDGE_PAD = 0.15      # отступ от краёв сегмента для точек, сек
OCR_MAG = 2.0            # увеличение вырезки перед OCR
OCR_MIN_CONF = 0.25      # отсечка мусорных боксов

# --- UI-шум в области титров (индикаторы, иконки) ---
# Регулярки удаляются ТОЛЬКО из конца фразы (регистр не важен; флаг задаётся
# в strip_ui_noise). Каждая строка -- конкретный известный артефакт; если
# какая-то цепляет легитимный текст -- просто закомментируйте её.
UI_NOISE_PATTERNS = [
    r"\bactl?i?\s*poi[lifeor]{0,3}\b",  # ACT POINT: AcT POII, ACTI POIF, ACT Poir...
    r"\bact[ile]?\b",                   # огрызки: AcT, AcTi, ACTE
    r"\bcj\s*#?",                       # мусорное чтение того же индикатора
]

# --- перевод ---
GOOGLE_URL = "https://translate-pa.googleapis.com/v1/translateHtml"
GOOGLE_KEY = "AIzaSyATBXajvzQLTDHEQbcpq0Ihe0vWDHmO520"
GOOGLE_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json+protobuf",
    "X-Goog-Api-Key": GOOGLE_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
}
TRANSLATE_SLEEP = 0.3    # пауза между запросами в одном потоке
TRANSLATE_RETRIES = 3
TRANSLATE_CONCURRENT = 4  # потоков перевода одновременно (риск 429 -> меньше)

# --- ЭТАП 5: ОЗВУЧКА (TTS) ---
# Русские фразы синтезируются нейроголосом (интернет, как и перевод) и
# подмешиваются поверх оригинальной дорожки видео. Каждая фраза стартует
# в свой тайминг; если предыдущая ещё не договорила -- следующая ждёт и
# включается сразу после, а дальше тайминг снова абсолютный ("нагон").
TTS_ENABLED = True
TTS_VOICE = "ru-RU-SvetlanaNeural"   # мужской голос: ru-RU-DmitryNeural
TTS_RATE = "+15%"                    # базовый темп речи ("+0%" -- обычный)
TTS_PITCH = "+0Hz"
# адаптивный догон: если фраза не влезает до старта следующей по таймингу,
# она пересинтезируется быстрее (вплоть до TTS_RATE_MAX суммарно)
TTS_ADAPTIVE = True
TTS_RATE_MAX = 50        # предельный суммарный темп, %
VOICE_GAP = 0.15         # мин. пауза между фразами при нагоне, сек
TTS_CONCURRENT = 5       # сколько фраз запрашивать у сервера одновременно
                         # (этап сетевой: ускоряет озвучку в несколько раз;
                         # при лимитах сервера уменьшите до 2-3)
ORIG_VOLUME = 0.7        # громкость оригинальной дорожки (специально чуть тише)
BG_LIMIT = 0.25          # граница громкости фона: пики оригинала плавно
                         # прижимаются к этому уровню и не заходят за него
DUCK_VOLUME = 0.35       # приглушение оригинала под голосом (0..1)
DUCK_ATTACK = 0.4        # плавное приглушение ПЕРЕД голосом, сек
DUCK_RELEASE = 3.0       # медленный возврат оригинала ПОСЛЕ голоса, сек
DUCK_HOLD = 2.0          # если до следующей фразы меньше этого -- оригинал
                         # НЕ успевает вернуться к полной громкости (близкие
                         # фразы склеиваются в группу без всплесков громкости)
MIX_SR = 48000           # частота микса, Гц

# ==================== ПОЛЬЗОВАТЕЛЬСКИЕ НАСТРОЙКИ ====================
# Файл Vid\settings.json (плоский словарь) накладывается поверх значений
# выше. Неизвестные ключи игнорируются; отсутствующий файл = значения кода.
SETTINGS_PATH = os.path.join(BASE, "settings.json")

_SETTING_MAP = {
    # зоны (списки из 4 чисел, координаты 1920x1080)
    "roi_titles": ("ROI", None),
    "roi_center": ("ROI_C", None),
    "roi_speaker": ("ROI_S", None),
    # пороги детекции
    "white_thr": ("WHITE_THR", float), "wr_on": ("WR_ON", float),
    "wr_off": ("WR_OFF", float), "off_hold_sec": ("OFF_HOLD_SEC", float),
    "min_segment_sec": ("MIN_SEGMENT_SEC", float),
    "merge_gap_sec": ("MERGE_GAP_SEC", float),
    "median_win": ("MEDIAN_WIN", int),
    "backtrack_sec": ("BACKTRACK_SEC", float),
    "gap_wr_thr": ("GAP_WR_THR", float), "gap_min_sec": ("GAP_MIN_SEC", float),
    "gap_min_conf": ("GAP_MIN_CONF", float),
    "black_frame_thr": ("BLACK_FRAME_THR", float),
    "bright_thr": ("BRIGHT_THR", float), "cz_wr_on": ("CZ_WR_ON", float),
    "cz_wr_off": ("CZ_WR_OFF", float),
    "dim_text_thr": ("DIM_TEXT_THR", float),
    "wr_on_c": ("WR_ON_C", float), "wr_off_c": ("WR_OFF_C", float),
    "spk_on": ("SPK_ON", float),
    # OCR
    "ocr_point_step": ("OCR_POINT_STEP", float),
    "ocr_point_window": ("OCR_POINT_WINDOW", float),
    "ocr_end_offsets": ("OCR_END_OFFSETS", None),
    "ocr_end_window": ("OCR_END_WINDOW", float),
    "ocr_edge_pad": ("OCR_EDGE_PAD", float),
    "ocr_mag": ("OCR_MAG", float), "ocr_min_conf": ("OCR_MIN_CONF", float),
    # перевод
    "translate_sleep": ("TRANSLATE_SLEEP", float),
    "translate_retries": ("TRANSLATE_RETRIES", int),
    "translate_concurrent": ("TRANSLATE_CONCURRENT", int),
    # озвучка
    "tts_enabled": ("TTS_ENABLED", bool), "tts_voice": ("TTS_VOICE", str),
    "tts_rate": ("TTS_RATE", str), "tts_pitch": ("TTS_PITCH", str),
    "tts_adaptive": ("TTS_ADAPTIVE", bool),
    "tts_rate_max": ("TTS_RATE_MAX", int), "voice_gap": ("VOICE_GAP", float),
    "tts_concurrent": ("TTS_CONCURRENT", int),
    "orig_volume": ("ORIG_VOLUME", float), "bg_limit": ("BG_LIMIT", float),
    "duck_volume": ("DUCK_VOLUME", float), "duck_attack": ("DUCK_ATTACK", float),
    "duck_release": ("DUCK_RELEASE", float), "duck_hold": ("DUCK_HOLD", float),
    "mix_sr": ("MIX_SR", int),
}

SETTINGS_APPLIED = []


def apply_settings():
    """Прочитать settings.json и наложить на константы. Вызывается при
    импорте модуля (в том числе в воркер-процессах)."""
    if not os.path.isfile(SETTINGS_PATH):
        return
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print("[!] settings.json не прочитан (%s) -- используются значения "
              "по умолчанию" % e)
        return
    if not isinstance(cfg, dict):
        return
    for key, val in cfg.items():
        if key.startswith("_"):
            continue
        if key == "output_dir":
            globals()["OUT_DIR"] = os.path.join(BASE, str(val))
            SETTINGS_APPLIED.append(key)
            continue
        if key in ("roi_titles", "roi_center", "roi_speaker"):
            if isinstance(val, list) and len(val) == 4:
                pref = {"roi_titles": ("ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2"),
                        "roi_center": ("CENTER_ROI_X1", "CENTER_ROI_Y1",
                                       "CENTER_ROI_X2", "CENTER_ROI_Y2"),
                        "roi_speaker": ("SPEAKER_ROI_X1", "SPEAKER_ROI_Y1",
                                        "SPEAKER_ROI_X2", "SPEAKER_ROI_Y2")}[key]
                for name, num in zip(pref, val):
                    globals()[name] = int(num)
                SETTINGS_APPLIED.append(key)
            continue
        if key == "hw_decode":
            if "DSH_HW" not in os.environ:
                globals()["_HW_OK"] = bool(val)
            SETTINGS_APPLIED.append(key)
            continue
        if key == "parallel":
            if "DSH_PARALLEL" not in os.environ:
                globals()["_PARALLEL"] = bool(val)
            SETTINGS_APPLIED.append(key)
            continue
        if key in ("scan_workers", "ocr_workers", "ocr_batch"):
            env_name = {"scan_workers": "DSH_SCAN_WORKERS",
                        "ocr_workers": "DSH_OCR_WORKERS",
                        "ocr_batch": "DSH_OCR_BATCH"}[key]
            # 0 / пусто = автоматически: не переопределяем
            if str(val).strip() not in ("", "0", "None", "auto") \
                    and env_name not in os.environ:
                os.environ[env_name] = str(val)
            SETTINGS_APPLIED.append(key)
            continue
        if key in _SETTING_MAP:
            name, cast = _SETTING_MAP[key]
            try:
                globals()[name] = cast(val) if cast else val
                SETTINGS_APPLIED.append(key)
            except (TypeError, ValueError):
                pass


apply_settings()

# ============================ УТИЛИТЫ ============================


def imdecode_u(path):
    """Чтение картинки с кириллическим путём."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def fmt_time(t):
    total = int(round(t * 10))          # десятые доли, чтобы не ловить ":60.0"
    m, rest = divmod(total, 600)
    s, d = divmod(rest, 10)
    return "%02d:%02d.%d" % (m, s, d)


def fmt_hms(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# Прогресс-бар: обновляется в одной строке консоли.
def show_progress(label, done, total, start_time, extra=""):
    frac = min(1.0, done / total) if total else 1.0
    pct = int(frac * 100)
    elapsed = time.perf_counter() - start_time
    eta = (elapsed / frac * (1 - frac)) if frac > 0.002 else 0.0
    filled = int(28 * frac)
    bar = "#" * filled + "." * (28 - filled)
    sys.stdout.write("\r%s [%s] %3d%% | прошло %s | ETA %s | %s   " % (
        label, bar, pct, fmt_hms(elapsed), fmt_hms(eta), extra))
    sys.stdout.flush()


# Многострочные прогресс-бары (по одному на воркер): воркеры-процессы пишут
# прогресс в файл "<выход>.prog" ("сделано/всего"), главный процесс опрашивает
# их и перерисовывает строки через ANSI-коды.
_MULTI_LINES = 0


def _bars_enabled():
    """Многострочные бары: живая консоль или явная пометка DSH_CONSOLE=1
    (некоторые оболочки/обёртки возвращают isatty=False даже для консоли)."""
    return sys.stdout.isatty() or os.environ.get("DSH_CONSOLE") == "1"


def _prog_write(path, done, total):
    try:
        with open(path, "w", encoding="ascii") as f:
            f.write("%d/%d" % (done, total))
    except OSError:
        pass


def _prog_read(path):
    try:
        with open(path, "r", encoding="ascii") as f:
            d, t = f.read().split("/")
        return int(d), int(t)
    except Exception:
        return None


def _draw_multi_bars(rows):
    """rows: [(label, done, total)] -- перерисовка N строк; курсор после
    отрисовки стоит на строке ПОД блоком, поэтому в конце цикла достаточно
    один раз дорисовать 100% -- бары остаются на экране как итог."""
    global _MULTI_LINES
    if not _bars_enabled():
        return False
    out = []
    if _MULTI_LINES:
        out.append("\x1b[%dA" % _MULTI_LINES)
    for label, done, total in rows:
        frac = min(1.0, done / total) if total else 1.0
        filled = int(24 * frac)
        bar = "#" * filled + "." * (24 - filled)
        out.append("\x1b[2K%s [%s] %3d%%  %d/%d\r\n" % (
            label, bar, int(frac * 100), done, total))
    sys.stdout.write("".join(out))
    sys.stdout.flush()
    _MULTI_LINES = len(rows)
    return True


def _final_multi_bars(rows):
    """Финальная отрисовка: все бары в 100%, затем курсор под блоком."""
    full = [(label, total, total) for (label, _d, total) in rows]
    global _MULTI_LINES
    if _draw_multi_bars(full):
        _MULTI_LINES = 0   # блок завершён: следующие print идут ниже баров


# ============================ ЭТАП 1: СКАН ============================


def region_of(frame_shape, W, roi1920):
    """ROI в пикселях текущего видео (координаты заданы для 1920x1080)."""
    s = W / 1920.0
    rx1, ry1, rx2, ry2 = roi1920
    x1, y1 = int(rx1 * s), int(ry1 * s)
    x2, y2 = int(rx2 * s), int(ry2 * s)
    H = frame_shape[0]
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def white_ratio(crop_bgr):
    """Доля почти-белых пикселей (текст) в вырезке."""
    b, g, r = cv2.split(crop_bgr)
    mn = cv2.min(cv2.min(b, g), r)
    return float((mn > WHITE_THR).mean())


def bright_ratio(crop_bgr):
    """Доля ярких пикселей по max-каналу: на чёрном фоне виден любой текст,
    включая цветной (красный 'Error!!'), который white_ratio не замечает."""
    b, g, r = cv2.split(crop_bgr)
    mx = cv2.max(cv2.max(b, g), r)
    return float((mx > BRIGHT_THR).mean())


def median_smooth(arr, win):
    if win <= 1 or len(arr) < win:
        return arr
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.median(windows, axis=1)


def _scan_chunk_impl(video_path, i0, i1, hw, prog_path=None):
    """Метрики зон по диапазону кадров [i0, i1)."""
    cap = _open_capture(video_path, hw)
    if not cap.isOpened():
        raise IOError("Не удалось открыть видео: " + video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = i1 - i0
    last_prog = 0.0
    wrA = np.zeros(n, dtype=np.float32)
    wrB = np.zeros(n, dtype=np.float32)
    wrS = np.zeros(n, dtype=np.float32)
    wrC = np.zeros(n, dtype=np.float32)
    ret = True
    idx = 0
    rois = None
    while ret and idx < n:
        ret, frame = cap.read()
        if not ret:
            break
        if rois is None:
            rois = (region_of(frame.shape, W, (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)),
                    region_of(frame.shape, W, (CENTER_ROI_X1, CENTER_ROI_Y1,
                                               CENTER_ROI_X2, CENTER_ROI_Y2)),
                    region_of(frame.shape, W, (SPEAKER_ROI_X1, SPEAKER_ROI_Y1,
                                               SPEAKER_ROI_X2, SPEAKER_ROI_Y2)))
        ra, rb, rs = rois
        cropA = frame[ra[1]:ra[3], ra[0]:ra[2]]
        cropB = frame[rb[1]:rb[3], rb[0]:rb[2]]
        cropS = frame[rs[1]:rs[3], rs[0]:rs[2]]
        if cropA.size:
            mnA = cv2.min(cv2.min(cropA[:, :, 0], cropA[:, :, 1]),
                          cropA[:, :, 2])
            wrA[idx] = float((mnA > 200).mean())
            wrC[idx] = float((mnA > DIM_TEXT_THR).mean())
        if cropS.size:
            mnS = cv2.min(cv2.min(cropS[:, :, 0], cropS[:, :, 1]),
                          cropS[:, :, 2])
            wrS[idx] = float((mnS > 200).mean())
        if cropB.size:
            small = cv2.resize(frame, (max(1, W // 8),
                                       max(1, frame.shape[0] // 8)))
            if float(np.median(small)) < BLACK_FRAME_THR:
                wrB[idx] = bright_ratio(cropB)
        idx += 1
        if prog_path and idx - last_prog >= 500:
            last_prog = idx
            _prog_write(prog_path, idx, n)
    cap.release()
    if prog_path:
        _prog_write(prog_path, idx, n)
    return wrA[:idx], wrB[:idx], wrS[:idx], wrC[:idx]


def _cmd_scan_chunk(argv):
    """Воркер-процесс скана: python analyze_video.py --scan-chunk ..."""
    video_path, i0, i1, hw, out_npy = argv
    a, b, s, c = _scan_chunk_impl(video_path, int(i0), int(i1), hw == "1",
                                  prog_path=out_npy + ".prog")
    np.savez_compressed(out_npy, a=a, b=b, s=s, c=c)


def _ocr_worker_impl(video_path, jobs, hw, prog_path=None):
    """OCR своего набора точек: батчи по (зона, масштаб), свой захват видео."""
    batch_n = max(1, int(os.environ.get("DSH_OCR_BATCH", "6")))
    prog_done = [0]
    prog_seen = set()   # каждая точка имеет 2 варианта (кроп + увеличенный):
                        # прогресс считаем по уникальным точкам, не по вариантам

    def _bump_prog(items):
        if not prog_path:
            return
        added = 0
        for (_f, si, pi, _img) in items:
            key = (si, pi)
            if key not in prog_seen:
                prog_seen.add(key)
                added += 1
        if added:
            prog_done[0] += added
            _prog_write(prog_path, prog_done[0], len(jobs))

    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False,
                            model_storage_directory=os.path.join(BASE,
                                                                 ".easyocr_models"))
    cap = _open_capture(video_path, hw)
    if not cap.isOpened():
        raise IOError("Не удалось открыть видео: " + video_path)

    queues = {}   # (zone, tag) -> [(f, si, pi, img), ...]
    raw = {}      # (si, pi, tag) -> (f, text, conf)
    cur = 0

    def _run_queue(key):
        items = queues.pop(key, [])
        zone, tag = key
        t_thr, l_thr = ((0.45, 0.2) if zone == 4 else
                        (0.5, 0.25) if zone == 3 else (0.6, 0.3))
        min_cf = SPEAKER_MIN_CONF if zone == 3 else OCR_MIN_CONF
        for (f, si, pi, img) in items:
            boxes = reader.readtext(img, detail=1, paragraph=False,
                                    text_threshold=t_thr, low_text=l_thr)
            boxes = [b for b in (boxes or []) if b[2] >= min_cf]
            if boxes:
                conf = float(np.mean([b[2] for b in boxes]))
                raw[(si, pi, tag)] = (f, cleanup_text(
                    ocr_boxes_to_text(boxes)), conf)
            else:
                raw[(si, pi, tag)] = (f, "", 0.0)
        _bump_prog(items)

    for f, si, pi, job_zone in sorted(jobs, key=lambda j: j[0]):
        while cur < f:
            if not cap.grab():
                break
            cur += 1
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
        s = frame.shape[1] / 1920.0
        if job_zone == 3:
            rx1, ry1, rx2, ry2 = (SPEAKER_ROI_X1, SPEAKER_ROI_Y1,
                                  SPEAKER_ROI_X2, SPEAKER_ROI_Y2)
        elif job_zone == 2:
            rx1, ry1, rx2, ry2 = (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_ROI_Y2)
        else:
            rx1, ry1, rx2, ry2 = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
        crop = frame[int(ry1 * s):int(ry2 * s), int(rx1 * s):int(rx2 * s)]
        if crop.size == 0:
            continue
        mag = SPEAKER_OCR_MAG if job_zone == 3 else OCR_MAG
        big = (cv2.resize(crop, None, fx=mag, fy=mag,
                          interpolation=cv2.INTER_CUBIC) if mag != 1.0 else crop)
        if job_zone == 3:
            for tag, img in (("s1", crop), ("s3", big)):
                queues.setdefault((3, tag), []).append((f, si, pi, img))
        else:
            for tag, img in (("t1", crop), ("t2", big)):
                queues.setdefault((job_zone, tag), []).append((f, si, pi, img))
        # распознаём очереди, набравшие полный батч
        for key in list(queues):
            if len(queues[key]) >= batch_n:
                _run_queue(key)
    cap.release()
    for key in list(queues):
        _run_queue(key)

    results, spk_results = {}, {}
    for (si, pi, tag), (f, text, conf) in raw.items():
        dst = spk_results if tag.startswith("s") else results
        cur_res = dst.get((si, pi))
        if cur_res is None or conf > cur_res[2]:
            dst[(si, pi)] = (f, text, conf)
    return results, spk_results


def _cmd_ocr_worker(argv):
    """Воркер-процесс OCR: python analyze_video.py --ocr-worker ..."""
    video_path, jobs_json, hw, out_json = argv
    with open(jobs_json, "r", encoding="utf-8") as f:
        jobs = [tuple(x) for x in json.load(f)]
    results, spk_results = _ocr_worker_impl(video_path, jobs, hw == "1",
                                            prog_path=out_json + ".prog")
    payload = {
        "results": [[si, pi, r[0], r[1], r[2]]
                    for (si, pi), r in results.items()],
        "spk": [[si, pi, r[0], r[1], r[2]]
                for (si, pi), r in spk_results.items()],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def scan_video(video_path):
    """Проход 1: метрики зон по каждому кадру + сегменты наличия текста.

    Зона 1 -- нижняя область титров (всегда активна).
    Зона 2 -- центр экрана, работает только на тёмных кадрах (титры на
    чёрном фоне, вне нижней области).
    Зона 4 -- "тусклые" реплики в зоне 1 (приглушённо-белый текст), активна
    только когда ярких субтитров нет и в зоне 3 горит плашка имени.

    Длинные видео считаются параллельно по чанкам кадров во все ядра CPU
    (с аппаратным декодированием, если доступно); короткие -- как раньше.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Не удалось открыть видео: " + video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    n_cpu = _mp.cpu_count() or 4
    n_workers = int(os.environ.get("DSH_SCAN_WORKERS", min(n_cpu, 12)))
    use_par = (_PARALLEL and total > 8000 and n_workers >= 4)
    bar_start = time.perf_counter()
    if use_par:
        n_chunks = min(n_workers, max(2, total // 4000))
        size = (total + n_chunks - 1) // n_chunks
        chunks = [(i, min(i + size, total)) for i in range(0, total, size)]
        print("Скан: %d кадров, %d воркеров (ядра CPU%s)..." % (
            total, len(chunks), ", HW-декод" if _HW_OK else ""))
        tmp = tempfile.mkdtemp(prefix="dshscan_")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        self_exe = os.path.abspath(__file__)
        try:
            procs = []
            for k, (i0, i1) in enumerate(chunks):
                out = os.path.join(tmp, "c%03d.npz" % k)
                procs.append((i0, out, subprocess.Popen(
                    [sys.executable, self_exe, "--scan-chunk",
                     video_path, str(i0), str(i1),
                     "1" if _HW_OK else "0", out], env=env)))
            pending = list(procs)
            is_tty = _bars_enabled()
            done_ct = 0

            def _scan_rows():
                rows = []
                for w, (i0w, outw, _pw) in enumerate(procs):
                    d_t = _prog_read(outw + ".prog")
                    tot_w = min(i0w + size, total) - i0w
                    if d_t is None:
                        d_t = (0, tot_w)
                    rows.append(("скан w%-2d" % (w + 1), d_t[0], d_t[1]))
                return rows

            while pending:
                still = []
                for it in pending:
                    if it[2].poll() is None:
                        still.append(it)
                    else:
                        rc = it[2].returncode
                        if rc != 0:
                            for _o, _f, p2 in still:
                                if p2.poll() is None:
                                    p2.kill()
                            raise RuntimeError(
                                "воркер скана упал (код %d)" % rc)
                        done_ct += 1
                        if not is_tty:
                            show_progress("Скан  ", done_ct * size, total,
                                          bar_start,
                                          extra="воркеров готово: %d/%d"
                                                % (done_ct, len(procs)))
                pending = still
                if pending and is_tty:
                    _draw_multi_bars(_scan_rows())
                    time.sleep(0.4)
            if is_tty:
                _final_multi_bars(_scan_rows())
                print("Скан: %d воркеров за %s" % (
                    len(procs), fmt_hms(time.perf_counter() - bar_start)))
            parts = []
            for _i0, out, _p in procs:
                with np.load(out) as z:
                    parts.append((z["a"], z["b"], z["s"], z["c"]))
            wrA = np.concatenate([p[0] for p in parts])
            wrB = np.concatenate([p[1] for p in parts])
            wrS = np.concatenate([p[2] for p in parts])
            wrC = np.concatenate([p[3] for p in parts])
            used = len(wrA)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Не удалось открыть видео: " + video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        wrA = np.zeros(total, dtype=np.float32)
        wrB = np.zeros(total, dtype=np.float32)
        wrS = np.zeros(total, dtype=np.float32)
        wrC = np.zeros(total, dtype=np.float32)
        ret, frame = cap.read()
        idx = 0
        roiA = roiB = roiS = None
        while ret:
            if idx == 0:
                roiA = region_of(frame.shape, W, (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2))
                roiB = region_of(frame.shape, W,
                                 (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_ROI_Y2))
                roiS = region_of(frame.shape, W,
                                 (SPEAKER_ROI_X1, SPEAKER_ROI_Y1,
                                  SPEAKER_ROI_X2, SPEAKER_ROI_Y2))
            cropA = frame[roiA[1]:roiA[3], roiA[0]:roiA[2]]
            cropB = frame[roiB[1]:roiB[3], roiB[0]:roiB[2]]
            cropS = frame[roiS[1]:roiS[3], roiS[0]:roiS[2]]
            if cropA.size:
                mnA = cv2.min(cv2.min(cropA[:, :, 0], cropA[:, :, 1]),
                              cropA[:, :, 2])
                wrA[idx] = float((mnA > 200).mean())
                wrC[idx] = float((mnA > DIM_TEXT_THR).mean())
            if cropS.size:
                mnS = cv2.min(cv2.min(cropS[:, :, 0], cropS[:, :, 1]),
                              cropS[:, :, 2])
                wrS[idx] = float((mnS > 200).mean())
            if cropB.size:
                # чернота фона -- по МЕДИАНЕ кадра (текст её не поднимает)
                small = cv2.resize(frame, (max(1, W // 8),
                                           max(1, frame.shape[0] // 8)))
                if float(np.median(small)) < BLACK_FRAME_THR:
                    wrB[idx] = bright_ratio(cropB)
            idx += 1
            if idx % 120 == 0:
                show_progress("Скан  ", idx, total, bar_start)
            ret, frame = cap.read()
        cap.release()
        used = min(idx, total)
        wrA, wrB, wrS, wrC = (wrA[:used], wrB[:used], wrS[:used], wrC[:used])
        show_progress("Скан  ", used, used, bar_start, extra="готово")
        sys.stdout.write("\n")

    wrA_s = median_smooth(wrA, MEDIAN_WIN)
    wrB_s = median_smooth(wrB, MEDIAN_WIN)
    wrS_s = median_smooth(wrS, MEDIAN_WIN)
    wrC_s = median_smooth(wrC, MEDIAN_WIN)

    # зона 2 смотрится ТОЛЬКО когда в зоне 1 нет текста (правило пользователя)
    wrB_s = np.where(wrA_s < WR_OFF, wrB_s, 0.0).astype(np.float32)

    # зона 1: штатная сегментация + добор из щелей
    segA = detect_segments(wrA_s, fps, used, WR_ON, WR_OFF, gap_fill=True)
    segA = [sgm + [1] for sgm in segA]
    # зона 2: титры на чёрном экране
    segB = detect_segments(wrB_s, fps, used, CZ_WR_ON, CZ_WR_OFF, gap_fill=False)
    segB = [sgm + [2] for sgm in segB]
    # зона 4: тусклые реплики (нет ярких титров + плашка имени + тусклый текст)
    wrC_g = np.where((wrA_s < WR_OFF) & (wrS_s > SPK_ON), wrC_s,
                     0.0).astype(np.float32)
    segC = detect_segments(wrC_g, fps, used, WR_ON_C, WR_OFF_C, gap_fill=False)
    segC = [sgm + [4] for sgm in segC]

    segments = sorted(segA + segB + segC, key=lambda sgm: sgm[0])
    n_cand = sum(1 for sgm in segments if sgm[2] == 1)
    return wrA_s, wrB_s, wrS_s, wrC_s, segments, fps, used, n_cand


def detect_segments(wr_s, fps, used, wr_on, wr_off, gap_fill=False):
    """Гистерезис + откат старта + слияние + фильтр коротких (+ щели)."""
    off_hold = max(1, int(OFF_HOLD_SEC * fps))
    segments = []
    in_seg = False
    start = 0
    low_run = 0
    for i in range(used):
        v = wr_s[i]
        if not in_seg:
            if v > wr_on:
                in_seg = True
                start = i
                low_run = 0
        else:
            if v < wr_off:
                low_run += 1
                if low_run >= off_hold:
                    segments.append([start, i - low_run + 1, 0])
                    in_seg = False
                    low_run = 0
            else:
                low_run = 0
    if in_seg:
        segments.append([start, used - 1, 0])

    # откат старта к моменту, где текст только начал печататься
    back = int(BACKTRACK_SEC * fps)
    for sgm in segments:
        s0 = sgm[0]
        j = s0
        while j > 0 and j > s0 - back and wr_s[j - 1] > wr_off:
            j -= 1
        sgm[0] = j

    # слить близкие сегменты
    merged = []
    gap = int(MERGE_GAP_SEC * fps)
    for sgm in segments:
        if merged and sgm[0] - merged[-1][1] <= gap:
            merged[-1][1] = sgm[1]
        else:
            merged.append(sgm)

    # отбросить короткие (вспышки)
    min_len = int(MIN_SEGMENT_SEC * fps)
    segments = [sgm for sgm in merged if sgm[1] - sgm[0] >= min_len]

    if gap_fill:
        # добор кандидатов из промежутков: короткие фразы с низким плато
        # (например "At the moment...") не дотягивают до порона, но видны.
        # Кандидаты помечаются третьим элементом (сортировка не ломает привязку)
        for c0, c1 in gap_candidates(segments, wr_s, fps, used):
            j = c0
            while j > 0 and j > c0 - back and wr_s[j - 1] > wr_off:
                j -= 1
            segments.append([j, c1, 1])
        segments.sort()
    return segments


def gap_candidates(segments, wr_s, fps, n_frames):
    """Поиск пропущенных реплик в промежутках между сегментами.

    Внутри каждой "щели" ищем непрерывные участки, где сглаженная метрика
    выше GAP_WR_THR дольше MIN_SEGMENT_SEC — это кандидаты в сегменты.
    Вспышки белого (переходы) короче и отсекаются длительностью.
    Текст распознается позже; пустые кандидаты отсеются сами.
    """
    if not segments:
        return []
    min_len = int(GAP_MIN_SEC * fps)
    gaps = [(0, segments[0][0] - 1)]
    for i in range(len(segments) - 1):
        gaps.append((segments[i][1] + 1, segments[i + 1][0] - 1))
    gaps.append((segments[-1][1] + 1, n_frames - 1))

    cands = []
    for g0, g1 in gaps:
        if g1 - g0 + 1 < min_len:
            continue
        run_start = None
        for i in range(g0, g1 + 2):
            inside = i <= g1 and wr_s[i] > GAP_WR_THR
            if inside and run_start is None:
                run_start = i
            elif not inside and run_start is not None:
                if i - run_start >= min_len:
                    cands.append([run_start, i - 1])
                run_start = None
    return cands


# ============================ ЭТАП 2: OCR ============================


def pick_frame(lo, hi, wr_s, fps, floor=0, on_thr=None):
    """Кадр с максимальной метрикой среди 'устойчивых' кадров окна.

    Отсеивает короткие белые вспышки (переходы сцен): кадр берётся, только
    если в ~окне 0.25 сек вокруг него метрика держится выше текстового порога
    большую часть времени. Если устойчивых кадров нет (окно целиком в
    вспышке/паузе) — откат к последнему текстовому кадру левее окна,
    но не левее floor (границы сегмента). on_thr -- текстовый порог канала
    (для тусклого текста -- WR_ON_C, иначе WR_ON).
    """
    n = len(wr_s)
    lo, hi = max(0, int(floor), lo), min(n - 1, hi)
    if hi < lo:
        return None
    on_thr = WR_ON if on_thr is None else on_thr
    w2 = max(1, (int(0.25 * fps) // 2) * 2)
    best, best_v = None, -1.0
    for i in range(lo, hi + 1):
        a, b = max(0, i - w2), min(n, i + w2 + 1)
        fill = float((wr_s[a:b] > on_thr * 0.7).mean())
        if fill >= 0.5 and wr_s[i] > best_v:
            best, best_v = i, wr_s[i]
    if best is not None:
        return best
    # откат: последний кадр левее окна, где метрика ещё текстовая
    i = lo
    while i > floor and lo - i <= int(1.5 * fps) and wr_s[i] < on_thr:
        i -= 1
    return i if wr_s[i] >= on_thr else None


def wr_peaks(s0, s1, wr_s, fps, min_dist_sec=1.0, max_peaks=10, thr=None):
    """Локальные максимумы метрики внутри сегмента.

    Каждое плато = момент, когда очередная реплика дописана (для сегментов,
    склеивших несколько фраз подряд). Возвращает прореженный список кадров.
    thr -- порог "текста" канала (для тусклого текста ниже).
    """
    thr = GAP_WR_THR if thr is None else thr
    pad = min(int(OCR_EDGE_PAD * fps), max(0, (s1 - s0) // 4))
    lo, hi = s0 + pad, s1 - pad
    if hi <= lo:
        return []
    w = max(3, int(0.5 * fps))
    seg = wr_s[lo:hi + 1]
    n = len(seg)
    raw = []
    for i in range(n):
        a, b = max(0, i - w), min(n, i + w + 1)
        if seg[i] > thr and seg[i] >= seg[a:b].max():
            raw.append(lo + i)
    # прореживание: один пик на min_dist_sec, при равенстве берём выше
    thinned = []
    for p in raw:
        if thinned and p - thinned[-1] < min_dist_sec * fps:
            if wr_s[p] > wr_s[thinned[-1]]:
                thinned[-1] = p
            continue
        thinned.append(p)
    return thinned[-max_peaks:]


def choose_ocr_points(segment, fps, wr_s, on_thr=None):
    """Точки OCR: равномерно + пики плато + плотно у конца (полный текст).

    on_thr -- текстовый порог канала: для тусклых реплик (зона 4) ниже.
    """
    s0, s1 = segment[0], segment[1]
    dur = (s1 - s0) / fps
    pad = min(int(OCR_EDGE_PAD * fps), max(0, (s1 - s0) // 4))
    win = int(OCR_POINT_WINDOW * fps)
    on_thr = WR_ON if on_thr is None else on_thr
    points = []
    floor = s0 + pad
    k = max(2, int(round(dur / OCR_POINT_STEP)))
    for i in range(k):
        t = (i + 0.5) / k
        f = s0 + int((s1 - s0) * t)
        f = min(max(f, s0 + pad), s1 - pad)
        pf = pick_frame(max(s0 + pad, f - win), min(s1 - pad, f + win),
                        wr_s, fps, floor=floor, on_thr=on_thr)
        if pf is not None:
            points.append(pf)
    # пики: моменты дописанных реплик внутри склеенного сегмента
    for pf in wr_peaks(s0, s1, wr_s, fps, thr=on_thr * 0.85):
        points.append(pf)
    # конечные точки: у конца реплики текст дописан; несколько позиций
    # страховка от вспышек/затухания в самом хвосте
    ewin = int(OCR_END_WINDOW * fps)
    for off in OCR_END_OFFSETS:
        f = s1 - int(off * fps)
        pf = pick_frame(f - ewin, f + ewin, wr_s, fps, floor=floor)
        if pf is not None and s0 + pad <= pf <= s1:
            points.append(pf)
    return sorted(set(points))


def ocr_boxes_to_text(boxes):
    """Собрать читаемый текст из боксов EasyOCR: строки сверху-вниз, слева-направо."""
    if not boxes:
        return ""
    items = []
    hs = []
    for pts, text, conf in boxes:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        items.append((min(ys), max(ys), sum(xs) / 4.0, text, conf))
        hs.append(max(ys) - min(ys))
    items.sort(key=lambda it: (it[0], it[2]))
    med_h = float(np.median(hs)) if hs else 20.0
    lines = []
    cur = []
    cur_lo = cur_hi = None
    for top, bot, cx, text, conf in items:
        # один ряд текста: боксы заметно перекрываются по вертикали И
        # их верхние края близки. Плотная вёрстка тусклых реплик (шаг строк
        # меньше высоты бокса) должна давать НОВЫЙ ряд, а не склейку
        if cur and (min(bot, cur_hi) - max(top, cur_lo)) >= 0.5 * min(
                bot - top, cur_hi - cur_lo) \
                and abs(top - cur_lo) <= med_h * 0.6:
            cur.append((cx, text))
            cur_lo, cur_hi = min(cur_lo, top), max(cur_hi, bot)
        else:
            if cur:
                cur.sort()
                lines.append(" ".join(t for _, t in cur))
            cur = [(cx, text)]
            cur_lo, cur_hi = top, bot
    if cur:
        cur.sort()
        lines.append(" ".join(t for _, t in cur))
    return " ".join(lines)


# Частые путаницы OCR -> исправления (порядок важен)
OCR_FIXES = [
    (r"!\s*just\b", "I just"),
    (r"\bIm\b", "I'm"),
    (r"\bIfyou\b", "If you"),
    (r"\bifit's\b", "if it's"),
    (r"\bNotjust\b", "Not just"),
    (r"\bIjust\b", "I just"),
    (r"\bItjust\b", "It just"),
    (r"\bThankyou\b", "Thank you"),
    (r"\byou'Il\b", "you'll"),
    (r"\bYou'Il\b", "You'll"),
    (r"\bWe'Il\b", "We'll"),
    (r"\b[sS]he'Il\b", lambda m: m.group(0).replace("'Il", "'ll")),
    # "Error!!": восклицательные знаки OCR читает как l/I/1 (Error! l!, Errorll!)
    (r"\bError\s*[.!]*\s*[l1]+\s*[.!]*$", "Error!!"),
    # логотип заставки: FRONTLINI/FRONTLINF -> FRONTLINE, Chanter -> Chapter
    (r"\bFRONTLIN[A-Z]\b", "FRONTLINE"),
    (r"\bChanter\b", "Chapter"),
    # "Morning; Commander." -> запятая
    (r"\bMorning;", "Morning,"),
    # обрыв начала реплики: 'know right? ...' -> 'I know, right? ...'
    (r"^know,? right\?", "I know, right?"),
    # 'hella cutel' -> 'cute!' (восклицательный знак прочитан как l)
    (r"\bcutel\b", "cute!"),
    # 'Lively Young Girl' -- y прочитан как v на мелком шрифте плашки
    (r"\bLivelv\b", "Lively"),
    # 'STEN Mk II' -- римская II прочитана как Il на мелком шрифте
    (r"\bMk Il\b", "Mk II"),
    # "I've" после OCR: "II [ve" (тусклый текст, апостроф как скобка)
    (r"II \[ve\b", "I've"),
    (r"II 've\b", "I've"),
    # тусклый текст: "identity yourself" -> "identify yourself"
    (r"\bidentity yourself\b", "identify yourself"),
    # тройная l на конце ("We certainly willl")
    (r"\bwilll\b", "will"),
    # терминальные строки: "IFILEC"/"IFILEE"/"JFILEE"/"IFILED" -- это
    # "FILE" (OCR добавляет палочку и путает крайние буквы)
    (r"\b[A-Z]FILE[A-Z]\b", "FILE"),
    (r"\bIFILE\b", "FILE"),
    # после ";" OCR теряет пробел ("yourself;or" -> "yourself; or")
    (r";(?=\w)", "; "),
    # 'M16A1' -- цифра 1 прочитана как l на мелком шрифте плашки
    (r"\bM16Al\b", "M16A1"),
    # мусорный "MI" перед "I've" (обрыв заголовка цитаты; OCR путает F/T)
    (r"\bMI [FT]'ve\b", "I've"),
    # хвостовой одиночный "S" -- обрывок GUI-кнопки ("in time! S", "Click S")
    (r"\s+S\.?\s*$", ""),
    # хвостовые осколки GUI-элементов рядом с титрами ("...hear m 074",
    # "...reinforce So", "...help you out. Sa")
    (r"\s+(?:074|So|Sa|Ue|5n|JU|CL)\.?$", ""),
    # 'I' читается как F/T/1 без точки: Fm sorry, Fve been, Tve been,
    # '1 heard', "William 1 s arrogance"
    (r"\bFm\b", "I'm"),
    (r"\bFve\b", "I've"),
    (r"\bTve\b", "I've"),
    (r"^1 (?=[a-z])", "I "),
    (r"\b(\w+) 1 s\b", r"\1's"),
    (r"(?<=\s)'\s*Il\b", "I'll"),
    (r"\bFI'll\b", "I'll"),
    (r"(?:/|F|!)\s*'Il\b", "I'll"),
    (r"(?:/|!)\s*'m\b", "I'm"),
    (r"\bdont\b", "don't"),
    (r"\bcant\b", "can't"),
    (r"(?<=\s)'eml\b", "'em!"),
    (r"\bThere s\b", "There's"),
    (r"\bYoujust\b", "You just"),
    (r"\bWe'Ilhave\b", "We'll have"),
    (r"\boftime\b", "of time"),
    # задвоение слова, когда OCR поймал правку текста
    (r"\bwer\s+were\b", "were"),
    (r"\btaler\s+talent\b", "talent"),
]


def strip_ui_noise(t):
    """Срезать хвостовую цепочку UI-мусора (индикатор ACT POINT и т.п.).

    Работает итеративно от конца фразы: '15. CJ# ACTE' -> '15. CJ#' -> '15.'.
    Если фраза целиком из шума -- зачистится в пустоту и отсеется позже.
    """
    for _ in range(6):
        stripped = t
        for pat in UI_NOISE_PATTERNS:
            m = re.search(r"\s*(" + pat + r")\s*$", stripped, flags=re.IGNORECASE)
            if m:
                stripped = stripped[:m.start()].strip()
        if stripped == t:
            break
        t = stripped
    return t


def cleanup_text(t):
    t = " ".join(t.split())
    t = t.replace(" '$", "'s").replace("'$", "'s")
    # путаница апострофа со скобкой: "you'[ 're" -> "you're"
    t = re.sub(r"'\[\s*'", "'", t)
    # артефакт правки текста: 'toug" > tough' -> 'tough'
    t = re.sub(
        r'\b(\w{2,})["\']?\s*>\s*(\w+)\b',
        lambda m: m.group(2) if m.group(2).lower().startswith(m.group(1).lower())
        else m.group(0),
        t)
    # UI-мусор из конца фразы
    t = strip_ui_noise(t)
    # изолированные артефакты курсора/скобок в середине и в конце
    t = re.sub(r"\s+[._0^<~|()\[\]#@]{1,3}(?=\s|$)", " ", t)
    t = re.sub(r"\s+$", "", t)
    # слитные артефакты после слова, в середине: 'talent._ и' -> 'talent. и'
    t = re.sub(r"(?<=\w)[._0^<~|\[\]]{2,}(?=\s)", ".", t)
    t = re.sub(r"(?<=\w)[_0^<~|\[\]](?=\s)", ".", t)
    # хвостовая цифра-курсор: 'And you are 2' -> 'And you are'
    t = re.sub(r"\s+\d(?=\s*$)", "", t)
    # слитный мусор в самом конце: 'jurisdiction:_' -> 'jurisdiction.'
    # (дефис включён: курсор терминала 'decrypting-_' -> 'decrypting.')
    t = re.sub(r"(?<=\w)[._0^<~|\[\]:;-]{2,}$", ".", t)
    t = re.sub(r"(?<=\w)[_0^<~|\[\]-]$", ".", t)
    # хвостовой мусор из многоточия: 'moment_..' -> 'moment.'
    t = re.sub(r"[._0^<~|-]{2,4}$", ".", t)
    # точки/двоеточия-в-конце: ":." -> ".", ":"/";" -> "."
    t = re.sub(r"[:;]\.$", ".", t)
    t = re.sub(r"[:;]$", ".", t)
    # исправления сокращений
    for pat, rep in OCR_FIXES:
        t = re.sub(pat, rep, t)
    # пробел после знака препинания, если слиплось
    t = re.sub(r"([!?.,:])([A-Za-z])", r"\1 \2", t)
    # висячие пробелы перед пунктуацией
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    # висячие кавычки/скобки/апострофы по краям
    t = t.strip(" '\"`~^[]()#").strip()
    # реплика-вопрос без знака: 'And you are' -> 'And you are?'
    words = t.split()
    if len(words) >= 3 and not t.endswith((".", "!", "?", ":", ";", ",")):
        if words[-1].lower() in QUESTION_HINT_WORDS:
            t += "?"
    return t.strip()


# Слова, с которых короткая реплика без пунктуации почти наверняка вопрос
QUESTION_HINT_WORDS = {"are", "is", "am", "ready", "okay", "do", "did",
                       "can", "will"}


def norm_for_compare(t):
    """Нормализация для сравнения фраз: только строчные буквы/цифры, без пробелов."""
    return re.sub(r"[^a-z0-9]", "", t.lower())


def same_phrase(a, b, thr=0.75):
    """True, если два текста — снапшоты одной реплики (печать по буквам)."""
    na, nb = norm_for_compare(a), norm_for_compare(b)
    if not na or not nb:
        return False
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    # длина общего префикса
    lcp = 0
    for ca, cb in zip(short, long_):
        if ca != cb:
            break
        lcp += 1
    return (lcp / len(short)) >= thr


def ocr_pass(video_path, segments, fps, wrA_s, wrB_s, wrC_s=None):
    """Проход 2: OCR выбранных кадров.

    Длинные наборы точек распределяются по GPU-воркерам (по одному на
    видеокарту), каждый воркер распознаёт батчами. Короткие наборы и
    аварийный фолбэк -- прежним последовательным способом.
    """
    jobs = []  # (frame_idx, seg_id, point_id, zone): 1/2/4 -- титры, 3 -- имя
    for si, sgm in enumerate(segments):
        zone = sgm[3] if len(sgm) > 3 else 1
        if zone == 2:
            wr_zone = wrB_s
        elif zone == 4 and wrC_s is not None:
            wr_zone = wrC_s
        else:
            wr_zone = wrA_s
        # Зоны 2 и 4: точки РАВНОМЕРНО шагом 1с напрямую --
        # терминалы и тусклые реплики дописываются медленно, а pick_frame
        # собирает все точки на вспышке начала ("IFILER U0835205303."
        # вместо полного текста, потерянный "Error!!")
        if zone in (2, 4):
            pad4 = min(int(OCR_EDGE_PAD * fps), max(0, (sgm[1] - sgm[0]) // 4))
            step4 = max(1, int(round(1.0 * fps)))
            points = sorted(set(
                list(range(sgm[0] + pad4, sgm[1] - pad4 + 1, step4))
                + [sgm[1] - int(off * fps) for off in OCR_END_OFFSETS
                   if sgm[0] + pad4 <= sgm[1] - int(off * fps) <= sgm[1]]))
        else:
            pts_thr = GAP_WR_THR if sgm[2] == 1 else None
            points = choose_ocr_points(sgm, fps, wr_zone, on_thr=pts_thr)
        for pi, f in enumerate(points):
            jobs.append((f, si, pi, zone))
            if zone in (1, 4):
                jobs.append((f, si, pi, 3))   # говорящий -- те же тайминги
    jobs.sort(key=lambda j: j[0])

    n_gpu = 0
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0
    n_ocr_workers = int(os.environ.get("DSH_OCR_WORKERS", str(min(n_gpu, 4))))

    if _PARALLEL and n_ocr_workers >= 1 and len(jobs) >= 120:
        # делим точки на непрерывные части: каждый воркер читает видео
        # последовательно, без прыжков; один воркер на GPU
        n_ocr_workers = min(n_ocr_workers, max(1, len(jobs) // 60))
        step = (len(jobs) + n_ocr_workers - 1) // n_ocr_workers
        parts = [jobs[w * step:(w + 1) * step]
                 for w in range(n_ocr_workers)]
        parts = [p for p in parts if p]
        if parts:
            print("OCR: %d точек, %d GPU-воркер(ов), батч %d..." % (
                len(jobs), len(parts),
                max(1, int(os.environ.get("DSH_OCR_BATCH", "6")))))
            tmp = tempfile.mkdtemp(prefix="dshocr_")
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            self_exe = os.path.abspath(__file__)
            time0_ocr = time.perf_counter()
            try:
                procs = []
                for w, part in enumerate(parts):
                    jf = os.path.join(tmp, "jobs%d.json" % w)
                    of = os.path.join(tmp, "out%d.json" % w)
                    with open(jf, "w", encoding="utf-8") as f:
                        json.dump([list(j) for j in part], f,
                                  ensure_ascii=True)
                    wenv = dict(env)
                    wenv["CUDA_VISIBLE_DEVICES"] = str(w % max(n_gpu, 1))
                    # torch/triton при старте пишут безвредные UserWarning
                    # про cuobjdump/nvdisasm -- в консоль они не нужны
                    wenv["PYTHONWARNINGS"] = "ignore"
                    procs.append((of, subprocess.Popen(
                        [sys.executable, self_exe, "--ocr-worker",
                         video_path, jf, "1" if _HW_OK else "0", of],
                        env=wenv)))
                pending = list(procs)
                is_tty = _bars_enabled()
                done_ct = 0

                def _ocr_rows():
                    rows = []
                    for w, (ofw, _pw) in enumerate(procs):
                        d_t = _prog_read(ofw + ".prog")
                        tot_w = len(parts[w])
                        if d_t is None:
                            d_t = (0, tot_w)
                        rows.append(("ocr w%-3d" % (w + 1), d_t[0], d_t[1]))
                    return rows

                while pending:
                    still = []
                    for it in pending:
                        if it[1].poll() is None:
                            still.append(it)
                        else:
                            rc = it[1].returncode
                            if rc != 0:
                                for _o, p2 in still:
                                    if p2.poll() is None:
                                        p2.kill()
                                raise RuntimeError("OCR-воркер упал (код %d)"
                                                   % rc)
                            done_ct += 1
                            if not is_tty:
                                print("  воркер %d/%d готов (%.1f с)" % (
                                    done_ct, len(procs),
                                    time.perf_counter() - time0_ocr))
                    pending = still
                    if pending and is_tty:
                        _draw_multi_bars(_ocr_rows())
                        time.sleep(0.4)
                if is_tty:
                    _final_multi_bars(_ocr_rows())
                    print("OCR: %d воркеров за %s" % (
                        len(procs), fmt_hms(time.perf_counter() - time0_ocr)))
            except Exception as e:
                print("  [!] параллельный OCR не удался (%s), "
                      "последовательный режим" % e)
                for _of, p in procs:
                    if p.poll() is None:
                        p.kill()
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                results, spk_results = {}, {}
                for of, _p in procs:
                    with open(of, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    for si, pi, fr, tx, cf in payload["results"]:
                        results[(si, pi)] = (fr, tx, cf)
                    for si, pi, fr, tx, cf in payload["spk"]:
                        spk_results[(si, pi)] = (fr, tx, cf)
                shutil.rmtree(tmp, ignore_errors=True)
                return results, spk_results

    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False,
                            model_storage_directory=os.path.join(BASE,
                                                                 ".easyocr_models"))

    results = {}       # (si, pi) -> (frame_idx, text, conf_avg) -- титры
    spk_results = {}   # (si, pi) -> (frame_idx, text, conf)     -- говорящий
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Не удалось открыть видео: " + video_path)
    bar_start = time.perf_counter()
    cur = 0
    done = 0
    for f, si, pi, job_zone in jobs:
        while cur < f:
            if not cap.grab():
                break
            cur += 1
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
        s = frame.shape[1] / 1920.0
        if job_zone == 3:
            rx1, ry1, rx2, ry2 = (SPEAKER_ROI_X1, SPEAKER_ROI_Y1,
                                  SPEAKER_ROI_X2, SPEAKER_ROI_Y2)
        elif job_zone == 2:
            rx1, ry1, rx2, ry2 = CENTER_ROI_X1, CENTER_ROI_Y1, CENTER_ROI_X2, CENTER_ROI_Y2
        else:
            rx1, ry1, rx2, ry2 = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
        x1, y1 = int(rx1 * s), int(ry1 * s)
        x2, y2 = int(rx2 * s), int(ry2 * s)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if job_zone == 3:
            # имя говорящего: мелкий текст, читаем в 3x
            big = (cv2.resize(crop, None, fx=SPEAKER_OCR_MAG, fy=SPEAKER_OCR_MAG,
                              interpolation=cv2.INTER_CUBIC)
                   if SPEAKER_OCR_MAG != 1.0 else crop)
            best = ("", 0.0)
            for img in (crop, big):
                boxes = reader.readtext(img, detail=1, paragraph=False,
                                        text_threshold=0.5, low_text=0.25)
                boxes = [b for b in boxes if b[2] >= SPEAKER_MIN_CONF]
                if not boxes:
                    continue
                conf = float(np.mean([b[2] for b in boxes]))
                if conf > best[1]:
                    best = (cleanup_text(ocr_boxes_to_text(boxes)), conf)
            spk_results[(si, pi)] = (f, best[0], best[1])
            done += 1
            continue

        # OCR в двух масштабах: мелкий текст лучше читается в 2x,
        # но часть фраз (низкий контраст) -- в 1x. Берём более уверенный вариант.
        # Зона 4 (тусклый текст) -- пониженные пороги детекции строк.
        t_thr = 0.45 if job_zone == 4 else 0.6
        l_thr = 0.2 if job_zone == 4 else 0.3
        big = (cv2.resize(crop, None, fx=OCR_MAG, fy=OCR_MAG,
                          interpolation=cv2.INTER_CUBIC)
               if OCR_MAG != 1.0 else crop)
        best = ("", 0.0)
        for img in (crop, big):
            boxes = reader.readtext(img, detail=1, paragraph=False,
                                    text_threshold=t_thr, low_text=l_thr)
            boxes = [b for b in boxes if b[2] >= OCR_MIN_CONF]
            if not boxes:
                continue
            conf = float(np.mean([b[2] for b in boxes]))
            if conf > best[1]:
                # чистим до кластеризации: снапшоты одной реплики должны
                # сравниваться уже без артефактов
                best = (cleanup_text(ocr_boxes_to_text(boxes)), conf)
        results[(si, pi)] = (f, best[0], best[1])
        done += 1
        if done % 5 == 0 or done == len(jobs):
            show_progress("OCR   ", done, len(jobs), bar_start,
                          extra="точек: %d/%d" % (done, len(jobs)))
    cap.release()
    show_progress("OCR   ", len(jobs), len(jobs), bar_start, extra="готово")
    sys.stdout.write("\n")
    return results, spk_results


def extract_phrases(segments, results, fps):
    """Выбрать фразы по сегментам.

    Точки OCR внутри сегмента кластеризуются: соседние снапшоты одной
    печатаемой реплики схлопываются в одну фразу (берём самый длинный вариант).
    Сегменты-кандидаты из щелей (помечены третьим элементом) дополнительно
    проходят проверку на вложенность в фразы соседних сегментов: осколок
    гаснущей реплики содержится в полной фразе и выбрасывается.
    """
    phrases = []
    for si, sgm in enumerate(segments):
        zone = sgm[3] if len(sgm) > 3 else 1
        pts = sorted(
            [(f, txt, cf) for (s_id, _p), (f, txt, cf) in results.items() if s_id == si],
            key=lambda x: x[0])
        pts = [(f, t, c) for f, t, c in pts if t]
        if not pts:
            continue
        # кластеризация: сравниваем с самым длинным текстом текущего кластера
        clusters = []
        for f, t, c in pts:
            if clusters:
                rep = max(clusters[-1], key=lambda x: len(x[1]))[1]
                if same_phrase(t, rep):
                    clusters[-1].append((f, t, c))
                    continue
            clusters.append([(f, t, c)])
        for cl in clusters:
            zone2 = zone == 2
            # зона 2 (заставки/терминал): берём самый УВЕРЕННЫЙ снапшот --
            # длинный может быть кашей перехода (шахматное поле)
            if zone2:
                # терминальные строки с номерами ("FILE U083..., decrypting")
                # печатаются постепенно и у всех снапшотов уверенность почти
                # одинакова -- микроразницы conf оставляют обрыв без хвоста.
                # Поэтому для строк с 6+ цифрами берём САМУЮ ДЛИННУЮ не-
                # обрывочную, для остальных (заставки, "Error!!") -- по conf
                def _z2key(x):
                    if len(re.sub(r"\D", "", x[1])) >= 6:
                        return (len(x[1]) >= 7, len(x[1]), x[2], x[0])
                    return (len(x[1]) >= 7, x[2], len(x[1]), x[0])
                best = max(cl, key=_z2key)
            else:
                best = max(cl, key=lambda x: (len(x[1]), x[0]))
            text = cleanup_text(best[1])  # повторно: нормы идемпотентны
            # фильтр мусора: короткие не-слова вроде "Jk", "JIlk".
            # исключения -- числовые фразы ("15...", "27") и короткие
            # легитимные слова из SHORT_WORDS_OK ("Good.", "END")
            alnum = re.sub(r"[^A-Za-z0-9]", "", text)
            short_norm = re.sub(r"[^a-z]", "", text.lower())
            if len(alnum) < 5 and short_norm not in SHORT_WORDS_OK \
                    and not re.match(r"^\d{2,4}([.,…·]*)?$", text):
                continue
            # короткие ЯРКИЕ сегменты (< 0.6с) -- вспышки GUI ("online",
            # переходы): у легитимных коротких реплик ("Good.", "END",
            # "15.") сегмент длиннее. Числовой текст и белый список живы.
            seg_dur = (sgm[1] - sgm[0]) / fps
            is_num = bool(re.match(r"^\d{2,4}([.,…·]*)?$", text))
            if zone == 1 and seg_dur < 0.6 and not is_num \
                    and short_norm not in SHORT_WORDS_OK:
                continue
            phrases.append({"seg": si, "frame": best[0], "text": text,
                            "conf": best[2],
                            "cand": sgm[2] == 1 or zone == 4,
                            "seg_start": segments[si][0], "seg_end": segments[si][1]})

    # кросс-сегментная дедупликация кандидатов: фраза из щели, вложенная
    # в фразу соседнего сегмента (или наоборот) -- осколок затухания
    def _near(p, q):
        return abs(p["frame"] - q["frame"]) <= 6.0 * fps
    normed = [(p, norm_for_compare(p["text"])) for p in phrases]
    kept = []
    for p, n_p in normed:
        if p["cand"] and len(n_p) >= 4 \
                and re.sub(r"[^a-z]", "", p["text"].lower()) not in SHORT_WORDS_OK:
            dup = False
            for q, n_q in normed:
                if q is p or not _near(p, q):
                    continue
                # p -- осколок более полной соседней реплики (p вложена в q);
                # обратное вложение не трогаем: полная фраза не должна гибнуть
                # из-за короткого обрывка рядом
                if n_p in n_q:
                    dup = True
                    break
            if dup:
                continue
        kept.append(p)

    # зона 2 (терминал/лого/заголовки на чёрном экране): в каждый момент
    # показана ОДНА строка, печать идёт с "пересмешкой" символов, поэтому
    # соседние снапшоты не склеиваются кластером. Оставляем один вариант
    # на зона-2 сегмент (сначала отбрасывая короткие неуверенные обрывки
    # скрамбла вроде "S HOUL": длина < 7 и уверенность < 0.8).
    def _lcp(a, b):
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        return n

    final = []
    for si in sorted(set(p["seg"] for p in kept)):
        sgm = segments[si]
        zone2 = len(sgm) > 3 and sgm[3] == 2
        plist = [p for p in kept if p["seg"] == si]
        if zone2:
            # короткие обрывки скрамбла заставки ("SHOUL") не пропускаем
            plist = [p for p in plist if len(p["text"]) >= 7]
            if not plist:
                continue
            # одна строка: самый УВЕРЕННЫЙ снапшот (длинный может быть
            # кашей перехода, чистый заголовок -- коротким, но точным)
            plist = [max(plist, key=lambda p: p["conf"])]
        # дедуп внутри сегмента: фазы печати одной реплики. Ошибка OCR в
        # первой букве ("Fve"/"Tve"/"1 heard") ломает общий префикс, поэтому
        # кроме LCP сравниваем похожесть строк (difflib) и префиксное вложение
        plist.sort(key=lambda p: -len(norm_for_compare(p["text"])))
        survivors = []
        for p in plist:
            n_p = norm_for_compare(p["text"])
            dup = False
            for q in survivors:
                n_q = norm_for_compare(q["text"])
                if len(n_p) < 10:
                    continue
                same = (_lcp(n_p, n_q) >= 0.6 * max(len(n_p), len(n_q))
                        or difflib.SequenceMatcher(None, n_p, n_q).ratio() >= 0.6
                        or n_q.startswith(n_p) or n_p.startswith(n_q)
                        # обрыв печати с ошибкой OCR у среза ("Youdor...abou"
                        # из "You don't..."): начало совпадает почти целиком
                        or (len(n_q) > len(n_p) + 4 and
                            difflib.SequenceMatcher(
                                None, n_p, n_q[:len(n_p) + 4]).ratio() >= 0.8))
                if same:
                    dup = True
                    break
            if not dup:
                survivors.append(p)
        final.extend(survivors)

    # осколок у обрезанного края видео (кандидат, прилегающий к концу файла)
    last_end = max(sgm[1] for sgm in segments)
    final = [p for p in final
             if not (p["cand"] and segments[p["seg"]][1] == last_end)]

    # дубли-осколки: короткая фраза, чей текст ТОЧНО равен тексту соседней
    # (±10с) фразы ("ST CLC" три раза подряд) -- оставляем первую.
    # Только зона 1: в зоне 2 одинаковые короткие события легитимны
    # (терминал может показать "Error!!" дважды подряд)
    def _key(p):
        return norm_for_compare(p["text"])
    def _is_z2(p):
        sg = segments[p["seg"]]
        return len(sg) > 3 and sg[3] == 2
    no_dup = []
    for p in final:
        n_p = _key(p)
        if not _is_z2(p) and len(n_p) < 10 \
                and any(_key(q) == n_p and q["frame"] < p["frame"]
                        and abs(q["frame"] - p["frame"]) <= 10 * fps
                        for q in final if q is not p and not _is_z2(q)):
            continue
        no_dup.append(p)
    final = no_dup

    # терминальные строки расшифровки: осколок с меньшим числом цифр перед
    # полной строкой ("FILED PACC354261" -> "IFILEE PACC35426159108760...")
    def _digits(s):
        return re.sub(r"\D", "", s)
    no_term = []
    for p in final:
        n_p = norm_for_compare(p["text"])
        d_p = _digits(n_p)
        drop = False
        if len(d_p) >= 6:
            for q in final:
                if q is p or abs(q["frame"] - p["frame"]) > 6 * fps:
                    continue
                n_q = norm_for_compare(q["text"])
                d_q = _digits(n_q)
                if len(d_q) > len(d_p) and d_q.startswith(d_p) \
                        and len(n_p) < len(n_q):
                    drop = True
                    break
        if not drop:
            no_term.append(p)
    final = no_term

    # числовые осколки GUI без пунктуации ("74", "074"): у легитимных
    # числовых реплик ("15.") точка на конце
    final = [p for p in final
             if not re.fullmatch(r"\d{1,4}", p["text"].strip())]

    # каша перестроения строк и обрывы печати внутри одного сегмента:
    # слова осколка целиком содержатся в полной фразе соседнего снапшота
    def _covers(short_text, n_long):
        words = re.findall(r"[a-z]{4,}", short_text.lower())
        if not words:
            return False
        hit = sum(1 for w in words if w in n_long)
        return hit >= max(3, int(0.85 * len(words)))
    no_cover = []
    for p in final:
        n_p = norm_for_compare(p["text"])
        drop = False
        if len(n_p) >= 8:
            for q in final:
                if q is p or q["seg"] != p["seg"] \
                        or len(norm_for_compare(q["text"])) <= len(n_p):
                    continue
                if _covers(p["text"], norm_for_compare(q["text"])):
                    drop = True
                    break
        if not drop:
            no_cover.append(p)
    final = no_cover

    # короткий недопечатанный осколок ("Do oue", "You 07", "No,"):
    # его начало почти совпадает с началом более длинной фразы того же или
    # примыкающего сегмента -- такая реплика там уже есть целиком.
    # Для примыкающих сегментов легитимные короткие слова ("Gray.")
    # защищены белым списком: соседняя длинная реплика начинается так же
    no_short = []
    for p in final:
        n_p = norm_for_compare(p["text"])
        drop = False
        if 2 <= len(n_p) <= 11 \
                and not re.match(r"^\d{2,4}([.,…·]*)?$", p["text"].strip()):
            short_norm = re.sub(r"[^a-z]", "", p["text"].lower())
            for q in final:
                if q is p or (q["seg"] != p["seg"]
                              and abs(segments[q["seg"]][0]
                                      - segments[p["seg"]][1]) >= 1.5 * fps):
                    continue
                if q["seg"] != p["seg"] and short_norm in SHORT_WORDS_OK:
                    continue
                n_q = norm_for_compare(q["text"])
                if len(n_q) > len(n_p) + 3 and \
                        difflib.SequenceMatcher(
                            None, n_p, n_q[:max(len(n_p) + 2, 6)]).ratio() >= 0.5:
                    drop = True
                    break
        if not drop:
            no_short.append(p)
    final = no_short

    # статичные фоновые надписи (зона 4): одна и та же тусклая надпись
    # мелькает в нескольких сегментах ("ACTE POII" на стене) -- это
    # декорация, а не речь. Реальные реплики не повторяются идентично
    deco_groups = []   # [rep_norm, [фразы]]
    for p in final:
        sgm = segments[p["seg"]]
        if not (len(sgm) > 3 and sgm[3] == 4 and p["cand"]):
            continue
        n_p = norm_for_compare(p["text"])
        for g in deco_groups:
            if difflib.SequenceMatcher(None, n_p, g[0]).ratio() >= 0.8:
                g[1].append(p)
                break
        else:
            deco_groups.append([n_p, [p]])
    deco_drop = set()
    for _rep, items in deco_groups:
        frames = [it["frame"] for it in items]
        if len(items) >= 3 and max(frames) - min(frames) > 15 * fps:
            deco_drop.update(id(it) for it in items)
    if deco_drop:
        final = [p for p in final if id(p) not in deco_drop]

    final.sort(key=lambda p: p["frame"])
    return final


# ============================ ЭТАП 3: ПЕРЕВОД ============================


def google_translate(text, src="en", dst="ru"):
    payload = json.dumps([[[text], src, dst], "wt_lib"])
    last_err = None
    for attempt in range(TRANSLATE_RETRIES):
        try:
            r = requests.post(GOOGLE_URL, headers=GOOGLE_HEADERS, data=payload,
                              timeout=15)
            r.raise_for_status()
            return r.json()[0][0]
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt * 2.0)
    raise RuntimeError("Перевод не удался: %r" % (last_err,))


def _translate_one(p):
    """Перевод одной фразы + правки имён. Возвращает ru (может быть "")."""
    en = p["text"]
    ru = html.unescape(google_translate(en))
    # "S.F." -- аббревиатура (не Сан-Франциско): правим перевод точечно
    if re.search(r"\bS\.\s?F\.", en) and "Сан-Франциско" in ru:
        ru = ru.replace("Сан-Франциско", "Эс Эф")
    # Dandelion -- имя (не "одуванчик"): склонение сохраняем
    ru = re.sub(r"[Оо]дуванчик(а|у|ом|е|и)?",
                lambda m: "Дэнделиан" + (m.group(1) or ""), ru)
    # Gray -- имя персонажа: "Грэй" (не "Грей" и не "Серый")
    if re.search(r"\bGray\b", en):
        ru = re.sub(r"\bГре(й|я|ю|ем|е)\b", r"Грэ\1", ru)
        if re.fullmatch(r"Gray\b[.!,;: ]*", en.strip()):
            ru = "Грэй."
    # Beak -- имя персонажа: "Биик" (не "Клюв" и не "Бик")
    if re.search(r"\bBeak\b", en):
        ru = ru.replace("Клюв", "Биик")
        ru = re.sub(r"\bБик\b", "Биик", ru)
        if re.fullmatch(r"Beak\b[.!,;: ]*", en.strip()):
            ru = "Биик."
    return ru


def translate_pass(phrases):
    bar_start = time.perf_counter()
    n = len(phrases)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done_ct = 0

    def _job(p):
        try:
            ru = _translate_one(p)
            # пауза-ограничитель в каждом потоке; в параллельном режиме
            # достаточно маленькой, иначе режим ~не отличается от одного
            # потока при большой задержке ответа
            time.sleep(min(TRANSLATE_SLEEP, 0.1))
            return p, ru, None
        except Exception as e:
            time.sleep(1.0)
            return p, "", str(e)

    with ThreadPoolExecutor(max_workers=max(1, TRANSLATE_CONCURRENT)) as ex:
        futs = [ex.submit(_job, p) for p in phrases]
        for fut in as_completed(futs):
            p, ru, err = fut.result()
            p["ru"] = ru
            if err:
                p["err"] = err
            done_ct += 1
            show_progress("Перевод", done_ct, n, bar_start)
    show_progress("Перевод", n, n, bar_start, extra="готово")
    sys.stdout.write("\n")


# ============================ ЭТАП 5: ОЗВУЧКА ============================


def _ffmpeg_exe():
    """Путь к ffmpeg: сначала локальный (imageio-ffmpeg), потом из PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _tts_synthesize(text, mp3_path, rate=None):
    """Синтез одной фразы нейроголосом edge-tts -> mp3."""
    import edge_tts

    async def _run():
        com = edge_tts.Communicate(text, TTS_VOICE,
                                   rate=rate or TTS_RATE, pitch=TTS_PITCH)
        await com.save(mp3_path)

    asyncio.run(_run())


def _mp3_to_wav(mp3_path, wav_path, ffmpeg):
    """mp3 -> wav (MIX_SR, стерео, 16 бит) и длительность в секундах."""
    subprocess.run(
        [ffmpeg, "-y", "-i", mp3_path, "-ar", str(MIX_SR), "-ac", "2",
         "-f", "wav", wav_path],
        capture_output=True, check=True)
    with wave.open(wav_path, "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
    return dur


def _read_wav_float(path):
    """wav -> float32 массив (N, 2) в диапазоне [-1, 1]."""
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch == 1:
        frames = np.repeat(frames, 2)
    data = frames.reshape(-1, 2).astype(np.float32) / 32768.0
    return data


def _build_env(n, sr, intervals, attack, release, hold, duck):
    """Огибающая громкости оригинала: 1.0 вдали от голосов, duck под голосом.

    intervals -- [(a0, a1)] интервалы голоса в сэмплах, отсортированы.
    Близкие интервалы (пауза < hold) склеиваются в группу: между фразами
    группы оригинал остаётся приглушённым, без всплесков громкости.
    Возврат к полной громкости -- медленный (release) и только если
    озвучки рядом долго нет и не планируется.
    """
    env = np.ones(n, dtype=np.float32)
    groups = []
    for a0, a1 in intervals:
        if groups and a0 - groups[-1][1] < hold * sr:
            groups[-1][1] = max(groups[-1][1], a1)
        else:
            groups.append([a0, a1])
    atk_n = int(attack * sr)
    rel_n = int(release * sr)
    for a0, a1 in groups:
        lo = max(0, a0 - atk_n)
        hi = min(n, a1 + rel_n)
        if hi <= lo:
            continue
        trap = np.full(hi - lo, duck, dtype=np.float32)
        up_n = a0 - lo
        if up_n > 0:
            trap[:up_n] = np.linspace(1.0, duck, up_n, dtype=np.float32)
        down_n = hi - a1
        if down_n > 0:
            trap[len(trap) - down_n:] = np.linspace(duck, 1.0, down_n,
                                                    dtype=np.float32)
        np.minimum(env[lo:hi], trap, out=env[lo:hi])
    return env


def assign_appearances(phrases, segments, wrA_s, wrB_s, fps, wrC_s=None):
    """Момент ПОЯВЛЕНИЯ текста каждой фразы (для озвучки).

    Озвучка должна стартовать, когда текст начал показываться, а не когда
    он дописался. Первая фраза сегмента -- seg_start (начало печати);
    следующая в том же сегменте -- провал метрики между кадрами соседних
    фраз (момент смены реплик). Для тусклых реплик (зона 4) метрика -- wrC.
    """
    by_seg = {}
    for p in phrases:
        by_seg.setdefault(p["seg"], []).append(p)
    for si, plist in by_seg.items():
        sgm = segments[si]
        z = sgm[3] if len(sgm) > 3 else 1
        if z == 2:
            wr = wrB_s
        elif z == 4 and wrC_s is not None:
            wr = wrC_s
        else:
            wr = wrA_s
        plist.sort(key=lambda p: p["frame"])
        prev_frame = sgm[0]
        for j, p in enumerate(plist):
            if j == 0:
                p["appear"] = sgm[0]
            else:
                lo, hi = prev_frame, p["frame"]
                if hi > lo + 1:
                    dip = lo + int(np.argmin(wr[lo:hi + 1]))
                else:
                    dip = lo
                p["appear"] = dip
            prev_frame = p["frame"]


def assign_speakers(phrases, segments, spk_results):
    """Имя говорящего для фраз зон 1/4 (зона 3, те же тайминги).

    Имя присваивается КАЖДОЙ ФРАЗЕ: плашка меняется внутри сегмента
    ("White Nyto" -> "Olga"), поэтому имя -- кластер снапшотов, чьё
    среднее время ближе всего к кадру фразы. Плашки нет (повествование) --
    пустая строка. Зона 2 (чёрный экран) без имён.
    """
    by_seg = {}
    for (si, _pi), (f, txt, cf) in spk_results.items():
        if txt:
            by_seg.setdefault(si, []).append((f, txt, cf))
    names_by_time = {}          # si -> [(среднее время, имя)]
    for si, snaps in by_seg.items():
        sgm = segments[si]
        if len(sgm) > 3 and sgm[3] == 2:
            continue
        clusters = []           # [текст-представитель, [время...], [conf...]]
        for f, txt, cf in snaps:
            for cl in clusters:
                if same_phrase(txt, cl[0]):
                    cl[1].append(f)
                    cl[2].append(cf)
                    break
            else:
                clusters.append([txt, [f], [cf]])
        entries = []
        for txt, frames, confs in clusters:
            name = re.sub(r"\s+", " ", txt).strip(" .,:;-_")
            if re.fullmatch(r"\?{1,3}", name):
                entries.append((float(np.median(frames)), "???"))
                continue                     # анонимный говорящий
            alnum = re.sub(r"[^A-Za-z0-9]", "", name)
            if not (1 <= len(alnum) <= 40):
                continue
            if alnum.isdigit() and len(alnum) > 1:
                continue                     # чистые цифры -- GUI, не имя
            if re.sub(r"[^a-z]", "", name.lower()) in SPEAKER_NOISE:
                continue                     # GUI-надпись, не имя говорящего
            # точечные фиксы мелкого шрифта плашек: y читается как v,
            # S как 5, римская II как Il (только имена, титры не трогаем)
            name = re.sub(r"\bMilitarv\b", "Military", name)
            name = re.sub(r"\bKrvuger\b", "Kryuger", name)
            name = re.sub(r"\bUMP4S\b", "UMP45", name)
            name = re.sub(r"\bIl\b", "II", name)
            name = re.sub(r"\bNvto\b", "Nyto", name)
            name = re.sub(r"\bRomv\b", "Romy", name)
            name = re.sub(r"\bGrav\b", "Gray", name)
            if name == "P9O":
                name = "Sterling"        # плашка OCR читает как P9O
            if name == "0":
                name = "Q"                   # одиночная цифра -- буква Q
            entries.append((float(np.median(frames)), name))
        if entries:
            names_by_time[si] = entries
    for p in phrases:
        entries = names_by_time.get(p["seg"], [])
        if entries:
            t = p["frame"]
            p["speaker"] = min(entries, key=lambda e: abs(e[0] - t))[1]
        else:
            p["speaker"] = ""


def voice_pass(video_path, phrases, fps, n_frames):
    """Синтез фраз, планирование таймингов без наложения, микс, сборка mp4.

    Тайминг фразы -- момент, когда её текст полностью виден (кадр OCR).
    Старт = max(свой тайминг, конец предыдущей фразы + VOICE_GAP):
    накладки нет, а после паузы таймлайн снова догоняется, т.к. все
    дальнейшие старты остаются абсолютными.
    """
    ffmpeg = _ffmpeg_exe()
    out_video = os.path.join(
        OUT_DIR, os.path.splitext(os.path.basename(video_path))[0]
        + "_озвучка.mp4")
    total_sec = n_frames / fps + 1.0
    workdir = tempfile.mkdtemp(prefix="tts_")
    n = len(phrases)

    # 1) предсинтез фраз: несколько запросов к серверу одновременно
    # (базовый синтез зависит только от текста и темпа, не от таймингов,
    # поэтому его можно делать заранее и параллельно; планировщик таймингов
    # ниже остаётся последовательным и при необходимости догоняет --
    # редкий пересинтез сжатой фразы -- как и раньше)
    print("Синтез речи (%d фраз, голос %s, темп %s, потоков %d)..." % (
        n, TTS_VOICE, TTS_RATE, TTS_CONCURRENT))
    m_rate = re.match(r"^([+-]?\d+)%$", TTS_RATE)
    base_pct = int(m_rate.group(1)) if m_rate else 0
    bar_start = time.perf_counter()
    order = sorted(range(n), key=lambda i: phrases[i]["appear"])
    sched_list = [phrases[j]["appear"] / fps for j in order]
    from concurrent.futures import ThreadPoolExecutor, as_completed
    base_cache = {}   # i -> длительность wav (базовый темп)

    def _make_base(i):
        p = phrases[i]
        mp3 = os.path.join(workdir, "p%03d.mp3" % i)
        wav = os.path.join(workdir, "p%03d.wav" % i)
        last_err = None
        for attempt in range(3):   # сеть ненадёжна: пара повторов
            try:
                _tts_synthesize(p["ru"] or p["text"], mp3, TTS_RATE)
                return i, _mp3_to_wav(mp3, wav, ffmpeg)
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        raise last_err

    done_ct = 0
    with ThreadPoolExecutor(max_workers=max(1, TTS_CONCURRENT)) as ex:
        futs = {ex.submit(_make_base, i): i for i in order}
        for fut in as_completed(futs):
            try:
                i, dur = fut.result()
                base_cache[i] = dur
            except Exception:
                pass   # упавший догоним последовательно в планировщике
            done_ct += 1
            show_progress("Озвучка", done_ct, n, bar_start)
    sys.stdout.write("\n")

    prev_end = 0.0
    ok = 0
    for k, i in enumerate(order):
        p = phrases[i]
        mp3 = os.path.join(workdir, "p%03d.mp3" % i)
        wav = os.path.join(workdir, "p%03d.wav" % i)
        # старт известен заранее: зависит только от конца предыдущей фразы
        start = max(p["appear"] / fps, prev_end + VOICE_GAP)
        try:
            rate = TTS_RATE
            used_pct = base_pct
            if i in base_cache:
                dur = base_cache[i]
            else:
                _tts_synthesize(p["ru"] or p["text"], mp3, rate)
                dur = _mp3_to_wav(mp3, wav, ffmpeg)
            # адаптивный догон: не влезаем до следующего тайминга --
            # пересинтез быстрее (вплоть до максимума; без наложения в любом
            # случае). Ограничения по окну нет: даже частичное сжатие
            # сокращает очередь
            if TTS_ADAPTIVE and k + 1 < len(order):
                allowed = sched_list[k + 1] - VOICE_GAP - start
                if allowed > 0.2 and dur > allowed:
                    extra = int(math.ceil((dur / allowed - 1.0) * 100))
                    target = min(TTS_RATE_MAX, base_pct + extra)
                    if target > used_pct:
                        rate = "+%d%%" % target
                        used_pct = target
                        _tts_synthesize(p["ru"] or p["text"], mp3, rate)
                        dur = _mp3_to_wav(mp3, wav, ffmpeg)
        except Exception as e:
            print("\n  [!] фраза %d не озвучена: %s" % (i + 1, e))
            continue
        p["voice_start"] = start
        p["voice_dur"] = dur
        p["voice_wav"] = wav
        p["voice_rate"] = used_pct
        prev_end = start + dur
        ok += 1
    if ok == 0:
        print("Ни одна фраза не озвучена -- видео не собираю.")
        shutil.rmtree(workdir, ignore_errors=True)
        return

    # 2) оригинальная дорожка
    print("Микширование аудио...")
    sr = MIX_SR
    total_n = int(total_sec * sr)
    orig = np.zeros((total_n, 2), dtype=np.float32)
    tmp_orig = os.path.join(workdir, "_orig.wav")
    r = subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-vn", "-ar", str(sr), "-ac", "2",
         "-f", "wav", tmp_orig], capture_output=True)
    if r.returncode == 0 and os.path.isfile(tmp_orig) \
            and os.path.getsize(tmp_orig) > 44:
        orig_data = _read_wav_float(tmp_orig)
        m = min(len(orig_data), total_n)
        orig[:m] = orig_data[:m]
    else:
        print("  (у видео нет аудиодорожки -- микс только с голосом)")

    # 3) голоса + огибающая приглушения оригинала
    voices = np.zeros((total_n, 2), dtype=np.float32)
    intervals = []
    for p in phrases:
        if "voice_wav" not in p:
            continue
        data = _read_wav_float(p["voice_wav"])
        offset = int(p["voice_start"] * sr)
        m = min(len(data), total_n - offset)
        if m <= 0:
            continue
        voices[offset:offset + m] += data[:m]
        intervals.append((offset, offset + m))
    env = _build_env(total_n, sr, intervals, DUCK_ATTACK, DUCK_RELEASE,
                     DUCK_HOLD, DUCK_VOLUME)

    bg = orig * (env[:, None] * ORIG_VOLUME)
    # граница громкости фона: всё, что громче BG_LIMIT, плавно прижимается
    # к потолку (мягкий лимитер без щелчков), фон не "перекрикивает" голос
    bg = BG_LIMIT * np.tanh(bg / BG_LIMIT)
    mixed = bg + voices
    np.clip(mixed, -1.0, 1.0, out=mixed)
    mix_wav = os.path.join(workdir, "_mix.wav")
    with wave.open(mix_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((mixed * 32767.0).astype(np.int16).tobytes())

    # 4) сборка видео: видеопоток копируется, звук -- микс
    print("Сборка видео: %s" % os.path.basename(out_video))
    r = subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-i", mix_wav,
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_video], capture_output=True)
    if r.returncode != 0:
        print("  [!] ffmpeg не собрал видео:")
        print(r.stderr.decode("utf-8", "ignore")[-800:])
    else:
        print("Готово: %s (%.1f МБ)" % (
            os.path.basename(out_video),
            os.path.getsize(out_video) / 1e6))
    shutil.rmtree(workdir, ignore_errors=True)


# ============================ ВЫВОД ============================


def write_output(out_path, video_path, fps, n_frames, phrases, elapsed):
    W = H = None
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = n_frames / fps

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Автоперевод титров из видео\n")
        f.write("# Зона 1 (нижняя): %d,%d-%d,%d | зона 2 (центр, только явный "
                "чёрный экран без текста в зоне 1): %d,%d-%d,%d | зона 3 "
                "(говорящий, не переводится): %d,%d-%d,%d  -- в координатах 1920x1080\n" %
                (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2,
                 CENTER_ROI_X1, CENTER_ROI_Y1, CENTER_ROI_X2, CENTER_ROI_Y2,
                 SPEAKER_ROI_X1, SPEAKER_ROI_Y1, SPEAKER_ROI_X2, SPEAKER_ROI_Y2))
        f.write("# Видео: %s\n" % video_path)
        f.write("# %sx%s, %.1f fps, длительность %s\n" %
                (W or "?", H or "?", fps, fmt_time(dur)))
        f.write("# Фраз: %d | Время анализа: %s\n" % (len(phrases), fmt_hms(elapsed)))
        m_rate = re.match(r"^([+-]?\d+)%$", TTS_RATE)
        base_rate = int(m_rate.group(1)) if m_rate else 0
        f.write("=" * 70 + "\n\n")
        for i, p in enumerate(phrases, 1):
            t1 = p["frame"] / fps
            f.write("[%03d] %s  (появление %s, сегмент %s-%s)\n" % (
                i, fmt_time(t1), fmt_time(p.get("appear", p["frame"]) / fps),
                fmt_time(p["seg_start"] / fps),
                fmt_time(p["seg_end"] / fps)))
            f.write("%s\n" % p.get("speaker", ""))   # имя говорящего или пусто
            f.write("EN: %s\n" % p["text"])
            f.write("RU: %s\n" % (p.get("ru") or "[перевод не удался]"))
            if "voice_start" in p:
                extra = (" темп +%d%%" % p["voice_rate"]
                         if p.get("voice_rate", base_rate) != base_rate else "")
                f.write("ОВ: старт %s, длительность %.1f с%s\n" % (
                    fmt_time(p["voice_start"]), p["voice_dur"], extra))
            f.write("-----------------\n")


# ============================ MAIN ============================


def main():
    # режимы воркеров (запускаются как отдельные процессы из scan/ocr)
    if len(sys.argv) > 1 and sys.argv[1] == "--scan-chunk":
        _cmd_scan_chunk(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--ocr-worker":
        _cmd_ocr_worker(sys.argv[2:])
        return

    t0 = time.perf_counter()
    if os.name == "nt":
        os.system("")   # включить ANSI-коды (многострочные бары воркеров)
    # torch/triton при первом импорте (подсчёт GPU, последовательный OCR в
    # главном процессе) пишут безвредные UserWarning -- гасим точечно их
    # (в GPU-воркерах всё предупреждение целиком выключено через окружение)
    warnings.filterwarnings("ignore", message="Failed to find cuobjdump")
    warnings.filterwarnings("ignore", message="Failed to find nvdisasm")
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    print("Видео:", video)
    print("Область (1920x1080): (%d,%d)-(%d,%d)" % (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2))
    os.makedirs(OUT_DIR, exist_ok=True)
    if SETTINGS_APPLIED:
        print("Настройки из settings.json: %s" % ", ".join(SETTINGS_APPLIED))
    print("=" * 70)

    # Этап 1
    wrA_s, wrB_s, wrS_s, wrC_s, segments, fps, n_frames, n_cand = scan_video(video)
    print("Сегментов с текстом: %d (из них кандидатов из щелей: %d)" %
          (len(segments), n_cand))
    for i, sgm in enumerate(segments):
        print("  [%02d] %s - %s  зона %d%s" % (
            i + 1, fmt_time(sgm[0] / fps), fmt_time(sgm[1] / fps),
            sgm[3] if len(sgm) > 3 else 1,
            " (канд.)" if sgm[2] == 1 else ""))

    # Этап 2
    results, spk_results = ocr_pass(video, segments, fps, wrA_s, wrB_s, wrC_s)
    phrases = extract_phrases(segments, results, fps)
    assign_speakers(phrases, segments, spk_results)
    assign_appearances(phrases, segments, wrA_s, wrB_s, fps, wrC_s)
    print("Извлечено фраз: %d" % len(phrases))

    # Этап 3
    translate_pass(phrases)

    # Этап 4: озвучка + сборка видео с миксом
    if TTS_ENABLED and phrases:
        try:
            voice_pass(video, phrases, fps, n_frames)
        except Exception as e:
            print("Озвучка не выполнена: %s" % e)

    # Вывод
    elapsed = time.perf_counter() - t0
    base = os.path.splitext(os.path.basename(video))[0]
    out_path = os.path.join(OUT_DIR, base + "_перевод.txt")
    write_output(out_path, video, fps, n_frames, phrases, elapsed)

    print("=" * 70)
    print("Готово! Фраз: %d" % len(phrases))
    print("Результат: %s" % out_path)
    if TTS_ENABLED:
        dubbed = os.path.join(OUT_DIR,
                              os.path.splitext(os.path.basename(video))[0]
                              + "_озвучка.mp4")
        if os.path.isfile(dubbed):
            print("Озвучка: %s" % dubbed)
    print("Общее время работы: %s" % fmt_hms(time.perf_counter() - t0))


if __name__ == "__main__":
    main()
