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

# Вторая зона (центр экрана): титры на ЧЁРНОМ экране и текст на БЕЛЫХ
# слайдах (письма/документы). Правило активации: экран явно чёрный
# (медиана яркости кадра ниже BLACK_FRAME_THR) ИЛИ явно белый
# (медиана выше WHITE_FRAME_THR -- тогда в зоне 1 меряем тёмные буквы),
# И в зоне 1 в этот момент нет текста. Тёмные, но не чёрные кадры
# (заставки, сцены) зону 2 не включают: в них текст ловит зона 1.
# Область широкая и высокая: строки диалогов и письмо на слайдах
# шире старой зоны и обрезались по краям ("No, there's nothin..."),
# а игровой UI (SKIP/LOG/Auto/HIDE, y 0-130) остаётся снаружи.
CENTER_ROI_X1, CENTER_ROI_Y1, CENTER_ROI_X2, CENTER_ROI_Y2 = 100, 300, 1850, 1080

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
WR_ON_C = 0.0013         # порог появления тусклого текста (обычная ступень)
WR_OFF_C = 0.0006        # порог исчезновения
WR_ON_C_LOW = 0.0006     # пониженная ступень (нейтрально-тёмные кадры):
                         # короткие розовые/красные вспышки 0.3-0.5с
WR_OFF_C_LOW = 0.0002    # порог исчезновения пониженной ступени
SPK_ON = 0.015           # доля ярких пикселей в зоне 3 = "плашка имени есть"

BLACK_FRAME_THR = 12     # медиана яркости кадра, ниже которой экран "явно чёрный"
                         # (было 8: "Chessmaster - Logged Out." идёт на кадрах
                         # с медианой 9-10 -- почти чёрных, и зона 2 их
                         # пропускала)
WHITE_FRAME_THR = 200    # медиана яркости кадра, выше которой экран "белый слайд"
BRIGHT_THR = 60          # зона 2: пиксель "яркий", если max(R,G,B) выше этого
                         # (на чёрном фоне виден и белый, и цветной текст,
                         # например красный "Error!!")
DARK_TEXT_THR = 60       # зона 2 на белом слайде: пиксель "тёмный" (текст),
                         # если min(R,G,B) ниже этого
CZ_WR_ON = 0.0009        # порог появления текста в зоне 2 (ROI 100,300-1850,1080:
                         # площадь в 4.6 раза больше старой 250,430-1550,660,
                         # поэтому порог пропорционально уменьшен -- та же
                         # чувствительность в абсолютных пикселях)
CZ_WR_OFF = 0.00033      # порог исчезновения в зоне 2

# --- зона 8: ЗАТЕМНЁННЫЙ ФОН (письма/записки на полупрозрачном затемнении
# сцены, "15.1 Summer Garden") ---
DARK_OVERLAY_THR = 70     # медиана яркости: чёрный <12 <= затемнённый <70 <= сцена
CZ_WR_ON_DARK = 0.0032    # порог БЕЛОГО текста (min RGB>200) в центре на затемнении
                          # (0.0036-0.0037 -- плато "Compose message." /
                          # "Reply to message." в 15.1: при 0.0038 эти экраны
                          # уходили патрулю и их текст терялся)
CZ_WR_OFF_DARK = 0.0012   # порог исчезновения белого текста
CENTER_READ_Y2 = 747      # нижняя граница чтения/метрики ЗОНЫ 8: ниже лежит
                          # полоса имени говорящего (760-811) -- зона 8 её
                          # не читает. Зона 2 читает полный центр (эталон)
Z8_NEIGHBOR_PAD_SEC = 1.0  # зазор до чужого сегмента: сегмент зоны 8, подошедший
                           # ближе, выбрасывается (текст рядом уже пойман другой
                           # зоной -- "The flesh...", 13.80_4 196-199с)

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
                  "you", "stop", "run", "sorry", "beak", "gray", "isee"}

# --- OCR ---
OCR_POINT_STEP = 2.0     # точка OCR внутри сегмента каждые N секунд
OCR_POINT_WINDOW = 0.20  # окно поиска кадра с максимальной метрикой вокруг точки, сек
OCR_END_OFFSETS = (0.08, 0.35, 0.70)  # конечные точки: столько секунд до конца сегмента
OCR_END_WINDOW = 0.12    # окно вокруг конечной точки, сек
OCR_EDGE_PAD = 0.15      # отступ от краёв сегмента для точек, сек
OCR_MAG = 2.0            # увеличение вырезки перед OCR
OCR_MIN_CONF = 0.25      # отсечка мусорных боксов
OCR_MIN_BOX_H = 34       # мин. высота бокса (нативных пикселей 1080p) для зон
                         # 1/4: ярлыки карты ("Control Terminal", "Nitrogen
                         # Squad", цифры HP/очков) мельче (h 16-32), реплики
                         # крупнее (h 38-66)
RED_TEXT_THR = 0.004     # доля "красных" пикселей в кропе, при которой кроп
                         # переводится в max-канал перед OCR (красный текст на
                         # тёмном EasyOCR читает плохо в цвете)
RED_ON = 0.004           # зона 5 (красные реплики): порог появления
RED_OFF = 0.0018         # порог исчезновения (шум полосы ~0.0013)
RED_MAX_SEG_SEC = 15.0   # максимальная длина сегмента зоны 5: длиннее --
                         # статичная красная подсветка интерфейса
RED_LOW_ANCHOR_SEC = 14.0  # радиус якоря пониженной ступени зоны 4
                           # вокруг сегментов зоны 5

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
# догоны: когда очередь фраз запаздывает (долгие реплики, вспышки подряд),
# фраза может начаться сильно позже своего тайминга. Чтобы отставание не
# росло, задержавшаяся фраза пересинтезируется быстрее: темп поднимается
# настолько, чтобы голос занял не больше 0.35..2.5с свободного хвоста до
# следующего тайминга (вплоть до TTS_RATE_MAX суммарно).
TTS_CATCHUP = True       # разрешить догон по отставанию старта
PATROL_STEP_SEC = 1.2    # шаг контрольного дозора (зона 6), сек
PATROL_MIN_ALNUM = 4     # минимум буквенно-цифровых знаков в фразе дозора
PATROL_WR_MAX = 0.008    # верхняя граница wrA для "тёмного" кадра дозора
                         # (выше WR_OFF: ".I see." имеет wrA 0.004-0.006)
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
TTS_WAV_SR = 24000       # частота wav-файлов фраз: edge-tts отдаёт 24кГц,
                         # стерео-расширение и ресемплинг делает миксер.
                         # 4x экономия диска: на 2685 фраз разница
                         # ~27 ГБ (48к стерео) против ~7 ГБ
TTS_TIMEOUT_SEC = 120    # жёсткий лимит синтеза фразы: websocket edge-tts
                         # может молча зависнуть (нет таймаута на приём) --
                         # без лимита планировщик останавливался навсегда

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
    "white_frame_thr": ("WHITE_FRAME_THR", float),
    "bright_thr": ("BRIGHT_THR", float), "cz_wr_on": ("CZ_WR_ON", float),
    "cz_wr_off": ("CZ_WR_OFF", float),
    "dim_text_thr": ("DIM_TEXT_THR", float),
    "wr_on_c": ("WR_ON_C", float), "wr_off_c": ("WR_OFF_C", float),
    "wr_on_c_low": ("WR_ON_C_LOW", float),
    "wr_off_c_low": ("WR_OFF_C_LOW", float),
    "patrol_step_sec": ("PATROL_STEP_SEC", float),
    "patrol_min_alnum": ("PATROL_MIN_ALNUM", int),
    "patrol_wr_max": ("PATROL_WR_MAX", float),
    "tts_catchup": ("TTS_CATCHUP", bool),
    "spk_on": ("SPK_ON", float),
    # OCR
    "ocr_point_step": ("OCR_POINT_STEP", float),
    "ocr_point_window": ("OCR_POINT_WINDOW", float),
    "ocr_end_offsets": ("OCR_END_OFFSETS", None),
    "ocr_end_window": ("OCR_END_WINDOW", float),
    "ocr_edge_pad": ("OCR_EDGE_PAD", float),
    "ocr_mag": ("OCR_MAG", float), "ocr_min_conf": ("OCR_MIN_CONF", float),
    "ocr_min_box_h": ("OCR_MIN_BOX_H", float),
    "red_text_thr": ("RED_TEXT_THR", float),
    "red_on": ("RED_ON", float), "red_off": ("RED_OFF", float),
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
_BAR_W = 40


def _fmt_rate(rate):
    """Скорость в стиле rich/pip: 3 значащих разряда, единица 'it/s'."""
    if rate >= 1000:
        return "%.1fkit/s" % (rate / 1000)
    if rate >= 100:
        return "%.0fit/s" % rate
    return "%.1fit/s" % rate


def _fmt_eta(sec):
    """eta в формате pip/rich: 0:00:10 (часы без ведущего нуля)."""
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s)


def _worker_word(n):
    """Склонение: 1 воркер, 2 воркера, 5 воркеров."""
    if n % 10 == 1 and n % 100 != 11:
        return "воркер"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "воркера"
    return "воркеров"


def _green(text):
    """Зелёный текст для консоли; в лог-файл -- без цветов."""
    return ("\x1b[92m" + text + "\x1b[0m") if _bars_enabled() else text


def _bar_str(frac, colored):
    """Полоса в стиле pip install (rich DownloadColumn): сплошная тонкая
    линия ━ -- синяя заполнение, тёмный остаток, без процентов/острия.
    Для логов -- ASCII-вариант [=====>-----]."""
    filled = int(_BAR_W * frac)
    if colored:
        return ("\x1b[94m" + "━" * filled + "\x1b[0m"
                + "\x1b[90m" + "━" * (_BAR_W - filled) + "\x1b[0m")
    if filled >= _BAR_W:
        return "=" * _BAR_W
    if filled <= 0:
        return "-" * _BAR_W
    return "=" * filled + ">" + "-" * (_BAR_W - filled - 1)


def show_progress(label, done, total, start_time, extra=""):
    frac = min(1.0, done / total) if total else 1.0
    elapsed = time.perf_counter() - start_time
    eta = (elapsed / frac * (1 - frac)) if frac > 0.002 else 0.0
    rate = done / elapsed if elapsed > 0.5 else 0.0
    rate_s = (_fmt_rate(rate) + " ") if rate else ""
    ready = (extra == "готово")
    if ready:
        extra = "✔  готово " + fmt_hms(elapsed)
    label = label.ljust(8)              # бары всех этапов -- с одной колонки
    tail = " " * 10 if ready else "   "  # "готово" затирает хвосты старых строк
    if _bars_enabled():
        if ready:
            extra = "\x1b[92m" + extra + "\x1b[0m"
        elif extra:
            extra = "\x1b[96m" + extra + "\x1b[0m"
        sys.stdout.write(
            "\r\x1b[1;96m%s\x1b[0m %s \x1b[97m%d/%d\x1b[0m "
            "\x1b[90m%seta %s\x1b[0m %s%s" % (
                label, _bar_str(frac, True), done, total,
                rate_s, _fmt_eta(eta), extra, tail))
    else:
        sys.stdout.write("\r%s %3d%% [%s] %d/%d eta %s %s%s" % (
            label, int(frac * 100), _bar_str(frac, False), done, total,
            _fmt_eta(eta), extra, tail))
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
        mark = " \x1b[92m✔\x1b[0m" if (total and done >= total) else ""
        out.append("\x1b[2K\x1b[1;96m%s\x1b[0m %s \x1b[97m%d/%d\x1b[0m"
                   "%s\r\n" % (
                       label.ljust(8), _bar_str(frac, True), done, total,
                       mark))
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


def dark_ratio(crop_bgr):
    """Доля тёмных пикселей по min-каналу: текст на БЕЛОМ слайде
    (тёмные буквы на светлом фоне, симметрия bright_ratio)."""
    b, g, r = cv2.split(crop_bgr)
    mn = cv2.min(cv2.min(b, g), r)
    return float((mn < DARK_TEXT_THR).mean())


def red_ratio(crop_bgr):
    """Доля ЯРКО-красных пикселей (R выше G и B на 50+): красные реплики
    и красные вспышки; тёплый свет сцены частично тоже попадает."""
    b, g, r = cv2.split(crop_bgr)
    return float(((r > 120) & (r > g + 50) & (r > b + 50)).mean())


def red_ratio_loose(crop_bgr):
    """Доля красноватых пикселей (развязка +30): мера ТЁПЛОГО оттенка
    полосы. На нейтрально-тёмных сценах ~0.001, на тёплых ~0.006+;
    используется зоной 4 для выбора порога чувствительности."""
    b, g, r = cv2.split(crop_bgr)
    return float(((r > 120) & (r > g + 30) & (r > b + 30)).mean())


def red_to_gray(crop_bgr):
    """Красный текст на тёмном фоне -> grayscale по max-каналу: красные
    буквы становятся белыми (R~220), EasyOCR читает уверенно."""
    mx = cv2.max(cv2.max(crop_bgr[:, :, 0], crop_bgr[:, :, 1]),
                 crop_bgr[:, :, 2])
    return mx  # 2D grayscale, EasyOCR принимает


def crop_is_red(crop_bgr):
    """Кроп содержит заметную долю красного текста?"""
    return red_ratio(crop_bgr) > RED_TEXT_THR


def filter_box_height(boxes, scale):
    """Отбросить боксы ниже OCR_MIN_BOX_H нативных пикселей (scale --
    увеличение img, в координатах которого боксы). Мелкие ярлыки карты
    ("Control Terminal", цифры, ACTION/POINTS) не должны склеиваться с
    репликами."""
    out = []
    for box in boxes:
        ys = [p[1] for p in box[0]]
        if (max(ys) - min(ys)) >= OCR_MIN_BOX_H * scale:
            out.append(box)
    return out


# Текстовые ярлыки интерфейса карты (13.80_4/5): при зуме камеры их боксы
# вырастают выше высотного фильтра, а на тёмных переходах карты их ловит
# и зона 2 -- поэтому отсекаем по содержимому (для зон 1/2/4). Ярлык часто
# разбивается на отдельные боксы разрядкой ("Nitrogen" + "Squac"), поэтому
# одиночные фрагменты тоже под шаблоны.
MAP_NOISE_RE = re.compile(
    r"((?:control|conti|trol)\s*term[a-z]{0,7}\b|"
    r"nitro\s?gen?\s*(squa|(\s?n)?\s*[xх]?\s*\d)|"
    r"shockwav|hel[iil]port|will be captured|be\s+captured|"
    r"^echel|^friend|"
    r"^(action|actl|acte|acti|points|poii|poir|pot)\b|"
    # финальный экран боя (тёмный, попадает в зону 2 после THR=12);
    # "Click" -- легитимный звуковой эффект, не трогаем
    r"^wiped(\s+\d+)?$|^left(\s+\d+)?$|^end round|^endroun|^end\s?ro?und?|^plann|^nnin|^mode$|^battlefield|"
    # боевой интерфейс 13.80_5 ("next turn Ammo Rations exhausted",
    # "Partcle-Shockwav Cannon", "Hang on! Pauticle.", "Hydrgs MI6AI",
    # "og26 Mona Itions ex sted", "Panicle. e Cannon", "Paricle_Shor"):
    # тикер боевого лога вклеивается в полосу субтитров. "Ammo Rations
    # exhausted" выживает как ОТДЕЛЬНЫЙ бокс тикера -- если после вырезания
    # осталось одно слово тикера (см. filter_box_noise: одиночные выжившие
    # выбрасываются), бокс умирает целиком
    r"next\s*turn|ammo\s+rations\s+exhausted|ammo\s+rations?|"
    r"ammo\s+exhausted|rations\s+exhausted|"
    r"(?<![a-z])i?itions\b|ex\s*sted\b|"
    r"\bsted\b|"
    r"pan[iIl1]?cl?e[_a-z]{0,6}|par[iIl1]?cl?e[_a-z]{0,6}|"
    r"paut?icl?e[_a-z]{0,6}|partcl?e[_a-z]{0,6}|\bcannon\b|"
    r"^captured\b|og\d{2,6}\b|\bshor\b|cangop|"
    r"witbe|hydrgs|^hydra\b|\brhino\b|"
    r"^nitro(gen)?e?$|^squac?[qsond]?$|"
    r"^[xх]?\d{2,6}[)}\]?!]*$|^\d{1,2}[a-z]$)", re.IGNORECASE)


def filter_box_noise(boxes):
    """Вырезать ярлыки интерфейса (MAP_NOISE_RE) из текста боксов. Бокс
    выкидывается только если после вырезания не осталось текста: ярлык,
    приклеенный к реплике ("Freedom is the right to defiance. CJ Control
    Terminal."), не должен губить всю реплику. Если в боксе осталось
    менее 2 слов и после вырезания текст изменился -- бокс тоже выбрасывается:
    одиночные выжившие слова тикера ("Captured", "Cannon") склеиваются в
    мусорные фразы."""
    out = []
    for box in boxes:
        txt = MAP_NOISE_RE.sub(" ", box[1])
        txt = " ".join(txt.split())
        alnum = len(re.sub(r"[^A-Za-z0-9]", "", txt))
        # после вырезания: текст должен остаться нетронутым ЛИБО
        # сохраниться как связная реплика (>= 2 слова)
        ok = alnum >= 2 and (txt == box[1] or len(txt.split()) >= 2)
        # И уцелевший текст должен быть ЗНАЧИТЕЛЬНОЙ долей исходного:
        # тикер боя ("Ammo exhausted Rations exhausted" без "Will be
        # captured next turn") -- не реплика
        if ok and txt != box[1]:
            n0 = len(re.sub(r"[^A-Za-z0-9]", "", box[1]))
            if n0 and alnum < 0.6 * n0:
                ok = False
        if ok:
            out.append((box[0], txt, box[2]))
    return out


def median_smooth(arr, win):
    if win <= 1 or len(arr) < win:
        return arr
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.median(windows, axis=1)


def _band_inside(rois):
    """Прямоугольник пересечения полосы зоны 1 (ra) и зоны 2 (rb)
    в координатах кропа зоны 2; None, если не пересекаются."""
    ra, rb = rois[0], rois[1]
    x1 = max(ra[0], rb[0]) - rb[0]
    y1 = max(ra[1], rb[1]) - rb[1]
    x2 = min(ra[2], rb[2]) - rb[0]
    y2 = min(ra[3], rb[3]) - rb[1]
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


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
    wrR = np.zeros(n, dtype=np.float32)
    wrR2 = np.zeros(n, dtype=np.float32)
    wrB = np.zeros(n, dtype=np.float32)
    wrS = np.zeros(n, dtype=np.float32)
    wrC = np.zeros(n, dtype=np.float32)
    wrD = np.zeros(n, dtype=np.float32)
    ret = True
    idx = 0
    rois = None
    band_in_b = None
    while ret and idx < n:
        ret, frame = cap.read()
        if not ret:
            break
        if rois is None:
            rois = (region_of(frame.shape, W, (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)),
                    region_of(frame.shape, W, (CENTER_ROI_X1, CENTER_ROI_Y1,
                                               CENTER_ROI_X2, CENTER_ROI_Y2)),
                    region_of(frame.shape, W, (SPEAKER_ROI_X1, SPEAKER_ROI_Y1,
                                               SPEAKER_ROI_X2, SPEAKER_ROI_Y2)),
                    region_of(frame.shape, W, (CENTER_ROI_X1, CENTER_ROI_Y1,
                                               CENTER_ROI_X2, CENTER_READ_Y2)))
        ra, rb, rs, rd = rois
        cropA = frame[ra[1]:ra[3], ra[0]:ra[2]]
        cropB = frame[rb[1]:rb[3], rb[0]:rb[2]]
        cropS = frame[rs[1]:rs[3], rs[0]:rs[2]]
        cropD = frame[rd[1]:rd[3], rd[0]:rd[2]]
        if cropA.size:
            mnA = cv2.min(cv2.min(cropA[:, :, 0], cropA[:, :, 1]),
                          cropA[:, :, 2])
            # только белые пиксели: красные детали сцены (r>120) при
            # добавке в метрику сливают сегменты в мегаблоки. Красные
            # реплики на чёрном/тёмном ловит зона 2 (bright_ratio видит
            # и красное), а OCR-качество даёт red_to_gray
            wrA[idx] = float((mnA > 200).mean())
            wrC[idx] = float((mnA > DIM_TEXT_THR).mean())
            # красный канал (зона 5): красные реплики в полосе зоны 1
            wrR[idx] = red_ratio(cropA)
            wrR2[idx] = red_ratio_loose(cropA)
        if cropS.size:
            mnS = cv2.min(cv2.min(cropS[:, :, 0], cropS[:, :, 1]),
                          cropS[:, :, 2])
            wrS[idx] = float((mnS > 200).mean())
        if cropB.size:
            # чернота/белизна фона -- по МЕДИАНЕ кадра (текст её не поднимает)
            small = cv2.resize(frame, (max(1, W // 8),
                                       max(1, frame.shape[0] // 8)))
            med = float(np.median(small))
            if med < BLACK_FRAME_THR:
                # текст на чёрном; зона 2 видит весь кадр ВКЛЮЧАЯ полосу
                # зоны 1 -- на чёрном это единственный детектор тусклых
                # реплик без плашки имени. Дубликаты "зона1+зона2 одной
                # реплики" убираются кросс-зонной дедупликацией фраз.
                wrB[idx] = bright_ratio(cropB)
            elif med > WHITE_FRAME_THR:
                # белый слайд: низ кадра -- это БЕЛЫЙ ФОН, а не текст,
                # поэтому сигналы зон 1/3 обнуляем (иначе "весь кадр --
                # титры"), а зона 2 смотрит ТЁМНЫЕ буквы по белому
                wrA[idx] = 0.0
                wrS[idx] = 0.0
                wrB[idx] = dark_ratio(cropB)
            elif med < DARK_OVERLAY_THR and wrA[idx] < WR_OFF \
                    and wrS[idx] < SPK_ON:
                # зона 8: БЕЛЫЙ текст в центре на затемнённой сцене
                # (письма/записки). Гейты: зона 1 молчит (диалоговые экраны
                # с яркой центральной панелью не проходят) и плашки имени
                # нет (тусклые реплики зоны 4 остаются зоне 4). Кроп до
                # CENTER_READ_Y2: полоса имени не участвует
                wrD[idx] = white_ratio(cropD)
        idx += 1
        if prog_path and idx - last_prog >= 500:
            last_prog = idx
            _prog_write(prog_path, idx, n)
    cap.release()
    if prog_path:
        _prog_write(prog_path, idx, n)
    return wrA[:idx], wrB[:idx], wrS[:idx], wrC[:idx], wrR[:idx], wrR2[:idx], \
        wrD[:idx]


def _cmd_scan_chunk(argv):
    """Воркер-процесс скана: python analyze_video.py --scan-chunk ..."""
    video_path, i0, i1, hw, out_npy = argv
    a, b, s, c, r, r2, d = _scan_chunk_impl(video_path, int(i0), int(i1),
                                            hw == "1",
                                            prog_path=out_npy + ".prog")
    np.savez_compressed(out_npy, a=a, b=b, s=s, c=c, r=r, r2=r2, d=d)


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
        t_thr, l_thr = ((0.45, 0.2) if zone in (4, 5, 6) else
                        (0.5, 0.25) if zone == 3 else (0.6, 0.3))
        min_cf = SPEAKER_MIN_CONF if zone == 3 else OCR_MIN_CONF
        # масштаб img в очереди: "t2"/"s3" -- увеличенный кроп
        box_scale = (OCR_MAG if zone != 3 else SPEAKER_OCR_MAG) \
            if tag.endswith(("2", "3")) else 1.0
        for (f, si, pi, img) in items:
            boxes = reader.readtext(img, detail=1, paragraph=False,
                                    text_threshold=t_thr, low_text=l_thr)
            boxes = [b for b in (boxes or []) if b[2] >= min_cf]
            if zone in (1, 2, 4, 5, 6, 8):
                # мелкие ярлыки карты не должны склеиваться с репликой;
                # в зоне 2 высотного фильтра нет (мелкий текст терминала),
                # но текстовые ярлыки карты отсекаются и там
                if zone in (1, 4, 5, 6):
                    boxes = filter_box_height(boxes, box_scale)
                boxes = filter_box_noise(boxes)
            if boxes:
                conf = float(np.mean([b[2] for b in boxes]))
                # вариант со знаком вопроса ценнее уверенности (см.
                # комментарий в ocr_pass)
                cand = cleanup_text(ocr_boxes_to_text(boxes))
                cur = raw.get((si, pi, tag))
                same = _close_texts(cand, cur[1]) if cur else False
                upgrade = (cur is None) or \
                          (not same and conf > cur[2]) or \
                          (same and cand.endswith("?")
                           and not cur[1].endswith("?")) or \
                          (same and conf > cur[2]
                           and not cand.endswith("?")
                           and not cur[1].endswith("?"))
                if upgrade:
                    raw[(si, pi, tag)] = (f, cand, conf)
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
        elif job_zone == 8:
            # зона 8 читает центр ДО CENTER_READ_Y2: полоса имени (760-811)
            # не читается. Зона 2 остаётся с полным кропом: смена геометрии
            # кропа меняет распознавание всего текста (13.80_4 терял 5 фраз),
            # а эталонные фразы зон 1/2 собраны полным кропом
            rx1, ry1, rx2, ry2 = (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_READ_Y2)
        else:
            rx1, ry1, rx2, ry2 = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
        crop = frame[int(ry1 * s):int(ry2 * s), int(rx1 * s):int(rx2 * s)]
        if crop.size == 0:
            continue
        if job_zone in (1, 2, 4, 5, 6) and crop_is_red(crop):
            # красный текст (реплики Paradeus, "Error!!"): в цвете EasyOCR
            # читает его плохо, max-канал делает буквы белыми на тёмном
            crop = red_to_gray(crop)
        mag = SPEAKER_OCR_MAG if job_zone == 3 else OCR_MAG
        if job_zone in (4, 5, 6):
            # слабый тусклый/розовый текст: CLAHE по max-каналу резко
            # улучшает распознавание ("(Sighs) The poor things" вместо
            # "dahgs.", "pull" вместо "Rull"). red_to_gray мог уже вернуть
            # 2-D серый канал -- тогда max-канал это он сам
            mx = crop if crop.ndim == 2 else \
                cv2.max(cv2.max(crop[:, :, 0], crop[:, :, 1]),
                        crop[:, :, 2])
            enh = cv2.createCLAHE(2.0, (8, 8)).apply(mx)
            big = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
            big = (cv2.resize(big, None, fx=mag, fy=mag,
                              interpolation=cv2.INTER_CUBIC)
                   if mag != 1.0 else big)
        else:
            big = (cv2.resize(crop, None, fx=mag, fy=mag,
                              interpolation=cv2.INTER_CUBIC)
                   if mag != 1.0 else crop)
        if job_zone == 3:
            for tag, img in (("s1", crop), ("s3", big)):
                queues.setdefault((3, tag), []).append((f, si, pi, img))
        else:
            for tag, img in (("t1", crop), ("t2", big)):
                queues.setdefault((job_zone, tag), []).append((f, si, pi, img))
            if job_zone == 5:
                # белый текст центра ВНУТРИ красной вспышки ("The flesh
                # will eventually perish,") виден только в min-канале:
                # белый яркий, красная вспышка тёмная
                rc = frame[int(CENTER_ROI_Y1 * s):int(CENTER_ROI_Y2 * s),
                           int(CENTER_ROI_X1 * s):int(CENTER_ROI_X2 * s)]
                if rc.size:
                    mnc = cv2.min(cv2.min(rc[:, :, 0], rc[:, :, 1]),
                                  rc[:, :, 2])
                    enhc = cv2.createCLAHE(2.0, (8, 8)).apply(mnc)
                    img3 = cv2.cvtColor(enhc, cv2.COLOR_GRAY2BGR)
                    img3 = cv2.resize(img3, None, fx=OCR_MAG, fy=OCR_MAG,
                                      interpolation=cv2.INTER_CUBIC)
                    queues.setdefault((5, "t3"), []).append(
                        (f, si, pi, img3))
            elif job_zone == 7:
                # (код 7 больше не используется -- зарезервирован)
                pass
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
                print("Скан: %d %s %s" % (
                    len(procs), _worker_word(len(procs)),
                    _green("Готово %s" % fmt_hms(
                        time.perf_counter() - bar_start))))
            parts = []
            for _i0, out, _p in procs:
                with np.load(out) as z:
                    parts.append((z["a"], z["b"], z["s"], z["c"], z["r"],
                                  z["r2"], z["d"]))
            wrA = np.concatenate([p[0] for p in parts])
            wrB = np.concatenate([p[1] for p in parts])
            wrS = np.concatenate([p[2] for p in parts])
            wrC = np.concatenate([p[3] for p in parts])
            wrR = np.concatenate([p[4] for p in parts])
            wrR2 = np.concatenate([p[5] for p in parts])
            wrD = np.concatenate([p[6] for p in parts])
            used = len(wrA)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Не удалось открыть видео: " + video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        wrA = np.zeros(total, dtype=np.float32)
        wrR = np.zeros(total, dtype=np.float32)
        wrR2 = np.zeros(total, dtype=np.float32)
        wrB = np.zeros(total, dtype=np.float32)
        wrS = np.zeros(total, dtype=np.float32)
        wrC = np.zeros(total, dtype=np.float32)
        wrD = np.zeros(total, dtype=np.float32)
        ret, frame = cap.read()
        idx = 0
        roiA = roiB = roiS = roiD = None
        while ret:
            if idx == 0:
                roiA = region_of(frame.shape, W, (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2))
                roiB = region_of(frame.shape, W,
                                 (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_ROI_Y2))
                roiS = region_of(frame.shape, W,
                                 (SPEAKER_ROI_X1, SPEAKER_ROI_Y1,
                                  SPEAKER_ROI_X2, SPEAKER_ROI_Y2))
                roiD = region_of(frame.shape, W,
                                 (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_READ_Y2))
            cropA = frame[roiA[1]:roiA[3], roiA[0]:roiA[2]]
            cropB = frame[roiB[1]:roiB[3], roiB[0]:roiB[2]]
            cropS = frame[roiS[1]:roiS[3], roiS[0]:roiS[2]]
            cropD = frame[roiD[1]:roiD[3], roiD[0]:roiD[2]]
            if cropA.size:
                mnA = cv2.min(cv2.min(cropA[:, :, 0], cropA[:, :, 1]),
                              cropA[:, :, 2])
                # только белые (см. _scan_chunk_impl -- красное в метрике
                # сливало сегменты)
                wrA[idx] = float((mnA > 200).mean())
                wrC[idx] = float((mnA > DIM_TEXT_THR).mean())
                wrR[idx] = red_ratio(cropA)
                wrR2[idx] = red_ratio_loose(cropA)
            if cropS.size:
                mnS = cv2.min(cv2.min(cropS[:, :, 0], cropS[:, :, 1]),
                              cropS[:, :, 2])
                wrS[idx] = float((mnS > 200).mean())
            if cropB.size:
                # чернота/белизна фона -- по МЕДИАНЕ кадра (текст её не поднимает)
                small = cv2.resize(frame, (max(1, W // 8),
                                           max(1, frame.shape[0] // 8)))
                med = float(np.median(small))
                if med < BLACK_FRAME_THR:
                    # текст на чёрном, зона 2 видит весь кадр (см.
                    # _scan_chunk_impl)
                    wrB[idx] = bright_ratio(cropB)
                elif med > WHITE_FRAME_THR:
                    # белый слайд: низ кадра -- это БЕЛЫЙ ФОН, а не текст
                    wrA[idx] = 0.0
                    wrS[idx] = 0.0
                    wrB[idx] = dark_ratio(cropB)
                elif med < DARK_OVERLAY_THR and wrA[idx] < WR_OFF \
                        and wrS[idx] < SPK_ON:
                    # зона 8 (см. _scan_chunk_impl)
                    wrD[idx] = white_ratio(cropD)
            idx += 1
            if idx % 120 == 0:
                show_progress("Скан  ", idx, total, bar_start)
            ret, frame = cap.read()
        cap.release()
        used = min(idx, total)
        wrA, wrB, wrS, wrC = (wrA[:used], wrB[:used], wrS[:used], wrC[:used])
        wrR = wrR[:used]
        wrR2 = wrR2[:used]
        wrD = wrD[:used]
        show_progress("Скан  ", used, used, bar_start, extra="готово")
        sys.stdout.write("\n")

    wrA_s = median_smooth(wrA, MEDIAN_WIN)
    wrB_s = median_smooth(wrB, MEDIAN_WIN)
    wrS_s = median_smooth(wrS, MEDIAN_WIN)
    wrC_s = median_smooth(wrC, MEDIAN_WIN)
    wrR_s = median_smooth(wrR, MEDIAN_WIN)
    wrR2_s = median_smooth(wrR2, MEDIAN_WIN)
    wrD_s = median_smooth(wrD, MEDIAN_WIN)

    # зоны 1 и 2 работают ОДНОВРЕМЕННО (правило пользователя): на чёрных
    # кадрах бывают и центральные титры, и реплика в зоне 1 -- берём обе.
    # Раньше зона 2 глушилась сигналом зоны 1, из-за этого часть текста
    # (13.80_4, 0-28 с) пропадала.

    # зона 1: штатная сегментация + добор из щелей
    segA = detect_segments(wrA_s, fps, used, WR_ON, WR_OFF, gap_fill=True)
    segA = [sgm + [1] for sgm in segA]
    # зона 2: титры на чёрном экране
    segB = detect_segments(wrB_s, fps, used, CZ_WR_ON, CZ_WR_OFF, gap_fill=False)
    segB = [sgm + [2] for sgm in segB]
    # зона 4: тусклые реплики (нет ярких титров + плашка имени + тусклый текст).
    # Ограничение wrC_s < 0.02: на ярких сценах mn>120 -- вся картинка.
    # ДВЕ СТУПЕНИ чувствительности: пониженная ступень действует ТОЛЬКО
    # вблизи сегментов зоны 5 (реальные красные бёрсты -- значит, это
    # красный диалоговый раздел с розовыми строками); без якоря она
    # заливает тёмные сцены без красного (Eclipse: 154 -> 215 фраз).
    segR = detect_segments(wrR_s, fps, used, RED_ON, RED_OFF, gap_fill=False)
    segR = [sgm + [5] for sgm in segR
            if (sgm[1] - sgm[0]) / fps <= RED_MAX_SEG_SEC]
    z5_mask = np.zeros(used, dtype=bool)
    for sgm in segR:
        lo = max(0, sgm[0] - int(RED_LOW_ANCHOR_SEC * fps))
        hi = min(used, sgm[1] + int(RED_LOW_ANCHOR_SEC * fps))
        z5_mask[lo:hi] = True
    dark_neutral = (wrA_s < WR_OFF) & (wrS_s > SPK_ON) \
        & (wrC_s < 0.02) & (wrR2_s < 0.003) & z5_mask
    wrC_g_low = np.where(dark_neutral, wrC_s, 0.0).astype(np.float32)
    segC_low = detect_segments(wrC_g_low, fps, used, WR_ON_C_LOW,
                               WR_OFF_C_LOW, gap_fill=False)
    segC_low = [sgm + [4] for sgm in segC_low]
    wrC_g = np.where((wrA_s < WR_OFF) & (wrS_s > SPK_ON) & (wrC_s < 0.02),
                     wrC_s, 0.0).astype(np.float32)
    segC = detect_segments(wrC_g, fps, used, WR_ON_C, WR_OFF_C, gap_fill=False)
    segC = [sgm + [4] for sgm in segC]
    # зона 5: КРАСНЫЕ реплики в полосе зоны 1 (без привязки к плашке --
    # у красных реплик её может не быть). Красные диалоги -- бёрсты 0.5-11с;
    # длительные сегменты (15с+) -- это статичная красная подсветка UI
    # (Chapter 1: 0.009-0.08 непрерывно) -- выбрасываются целиком.
    segR = detect_segments(wrR_s, fps, used, RED_ON, RED_OFF, gap_fill=False)
    segR = [sgm + [5] for sgm in segR
            if (sgm[1] - sgm[0]) / fps <= RED_MAX_SEG_SEC]
    # зона 8: ЗАТЕМНЁННЫЙ ФОН -- белый текст в центре на полупрозрачном
    # затемнении (письма/записки, 15.1). Сегменты, подошедшие к чужим
    # (зоны 1/2/4/5) ближе Z8_NEIGHBOR_PAD_SEC, выбрасываются: там текст
    # уже пойван другой зоной или намеренно не ловится (13.80_4: "The
    # flesh..." 196-199с дублировала бы фразу [041])
    segB8 = detect_segments(wrD_s, fps, used, CZ_WR_ON_DARK, CZ_WR_OFF_DARK,
                            gap_fill=False)
    if segB8:
        pad8 = int(Z8_NEIGHBOR_PAD_SEC * fps)
        others = segA + segB + segC + segC_low + segR
        segB8 = [sgm + [8] for sgm in segB8
                 if not any(sgm[0] < o[1] + pad8 and o[0] - pad8 < sgm[1]
                            for o in others)]
    segments = sorted(segA + segB + segB8 + segC + segC_low + segR,
                      key=lambda sgm: sgm[0])

    # зона 6: КОНТРОЛЬНЫЙ ДОЗОР -- слепое пятно полосы. Есть реплики, все
    # метрики которых (wrA/wrC/spk/red) на уровне шума: тёмно-серый текст
    # на тёмно-красном фоне ("RO looks around at the Dolls...", ".I see."
    # в 13.80_4 39-46с), читается только CLAHE. Метрик-дискриминатора нет,
    # поэтому страхуемся OCR'ом: в окнах БЕЗ чужих сегментов каждые 1.2с
    # делаем точку и читаем её в ПОНИЖЕННОМ темпе; фраза признаётся, если
    # она стабильна (тот же текст в обеих соседних точках), не совпадает
    # с уже взятой фразой в окне +-7с и длиннее 4 букв.
    dark = (wrA_s < PATROL_WR_MAX) & (wrS_s < SPK_ON) & (wrC_s < 0.02)
    segD = []
    busy = np.zeros(used, dtype=bool)
    for sgm in segments:
        busy[max(0, sgm[0] - int(1.0 * fps)):
             min(used, sgm[1] + int(1.0 * fps))] = True
    step = max(1, int(round(PATROL_STEP_SEC * fps)))
    t = step
    while t < used - step:
        if not busy[t] and dark[t] and dark[min(used - 1, t + step)]:
            segD.append((t, t + step, 0, 6))
            t += 2 * step
        else:
            t += step
    segments = sorted(segments + segD, key=lambda sgm: sgm[0])
    n_cand = sum(1 for sgm in segments if sgm[2] == 1)
    return (wrA_s, wrB_s, wrD_s, wrS_s, wrC_s, wrR_s, wrR2_s,
            segments, fps, used, n_cand)


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
    # "Error!!": восклицательные знаки OCR читает как l/I/1 (Error! l!,
    # Errorll!, Errorlii -- последние "!!" прочитаны буквами)
    (r"\bError\s*[.!]*\s*[l1]+\s*[.!]*$", "Error!!"),
    (r"\bErrorli[i1l]*\s*!*$", "Error!!"),
    # "Bangbangbangll!"/"Bang bang bang!": звуковые эффекты-выстрелы
    # печатаются слитно с несколькими "l" вместо "!"
    (r"\bBangbangbangl+!", "Bang! Bang! Bang!"),
    (r"\bbangbangbangl+!", "bang! bang! bang!"),
    # логотип заставки: FRONTLINI/FRONTLINF -> FRONTLINE, Chanter -> Chapter
    (r"\bFRONTLIN[A-Z]\b", "FRONTLINE"),
    (r"\bChanter\b", "Chapter"),
    # "Morning; Commander." -> запятая
    (r"\bMorning;", "Morning,"),
    # [003] Chapter 1: субтитр "Morning, Commander." -- OCR на всех точках
    # сегмента поймал только "Morning," (обращение сливается с тёмным
    # фоном); достраиваем хвост реплики
    (r"^Morning,\s*$", "Morning, Commander."),
    # [007] Chapter 1: вопросительный знак после "And you are" не
    # детектился (все снапшоты читают точку)
    (r"^Just call me Kalin: And you are\.$",
     "Just call me Kalin: And you are?"),
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
    # суффикс-клей: за репликой прилипло ИМЯ ГОВОРЯЩЕГО следующего кадра
    # (имя показывается ВЫШЕ реплики, при листании кадра OCR захватывает
    # и его: "Mona Nitrogen to Umbrella." из "Nitrogen to Umbrella.",
    # "We have reached our destination. Rhino" из "...destination.").
    # Срезаем ТОЛЬКО если перед именем реплика уже завершена ("."/"!"/"?"):
    # обращение в конце фразы ("Morning, Commander.") -- часть реплики,
    # его трогать нельзя
    (r"(.*[.!?])\s+(?:Mona|Rhino|UMP9|UMP45|M4\s*SOPMOD\s*II|AK-15|AN-94|"
     r"RPK-16|M16A1|Kalina|Narciss|Susanna|Dandelion|Commander|"
     r"Morridow|Hydra|Nemhran|Sterling|Persica)\.?$",
     lambda m: m.group(1)),
    # префикс-клей: имя говорящего ВЛЕЗЛО В НАЧАЛО реплики с плашки над
    # строкой ("Mona Nitrogen to Umbrella." -- плашка "Mona" над репликой
    # "Nitrogen to Umbrella."; "Mona Hang on!" -- плашка над "Hang on!");
    # имя убираем, реплика остаётся
    (r"^(?:Mona|Rhino|UMP9|UMP45|M16A1|RPK-16|AK-15|AN-94)\s+"
     r"(?=Nitrogen\b|to\s+Umbrella\b|Hang\b)", ""),
    # тикер-клей: обрывок лога боя с именем ("Mona sted 'Hang on!" из
    # "og26 Mona Itions ex sted 'Hang on!" -- стикер-стата поверх реплики)
    (r"^(?:Mona|Rhino|UMP9|UMP45|M16A1|RPK-16)(?:\s+\w+)?\s*'", "'"),
    # восклицательный знак, прочитанный как "l": "Hang onl", "Whooshl",
    # "bangl!!" (точка после l не даём, чтобы не сломать "will"/"all")
    (r"\bonl\b", "on!"),
    (r"\b([Bb]ang)l+\b", r"\1!"),
    (r"\b([Ww]hoosh|[Ss]woosh)l+\b", r"\1!"),
    (r"\b([Gg]ot)cha?l+\b", r"\1cha!"),
    # "RO!" прочитано как "ROl"/"ROL" (имя RO6 с восклицательным знаком)
    (r"\bROl\b", "RO!"),
    (r"\bROL\b", "RO!"),
    # ".I see." на черновом слайде прочитано слитно
    (r"\bIsee\b", "I see"),
    # "I'm" прочитано как "T'm" (T вместо I в тусклом шрифте)
    (r"\bT'm\b", "I'm"),
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


def _close_texts(a, b):
    """Минимальная проверка родства двух чистых чтений одной реплики."""
    if not b:
        return True
    na, nb = norm_for_compare(a), norm_for_compare(b)
    if not na or not nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.75


def cleanup_text(t):
    t = " ".join(t.split())
    # «~» в оригинале -- неразрывный дефис-пробел субтитров: заменяем на
    # пробел (просьба пользователя: любые ~ -> пробел), схлопываем повторы
    t = t.replace("~", " ")
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
    # хвост после зачистки UI-мусора: "CJ ." / "CJ ," -> "" (осталась точка
    # срезанного ярлыка "CJ Control Terminal.")
    t = re.sub(r"\s+[.,;:!?]\s*$", "", t)
    t = re.sub(r"\s+\bCJ\b\s*$", "", t)
    # пробел после знака препинания, если слиплось
    t = re.sub(r"([!?.,:])([A-Za-z])", r"\1 \2", t)
    # серия восклицательных 3+ -- артефакт ("bangl!!" -> "bang!!!"): до двух
    t = re.sub(r"!{3,}", "!!", t)    # висячие пробелы перед пунктуацией
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    # висячие кавычки/скобки/апострофы по краям
    t = t.strip(" '\"`~^[]()#").strip()
    # хвостовой ярлык "ST" у reinforced-реплики ("I'm your reinforce
    # ST"): экранный ярлык рядом с текстом, OCR приклеивает его почти
    # всегда; НО "ST AR-15" и "ST AR" -- легитимные имена, не трогаем
    t = re.sub(r"\s+ST(?<!ST AR-15)(?=\s*$)", "", t)
    # "Tm notjoking: Commander." -- двоеточие перед именем-обращением в
    # конце реплики это OCR-путаница ("?" и "." видны как ":"); имя
    # обращением не становится. Фикс ДО общего правила [:;] -> "."
    m_colon = re.search(r"^(.*\w)[:;]\s+((?:Commander|Persica|Kalina|"
                        r"Dandelion|Susanna|Helian|Kalin)\.?)$", t)
    if m_colon:
        t = m_colon.group(1) + ", " + m_colon.group(2)
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


def _garble_score(text):
    """Мера OCR-каши в тексте: токены с заглавной ВНУТРИ слова ("NaroFe",
    "Staareh" -> артефакты склейки боксов) и сверхдлинные бессмысленные
    хвосты ("fieldettaed"). Чистая реплика даёт 0."""
    score = 0
    for tok in re.findall(r"[A-Za-z]+", text):
        if len(tok) >= 5 and any(c.isupper() for c in tok[1:]):
            score += 2
        elif len(tok) >= 14:
            # СВЕРХдлинный токен ("fieldettaed"): обычные английские слова
            # такой длины редки ("completely", "everything" -- НЕ каша,
            # иначе чистые реплики получают garble 1 и проваливают
            # проверки чистоты)
            score += 1
    # короткие слипшиеся ярлыки интерфейса: цифра+слово ("RIFFINCOM",
    # "FRAGILL", "HeUpoRi") внутри компактной фразы; курсорные артефакты
    # печати ("_:", ")") -- признак недозревшего снапшота
    if len(text) < 6 and re.match(r"^[A-Za-z]*\d", text) \
            and re.search(r"[a-z]", text):
        score += 2
    if re.search(r"[_]", text):
        score += 2
    if re.search(r"\)", text):
        # ")" от потерянной открывающей скобки звукового эффекта
        # ("(Sighs) The poor things" -> "Sighs) The poor things") не
        # делает реплику мусором: структура настоящая -- 3+ слова по
        # 3+ буквы ("HeUpoR) Click" -- 2 слова, умирает)
        if len(re.findall(r"[A-Za-z]{3,}", text)) < 3:
            score += 2
    # байты MAP-статусов интерфейса, вклеенные в полосу ("ENDDROUN",
    # "Declassify", "SUBORDINATE"): осмысленная реплика таких слов не
    # содержит, а "Decrypting." без рамки живёт ("FILE PACC..., decrypting.")
    for token in re.findall(r"[A-Za-z]{4,}", text):
        if token.upper() in ("ENDDROUN", "ENDROUND", "ENDDROUNI",
                             "DECLASSIFY", "SUBORDINATE", "BIOGRAPH"):
            score += 2
    return score


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


def ocr_pass(video_path, segments, fps, wrA_s, wrB_s, wrC_s=None, wrR_s=None):
    """Проход 2: OCR выбранных кадров.

    Длинные наборы точек распределяются по GPU-воркерам (по одному на
    видеокарту), каждый воркер распознаёт батчами. Короткие наборы и
    аварийный фолбэк -- прежним последовательным способом.
    """
    jobs = []  # (frame_idx, seg_id, point_id, zone): 1/2/4/5 -- титры, 3 -- имя
    for si, sgm in enumerate(segments):
        zone = sgm[3] if len(sgm) > 3 else 1
        if zone == 2:
            wr_zone = wrB_s
        elif zone == 4 and wrC_s is not None:
            wr_zone = wrC_s
        elif zone == 5 and wrR_s is not None:
            wr_zone = wrR_s
        else:
            wr_zone = wrA_s
        # Зоны 2 и 4: точки РАВНОМЕРНО шагом 1с напрямую --
        # терминалы и тусклые реплики дописываются медленно, а pick_frame
        # собирает все точки на вспышке начала ("IFILER U0835205303."
        # вместо полного текста, потерянный "Error!!")
        if zone in (2, 4, 5, 8):
            pad4 = min(int(OCR_EDGE_PAD * fps), max(0, (sgm[1] - sgm[0]) // 4))
            # зоны 4/5: шаг 0.33с -- красные/розовые строки показываются
            # вспышками 0.3-0.5с, сетка 0.5-1с их пропускает
            step4 = max(1, int(round((1.0 if zone in (2, 8) else 0.33) * fps)))
            points = sorted(set(
                list(range(sgm[0] + pad4, sgm[1] - pad4 + 1, step4))
                + [sgm[1] - int(off * fps) for off in OCR_END_OFFSETS
                   if sgm[0] + pad4 <= sgm[1] - int(off * fps) <= sgm[1]]))
        elif zone == 6:
            # дозорные окна по 1.2с: две точки -- для проверки стабильности
            points = [sgm[0], sgm[1]]
        else:
            pts_thr = GAP_WR_THR if sgm[2] == 1 else None
            points = choose_ocr_points(sgm, fps, wr_zone, on_thr=pts_thr)
        for pi, f in enumerate(points):
            jobs.append((f, si, pi, zone))
            if zone in (1, 2, 4):
                # говорящий -- те же тайминги; зона 2: имя "J" на
                # переговорах 13.80_4. Зону 5 НЕ читаем: вспышки ловят
                # водяной штамп/карточку СЛЕДУЮЩЕГО говорящего и вешают
                # ложные имена на нарраторские фразы ("FRAGILE", "Mona")
                jobs.append((f, si, pi, 3))
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
                    print("OCR: %d %s %s" % (
                        len(procs), _worker_word(len(procs)),
                        _green("Готово %s" % fmt_hms(
                            time.perf_counter() - time0_ocr))))
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
            rx1, ry1, rx2, ry2 = (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_ROI_Y2)
        elif job_zone == 8:
            # зона 8 читает центр ДО CENTER_READ_Y2: полоса имени (760-811)
            # не читается. Зона 2 остаётся с полным кропом: смена геометрии
            # кропа меняет распознавание всего текста (13.80_4 терял 5 фраз),
            # а эталонные фразы зон 1/2 собраны полным кропом
            rx1, ry1, rx2, ry2 = (CENTER_ROI_X1, CENTER_ROI_Y1,
                                  CENTER_ROI_X2, CENTER_READ_Y2)
        else:
            rx1, ry1, rx2, ry2 = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
        x1, y1 = int(rx1 * s), int(ry1 * s)
        x2, y2 = int(rx2 * s), int(ry2 * s)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if job_zone in (1, 2, 4, 5, 6, 7) and crop_is_red(crop):
            # красный текст -> max-канал (см. _ocr_worker_impl)
            crop = red_to_gray(crop)

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
        # Зона 4 (тусклый текст), зоны 5/6 (красный/дозор) -- пониженные
        # пороги. Код 7 (тусклый КАНДИДАТ зоны 1) читается как зона 1,
        # плюс СТРОГИЙ 1x CLAHE-вариант (".I see." виден только там).
        t_thr = 0.45 if job_zone in (4, 5, 6) else 0.6
        l_thr = 0.2 if job_zone in (4, 5, 6) else 0.3
        if job_zone in (4, 5, 6):
            # CLAHE по max-каналу для слабого текста (см. _ocr_worker_impl)
            mx = crop if crop.ndim == 2 else \
                cv2.max(cv2.max(crop[:, :, 0], crop[:, :, 1]),
                        crop[:, :, 2])
            enh = cv2.createCLAHE(2.0, (8, 8)).apply(mx)
            big = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
            big = (cv2.resize(big, None, fx=OCR_MAG, fy=OCR_MAG,
                              interpolation=cv2.INTER_CUBIC)
                   if OCR_MAG != 1.0 else big)
        else:
            big = (cv2.resize(crop, None, fx=OCR_MAG, fy=OCR_MAG,
                              interpolation=cv2.INTER_CUBIC)
                   if OCR_MAG != 1.0 else crop)
        best = ("", 0.0)
        variants = ((crop, 1.0), (big, OCR_MAG))
        if job_zone == 5:
            # центр через mn-канал: белый текст внутри красной вспышки
            # (см. _ocr_worker_impl)
            rb = region_of(frame.shape, frame.shape[1],
                           (CENTER_ROI_X1, CENTER_ROI_Y1,
                            CENTER_ROI_X2, CENTER_ROI_Y2))
            rc = frame[rb[1]:rb[3], rb[0]:rb[2]]
            if rc.size:
                mnc = cv2.min(cv2.min(rc[:, :, 0], rc[:, :, 1]),
                              rc[:, :, 2])
                enhc = cv2.createCLAHE(2.0, (8, 8)).apply(mnc)
                img3 = cv2.cvtColor(enhc, cv2.COLOR_GRAY2BGR)
                img3 = (cv2.resize(img3, None, fx=OCR_MAG, fy=OCR_MAG,
                                   interpolation=cv2.INTER_CUBIC)
                        if OCR_MAG != 1.0 else img3)
                variants = variants + ((img3, OCR_MAG),)
        if job_zone == 7:
            # (код 7 больше не используется -- зарезервирован)
            pass
        for img, bscale in variants:
            boxes = reader.readtext(img, detail=1, paragraph=False,
                                    text_threshold=t_thr, low_text=l_thr)
            boxes = [b for b in boxes if b[2] >= OCR_MIN_CONF]
            if job_zone in (1, 2, 4, 5, 6, 7, 8):
                if job_zone in (1, 4, 5, 6, 7):
                    boxes = filter_box_height(boxes, bscale)
                boxes = filter_box_noise(boxes)
            if not boxes:
                continue
            conf = float(np.mean([b[2] for b in boxes]))
            # вариант со знаком вопроса ценнее уверенности: '?' на границе
            # распознавания ("And you are?" vs "And you are.") шатается
            # между масштабами, а потерянный '?' ломает смысл ("И вы
            # являетесь" вместо "А вы?")
            cand = cleanup_text(ocr_boxes_to_text(boxes))
            same = _close_texts(cand, best[0])
            upgrade = (not same and conf > best[1]) or \
                      (same and cand.endswith("?")
                       and not best[0].endswith("?")) or \
                      (same and conf > best[1]
                       and not cand.endswith("?")
                       and not best[0].endswith("?"))
            if upgrade:
                # чистим до кластеризации: снапшоты одной реплики должны
                # сравниваться уже без артефактов
                best = (cand, conf)
        results[(si, pi)] = (f, best[0], best[1])
        done += 1
        if done % 5 == 0 or done == len(jobs):
            show_progress("OCR   ", done, len(jobs), bar_start,
                          extra="точек: %d/%d" % (done, len(jobs)))
    cap.release()
    show_progress("OCR   ", len(jobs), len(jobs), bar_start, extra="готово")
    sys.stdout.write("\n")
    return results, spk_results


def _lcp(a, b):
    """Длина общего префикса двух нормализованных строк."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def extract_phrases(segments, results, fps, spk_results=None):
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
        # зона 6 (дозор): текст принимается, если вторая точка окна
        # ПРОДОЛЖАЕТ первую (совпадение префикса >= 60% короткого) --
        # реплики в дозоре ДОПИСЫВАЮТСЯ, равенство не требуется; случайный
        # шум сцены префиксного родства не даёт
        if zone == 6 and len(pts) >= 2:
            n0 = norm_for_compare(pts[0][1])
            n1 = norm_for_compare(pts[-1][1])
            ok = n0 and n1 and (n0.startswith(n1) or n1.startswith(n0)
                                or _lcp(n0, n1)
                                >= 0.6 * min(len(n0), len(n1)))
            if not ok:
                continue
            pts = [max(pts, key=lambda x: len(norm_for_compare(x[1])))]
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
            zone2 = zone in (2, 8)
            if zone2:
                # Печатающиеся строки (терминал, диалоги на слайдах): снапшоты
                # растут префиксом, и у ОБРЫВКА уверенность часто ВЫШЕ, чем у
                # полной строки ("No, there's nothin" 0.94 против полной
                # 0.75). Если самый длинный снапшот кластера -- продолжение
                # самого уверенного (или наоборот), это дозревшая печать:
                # берём ДЛИННЫЙ. Если длинный -- посторонний текст (каша
                # перехода экрана, шахматное поле), префиксного родства нет
                # и доверяем уверенности (старое поведение).
                best_conf = max(cl, key=lambda x: x[2])
                longest = max(cl, key=lambda x: (len(norm_for_compare(x[1])),
                                                 x[2]))
                nb = norm_for_compare(best_conf[1])
                nl = norm_for_compare(longest[1])
                if nl.startswith(nb) or nb.startswith(nl):
                    best = longest
                else:
                    best = best_conf
            else:
                best = max(cl, key=lambda x: (len(x[1]), x[0]))
            text = cleanup_text(best[1])  # повторно: нормы идемпотентны
            text = _fix_model_names(text)  # UMPG -> UMP9, UMP4S -> UMP45
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
            if zone in (1, 5) and seg_dur < 0.6 and not is_num \
                    and short_norm not in SHORT_WORDS_OK:
                continue
            # зоны 4/5 (тусклые/красные вспышки): короткие неуверенные
            # обрывки ("dahgs.", "You migai") -- мусор распознавания кадра
            # между строками; осмысленные короткие ("Hello?", "I see")
            # читаются уверенно
            if zone in (4, 5) and len(alnum) < 6 and best[2] < 0.75 \
                    and short_norm not in SHORT_WORDS_OK and not is_num:
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
                # из-за короткого обрывка рядом. Равные тексты (близнецы из
                # двух ступеней зоны 4) здесь не выбрасываем -- их снимает
                # dedup близнецов ниже, иначе фразы уничтожают друг друга.
                # ПОГЛОЩАЮЩАЯ зона 5 ("Mona An) 'Hang on!") -- не "более
                # полная реплика", а вспышка с приклеенным интерфейсом:
                # настоящий диалог читает зона 1 ("Hang on" 13.80_5);
                # иначе z1-осколок гибнет здесь, а z5-клей -- в click-
                # правиле, и реплика исчезает целиком.
                if len(n_q) > len(n_p) and n_p in n_q \
                        and not (len(segments[q["seg"]]) > 3
                                 and segments[q["seg"]][3] == 5):
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
    # (_lcp перенесена на уровень модуля: локальное определение ниже
    # первого использования в блоке зоны 6 давало UnboundLocalError)

    final = []
    for si in sorted(set(p["seg"] for p in kept)):
        sgm = segments[si]
        zone2 = len(sgm) > 3 and sgm[3] in (2, 8)
        plist = [p for p in kept if p["seg"] == si]
        if zone2:
            # короткие обрывки скрамбла заставки ("SHOUL") не пропускаем.
            # РАЗНЫЕ строки одного слайда (заголовок + реплика, две строки
            # диалога) выживают ОБЕ: фазы печати/скрамбла ОДНОЙ строки
            # разводит общая дедупликация по похожести ниже. А вот
            # моментные снапшоты -- конечные точки (0.08/0.35/0.70с до
            # края сегмента) и каша перехода экрана -- это ОДИН момент:
            # в окне 0.75с оставляем только самый уверенный снапшот
            # (настоящие разные строки разнесены сильнее секунды).
            # короткие обрывки скрамбла заставки ("SHOUL") не пропускаем,
            # НО короткие чистые надписи с высокой уверенностью ("Ascent",
            # название локации на чёрном; ".I see." на черновом слайде --
            # 4 буквы при conf 0.97) -- оставляем: минимум 4 знака,
            # conf >= 0.9 и ЕСТЬ строчные буквы (CAPS-обрывки скрамбла
            # типа "SHOUL" строчных не имеют).
            plist = [p for p in plist if len(p["text"]) >= 7
                     or (p["conf"] >= 0.9 and not p["text"].isupper()
                         and re.search(r"[a-z]", p["text"])
                         and len(re.sub(r"[^A-Za-z0-9]", "", p["text"])) >= 4)]
            plist.sort(key=lambda p: -p["conf"])
            kept2 = []
            for p in plist:
                # короткий осколок кроссфейда ("After it my YAY theirs?")
                # держится на экране дольше 0.75с и переживает окно выше;
                # против СОСЕДНЕГО длинного чтения той же строки окно
                # шире -- 1.8с. Настоящие разные строки разнесены сильнее
                window = 1.8 * fps if len(p["text"]) < 40 else 0.75 * fps
                if any(abs(p["frame"] - q["frame"]) < window
                       for q in kept2):
                    continue
                kept2.append(p)
            plist = kept2
            if not plist:
                continue
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

    # имя-карточка зоны 5 (ДО twin-цикла): платный glue ("Grig Gray",
    # "MI6A1 Beak", "Helian") с живой платой говорящего z3 поблизости --
    # снапшот карточки персонажа, НЕ диалог. Если не снять его сейчас,
    # он выигрывает twin-конкурс у настоящей короткой реплики
    # (z1 "Gray." в той же сцене) по длине/уверенности и топит её.
    # Ищем z3 fuzzy: плата мелкая, OCR читает "Gray" как "Grig"/"Grav".
    name_drop_early = set()

    def _z3_has_early(name_norm, p, width=6.0):
        if not spk_results or not name_norm:
            return False
        for (_si, _pi), (fr, txt, _cf) in spk_results.items():
            if not txt or abs(fr - p["frame"]) > width * fps:
                continue
            n_txt = norm_for_compare(txt)
            if name_norm in n_txt or (len(name_norm) >= 4
                                      and len(n_txt) <= 12
                                      and difflib.SequenceMatcher(
                                          None, name_norm,
                                          n_txt).ratio() >= 0.70):
                return True
            # OCR платы шатается сильнее, чем длинное fuzzy: "Gray" на
            # карточке читается "Grig"/"Grav" -- сравниваем головы на 3
            if len(name_norm) >= 4 and len(n_txt) >= 3 \
                    and n_txt[:3] == name_norm[:3]:
                return True
        return False

    for p in final:
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        if z_p != 5:
            continue
        t = p["text"].rstrip(".:;,!? ")
        n_t = norm_for_compare(t)
        hit = False
        if re.fullmatch(r"[A-Z][a-z]{1,12}", t) and \
                _z3_has_early(n_t, p):
            hit = True
        m = re.fullmatch(r"[A-Za-z0-9]+\s+([A-Z][a-z]{1,12})", t)
        if not hit and m and _z3_has_early(norm_for_compare(m.group(1)), p):
            hit = True
        # карточка-платка: все слова Title Case ("HQ Secret Operative");
        # "Dandelion Hold on." не карточка -- "on" с маленькой буквы
        m = re.fullmatch(r"[A-Z][A-Za-z]{1,11}"
                         r"(?:\s+[A-Z][a-z]{2,12}){1,3}", t)
        if not hit and m and _z3_has_early(n_t, p):
            hit = True
        if hit:
            name_drop_early.add(id(p))
    if name_drop_early:
        final = [p for p in final if id(p) not in name_drop_early]

    # осколок у обрезанного края видео (кандидат, прилегающий к концу файла)
    # близнецы: одна и та же реплика, пойманная обеими ступенями зоны 4
    # (или зонами 1/4 на стыке), плюс склейки с именем из диалог-бокса
    # ("Echelon 1 With a nod..." рядом с "With a nod..."). Окно 6с;
    # близки по difflib >= 0.82 ИЛИ вложены целиком. Оставляем более
    # длинный вариант, при равенстве -- ранний. Реплики с РАЗНЫМИ цифрами
    # ("FILE U083..."/"FILE U034...") -- не близнецы. Зона 2: печать одной
    # строки на слайде оставляет финальные снапшоты в СОСЕДНИХ сегментах
    # ("When I set foot upon it; I did not know." в сегменте строки и её
    # повтор в сегменте следующей строки -- 13.80_4 [035]/[036]): в зоне 2
    # сравниваем и с НАКЛОНЁННЫМ (каждый снапшот читает текст, который
    # уходит вверх) -- расстояние >= 0.82 ИЛИ одно содержит другое.
    no_twin = []
    for p in final:
        n_p = norm_for_compare(p["text"])
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        d_p = re.sub(r"\D", "", p["text"])
        drop = False
        best = None
        best_dist = 0.60
        for q in no_twin:
            sgm_q = segments[q["seg"]]
            z_q = sgm_q[3] if len(sgm_q) > 3 else 1
            d_q = re.sub(r"\D", "", q["text"])
            if d_p and d_q and d_p != d_q:
                continue
            if abs(q["frame"] - p["frame"]) > 6.0 * fps:
                continue
            n_q = norm_for_compare(q["text"])
            g_p = _garble_score(p["text"])
            g_q = _garble_score(q["text"])
            # хвосты после первого токена: для z5-клея ("Имя Реплика...")
            # это сама реплика без имени
            tail_p = norm_for_compare(
                " ".join(p["text"].split()[1:])) \
                if len(p["text"].split()) > 1 else ""
            tail_q = norm_for_compare(
                " ".join(q["text"].split()[1:])) \
                if len(q["text"].split()) > 1 else ""
            # z5-клей с именем-префиксом против чистого диалогового близнеца:
            # даже если dist не дотягивает до порога (префикс имени съедает
            # похожесть: "Border Guard We are..." vs "We are..."), структура
            # "Имя + хвост=q" однозначна: q выживает с ЧИСТЫМ текстом,
            # имя уходит в speaker, клей гибнет
            if z_p == 5 and z_q in (1, 2, 4) \
                    and not q.get("_spk_override"):
                toks = p["text"].split()
                for k in range(1, min(4, len(toks))):
                    tail_k = norm_for_compare(" ".join(toks[k:]))
                    if len(tail_k) < 12:
                        continue
                    if n_q.startswith(tail_k[:12]) \
                            and len(tail_k) <= len(n_q) * 1.15 + 6:
                        pref_toks = toks[:k]
                        pref_str = " ".join(pref_toks)
                        if len(pref_toks) <= 3 and pref_str \
                                and all(re.fullmatch(r"[A-Z][A-Za-z0-9-]*", t)
                                        for t in pref_toks) \
                                and pref_toks[-1] not in (
                                    "She", "The", "A", "An", "It", "System",
                                    "FILE", "When", "M4", "Squad", "Echelon",
                                    "Click", "Comrades", "Comredes"):
                            q["_spk_override"] = pref_str
                            p["_drop_twin"] = True
                            drop = True
                    break
            if drop:
                break
            dist = difflib.SequenceMatcher(None, n_p, n_q).ratio()
            # z5-клей ("Dupieux It is not Statesec...") против чистой
            # диалоговой реплики: префикс имени снижает dist до 0.66
            # (порог 0.70 для длинных) -- узнаём клей по структуре
            # "Имя + реплика" и родству хвоста клея с q
            glue_match = (
                (z_p == 5 and z_q in (1, 2, 4)
                 and re.match(r"^[A-Z][A-Za-z0-9-]+\s+\S", p["text"])
                 and len(tail_p) >= 12
                 and (tail_p[:12] == n_q[:12]
                      or difflib.SequenceMatcher(
                          None, tail_p, n_q).ratio() >= 0.75))
                or (z_q == 5 and z_p in (1, 2, 4)
                    and re.match(r"^[A-Z][A-Za-z0-9-]+\s+\S", q["text"])
                    and len(tail_q) >= 12
                    and (tail_q[:12] == n_p[:12]
                         or difflib.SequenceMatcher(
                             None, tail_q, n_p).ratio() >= 0.75)))
            # ДЛИННЫЕ тексты (>= 60 знаков): OCR той же реплики из другого
            # канала даёт расплывчатые вариации ("...Squad NaroFe Staareh
            # fieldettaed Rrient..." против "...Squad Nitrogen..."): 0.72
            if dist >= 0.82 or (len(n_p) >= 60 and dist >= 0.70) \
                    or n_p in n_q or n_q in n_p or glue_match:
                # мелкое слово случайно вложено в длинную фразу ("you" внутри
                # "your" из "T'm your reinforce"): это НЕ близнецы -- длинную
                # фразу сохраняем, иначе она гибнет из-за conf-замены, а её
                # "убийца" потом всё равно удаляется другими правилами.
                # "_"-снапшот печати не вкладывается в gluing-мусор
                # (z5 "MI6A1 Beak" vs z2 "Beak"). ОДИНОЧНАЯ ЦИФРА-ярлык
                # (z5 "1" из тикера) не должна спасать gluing-мусор от
                # близнеца с "Echelon 1 ..." -- НО 4+ цифры ("FILE
                # U0835205303") -- это настоящая защита осколков расшифровки
                if z_q != z_p and min(len(n_p), len(n_q)) < 6 \
                        and min(len(n_p), len(n_q)) < 0.45 * max(len(n_p),
                                                                 len(n_q)) \
                        and "_" not in p["text"] + q["text"] \
                        and not (len(re.sub(r"\D", "", n_p + n_q)) >= 4
                                 and (re.fullmatch(r"\d{1,2}", n_p)
                                      or re.fullmatch(r"\d{1,2}", n_q))):
                    continue
                # короткое слово из ЗАКОНЧИВШЕЙся реплики не вкладывается в
                # следующую (z1 "Gray." 305.1-306.5 vs z1 "Gray grinds her
                # teeth." 309.7-311.7 -- разные субтитры, у фаз печати одной
                # реплики сегменты перекрываются)
                if min(len(n_p), len(n_q)) < 8 \
                        and (sgm_q[0] > sgm_p[1] or sgm_p[0] > sgm_q[1]):
                    continue
                # p -- дубликат q. Внутри одной зоны -- более полный;
                # между РАЗНЫМИ зонами (один и тот же текст, прочитанный
                # двумя каналами): уверенность, затем МЕНЬШЕ OCR-каши
                # ("NaroFe Staareh fieldettaed" против "Squad Nitrogen"),
                # затем длина
                if z_q == z_p:
                    if len(n_p) > len(n_q):
                        no_twin[no_twin.index(q)] = p
                elif z_p in (1, 2, 4) and z_q == 5 \
                        and g_p <= g_q:
                    # близнец одной реплики: диалоговый канал (z1/2/4)
                    # каноничнее вспышки зоны 5 ПРИ ЛЮБОМ исходе сравнения
                    # conf -- z5-версия то проигрывает twin по conf, то
                    # гибнет в click/H2b-правиле, и реплика исчезает целиком
                    # ("Hang on!", "Understood!", "Protest! ...")
                    no_twin[no_twin.index(q)] = p
                elif z_p == 5 and z_q in (1, 2, 4) \
                        and g_q <= g_p:
                    # встречное направление: p (z5) уступает q (z1/2/4)
                    pass
                elif (-g_p, p["conf"], len(n_p)) > (-g_q, q["conf"],
                                                    len(n_q)):
                    no_twin[no_twin.index(q)] = p
                # умирающий z5-клей ("Border Guard We are inspectors...")
                # отдаёт своё имя-префикс выжившему диалоговому близнецу:
                # у того плашки нет, а имя говорящего -- именно этот префикс.
                # Строгие условия: q начинается БЕЗ префикса (его текст --
                # это хвост клея), префикс похож на имя (имена-предложения
                # "She"/"System" исключены), и q -- не ДРУГАЯ фраза с
                # собственным началом (тогда клей совпадает с q целиком)
                if glue_match and z_p == 5 and z_q in (1, 2, 4) \
                        and not q.get("_spk_override"):
                    toks = p["text"].split()
                    for k in range(1, min(5, len(toks))):
                        tail_k = norm_for_compare(" ".join(toks[k:]))
                        # ТОЛЬКО "q начинается с хвоста клея": близнец с тем
                        # же текстом, где префикс -- подлежащее-предложение
                        # ("AK-15 studies..."), вкладом не является
                        if n_q.startswith(tail_k[:12]):
                            pref_toks = toks[:k]
                            pref_str = " ".join(pref_toks)
                            # q длиннее хвоста => q несёт собственный текст,
                            # префикс не имя ("Echelon 1 With a nod...")
                            if len(tail_k) <= len(n_q) * 1.15 + 6 \
                                    and len(pref_toks) <= 3 \
                                    and "&" not in pref_str \
                                    and re.search(r"\b(?:we|our|ours|us|"
                                                  r"you|your|yours|i|my|"
                                                  r"me|mine)\b", q["text"],
                                                  re.I) \
                                    and all(re.fullmatch(r"[A-Z][A-Za-z0-9-]*",
                                                         t) for t in pref_toks) \
                                    and pref_toks[-1] not in (
                                        "She", "The", "A", "An", "It",
                                        "System", "FILE", "When", "M4",
                                        "Squad", "Echelon", "Click",
                                        "Comrades", "Comredes", "Iop"):
                                q["_spk_override"] = pref_str
                                # takeover: чистый близнец должен выжить,
                                # клеевой вариант умирает как дубликат
                            break
                elif glue_match and z_q == 5 and z_p in (1, 2, 4) \
                        and not p.get("_spk_override"):
                    # зеркальный случай: клеевой z5 -- это q (уже в no_twin),
                    # чистый диалоговый близнец -- p. Имя-префикс клея уходит
                    # в speaker p, чистый забирает слот клея (без условия
                    # garble/conf: у "Bundesgrenzschutz..." каша выше, и
                    # без этого чистая реплика гибнет целиком)
                    toks = q["text"].split()
                    for k in range(1, min(5, len(toks))):
                        tail_k = norm_for_compare(" ".join(toks[k:]))
                        if n_p.startswith(tail_k[:12]):
                            pref_toks = toks[:k]
                            pref_str = " ".join(pref_toks)
                            if len(tail_k) <= len(n_p) * 1.15 + 6 \
                                    and len(pref_toks) <= 3 \
                                    and "&" not in pref_str \
                                    and re.search(r"\b(?:we|our|ours|us|"
                                                  r"you|your|yours|i|my|"
                                                  r"me|mine)\b", p["text"],
                                                  re.I) \
                                    and all(re.fullmatch(r"[A-Z][A-Za-z0-9-]*",
                                                         t) for t in pref_toks) \
                                    and pref_toks[-1] not in (
                                        "She", "The", "A", "An", "It",
                                        "System", "FILE", "When", "M4",
                                        "Squad", "Echelon", "Click",
                                        "Comrades", "Comredes", "Iop"):
                                p["_spk_override"] = pref_str
                                if q in no_twin:
                                    no_twin[no_twin.index(q)] = p
                            break
                p["_drop_twin"] = True
                drop = True
                break
            # зона 2: печать одной строки на слайде оставляет финальные
            # снапшоты в соседних сегментах; берём более полный из близких
            # (>= 0.60) вариантов
            if z_p == 2 and z_q == 2 and dist >= best_dist \
                    and len(n_p) > len(n_q):
                best = q
                best_dist = dist
        if drop:
            continue
        if best is not None:
            no_twin[no_twin.index(best)] = p
            continue
        no_twin.append(p)
    final = no_twin

    # gluing-мусор: компактные OCR-каши из ярлыков интерфейса, вклеенных в
    # полосу субтитров ("RIFFINCOM", "FRAGILL", "HeUpoRi SIG MCX",
    # "HeUpoR) Click", "K36899 ENDDROUNI", "nelon 1 708T7 ENDROUN",
    # "lelon 1 7087 ENDROUN"). Признак: высокий счёт OCR-каши при
    # малой длине. Осмысленные короткие реплики кашу не набирают
    # ("Beak", "Helian", "Grig Gray" -- garble 0; "I'm your reinforce" --
    # длиннее порога).
    junk_drop = []
    for p in final:
        g = _garble_score(p["text"])
        alnum_ct = len(re.sub(r"[^A-Za-z0-9]", "", p["text"]))
        if g >= 2 and alnum_ct < 20:
            p["_drop_junk"] = True
            junk_drop.append(p)
            continue
        # мусор титульных заставок и бегущих строк: много заглавных
        # "слов" при высокой каше ("RLS FRONTLIN Chapter ASTORY PNL",
        # "GIRLS FRONTLINE Chajter ^sTORY ONLY SHOUL 202."). Осмысленные
        # реплики такие счёты не набирают ("AK-12 From now on..." -- одна
        # аббревиатура; "DEFY, M16, you're..." -- тоже)
        caps_words = re.findall(r"\b[A-Z]{2,}\b", p["text"])
        if g >= 4 and len(caps_words) >= 3 and alnum_ct >= 20:
            p["_drop_junk"] = True
            junk_drop.append(p)
    if junk_drop:
        final = [p for p in final if not p.get("_drop_junk")]

    # обрыв печати в НАЧАЛЕ сегмента ("Common destiny\" I've grown",
    # "Dupieux ... thoro", "Military Officer B If we execute..."): рядом
    # (внутри того же или соседнего сегмента, окно 2.5с) есть ДОЗРЕВШАЯ
    # версия. Правило "короткий близнец длинного": длина меньше 0.85
    # И похожесть высокая (0.86 внутри одной зоны, 0.80 между зонами --
    # имя говорящего в клеевой версии сбивает ratio). Победитель --
    # длинный; между зонами приоритет "меньше OCR-каши". Эталонная
    # короткая реплика соседа не имеет ("Hold on." в 13.80_3 соседа
    # не имеет -- проверено).
    no_frag = []
    for p in final:
        n_p = norm_for_compare(p["text"])
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        drop = False
        for q in final:
            if q is p:
                continue
            sgm_q = segments[q["seg"]]
            z_q = sgm_q[3] if len(sgm_q) > 3 else 1
            if abs(q["frame"] - p["frame"]) > 2.5 * fps:
                continue
            if sgm_q[0] > sgm_p[1] + 0.5 * fps \
                    or sgm_p[0] > sgm_q[1] + 0.5 * fps:
                continue
            n_q = norm_for_compare(q["text"])
            if len(n_p) >= len(n_q) or not n_q:
                continue
            dist = difflib.SequenceMatcher(None, n_p, n_q).ratio()
            if len(n_p) < 0.85 * len(n_q) and dist >= (0.80 if z_q != z_p
                                                       else 0.86):
                if z_q == z_p or z_p == 5 or _garble_score(p["text"]) > \
                        _garble_score(q["text"]):
                    drop = True
                    break
            # обрыв в начале (имя+первые слова, потом OCR-срез): длинная
            # версия содержит начало короткой почти целиком ("Military
            # Officer B If we execute..." содержит "Ifwe execute...")
            if len(n_p) >= 25 and len(n_q) > len(n_p) + 4 \
                    and not n_q.startswith(n_p) \
                    and difflib.SequenceMatcher(
                        None, n_p, n_q[:len(n_p) + 4]).ratio() >= 0.90 \
                    and (z_q == z_p or z_p == 5):
                drop = True
                break
            # вспышка зоны 5 поймала ТОЛЬКО голову реплики (первые 1-2
            # слова, "Common destiny\" I've grown" / "Dupieux ... thoro"):
            # длинная версия той же реплики живёт в z1/z4 рядом. Головы
            # сравниваем с ОГОВОРКОЙ на OCR-ошибку ("I've" -> "F've"):
            # достаточно совпадения первых 14 знаков и ratio >= 0.85 на
            # общем префиксе. У одиночных коротких реплик ("Hold on.")
            # такой z1-версии нет.
            if z_p == 5 and z_q != 5 and len(n_p) >= 12 \
                    and len(n_q) > len(n_p) + 12 \
                    and len(n_p) < 0.85 * len(n_q) \
                    and abs(q["frame"] - p["frame"]) <= 4.0 * fps:
                head = min(len(n_p), 24)
                if difflib.SequenceMatcher(
                        None, n_p[:12], n_q[:12]).ratio() >= 0.90 \
                        and difflib.SequenceMatcher(
                            None, n_p[:head],
                            n_q[:head]).ratio() >= 0.85:
                    drop = True
                    break
            # ПОЛНАЯ чистая z1/z4-версия той же реплики (длина >= z5,
            # похожесть по 60-знаковому окну, без OCR-каши) -- z5-версия
            # (с именем, штампом, хуже читаемая) уступает всегда: зона 5
            # тусклее диалогового канала ("Earl Yes. In exchange..." /
            # "AK-12 From now on..." против чистых z1-версий)
            if z_p == 5 and z_q in (1, 4) \
                    and _garble_score(p["text"]) > 0 \
                    and _garble_score(q["text"]) == 0 \
                    and len(n_q) >= len(n_p) \
                    and abs(q["frame"] - p["frame"]) <= 6.0 * fps \
                    and difflib.SequenceMatcher(
                        None, n_p[:60], n_q[:60]).ratio() >= 0.60:
                drop = True
                break
            if not drop and z_p == 5:
                # z5-glue с ИМЕНЕМ в начале ("Dupieux It is not Statesec..."):
                # после снятия первого токена остаётся голова реплики,
                # которую сосед читает целиком; окно шире (10с) -- вспышка
                # бьёт задолго до появления субтитра. Голова "Имя+первое
                # слово" должна совпасть -- ложных срабатываний у соседних
                # реплик нет. Штамп водяного знака ("FRAGILE") может стоять
                # ВТОРЫМ токеном ("Commander FRAGILE Kalina, I'm...") --
                # пробуем и один, и два первых токена; чистая версия
                # соседа длиннее хвоста.
                for q in final:
                    if q is p:
                        continue
                    sgm_q = segments[q["seg"]]
                    n_q = norm_for_compare(q["text"])
                    if len(n_q) > len(n_p) \
                            and abs(q["frame"] - p["frame"]) <= 10.0 * fps:
                        toks_h = p["text"].split()
                        heads = []
                        if len(toks_h) >= 2 and toks_h[0][:1].isupper():
                            heads.append(" ".join(toks_h[1:]))
                        if len(toks_h) >= 3 and toks_h[1].isupper() \
                                and len(toks_h[1]) >= 5:
                            heads.append(" ".join(toks_h[2:]))
                        for h in heads:
                            n_rest = norm_for_compare(h)
                            if len(n_rest) >= 20 \
                                    and len(n_q) > len(n_rest) + 8 \
                                    and difflib.SequenceMatcher(
                                        None, n_rest[:20],
                                        n_q[:20]).ratio() >= 0.90:
                                drop = True
                                break
                        if drop:
                            break
        if not drop:
            no_frag.append(p)
    final = no_frag

    # glue-дубль зоны 5: имя говорящего/ярлык приклеены к реплике, которую
    # сосед читает чище ("Dupieux It is not..." / "Military Officer B If
    # we execute..." / "Dandelion m not joking:_ Command"). p уничтожается
    # только при СОБСТВЕННОМ признаке склейки: (A) общая подстрока
    # начинается в p не с нуля -- впереди мусор-префикс; (B) после блока
    # остаётся хвост с курсорным мусором; (C) после снятия первого токена
    # хвост совпадает с q. Партнёр q должен быть не грязнее p ("FILE
    # PACC..., decrypting." чист -- его "партнёр" STATUS-клей грязнее,
    # файловая строка выживает).
    glue_drop = set()
    for p in final:
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        if z_p != 5:
            continue
        t_p = p["text"]
        toks = t_p.split()
        if len(toks) < 2:
            continue
        n_full = norm_for_compare(t_p)
        n_tail = norm_for_compare(" ".join(toks[1:]))
        g_p = _garble_score(t_p)
        for q in final:
            if q is p:
                continue
            # окно 5с: зрелая зона-1 версия часто появляется через 3-4с
            # после вспышки (печать субтитра запаздывает за карточкой
            # говорящего: "Commander FRAGILE Theard..." 162.5s ->
            # "heard from Kalina..." 167.1s в Eclipse). Для хвостовых
            # правил z5-осколка против диалоговой фразы -- 6.5с (осколок
            # живёт до конца печати, зрелый вариант приходит позже).
            dt = abs(q["frame"] - p["frame"]) / fps
            if dt > 6.5:
                continue
            n_q = norm_for_compare(q["text"])
            if not n_q:
                continue
            g_q = _garble_score(q["text"])
            z_q = segments[q["seg"]][3] if len(segments[q["seg"]]) > 3 else 1
            if dt <= 5.0:
                m = difflib.SequenceMatcher(None, n_full, n_q) \
                    .find_longest_match(0, len(n_full), 0, len(n_q))
                if m.size >= 8 \
                        and m.size >= 0.5 * min(len(n_full), len(n_q)):
                    sig_a = m.a > 0                  # мусор-префикс в p
                    sig_b = m.a + m.size < len(n_full) \
                        and re.search(r"[_():;]", t_p)   # хвост-курсор
                    if (sig_a or sig_b) and g_q <= g_p + 1:
                        glue_drop.add(id(p))
                        break
            if len(n_tail) >= 8:
                m2 = difflib.SequenceMatcher(None, n_tail, n_q) \
                    .find_longest_match(0, len(n_tail), 0, len(n_q))
                if dt <= 5.0 and m2.size >= 8 \
                        and m2.size >= 0.5 * min(len(n_tail), len(n_q)):
                    # снятие первого токена обнажает реплику: q читает её
                    # чище ИЛИ q начинается ровно с неё (p -- "Имя: текст")
                    if (g_p > g_q or (m2.b == 0 and g_p >= g_q)) \
                            and g_q <= g_p + 1:
                        glue_drop.add(id(p))
                        break
                # хвост p -- ЧАСТЬ чистого диалогового текста: p -- осколок
                # середины/середины-конца печати. Случаи: точное вхождение
                # хвоста ("smiles and no of stopping" -- кусок z1-версии);
                # крупный общий блок; все токены хвоста в порядке внутри q
                # с малыми промежутками (OCR съел слова середины: "This
                # smiles and no of stopp" из "...simply smiles and shows
                # no intention of stopping them.")
                if z_p == 5 and z_q in (1, 2, 4) and g_q == 0 \
                        and (n_tail in n_q
                             or (m2.size >= 12
                                 and m2.size >= 0.7 * len(n_tail))):
                    glue_drop.add(id(p))
                    break
                tail_toks = [norm_for_compare(w)
                             for w in t_p.split()[1:]]
                tail_toks = [w for w in tail_toks if w]
                if z_p == 5 and z_q in (1, 2, 4) and g_q == 0 \
                        and len(tail_toks) >= 4:
                    pos = 0
                    ok = True
                    for w in tail_toks:
                        j = n_q.find(w, pos)
                        if j < 0 or (pos and j - pos > 14):
                            ok = False
                            break
                        pos = j + len(w)
                    if ok:
                        glue_drop.add(id(p))
                        break
    if glue_drop:
        final = [p for p in final if id(p) not in glue_drop]

    # говорящий-склейка: зона 5 (вспышка) приклеила имя говорящего к
    # началу реплики, которую зона 1 читает целиком ("RPK-16 As I said,
    # M16..." vs "As said M16..."; "Dandelion m not joking:_ Command" vs
    # "Tm notjoking."). Родство по ЗОНЕ 1-фразе без имени или по имени:
    # зона 5 тусклее, OCR-каши больше. Белого списка зон нет: правило
    # работает только когда склейка реально короче полной версии.
    spk_drop = set()
    for i, p in enumerate(final):
        sgm_p = segments[p["seg"]]
        if (len(sgm_p) <= 3 or sgm_p[3] != 5) or len(p["text"]) < 12:
            continue
        for q in final:
            if q is p:
                continue
            if abs(q["frame"] - p["frame"]) > 2.5 * fps:
                continue
            sgm_q = segments[q["seg"]]
            z_q = sgm_q[3] if len(sgm_q) > 3 else 1
            if z_q not in (1, 4):
                continue
            n_q = norm_for_compare(q["text"])
            # вариант 1: зона 1 читает ту же реплику БЕЗ имени
            if norm_for_compare(p["text"]) in n_q:
                spk_drop.add(id(p))
                break
            # вариант 2: зона 1-фраза начинается с имени из склейки,
            # хвост склейки после имени совпадает с зона-1 началом
            m = re.match(r"^(.{3,25}?)\s+([A-Z][a-z]+)", p["text"])
            if m and norm_for_compare(m.group(1)) in n_q[:30]:
                tail = norm_for_compare(p["text"][m.end(1):])
                if tail and tail[:12] == n_q[:len(tail[:12])]:
                    spk_drop.add(id(p))
                    break
    if spk_drop:
        final = [p for p in final if id(p) not in spk_drop]

    # водяной знак ВНУТРИ клея: "Dandelion FRAGILE That's an absurd..."
    # -- имя говорящего + штамп FRAGILE + реплика. Хвост после штампа
    # родственен чистой диалоговой фразе (голова совпадает), сама фраза
    # живёт рядом в z1/2/4 -- клей умирает, реплика остаётся
    wm_drop = set()
    for p in final:
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        if z_p != 5:
            continue
        m = re.match(r"^[A-Z][A-Za-z0-9-]+\s+FRAGILE[\s'.,:;]*\s*(\S.*)$",
                     p["text"])
        if not m:
            continue
        tail = norm_for_compare(m.group(1))
        if len(tail) < 12:
            continue
        for q in final:
            if q is p:
                continue
            sgm_q = segments[q["seg"]]
            z_q = sgm_q[3] if len(sgm_q) > 3 else 1
            if z_q not in (1, 2, 4):
                continue
            if abs(q["frame"] - p["frame"]) > 6.5 * fps:
                continue
            n_q = norm_for_compare(q["text"])
            if not n_q or _garble_score(q["text"]) > 0:
                continue
            if tail[:12] == n_q[:12] \
                    or difflib.SequenceMatcher(None, tail, n_q).ratio() \
                    >= 0.70:
                wm_drop.add(id(p))
                break
    if wm_drop:
        final = [p for p in final if id(p) not in wm_drop]

    # имя-реклама из зоны 5: имя персонажа ("Helian") или glue ("Grig
    # Gray", "MI6A1 Beak") при живой зона-3 паре ("Helian" в плате
    # говорящего рядом). Диалог -- зона 1/2/4/5, имя -- зона 3 (spk_results).
    def _z3_has(name_norm, p, width=6.0):
        if not spk_results or not name_norm:
            return False
        for (_si, _pi), (fr, txt, _cf) in spk_results.items():
            if not txt or abs(fr - p["frame"]) > width * fps:
                continue
            n_txt = norm_for_compare(txt)
            # точное вложение ИЛИ нечёткое родство ("Gray" на плате
            # читается "Grig"/"Grav" -- плата мелкая, OCR шатается,
            # уверенность при этом 1.00). Голова на 3 знака: "gra"=="gra"
            # ("gray" vs "grig"/"grav")
            if name_norm in n_txt or (len(name_norm) >= 4
                                      and len(n_txt) <= 12
                                      and difflib.SequenceMatcher(
                                          None, name_norm,
                                          n_txt).ratio() >= 0.70) \
                    or (len(name_norm) >= 4 and len(n_txt) >= 3
                        and n_txt[:3] == name_norm[:3]):
                return True
        return False

    name_drop = set()
    for p in final:
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        if z_p != 5:
            continue
        t = p["text"]
        n_t = norm_for_compare(t)
        if re.match(r"^[A-Z][a-z]{1,12}$", t) and _z3_has(n_t, p):
            name_drop.add(id(p))
            continue
        m = re.fullmatch(r"[A-Za-z0-9]+\s+([A-Z][a-z]{1,12})", t)
        if m and _z3_has(norm_for_compare(m.group(1)), p):
            name_drop.add(id(p))
            continue
        # картишка говорящего: glue z5 ("HQ Secret Operative") повторяет
        # текст платы говорящего z3 ("HQ Secret Operative") -- это название
        # карточки персонажа, не реплика; диалог в это время несёт зона 1
        m = re.fullmatch(r"[A-Za-z]{2,12}(?:\s+[A-Za-z]{2,12}){1,3}", t)
        if m and _z3_has(n_t, p):
            name_drop.add(id(p))
    if name_drop:
        final = [p for p in final if id(p) not in name_drop]

    # UI-клейка зоны 5 ("Click Sia", "HeUpoR) Click"): снапшоты клика
    # интерфейса. Осмысленная короткая реплика ("Hold on.", "I see.")
    # имеет зона-1/4 пару или живёт в z2; осиротевший клей умирает.
    click_drop = set()
    for p in final:
        sgm_p = segments[p["seg"]]
        z_p = sgm_p[3] if len(sgm_p) > 3 else 1
        if z_p != 5:
            continue
        n_p = norm_for_compare(p["text"])
        if len(n_p) >= 12:
            continue
        # голова печати: сосед читает ДОЛГИЙ текст, начинающийся с p
        # ("Command!" при живом "Commander!") -- p недопечатан, умирает
        # даже при высокой уверенности
        has_longer = False
        for q in final:
            if q is p:
                continue
            if abs(q["frame"] - p["frame"]) > 2.5 * fps:
                continue
            n_q = norm_for_compare(q["text"])
            if len(n_q) > len(n_p) + 1 and n_q.startswith(n_p):
                has_longer = True
                break
        if has_longer:
            click_drop.add(id(p))
            continue
        has_pair = False
        for q in final:
            if q is p:
                continue
            sgm_q = segments[q["seg"]]
            z_q = sgm_q[3] if len(sgm_q) > 3 else 1
            if z_q in (1, 4) and abs(q["frame"] - p["frame"]) <= 2.5 * fps \
                    and norm_for_compare(q["text"])[:8] == n_p[:8]:
                has_pair = True
                break
        # одиночное чистое слово с высокой уверенностью ("Beak" 1.00,
        # "Helian" 0.96, "Gray." 0.98) -- реплика, а не клей: у клея
        # ("Click Sia", "HeUpoR) Click") уверенность ниже или внутри
        # есть цифры/скобки/курсор. Восклицание/вопрос ("Understood!")
        # не делают слово клейкой
        if not has_pair and p["conf"] >= 0.9 \
                and re.fullmatch(r"[A-Za-z.!?]+", p["text"]):
            continue
        if not has_pair:
            click_drop.add(id(p))
    if click_drop:
        final = [p for p in final if id(p) not in click_drop]

    # повторяющийся красный UI-штамп ("FRAGILE" в Eclipse): одна и та же
    # надпись накладывается на кадр многократно и читается каждый раз
    # с вариациями. Группа из 3+ похожих реплик (difflib >= 0.6, без
    # разных цифр) на промежутке 20с+ -- водяной знак интерфейса,
    # выбрасывается целиком; настоящие реплики так не повторяются.
    def _zone_of(ph):
        sgm = segments[ph["seg"]]
        return sgm[3] if len(sgm) > 3 else 1

    stamp_drop = set()
    cand_list = [p for p in final if _zone_of(p) != 2]
    for i in range(len(cand_list)):
        p = cand_list[i]
        if id(p) in stamp_drop:
            continue
        grp = [p]
        n_p = norm_for_compare(p["text"])
        d_p = re.sub(r"\D", "", p["text"])
        for q in cand_list[i + 1:]:
            if id(q) in stamp_drop:
                continue
            d_q = re.sub(r"\D", "", q["text"])
            if d_p and d_q and d_p != d_q:
                continue
            n_q = norm_for_compare(q["text"])
            # штамп короткий и повторяется ПОЧТИ ТОЧНО. Одно общее слово
            # ("Commander" в диалогах, "The Commander" в нарраторе) — НЕ
            # родство: длинные тексты в группу не берём вовсе, короткие
            # (до 9 знаков) сравниваем жёстко (0.75, оба короткие).
            # Иначе "Commander!" + "Welcome back, Commander." +
            # "Commander Earl, 1." собираются в "штамп" по одному слову
            same = (n_q == n_p
                    or (len(n_p) <= 9 and len(n_q) <= 9
                        and difflib.SequenceMatcher(
                            None, n_p, n_q).ratio() >= 0.75))
            if same:
                grp.append(q)
        if len(grp) >= 3:
            t_min = min(x["frame"] for x in grp)
            t_max = max(x["frame"] for x in grp)
            if t_max - t_min > 20.0 * fps:
                for x in grp:
                    stamp_drop.add(id(x))
    if stamp_drop:
        final = [p for p in final if id(p) not in stamp_drop]

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
        return len(sg) > 3 and sg[3] in (2, 8)
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
                    # завершённая короткая реплика ("No. 1 going in.") --
                    # не обрывок печати длинной соседки ("No. 2 going in..."),
                    # даже если их начала похожи
                    complete = (p["text"].rstrip().endswith((".", "!", "?"))
                                and len(p["text"].split()) >= 3)
                    if complete:
                        continue
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

    # дубликат зон: ROI зоны 2 включает полосу зоны 1, поэтому на чёрных
    # кадрах одна и та же реплика может быть поймана несколькими зонами
    # (1/2/4/5). Оставляем фразу родного канала полосы (зона 1 или 4),
    # копии зон 2/5 -- только если рядом нет похожей фразы зоны 1/4.
    def _zone_of(p):
        sgm = segments[p["seg"]]
        return sgm[3] if len(sgm) > 3 else 1

    z1 = [(q, norm_for_compare(q["text"])) for q in final
          if _zone_of(q) in (1, 4)]
    kept2 = []
    for p in final:
        if _zone_of(p) in (2, 5):
            n_p = norm_for_compare(p["text"])
            dup = False
            for q, n_q in z1:
                if abs(q["frame"] - p["frame"]) > 6.0 * fps:
                    continue
                if n_p and n_q and (n_p in n_q or n_q in n_p
                                    or _lcp(n_p, n_q) >= 0.6
                                    * max(len(n_p), len(n_q))):
                    dup = True
                    break
            if dup:
                continue
        kept2.append(p)
    final = kept2

    # контрольный дозор (зона 6): страховка от слепого пятна метрик.
    # Оставляем только фразы, которые НЕ дублируют уже пойманное (окно
    # 7с, difflib >= 0.7 / вложение / общий префикс) и содержат минимум
    # 4 буквенно-цифровых знака; короткое запрещаем всегда (шум)
    final = [p for p in final
             if _zone_of(p) != 6 or len(re.sub(r"[^A-Za-z0-9]", "",
                                               p["text"]))
             >= PATROL_MIN_ALNUM]
    no_patrol = []
    for p in final:
        if _zone_of(p) != 6:
            no_patrol.append(p)
            continue
        n_p = norm_for_compare(p["text"])
        dup = False
        for q in no_patrol:
            if abs(q["frame"] - p["frame"]) > 7.0 * fps:
                continue
            n_q = norm_for_compare(q["text"])
            if n_q and (n_p in n_q or n_q in n_p
                        or difflib.SequenceMatcher(None, n_p, n_q).ratio()
                        >= 0.7
                        or _lcp(n_p, n_q) >= 0.7 * min(len(n_p), len(n_q))):
                dup = True
                break
        if not dup:
            no_patrol.append(p)
    final = no_patrol

    final.sort(key=lambda p: p["frame"])
    return final


def _fix_model_names(text):
    """EasyOCR путает цифры в именах моделей оружия: UMP9 -> 'UMPG',
    UMP45 -> 'UMP4S'. Буква на цифровой позиции токена UMPxx восстанавливается
    по частым OCR-подменам (G<->9, S<->5, O<->0, B<->8)."""
    def _rep(m):
        suf = m.group(1)
        if not suf or not re.search(r"[GSOB]", suf):
            return m.group(0)
        return "UMP" + suf.translate(str.maketrans("GSOB", "9508"))
    # имя RO (или RO6): EasyOCR дорисовывает хвост -- "ROI"
    text = re.sub(r"\bROI\b", "RO", text)
    return re.sub(r"\bUMP([A-Z0-9]{0,2})(?![A-Za-z0-9])", _rep, text)


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
    # Rhino -- имя (не "Носорог"): транслитерация "Райно"
    if re.search(r"\bRhino\b", en):
        ru = re.sub(r"\bН[оо]сорог\w*", "Райно", ru)
    # Desert Eagle -- имя (позывной): "Дэзерт Игл" (не "Пустынный орёл")
    if re.search(r"\bDesert\s+Eagle\b", en):
        ru = re.sub(r"«?Пустынн\w* ор[её]л»?", "«Дэзерт Игл»", ru)
        ru = re.sub(r"\bПустынн\w*\s+ор[её]л\w*", "Дэзерт Игл", ru)
    # "подкрепление" -- средний род: Google согласует "ваша подкрепление"
    ru = re.sub(r"\bваша подкрепление\b", "ваше подкрепление", ru)
    ru = re.sub(r"\bВаша подкрепление\b", "Ваше подкрепление", ru)
    # модели оружия M16/M4 -- латиницей (Google даёт кириллическую М)
    ru = re.sub(r"М(?=1?[64])", "M", ru)
    return ru


def translate_pass(phrases):
    bar_start = time.perf_counter()
    n = len(phrases)
    print("Перевод (%d фраз, одновременных запросов: %d)..." % (
        n, TRANSLATE_CONCURRENT))
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
        await asyncio.wait_for(com.save(mp3_path), timeout=TTS_TIMEOUT_SEC)

    asyncio.run(_run())
    # сервис мог отдать пустой/битый ответ без исключения: ловим здесь,
    # чтобы ошибка была понятной, а не "ffmpeg exit 4294967283"
    if not os.path.isfile(mp3_path) or os.path.getsize(mp3_path) < 500:
        raise RuntimeError("сервис TTS вернул пустой файл "
                           "(сеть/антивирус/блокировка службы речи)")


def _mp3_to_wav(mp3_path, wav_path, ffmpeg):
    """mp3 -> wav (TTS_WAV_SR моно, 16 бит) и длительность в секундах.

    Для длинных видео (2685 фраз = ~27 ГБ в формате 48кГц стерео) диск
    C: заканчивался -- ffmpeg падал на записи КАЖДОГО wav. 24кГц моно
    качеству голоса не вредит (edge-tts отдаёт 24кГц), а места нужно
    вчетверо меньше. mp3 удаляется сразу после конвертации.
    """
    p = subprocess.run(
        [ffmpeg, "-y", "-i", mp3_path, "-ar", str(TTS_WAV_SR), "-ac", "1",
         "-f", "wav", wav_path],
        capture_output=True)
    if p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", "ignore")[-300:]
        raise RuntimeError("ffmpeg (mp3->wav): %s" % (err or
                         "exit %d" % p.returncode))
    try:
        os.remove(mp3_path)
    except OSError:
        pass
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


def _env_groups(intervals, hold, sr):
    """Группы близких интервалов голоса (та же логика, что в _build_env)."""
    groups = []
    for a0, a1 in sorted(intervals):
        if groups and a0 - groups[-1][1] < hold * sr:
            groups[-1][1] = max(groups[-1][1], a1)
        else:
            groups.append([a0, a1])
    return groups


def _env_block(groups, bs, be, attack, release, duck, sr):
    """Огибающая для сэмплов [bs, be) -- тот же результат, что _build_env
    на всей дорожке сразу, но вычисляется блоком (память O(длины блока))."""
    atk_n = int(attack * sr)
    rel_n = int(release * sr)
    env = np.ones(be - bs, dtype=np.float32)
    for a0, a1 in groups:
        lo = a0 - atk_n
        hi = a1 + rel_n
        if hi <= bs or lo >= be:
            continue
        c0 = max(lo, bs)
        c1 = min(hi, be)
        if c1 <= c0:
            continue
        vals = np.full(c1 - c0, duck, dtype=np.float32)
        up_n = a0 - lo
        if up_n > 0:
            i0, i1 = max(c0, lo), min(a0, c1)
            if i1 > i0:
                if up_n > 1:
                    vals[i0 - c0:i1 - c0] = np.linspace(
                        1.0, duck, up_n)[i0 - lo:i1 - lo]
                else:
                    vals[i0 - c0:i1 - c0] = 1.0
        down_n = hi - a1
        if down_n > 0:
            i0, i1 = max(c0, a1), min(hi, c1)
            if i1 > i0:
                if down_n > 1:
                    vals[i0 - c0:i1 - c0] = np.linspace(
                        duck, 1.0, down_n)[i0 - a1:i1 - a1]
                else:
                    vals[i0 - c0:i1 - c0] = 1.0
        np.minimum(env[c0 - bs:c1 - bs], vals, out=env[c0 - bs:c1 - bs])
    return env


def _voice_frames(path):
    """Длительность wav-файла фразы в сэмплах (по заголовку)."""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes()
    except Exception:
        return 0


def _voice_slice(path, smp0, count):
    """count стерео-сэмплов голоса (float32, [-1..1]) из wav с позиции smp0,
    ОТМАСШТАБИРОВАННЫЕ в сэмплы MIX_SR (wav хранится в TTS_WAV_SR моно);
    за краем файла -- нули."""
    out = np.zeros((count, 2), dtype=np.float32)
    if count <= 0:
        return out
    try:
        with wave.open(path, "rb") as w:
            ch = w.getnchannels()
            wr = w.getframerate()
            total = w.getnframes()
            r = wr / float(MIX_SR)          # wav-кадр на сэмпл микса
            w0 = smp0 * r
            w1 = (smp0 + count) * r
            i0 = max(0, int(math.floor(w0)))
            i1 = min(total, int(math.ceil(w1)) + 1)
            if i1 <= i0:
                return out
            w.setpos(i0)
            arr = np.frombuffer(w.readframes(i1 - i0), dtype=np.int16)
            n = len(arr) // ch
            seg = arr[:n * ch].reshape(-1, ch)[:, 0].astype(np.float32) \
                / 32768.0
            if len(seg) == 0:
                return out
            x_old = i0 + np.arange(len(seg), dtype=np.float64)
            # соседние сэмплы микса отстоят друг от друга на r wav-кадров:
            # шаг В ТКАНИ wav-времени, иначе срез растягивает аудио вдвое
            x_new = w0 + np.arange(count, dtype=np.float64) * r
            good = (x_new >= x_old[0]) & (x_new <= x_old[-1])
            if good.any():
                vals = np.interp(x_new[good], x_old, seg)
                out[good, 0] = vals
                out[good, 1] = vals
    except Exception:
        pass
    return out


def assign_appearances(phrases, segments, wrA_s, wrB_s, fps, wrC_s=None,
                       wrR_s=None, wrD_s=None):
    """Момент ПОЯВЛЕНИЯ текста каждой фразы (для озвучки).

    Озвучка должна стартовать, когда текст начал показываться, а не когда
    он дописался. Первая фраза сегмента -- seg_start (начало печати);
    следующая в том же сегменте -- провал метрики между кадрами соседних
    фраз (момент смены реплик). Для тусклых реплик (зона 4) метрика -- wrC,
    для красных (зона 5) -- wrR, для зоны 8 (затемнённый фон) -- wrD.
    """
    by_seg = {}
    for p in phrases:
        by_seg.setdefault(p["seg"], []).append(p)
    for si, plist in by_seg.items():
        sgm = segments[si]
        z = sgm[3] if len(sgm) > 3 else 1
        if z == 2:
            wr = wrB_s
        elif z == 8 and wrD_s is not None:
            wr = wrD_s
        elif z == 4 and wrC_s is not None:
            wr = wrC_s
        elif z == 5 and wrR_s is not None:
            wr = wrR_s
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


def assign_speakers(phrases, segments, spk_results, fps=60.0):
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
    names_by_time = {}          # si -> [(время снимка, имя)]
    for si, snaps in by_seg.items():
        sgm_si = segments[si]
        zone2_si = len(sgm_si) > 3 and sgm_si[3] == 2
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
            if zone2_si and len(frames) < 4:
                continue        # z2: имя-мусор с одного кадра не проходит
            # каждый СНИМОК плашки -- самостоятельная точка: медиана кластера
            # промахивается, когда склейка-мусор ("Friend s ontr Echelon",
            # 13.80_5 [014]) или имя-вопрос ("William?") меняет центр массы
            # кластера, а настоящая плашка стоит ровно на кадре реплики
            for f in frames:
                entries.append((float(f), name))
        if entries:
            names_by_time[si] = entries
    for p in phrases:
        ov = p.pop("_spk_override", None)
        entries = names_by_time.get(p["seg"], [])
        if entries:
            t = p["frame"]
            best_e = min(entries, key=lambda e: abs(e[0] - t))
            best_d = abs(best_e[0] - t)
            # откат к старому поведению на смене плашки: ближайший снимок
            # (расстояние <= 0.5с) может принадлежать УХОДЯЩЕЙ плашке, пока
            # новая ещё не сменила её. Если у другого кластера >= 4 снимков,
            # есть снимок в <= 1.0с и его медиана ближе к кадру фразы --
            # говорим именно он (кластер-сосед устойчиво висит на реплике)
            if best_d <= 0.5 * fps:
                by_name = {}
                for e in entries:
                    by_name.setdefault(e[1], []).append(e)
                best_med = float(np.median(
                    [e[0] for e in by_name[best_e[1]]]))
                for nm, es in by_name.items():
                    if nm == best_e[1] or len(es) < 4:
                        continue
                    d_near = min(abs(e[0] - t) for e in es)
                    med = float(np.median([e[0] for e in es]))
                    if d_near <= 1.0 * fps and abs(med - t) < abs(best_med - t):
                        best_e = (t, nm)
                        best_med = med
            p["speaker"] = best_e[1]
        elif ov:
            # имя снято с умирающего клеевого близнеца -- только так
            # Border Guard попадает на фразу; плашек у реплик без клея нет.
            # Плашечные имена ВАЖНЕЕ: override не перебивает их
            # ("Mk I" из клеевого хвоста не глотает "STEN Mk II")
            p["speaker"] = ov
        else:
            p["speaker"] = ""


def _voice_disk_ok(workdir):
    """True, пока на диске достаточно места для продолжения озвучки.
    Фразы занимают ~0.5 МБ каждая; запас 2 ГБ (включая итоговый mp4).
    Прогон 13.90 упёрся в заполненный диск: Windows отвечала
    "Permission denied" на КАЖДЫЙ wav -- теперь останавливаемся заранее."""
    try:
        return shutil.disk_usage(workdir).free > 2_000_000_000
    except Exception:
        return True


def _voice_disk_stop(workdir):
    print("\n  [!] на диске C: закончилось место -- озвучка остановлена.")
    print("      Освободите диск и запустите озвучку снова:")
    print("      analyze_video.py --voice-txt \"путь\\к\\перевод.txt\"")
    print("      (перевод сохранён, заново анализировать видео не нужно)")
    shutil.rmtree(workdir, ignore_errors=True)


def voice_pass(video_path, phrases, fps, n_frames, attempts=2):
    """Озвучка с автоповтором: если сборка оборвалась (сеть, диск, служба
    TTS) и итоговый mp4 не собран -- попытка повторяется сама, а не по
    команде. Команда --voice-txt остаётся как запасной вариант."""
    out_video = os.path.join(
        OUT_DIR, os.path.splitext(os.path.basename(video_path))[0]
        + "_озвучка.mp4")
    pre = os.path.getmtime(out_video) if os.path.isfile(out_video) else 0.0
    for k in range(attempts):
        if k:
            # чистим следы незавершённой попытки: поля голоса не должны
            # остаться от оборванной сборки
            for p in phrases:
                for key in ("voice_start", "voice_dur", "voice_wav",
                            "voice_rate"):
                    p.pop(key, None)
            print("\n=== Повторная сборка озвучки (попытка %d из %d)... "
                  "===" % (k + 1, attempts))
            time.sleep(5)
        _voice_pass_once(video_path, phrases, fps, n_frames)
        # успех = mp4 собран ЭТОЙ попыткой (свежее, чем до старта)
        if os.path.isfile(out_video) and os.path.getmtime(out_video) > pre:
            return True
    print("\n  [!] озвучка не удалась после %d попыток. Перевод сохранён "
          "в txt; когда проблема устранена, соберите озвучку командой:\n"
          "      analyze_video.py --voice-txt \"путь\\к\\перевод.txt\""
          % attempts)
    return False


def _voice_pass_once(video_path, phrases, fps, n_frames):
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
    # сборка во временный файл: при неудаче битый mp4 не подменит старый
    out_tmp = out_video + ".part.mp4"
    try:
        if os.path.isfile(out_tmp):
            os.remove(out_tmp)
    except OSError:
        pass
    total_sec = n_frames / fps + 1.0
    workdir = tempfile.mkdtemp(prefix="tts_")
    n = len(phrases)
    # предпроверка диска: 15.2 (2685 фраз) упёрся в полное C: -- каждый
    # ffmpeg падал на записи wav, "не озвучена" по кругу 5 часов. Оценка:
    # ~0.5 МБ на фразу (24кГц моно) + запас на итоговый mp4
    try:
        free = shutil.disk_usage(workdir).free
        need = int(n * 0.5e6) + 3_000_000_000
        if free < need:
            print("Не хватает места на диске: свободно %.1f ГБ, нужно "
                  "примерно %.1f ГБ (фразы + итоговое видео). Освободите "
                  "диск и запустите озвучку снова." % (free / 1e9, need / 1e9))
            shutil.rmtree(workdir, ignore_errors=True)
            return
    except Exception:
        pass

    # 1) предсинтез фраз: несколько запросов к серверу одновременно
    # (базовый синтез зависит только от текста и темпа, не от таймингов,
    # поэтому его можно делать заранее и параллельно; планировщик таймингов
    # ниже остаётся последовательным и при необходимости догоняет --
    # редкий пересинтез сжатой фразы -- как и раньше)
    print("Синтез речи (%d фраз, голос %s, темп %s, "
          "одновременных запросов: %d)..." % (
              n, TTS_VOICE, TTS_RATE, TTS_CONCURRENT))
    m_rate = re.match(r"^([+-]?\d+)%$", TTS_RATE)
    base_pct = int(m_rate.group(1)) if m_rate else 0
    bar_start = time.perf_counter()
    # Порядок озвучки = порядок фраз в файле (хронология субтитров).
    # Сортировка по "появлению" давала инверсии: "появление" дрожит на
    # десятые доли секунды между соседними сегментами одной сцены, и
    # фраза обгоняла соседей (13.80_2 [002] 00:00.0 против [001] 00:00.2;
    # 13.80_4 [022] 01:28.2 против [021] 01:30.7) -- голос звучал раньше
    # своей очереди. Старт каждой фразы по-прежнему >= её появления и
    # >= конца предыдущей (наложений нет): сдвигается только очередь.
    order = list(range(n))
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
                if re.search(r"getaddrinfo|timeout|timed out|SSL|Connection"
                             r"|Temporary failure", str(e), re.I):
                    time.sleep(10)   # краткий обрыв сети -- переждать
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
            # контроль места: на 7378 фраз wav+mp3 ~7 ГБ; если диск кончился
            # в середине синтеза -- останавливаемся сразу, не ждём отказов
            if done_ct % 300 == 0 and not _voice_disk_ok(workdir):
                print("\n  [!] на диске закончилось место в середине синтеза")
                _voice_disk_stop(workdir)
                return
            show_progress("Озвучка", done_ct, n, bar_start)
    show_progress("Озвучка", done_ct, n, bar_start, extra="готово")
    sys.stdout.write("\n")

    prev_end = 0.0
    ok = 0
    fail_streak = 0
    sched_t0 = time.perf_counter()
    for k, i in enumerate(order):
        p = phrases[i]
        mp3 = os.path.join(workdir, "p%03d.mp3" % i)
        wav = os.path.join(workdir, "p%03d.wav" % i)
        # старт известен заранее: зависит только от конца предыдущей фразы
        start = max(p["appear"] / fps, prev_end + VOICE_GAP)
        if fail_streak >= 30:
            print("\n  [!] %d фраз подряд не озвучены -- остановка озвучки "
                  "(сеть/диск/служба TTS). Видео не собираю." % fail_streak)
            shutil.rmtree(workdir, ignore_errors=True)
            return
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
            # догон по отставанию: если фраза стартовала позже своего
            # тайминга (очередь накопила долг), синтезируем её быстрее,
            # пока не вернёмся к синхронности (задержка -> 0). Долг больше
            # 0.3с только, чтобы не гнать микроподгонками
            if TTS_CATCHUP and k + 1 < len(order):
                late = start - p["appear"] / fps
                if late > 0.3:
                    tail = sched_list[k + 1] - start
                    need = max(0.35, min(2.5, tail))
                    if dur > need:
                        extra = int(math.ceil(
                            (dur / need - 1.0) * 100))
                        target = min(TTS_RATE_MAX, used_pct + extra)
                        if target > used_pct:
                            rate = "+%d%%" % target
                            used_pct = target
                            _tts_synthesize(p["ru"] or p["text"], mp3, rate)
                            dur = _mp3_to_wav(mp3, wav, ffmpeg)
        except Exception as e:
            if not _voice_disk_ok(workdir):
                _voice_disk_stop(workdir)
                return
            print("\n  [!] фраза %d не озвучена: %s" % (i + 1, e))
            fail_streak += 1
            # сетевой сбой (обрыв DNS/интернета на длинном видео) --
            # даём сети восстановиться, прежде чем продолжить
            if re.search(r"getaddrinfo|timeout|timed out|SSL|Connection"
                         r"|Temporary failure", str(e), re.I):
                time.sleep(15)
            continue
        fail_streak = 0
        if k % 100 == 0 and k:
            # планировщик долгий на длинных видео (адаптивные пересинтезы),
            # без индикации он выглядит как зависание
            print("  планировщик %d/%d (%.1f мин)..." % (
                k, n, (time.perf_counter() - sched_t0) / 60.0))
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

    # 2-4) потоковое микширование и сборка: ffmpeg декодирует оригинал
    # блоками, микс считается на лету и сразу подаётся на stdin второго
    # ffmpeg (видеопоток копируется как есть). Память O(1), гигантских
    # WAV на диске нет -- длинные видео больше не упираются ни в лимит
    # формата WAV 4 ГБ, ни в объём оперативной памяти.
    print("Микширование аудио и сборка (потоково): %s" %
          os.path.basename(out_video))
    sr = MIX_SR
    total_n = int(total_sec * sr)
    blk_n = int(2.0 * sr)
    phrases_v = []
    for p in phrases:
        if "voice_wav" not in p:
            continue
        off = int(p["voice_start"] * sr)
        # wav хранится в TTS_WAV_SR: длительность переводим в сэмплы микса
        m = int(_voice_frames(p["voice_wav"]) * sr / float(TTS_WAV_SR))
        m = min(m, total_n - off)
        if m > 0:
            phrases_v.append((off, off + m, p["voice_wav"]))
    phrases_v.sort(key=lambda t: t[0])
    groups = _env_groups([(a0, a1) for a0, a1, _w in phrases_v],
                         DUCK_HOLD, sr)

    dec = subprocess.Popen(
        [ffmpeg, "-y", "-i", video_path, "-vn", "-ar", str(sr), "-ac", "2",
         "-f", "s16le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    err_path = os.path.join(workdir, "_mux_err.txt")
    with open(err_path, "wb") as errf:
        mux = subprocess.Popen(
            [ffmpeg, "-y", "-i", video_path,
             "-f", "s16le", "-ar", str(sr), "-ac", "2", "-i", "pipe:0",
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", out_tmp],
            stdin=subprocess.PIPE, stderr=errf)
        vi = 0            # указатель по фразам (отсортированы по старту)
        written = 0
        dec_err = False   # аудио нет / декодер закончился раньше дорожки
        bar_start2 = time.perf_counter()
        broken = False
        try:
            while written < total_n:
                n_blk = min(blk_n, total_n - written)
                raw = b""
                if not dec_err:
                    raw = dec.stdout.read(n_blk * 4)
                    if len(raw) < n_blk * 4:
                        dec_err = True
                        if written == 0:
                            print("\n  (у видео нет аудиодорожки -- микс "
                                  "только с голосом)")
                orig_blk = np.frombuffer(raw, dtype=np.int16)
                nf = min(len(orig_blk) // 2, n_blk)
                blk = np.zeros((n_blk, 2), dtype=np.float32)
                if nf:
                    blk[:nf] = (orig_blk[:nf * 2].reshape(-1, 2)
                                .astype(np.float32) / 32768.0)
                env_blk = _env_block(groups, written, written + n_blk,
                                     DUCK_ATTACK, DUCK_RELEASE,
                                     DUCK_VOLUME, sr)
                bg_blk = BG_LIMIT * np.tanh(
                    blk * (env_blk[:, None] * ORIG_VOLUME) / BG_LIMIT)
                voices_blk = np.zeros((n_blk, 2), dtype=np.float32)
                while vi < len(phrases_v) and phrases_v[vi][1] <= written:
                    vi += 1
                j = vi
                while j < len(phrases_v) and phrases_v[j][0] < written + n_blk:
                    a0, a1, wpath = phrases_v[j]
                    s0 = max(written, a0)
                    s1 = min(written + n_blk, a1)
                    if s1 > s0:
                        voices_blk[s0 - written:s1 - written] += \
                            _voice_slice(wpath, s0 - a0, s1 - s0)
                    j += 1
                mixed_blk = bg_blk + voices_blk
                np.clip(mixed_blk, -1.0, 1.0, out=mixed_blk)
                try:
                    mux.stdin.write(
                        (mixed_blk * 32767.0).astype(np.int16).tobytes())
                except OSError:
                    broken = True
                    dec.kill()
                    break
                written += n_blk
                show_progress("Микс  ", written, total_n, bar_start2)
        finally:
            try:
                mux.stdin.close()
            except Exception:
                pass
        if not broken:
            show_progress("Микс  ", written, total_n, bar_start2,
                          extra="готово")
        sys.stdout.write("\n")
        dec.stdout.close()
        dec.wait()
        mux.wait()
    if mux.returncode != 0:
        print("  [!] ffmpeg не собрал видео:")
        with open(err_path, "rb") as f:
            print(f.read().decode("utf-8", "ignore")[-800:])
        try:
            if os.path.isfile(out_tmp):
                os.remove(out_tmp)
        except OSError:
            pass
    elif os.path.isfile(out_tmp):
        try:
            os.replace(out_tmp, out_video)   # атомарная замена по успеху
        except OSError:
            shutil.move(out_tmp, out_video)
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
        f.write("# Зона 1 (нижняя): %d,%d-%d,%d | зона 2 (центр, явный чёрный "
                "экран ИЛИ белый слайд, без текста в зоне 1) | зона 8 "
                "(затемнённый фон, письма): %d,%d-%d,%d | зона 3 (говорящий, "
                "не переводится): %d,%d-%d,%d  -- в координатах 1920x1080\n" %
                (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2,
                 CENTER_ROI_X1, CENTER_ROI_Y1, CENTER_ROI_X2, CENTER_READ_Y2,
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


def _parse_ts(s):
    """'MM:SS.d' | 'H:MM:SS.d' -> секунды (float)."""
    parts = s.replace(",", ".").split(":")
    while len(parts) < 3:
        parts.insert(0, "0")
    return (int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))


def _cmd_voice_txt(argv):
    """python analyze_video.py --voice-txt "путь\\к\\..._перевод.txt"

    Дозапуск ТОЛЬКО озвучки: фразы, тайминги и перевод берутся из
    готового txt (после падения на этапе озвучки). Совпадает с полным
    прогоном: "появление"/"сегмент" парсятся в кадры, как их и записал
    write_output."""
    if not argv or not os.path.isfile(argv[0]):
        print("использование: analyze_video.py --voice-txt "
              "\"путь\\к\\перевод.txt\"")
        return
    txt_path = argv[0]
    lines = open(txt_path, encoding="utf-8").read().splitlines()
    video = None
    fps = None
    elapsed = 0.0
    phrases = []
    cur = None
    re_hdr = re.compile(r"^\[(\d+)\]\s+([\d:.]+)\s+\(появление ([\d:.]+), "
                        r"сегмент ([\d:.]+)-([\d:.]+)\)\s*$")
    for ln in lines:
        m = re.match(r"^# Видео: (.+)$", ln)
        if m:
            video = m.group(1).strip()
            continue
        m = re.match(r"^# (\d+)x(\d+), ([\d.]+) fps", ln)
        if m:
            fps = float(m.group(3))
            continue
        m = re.match(r"^# Фраз: \d+ \| Время анализа: (.+)$", ln)
        if m:
            e = m.group(1).split(":")
            elapsed = sum(float(x) * 60 ** i
                          for i, x in enumerate(reversed(e)))
            continue
        m = re_hdr.match(ln)
        if m:
            cur = {
                "frame": _parse_ts(m.group(2)) * fps,
                "appear": _parse_ts(m.group(3)) * fps,
                "seg_start": _parse_ts(m.group(4)) * fps,
                "seg_end": _parse_ts(m.group(5)) * fps,
                "speaker": "",
                "text": "",
                "ru": "",
            }
            phrases.append(cur)
            continue
        if cur is None or ln.startswith(("-", "=")) or ln.startswith("ОВ:"):
            continue
        if ln.startswith("EN: "):
            cur["text"] = ln[4:]
        elif ln.startswith("RU: "):
            cur["ru"] = ln[4:]
        elif ln and not cur["speaker"] and not cur["text"]:
            cur["speaker"] = ln

    if not video or not os.path.isfile(video):
        print("видео из txt не найдено: %s" % video)
        return
    if not phrases:
        print("в txt нет фраз")
        return
    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_real = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()
    print("Дозапуск озвучки: %d фраз, видео %s" % (
        len(phrases), os.path.basename(video)))
    t0 = time.perf_counter()
    voice_pass(video, phrases, fps_real, n_frames)
    n_v = sum(1 for p in phrases if "voice_start" in p)
    if n_v == 0:
        print("Озвучка не удалась -- txt не перезаписан.")
        return
    # перезаписываем ТОТ ЖЕ txt: кадры остались кадрами, добавились ОВ
    write_output(txt_path, video, fps_real, n_frames, phrases,
                 elapsed + time.perf_counter() - t0)
    print("txt обновлён: %s" % txt_path)


def main():
    # режимы воркеров (запускаются как отдельные процессы из scan/ocr)
    if len(sys.argv) > 1 and sys.argv[1] == "--scan-chunk":
        _cmd_scan_chunk(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--ocr-worker":
        _cmd_ocr_worker(sys.argv[2:])
        return
    # дозапуск озвучки из готового txt (после падения/отключения диска):
    # перевод и тайминги не пересчитываются, только синтез + микс.
    # Термин "кадры" тут -- внутренние числа пайплайна (как они и писались),
    # так что собранное видео совпадает с обычным полным прогоном
    if len(sys.argv) > 1 and sys.argv[1] == "--voice-txt":
        _cmd_voice_txt(sys.argv[2:])
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
    if SETTINGS_APPLIED and os.environ.get("DSH_DEBUG") == "1":
        print("Настройки из settings.json: %s" % ", ".join(SETTINGS_APPLIED))
    print("=" * 70)

    # Этап 1
    (wrA_s, wrB_s, wrD_s, wrS_s, wrC_s, wrR_s, wrR2_s,
     segments, fps, n_frames, n_cand) = scan_video(video)
    print("Сегментов с текстом: %d (из них добор: %d)" %
          (len(segments), n_cand))
    if os.environ.get("DSH_DEBUG") == "1":   # подробный список -- по флагу
        for i, sgm in enumerate(segments):
            print("  [%02d] %s - %s  зона %d%s" % (
                i + 1, fmt_time(sgm[0] / fps), fmt_time(sgm[1] / fps),
                sgm[3] if len(sgm) > 3 else 1,
                " (добор)" if sgm[2] == 1 else ""))

    # Этап 2
    results, spk_results = ocr_pass(video, segments, fps, wrA_s, wrB_s,
                                    wrC_s, wrR_s)
    phrases = extract_phrases(segments, results, fps,
                              spk_results=spk_results)
    assign_speakers(phrases, segments, spk_results, fps)
    assign_appearances(phrases, segments, wrA_s, wrB_s, fps, wrC_s, wrR_s,
                       wrD_s)
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
