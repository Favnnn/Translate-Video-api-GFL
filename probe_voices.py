# -*- coding: utf-8 -*-
# Проба голосов Windows (SAPI): генерирует wav ОДНОЙ фразы каждым
# установленным русским SAPI-голосом. Натуральные голоса Microsoft
# (Svetlana/Dmitry Online Natural из NaturalVoiceSAPIAdapter) помечены
# словом НАТУРАЛЬНЫЙ. Результаты -- в папке "Пробы голосов".
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "Пробы голосов")
os.makedirs(OUT, exist_ok=True)
TEXT = ("Сегодня мы проверяем качество синтеза речи. "
        "Скорость работы и разборчивость каждого голоса.")

try:
    import pythoncom
    import win32com.client
except ImportError:
    print("pywin32 не найден -- проба невозможна")
    input("Enter для выхода...")
    sys.exit(1)

pythoncom.CoInitialize()
v = win32com.client.Dispatch("SAPI.SpVoice")
fs = win32com.client.Dispatch("SAPI.SpFileStream")
fs.Format.Type = 22   # SAFT24kHz16BitMono (родной формат Edge-голосов)

rus = []
for t in v.GetVoices():
    d = t.GetDescription()
    dl = d.lower()
    if ("russia" in dl or "ru-ru" in dl or "irina" in dl
            or "svetlana" in dl or "dmitry" in dl or "russian" in dl):
        rus.append((t, d))
if not rus:
    print("Русские голоса не найдены.")
    input("Enter для выхода...")
    sys.exit(1)

print("Найдено ru-голосов: %d" % len(rus))
ok = 0
for t, d in rus:
    dl = d.lower()
    nat = " [НАТУРАЛЬНЫЙ]" if "natural" in dl else ""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in d)
    path = os.path.join(OUT, "SAPI_" + safe + ".wav")
    try:
        v.Voice = t
        fs.Open(path, 3)   # SSFMCreateForWrite
        v.AudioOutputStream = fs
        v.Speak(TEXT)
        fs.Close()
        v.AudioOutputStream = None
        ok += 1
        print("  OK  %s%s" % (d, nat))
    except Exception as e:
        try:
            fs.Close()
        except Exception:
            pass
        print("  FAIL %s: %s" % (d, e))
print("Озвучено: %d из %d. Папка: %s" % (ok, len(rus), OUT))
input("Нажмите Enter для закрытия...")
