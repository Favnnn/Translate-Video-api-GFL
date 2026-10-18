# -*- coding: utf-8 -*-
# Установка натуральных голосов Microsoft (Svetlana Natural и др.) как
# SAPI-голосов. Делает всё сам:
#   1) качает NaturalVoiceSAPIAdapter в C:\ProgramData (НЕ в Vid --
#      Vid остаётся переносимым, реестр запоминает системный путь);
#   2) включает онлайн-голоса Edge и русский язык (HKCU, без админа);
#   3) запускает Installer.exe от администратора -- в его окне нужно
#      нажать Install, дождаться галочек и ЗАКРЫТЬ окно;
#   4) после закрытия проверяет регистрацию и печатает результат.
import ctypes
import os
import shutil
import sys
import tempfile
import zipfile

import winreg

PROG = os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
                    "NaturalVoiceSAPIAdapter")
ZIP_URL = ("https://github.com/gexgd0419/NaturalVoiceSAPIAdapter"
           "/releases/download/v0.2.10/"
           "NaturalVoiceSAPIAdapter_v0.2.10_x86_x64.zip")
TOKEN_KEYS = (
    r"SOFTWARE\Microsoft\Speech\Voices\Tokens",
    r"SOFTWARE\WOW6432Node\Microsoft\Speech\Voices\Tokens",
)


def installed():
    """Голоса уже зарегистрированы в SAPI? Адаптер регистрирует их
    через CLSID, а не через ключи Tokens -- проверяем перечислением
    SAPI (как это видит программа), запасной путь -- известные CLSID."""
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        v = win32com.client.Dispatch("SAPI.SpVoice")
        for t in v.GetVoices():
            d = t.GetDescription().lower()
            if ("svetlana online" in d or "natural" in d) \
                    and ("russian" in d or "ru-ru" in d or "ru" in d):
                return True
    except Exception:
        pass
    for clsid in ("{013ab33b-ad1a-401c-8bee-f6e2b046a94e}",
                  "{b8b9e38f-e5a2-4661-9fde-4ac7377aa6f6}"):
        for hive_sub in (r"SOFTWARE\Classes\CLSID",
                         r"SOFTWARE\Classes\WOW6432Node\CLSID"):
            try:
                winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               hive_sub + "\\" + clsid)
                return True
            except OSError:
                pass
    return False


def download():
    """Скачивание zip в TEMP (прокси берётся из настроек Windows сам)."""
    import urllib.request
    dst = os.path.join(tempfile.gettempdir(), "NaturalVoiceSAPIAdapter.zip")
    print("Скачиваю пакет (~22 МБ)...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
        total = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            total += len(b)
    print("Скачано: %.1f МБ" % (total / 1048576.0))
    return dst


def prepare():
    """Распаковка в ProgramData и конфиг (русский язык, Edge-голоса)."""
    if not os.path.isfile(os.path.join(PROG, "Installer.exe")):
        zp = download()
        print("Распаковываю в %s ..." % PROG)
        if os.path.isdir(PROG):
            shutil.rmtree(PROG, ignore_errors=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(PROG)
        try:
            os.remove(zp)
        except OSError:
            pass
    else:
        print("Пакет уже скачан: %s" % PROG)
    # конфиг enumerator'а: включаем Edge-онлайн голоса и русский
    # (это HKCU -- права администратора не нужны)
    try:
        k = winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                             r"Software\NaturalVoiceSAPIAdapter\Enumerator")
        winreg.SetValueEx(k, "NoEdgeVoices", 0, winreg.REG_DWORD, 0)
        winreg.SetValueEx(k, "EdgeVoiceAllLanguages", 0,
                          winreg.REG_DWORD, 0)
        winreg.SetValueEx(k, "EdgeVoiceLanguages", 0,
                          winreg.REG_MULTI_SZ, ["ru"])
        winreg.CloseKey(k)
        print("Русский язык для онлайн-голосов включён.")
    except OSError as e:
        print("Не удалось записать конфиг (не критично): %s" % e)


def run_installer():
    """Установщик от администратора; ждём закрытия его окна."""
    exe = os.path.join(PROG, "Installer.exe")
    print("\nОткрываю установщик. В его окне:")
    print("  1) нажмите Install (и подтвердите запрос Windows);")
    print("  2) дождитесь отметок 32-bit / 64-bit в Installation Status;")
    print("  3) ЗАКРОЙТЕ окно установщика -- изменения применятся.")
    rc = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", exe, None, PROG, 1)   # SW_SHOWNORMAL
    if rc <= 32:
        print("Не удалось запустить от админа (код %s) -- запускаю обычно."
              % rc)
        os.startfile(exe)
    # ждём, пока окно закроют
    input("После закрытия окна установщика нажмите здесь Enter...")


def main():
    if installed():
        print("Натуральные голоса УЖЕ установлены в SAPI.")
        print("Если нужно переустановить -- сначала Uninstall в Installer.")
        return 0
    prepare()
    run_installer()
    if installed():
        print("\nГОТОВО: натуральные голоса зарегистрированы в SAPI.")
        print("Теперь запустите 'Проба голосов.bat' и послушайте Svetlana.")
        return 0
    print("\nПока НЕ зарегистрировано. Откройте Installer.exe в %s и"
          % PROG)
    print("нажмите Install, затем закройте окно -- и запустите этот")
    print("батник снова для проверки.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("ОШИБКА: %s" % e)
        input("Enter для выхода...")
        sys.exit(1)
