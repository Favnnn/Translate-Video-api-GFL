@echo off
rem ============================================================
rem  Drag and drop a *_perevod.txt file onto this bat:
rem  it runs ONLY the voice-over stage (no re-analysis).
rem  Interrupted jobs resume from where they stopped.
rem  Put this bat next to the "app" folder (Vid\...).
rem ============================================================
rem UTF-8: otherwise python crashes on the "✔" progress char
rem in a console with a legacy codepage (cp866/cp1251)
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "APP=%~dp0app\analyze_video.py"
if not exist "%APP%" (
  echo [!] analyze_video.py not found:
  echo     "%APP%"
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag and drop a txt file onto this bat.
  pause
  exit /b 0
)
:loop
if "%~1"=="" goto done
echo.
echo ============================================================
echo  Voice-over: %~nx1
echo ============================================================
python "%APP%" --voice-txt "%~f1"
shift
goto loop
:done
echo.
echo Done.
pause
