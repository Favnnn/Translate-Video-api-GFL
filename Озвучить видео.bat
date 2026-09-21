@echo off
setlocal
chcp 65001 >nul
title Video dubbing
set "PYTHONIOENCODING=utf-8"
set "DSH_CONSOLE=1"
cd /d "%~dp0app"

if "%~1"=="" goto noparam

:next
if "%~1"=="" goto done
echo.
echo ================================================================
echo   Obrabotka: %~nx1
echo ================================================================
python analyze_video.py "%~1"
shift
goto next

:done
echo.
echo Vse fayly obrabotany.
pause
exit /b 0

:noparam
echo.
echo   Peretaschite videofayl na etot batnik.
echo   Budet sozdano: NAME_perevod.txt + NAME_ozvuchka.mp4 (v papke Output)
echo.
pause
exit /b 0
