@echo off
setlocal
chcp 65001 >nul
title Obrabotka: video = polnyy put, txt = tolko ozvuchka
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "DSH_CONSOLE=1"

rem ODIN universalnyy batnik vmesto dvukh:
rem   - perekin'te VIDEO [mp4/mkv/avi/mov/webm/wmv/flv/ts] --
rem     POLNAYA obrabotka: analiz + perevod + ozvuchka + sborka;
rem   - perekin'te TXT [*_перевод.txt] -- TOLKO ozvuchka po gotovomu
rem     perevodu, bez povtornogo analiza, prodolzhaet s mesta obryva;
rem   - mozhno klast neskolko faylov razom, i video i txt --
rem     kazhdyy obrabotaetsya po svoyemu tipu po ocheredi;
rem   - zapusk bez argumentov -- obrabotayutsya vse video,
rem     lezhashchie ryadom s batnikom, po poryadku.
rem Papka app ishetsya ryadom s batnikom ili na uroven vyshe.

if exist "%~dp0app\analyze_video.py" (cd /d "%~dp0app") else (cd /d "%~dp0..\app")
if not exist "analyze_video.py" (
  echo Ne nayden app\analyze_video.py otnositelno batnika.
  echo Polozhite batnik v papku proekta, ryadom s papkoy app.
  pause
  exit /b 1
)

set /a N=0
if not "%~1"=="" goto dropped

rem --- bez argumentov: vse video iz papki batnika, po poryadku ---
set /a COUNT=0
for %%e in (mp4 mkv avi mov webm wmv flv ts) do for /f "delims=" %%f in ('dir /b /a-d /on "%~dp0*.%%e" 2^>nul') do set /a COUNT+=1
if %COUNT%==0 (
  echo.
  echo   Videofaylov ryadom s batnikom ne naydeno.
  echo   Perekin'te na batnik video -- polnaya obrabotka,
  echo   ili txt-perevod -- tolko ozvuchka.
  echo.
  pause
  exit /b 1
)
for %%e in (mp4 mkv avi mov webm wmv flv ts) do for /f "delims=" %%f in ('dir /b /a-d /on "%~dp0*.%%e" 2^>nul') do call :one "%~dp0%%f"
echo.
echo Vse fayly obrabotany.
pause
exit /b 0

:dropped
for %%f in (%*) do call :one "%%~f"
echo.
echo Vse fayly obrabotany.
pause
exit /b 0

:one
if /i "%~x1"==".txt" goto voicetxt
for %%e in (mp4 mkv avi mov webm wmv flv ts) do if /i "%~x1"==".%%e" goto video
echo.
echo [!] Propuskayu -- ne video i ne txt: %~nx1
goto :eof

:video
set /a N+=1
echo.
echo ================================================================
echo   [%N%] POLNAYA OBRABOTKA: %~nx1
echo ================================================================
python analyze_video.py %1
if errorlevel 1 (
  echo [!] Oshibka na fayle: %~nx1 -- prodolzhayu ochered
)
goto :eof

:voicetxt
set /a N+=1
echo.
echo ================================================================
echo   [%N%] OZVUCHKA PO TXT: %~nx1
echo ================================================================
python analyze_video.py --voice-txt %1
if errorlevel 1 (
  echo [!] Oshibka na fayle: %~nx1 -- prodolzhayu ochered
)
goto :eof
