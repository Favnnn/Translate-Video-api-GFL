@echo off
setlocal
chcp 65001 >nul
title Video dubbing
set "PYTHONIOENCODING=utf-8"
set "DSH_CONSOLE=1"

rem Odinakovyy batnik dlya lyuboy papki:
rem   - perekin'te video na batnik (mozhno neskolko, vydelyv ih) --
rem     obrabotka po ocheredi, fayl s oshibkoy propuskaetsya;
rem   - zapustite bez argumentov -- obrabotatsya vse video
rem     iz papki, gde lezhit batnik (po poryadku).
rem Papka app ishetsya ryadom s batnikom ili na uroven vyshe.

if exist "%~dp0app\analyze_video.py" (cd /d "%~dp0app") else (cd /d "%~dp0..\app")
if not exist "analyze_video.py" (
  echo Ne nayden app\analyze_video.py otnositelno batnika.
  pause
  exit /b 1
)

set /a N=0
if not "%~1"=="" goto dropped

rem --- bez argumentov: vse video iz papki batnika, po poryadku ---
set /a COUNT=0
for /f "delims=" %%f in ('dir /b /a-d /on "%~dp0*.mp4" "%~dp0*.mkv" "%~dp0*.avi" "%~dp0*.mov" "%~dp0*.webm" "%~dp0*.wmv" "%~dp0*.flv" "%~dp0*.ts" 2^>nul') do set /a COUNT+=1
if %COUNT%==0 (
  echo.
  echo   Videofaylov ryadom s batnikom ne naydeno.
  echo   Perekin'te fayly na batnik myshkoy.
  echo.
  pause
  exit /b 1
)
for /f "delims=" %%f in ('dir /b /a-d /on "%~dp0*.mp4" "%~dp0*.mkv" "%~dp0*.avi" "%~dp0*.mov" "%~dp0*.webm" "%~dp0*.wmv" "%~dp0*.flv" "%~dp0*.ts" 2^>nul') do call :one "%~dp0%%f"
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
set /a N+=1
echo.
echo ================================================================
echo   [%N%] Obrabotka: %~nx1
echo ================================================================
python analyze_video.py %1
if errorlevel 1 (
  echo [!] Oshibka na fayle: %~nx1 -- prodolzhayu ochered
)
goto :eof
