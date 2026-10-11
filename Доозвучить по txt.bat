@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Dozvuchka po txt (--voice-txt)
set "PYTHONIOENCODING=utf-8"

rem Takoy zhe batnik dlya lyuboy papki proekta:
rem   - peretashchite na nego txt-fayl (mozhno neskolko);
rem   - ili zapustite bez argumentov: vyberite nomer iz spiska
rem     (spisok stroyitsya iz failov *_перевод.txt v papke Output),
rem     libo vstavte polnyy put k txt.
rem Papka app ishetsya ryadom s batnikom ili na uroven vyshe.

if exist "%~dp0app\analyze_video.py" (cd /d "%~dp0app") else (cd /d "%~dp0..\app")
if not exist "analyze_video.py" (
  echo.
  echo   Ne nayden app\analyze_video.py otnositelno batnika.
  echo   Polozhite batnik v papku proekta (ryadom s papkoy app).
  echo.
  pause
  exit /b 1
)

if not "%~1"=="" goto dropped

:menu
set /a CNT=0
for /f "delims=" %%f in ('dir /b /on "%~dp0Output\*_перевод.txt" 2^>nul') do (
  set /a CNT+=1
  set "F!CNT!=%%~ff"
)
echo.
echo   Faily perevoda iz papki Output:
if %CNT%==0 (
  echo   (net failov *_перевод.txt -- vvedite put vrychnuyu)
) else (
  for /l %%i in (1,1,%CNT%) do echo     %%i. !F%%i!
)
echo.
set "CHOICE="
set /p "CHOICE=  Nomer iz spiska ILI polnyy put k txt (Enter -- vyhod): "
if not defined CHOICE exit /b 0
set "CHOICE=%CHOICE:"=%"
set "CHOICE=%CHOICE:&=%"
set "CHOICE=%CHOICE:|=%"
set "TARGET="
if defined F%CHOICE% call set "TARGET=%%F%CHOICE%%"
if not defined TARGET set "TARGET=%CHOICE%"
if not exist "%TARGET%" (
  echo   Ne nayden fayl: "%TARGET%"
  goto menu
)
call :one "%TARGET%"
echo.
echo Vse gotovo.
pause
exit /b 0

:dropped
for %%f in (%*) do call :one "%%~f"
echo.
echo Vse fayly obrabotany.
pause
exit /b 0

:one
if /i not "%~x1"==".txt" (
  echo   [!] Propuskayu (ne txt-fayl): %~nx1
  goto :eof
)
echo.
echo ================================================================
echo   Dozvuchka: %~nx1
echo ================================================================
python analyze_video.py --voice-txt %1
if errorlevel 1 (
  echo   [!] Oshibka na fayle: %~nx1
)
goto :eof
