@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Пробная озвучка голосов Windows.
echo Каждому русскому голосу -- свой wav в папке 'Пробы голосов'.
echo Натуральные голоса помечены словом НАТУРАЛЬНЫЙ.
python "%~dp0probe_voices.py"
pause
