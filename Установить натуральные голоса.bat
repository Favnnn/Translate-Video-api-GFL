@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Установка натуральных голосов Microsoft как SAPI-голосов.
echo Голоса ставятся в C:\ProgramData -- папка Vid остаётся переносимой.
python "%~dp0voice_setup.py"
pause
