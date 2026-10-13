@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "outputs\oneshot\latest.pt" (
  echo ERROR: no One-shot checkpoint to resume.
  pause
  exit /b 1
)
python -u src\train_oneshot.py --cddim-checkpoint outputs\paper\latest.pt --output outputs\oneshot --resume --device auto
if errorlevel 1 (pause & exit /b 1)
python src\plot_oneshot_loss.py --input outputs\oneshot\loss.csv --output outputs\oneshot\loss.png
pause
