@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "outputs\paper\latest.pt" (
  echo ERROR: outputs\paper\latest.pt missing. Do not rerun cDDIM; restore its 200 epoch weights.
  pause
  exit /b 1
)
if not exist "data\QuaDRiGa\NumUEs_100000_num_BerUMaL_ULA.mat" (
  echo ERROR: Missing QuaDRiGa training MAT file.
  pause
  exit /b 1
)
if not exist "data\QuaDRiGa\NumUEs_500_BerUMaL_ULA_OOD_100to200m.mat" (
  echo ERROR: Missing OOD MAT file. Please run earlier MATLAB OOD generator.
  pause
  exit /b 1
)
if not exist "outputs\paper\fig6_id\fig6_cdf.csv" (
  echo ERROR: Missing previous cDDIM Fig6 CSV. Run run_fig6_id.bat first.
  pause
  exit /b 1
)
if not exist "outputs\paper\fig7_ood\fig7_ood_cdf.csv" (
  echo ERROR: Missing previous cDDIM Fig7 CSV. Run run_fig7_ood.bat first.
  pause
  exit /b 1
)
python -c "import torch; print('CUDA available:',torch.cuda.is_available())"
if errorlevel 1 (pause & exit /b 1)
if not exist "outputs\oneshot\latest.pt" (
  echo Training new One-shot U-Net baseline with SAME train split and epochs as saved cDDIM.
  python -u src\train_oneshot.py --cddim-checkpoint outputs\paper\latest.pt --output outputs\oneshot --device auto
  if errorlevel 1 (pause & exit /b 1)
) else (
  echo Existing One-shot weights found. Skipping training. Use resume_oneshot.bat if needed.
)
python src\plot_oneshot_loss.py --input outputs\oneshot\loss.csv --output outputs\oneshot\loss.png
if errorlevel 1 (pause & exit /b 1)
python -u src\compare_oneshot.py --oneshot-checkpoint outputs\oneshot\latest.pt --cddim-checkpoint outputs\paper\latest.pt --ood-data data\QuaDRiGa\NumUEs_500_BerUMaL_ULA_OOD_100to200m.mat --output outputs\oneshot\comparison --id-start 90000 --count 500 --batch-size 64 --seed 42 --device auto
if errorlevel 1 (pause & exit /b 1)
echo SUCCESS. See outputs\oneshot\comparison\ for the new ID and OOD comparison graphs.
pause
