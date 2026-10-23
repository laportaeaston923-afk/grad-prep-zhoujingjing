param([int]$Epochs=30,[string]$RunName='paper_run',[string]$Stage='all',[string]$Dataset='WiGesture',[switch]$PaperMask=$true)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { & .\setup.ps1 }
if ($Dataset -ne 'WiGesture' -and -not (Test-Path "data\$Dataset\processed\data_sequence.pkl")) {
    & .\.venv\Scripts\python.exe prepare_extra.py --dataset $Dataset --download
    if ($LASTEXITCODE -ne 0) { throw 'Extra data preparation failed' }
}
if ($Dataset -eq 'WiGesture' -and -not (Test-Path 'data\processed\data_sequence.pkl')) {
    & .\.venv\Scripts\python.exe prepare_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
}
$trainArgs = @('run.py','--stage',$Stage,'--epochs',$Epochs,'--run-name',$RunName,'--dataset',$Dataset)
if ($PaperMask) { $trainArgs += '--paper-mask' }
& .\.venv\Scripts\python.exe @trainArgs
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect console output' }
