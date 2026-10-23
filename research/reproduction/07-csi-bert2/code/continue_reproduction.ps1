param([int]$Epochs=30)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { & .\setup.ps1 }
foreach ($dataset in @('WiCount','WiFall')) {
    if (-not (Test-Path "data\$dataset\processed\data_sequence.pkl")) {
        & .\.venv\Scripts\python.exe prepare_extra.py --dataset $dataset --download
        if ($LASTEXITCODE -ne 0) { throw "Failed to prepare $dataset" }
    }
}
& .\.venv\Scripts\python.exe test_protocol.py
if ($LASTEXITCODE -ne 0) { throw 'Protocol checks failed' }
& .\.venv\Scripts\python.exe train_suite.py --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Model training failed' }
& .\.venv\Scripts\python.exe train_baselines.py --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Baseline training failed' }
& .\.venv\Scripts\python.exe verify_baselines.py
if ($LASTEXITCODE -ne 0) { throw 'Baseline verification failed' }
& .\.venv\Scripts\python.exe evaluate_suite.py --runs wigesture_paper_v2 wifall_paper_v2 wicount_paper_v2 wigesture_no_gan wigesture_no_arl wigesture_no_time wigesture_scratch
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
foreach ($run in @('wigesture_paper_v2','wifall_paper_v2','wicount_paper_v2','wigesture_no_gan','wigesture_no_arl','wigesture_no_time','wigesture_scratch')) {
    & .\.venv\Scripts\python.exe verify.py $run
    if ($LASTEXITCODE -ne 0) { throw "Inference verification failed: $run" }
}
& .\.venv\Scripts\python.exe report_suite.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
& .\.venv\Scripts\python.exe audit_suite.py
if ($LASTEXITCODE -ne 0) { throw 'Final experiment audit failed' }
