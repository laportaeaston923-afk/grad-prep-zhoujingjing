$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$basePython = 'D:\anaconda\python.exe'
if (-not (Test-Path -LiteralPath $basePython)) { $basePython = (Get-Command python).Source }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $basePython -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment' }
}
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: transformers==4.57.6 'matplotlib>=3.9'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv\Scripts\python.exe -c "import torch; from transformers import BertModel; print('Torch:',torch.__version__,'CUDA:',torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
