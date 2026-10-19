$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -B src/server.py --port 8083
