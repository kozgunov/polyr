$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $projectRoot
& ".\.venv\Scripts\python.exe" ".\запуск\run_dashboard.py"
