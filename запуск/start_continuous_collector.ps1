$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot
& '.\.venv\Scripts\python.exe' '.\запуск\continuous_btc_collector.py'
