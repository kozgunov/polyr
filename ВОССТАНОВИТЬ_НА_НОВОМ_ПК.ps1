$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -r .\настройка_проекта\requirements.txt
& .\.venv\Scripts\python.exe .\проверить_целостность_переноса.py
Write-Host "Среда восстановлена. Сначала запустите PAPER:"
Write-Host ".\.venv\Scripts\python.exe .\запустить_проект.py --mode paper"
