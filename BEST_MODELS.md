# Воспроизводимая сборка best-models

Ветка фиксирует код, конфигурацию и компактные обученные артефакты рабочей связки.
Секреты `api_config.py`, рабочая SQLite и многогигабайтные базовые LLM-веса намеренно не входят в Git.

На новом компьютере:

1. Клонировать ветку: `git clone --branch best-models git@github.com:kozgunov/polyr.git poly_project`.
2. Создать `api_config.py` из локальной защищённой копии.
3. Выполнить `powershell -ExecutionPolicy Bypass -File .\ВОССТАНОВИТЬ_НА_НОВОМ_ПК.ps1`.
4. Применить связку: `.\.venv\Scripts\python.exe .\настройка_проекта\применить_best_models.py`.
5. Запустить безопасно: `.\.venv\Scripts\python.exe .\запустить_проект.py start --mode paper`.

LIVE всегда включается вручную с дашборда после preflight. Полный локальный датасет переносится отдельно,
так как GitHub не предназначен для SQLite размером в несколько гигабайт.
