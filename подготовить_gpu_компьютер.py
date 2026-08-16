"""Одна команда для будущего перехода проекта на компьютер с NVIDIA GPU."""

# Сейчас запускайте без аргументов: скрипт только покажет аудит и план.
# На новом компьютере после установки Python 3.12 и драйвера NVIDIA:
# py -3.12 подготовить_gpu_компьютер.py --execute --build-data

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    from polybot.tools.gpu_bootstrap import main
except Exception:
    print("error_in_подготовить_gpu_компьютер.py", flush=True)
    raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("error_in_подготовить_gpu_компьютер.py", flush=True)
        raise

