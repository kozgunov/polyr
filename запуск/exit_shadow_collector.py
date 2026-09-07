"""Запускает безопасный shadow-сборщик исполнимости выходных GTD-заявок."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "исходный_код"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from polybot.trading.exit_shadow_collector import main


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"error_in_{Path(__file__).name}", flush=True)
        raise
