"""Ручной запуск проверки готовности к реальной торговле."""

# Файл пока выполняет только защитные проверки и не отправляет реальные ордера.
# Live останется заблокированным до достаточной demo-статистики и отдельного аудита.

try:
    from polybot.runtime import run_sync
    from polybot.trading.live_guard import main
except Exception:
    print("error_in_запустить_реальную_торговлю.py", flush=True)
    raise


if __name__ == "__main__":
    run_sync(__file__, main)
