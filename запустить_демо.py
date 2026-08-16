"""Ручной запуск безопасной демо-торговли BTC Up/Down 5m."""

# Запускайте этот файл обычной кнопкой Run в IDE.
# Реальные заявки здесь никогда не подписываются и не отправляются.
# Остановка, продолжение, новая сессия и выбор модели доступны на дашборде.

try:
    from polybot.runtime import run_async
    from polybot.trading.paper_engine import main
except Exception:
    print("error_in_запустить_демо.py", flush=True)
    raise


if __name__ == "__main__":
    run_async(__file__, main)
