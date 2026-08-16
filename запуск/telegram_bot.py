"""Compatibility launcher for the Telegram bot."""
try:
    from polybot.integrations.telegram_bot import main
    from polybot.runtime import run_async
except Exception:
    print("error_in_telegram_bot.py", flush=True)
    raise

if __name__ == "__main__":
    run_async(__file__, main)
