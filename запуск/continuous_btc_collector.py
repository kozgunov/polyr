"""Compatibility launcher for continuous BTC 5m collection."""
try:
    from polybot.collectors.continuous import main
    from polybot.runtime import run_async
except Exception:
    print("error_in_continuous_btc_collector.py", flush=True)
    raise

if __name__ == "__main__":
    run_async(__file__, main)
