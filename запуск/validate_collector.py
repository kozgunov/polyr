"""Compatibility launcher for database validation."""
try:
    from polybot.diagnostics.validator import main
    from polybot.runtime import run_sync
except Exception:
    print("error_in_validate_collector.py", flush=True)
    raise

if __name__ == "__main__":
    run_sync(__file__, main)
