"""Compatibility launcher for the one-time configuration migration."""
try:
    from polybot.runtime import run_sync
    from polybot.tools.migrate_api_config import main
except Exception:
    print("error_in_migrate_api_config.py", flush=True)
    raise

if __name__ == "__main__":
    run_sync(__file__, main)
