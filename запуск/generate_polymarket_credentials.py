"""Compatibility launcher for local CLOB credential derivation."""
try:
    from polybot.runtime import run_sync
    from polybot.tools.generate_polymarket_credentials import main
except Exception:
    print("error_in_generate_polymarket_credentials.py", flush=True)
    raise

if __name__ == "__main__":
    run_sync(__file__, main)
