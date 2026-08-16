"""Compatibility launcher for oracle diagnostics."""
try:
    from polybot.diagnostics.oracles import main
    from polybot.runtime import run_async
except Exception:
    print("error_in_check_oracles.py", flush=True)
    raise

if __name__ == "__main__":
    run_async(__file__, main)
