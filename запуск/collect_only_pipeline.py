"""Compatibility launcher for the packaged read-only collector."""
try:
    from polybot.collectors.pipeline import main
    from polybot.runtime import run_async
except Exception:
    print("error_in_collect_only_pipeline.py", flush=True)
    raise

if __name__ == "__main__":
    run_async(__file__, main)
