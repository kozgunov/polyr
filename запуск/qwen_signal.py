"""Compatibility launcher for the Qwen signal classifier."""
try:
    from polybot.models.qwen_signal import main
    from polybot.runtime import run_sync
except Exception:
    print("error_in_qwen_signal.py", flush=True)
    raise

if __name__ == "__main__":
    run_sync(__file__, main)
