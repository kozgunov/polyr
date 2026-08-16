"""Compatibility launcher for outcome labeling."""
try:
    from polybot.labeling.resolution import main
    from polybot.runtime import run_sync
except Exception:
    print("error_in_label_training_data.py", flush=True)
    raise

if __name__ == "__main__":
    run_sync(__file__, main)
