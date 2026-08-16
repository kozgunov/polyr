"""Create compact CSV/JSONL files that can be opened directly."""

from polybot.exporting.data_export import main
from polybot.runtime import run_sync

if __name__ == "__main__":
    run_sync(__file__, main)
