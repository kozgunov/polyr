"""Run event-level numeric model evaluation; small samples require explicit CLI flag."""

from polybot.models.train_direction_model import main
from polybot.runtime import run_sync

if __name__ == "__main__":
    run_sync(__file__, main)
