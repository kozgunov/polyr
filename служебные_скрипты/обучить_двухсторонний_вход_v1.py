"""Обучает Up/Down challenger, не заменяя текущий champion."""

from __future__ import annotations

import json

from polybot.models.bidirectional_entry import train


def main() -> None:
    print(json.dumps(train(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
