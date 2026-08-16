"""Строит диагностический отчёт о расхождениях PAPER и LIVE исполнения."""

from polybot.analytics.paper_live_gap import build


if __name__ == "__main__":
    import json
    print(json.dumps(build(), ensure_ascii=False, indent=2))
