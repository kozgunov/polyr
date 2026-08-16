"""Обучает отдельную модель выхода v14; рабочую модель автоматически не заменяет."""

from polybot.models.train_exit_sequence import train


if __name__ == "__main__":
    import json
    print(json.dumps(train(), ensure_ascii=False, indent=2))
