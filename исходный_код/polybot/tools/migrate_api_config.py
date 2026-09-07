"""One-time safe migration from token.py to api_config.py."""

from __future__ import annotations

import re

import app_config as settings

ROOT = settings.PROJECT_ROOT
OLD_PATH = ROOT / "token.py"
NEW_PATH = ROOT / "api_config.py"


def append_if_missing(text: str, name: str, block: str) -> str:
    if re.search(rf"(?m)^{re.escape(name)}\s*=", text):
        return text
    return text.rstrip() + "\n\n" + block.strip() + "\n"


def main() -> None:
    source = NEW_PATH if NEW_PATH.exists() else OLD_PATH
    if not source.exists():
        raise SystemExit("Neither token.py nor api_config.py exists")
    text = source.read_text(encoding="utf-8")

    # Remove forbidden/unused providers without exposing their values.
    text = "\n".join(
        line
        for line in text.splitlines()
        if "BINANCE_" not in line and "CRYPTORANK_" not in line
    ) + "\n"

    text = text.replace(
        "# Обычно 0 = EOA, 1 = POLY_PROXY, 2/3 зависят от версии клиента/типа кошелька.\n"
        "# Не задавайте значение наугад: определим его при подключении кошелька.\n",
        "# 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE, 3=POLY_1271 deposit wallet.\n"
        "# Значение должно соответствовать реальному типу funder wallet.\n",
    )

    text = append_if_missing(
        text,
        "PYTH_HERMES_URL",
        '''
# ---------------------------------------------------------------------------
# Pyth Core BTC/USD: независимый oracle для validation.
# Public Hermes требует API key с 18.08.2026.
# ---------------------------------------------------------------------------
PYTH_HERMES_URL = "https://pyth.dourolabs.app/hermes"
PYTH_API_KEY = ""
PYTH_BTC_USD_FEED_ID = (
    "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
)
''',
    )
    text = append_if_missing(
        text,
        "QWEN_MODEL_ID",
        '''
# ---------------------------------------------------------------------------
# LLM providers. Local Qwen is the default low-cost classifier.
# ---------------------------------------------------------------------------
LLM_PRIMARY_PROVIDER = "qwen_local"
QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN_DEVICE = "cpu"
QWEN_LOCAL_FILES_ONLY = False

OPENAI_API_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5.6-luna"

HUGGINGFACE_API_TOKEN = ""
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = ""
GROQ_API_KEY = ""
GROQ_MODEL = ""
''',
    )
    text = append_if_missing(
        text,
        "POLYMARKET_RELAYER_API_KEY",
        '''
# ---------------------------------------------------------------------------
# Optional Polymarket builder/relayer metadata (not CLOB credentials).
# ---------------------------------------------------------------------------
POLYMARKET_BUILDER_ADDRESS = ""
POLYMARKET_BUILDER_CODE = ""
POLYMARKET_RELAYER_API_KEY = ""
POLYMARKET_RELAYER_API_KEY_ADDRESS = ""
''',
    )

    temporary = NEW_PATH.with_suffix(".py.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(NEW_PATH)
    if OLD_PATH.exists() and OLD_PATH != NEW_PATH:
        OLD_PATH.unlink()

    gitignore = ROOT / ".gitignore"
    ignored = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if "api_config.py" not in ignored.splitlines():
        gitignore.write_text(ignored.rstrip() + "\napi_config.py\n", encoding="utf-8")
    print("OK: configuration migrated to api_config.py; secrets were not printed.")


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
