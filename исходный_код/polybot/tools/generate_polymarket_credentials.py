"""Create or derive Polymarket CLOB L2 credentials without printing secrets."""

from __future__ import annotations

import importlib.util
import re

import app_config as settings
from py_clob_client_v2 import ClobClient

ROOT = settings.PROJECT_ROOT
CONFIG_PATH = ROOT / "api_config.py"


def load_config():
    spec = importlib.util.spec_from_file_location("project_api_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def credential_value(credentials, *names: str) -> str:
    if isinstance(credentials, dict):
        for name in names:
            value = credentials.get(name)
            if value:
                return str(value)
    for name in names:
        value = getattr(credentials, name, None)
        if value:
            return str(value)
    raise RuntimeError(f"Credential field not found: {names[0]}")


def replace_string_assignment(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=\s*.*$")
    replacement = f"{name} = {value!r}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one assignment for {name}, found {count}")
    return updated


def main() -> None:
    if not CONFIG_PATH.exists():
        raise SystemExit("api_config.py not found")

    config = load_config()
    private_key = str(getattr(config, "POLYMARKET_PRIVATE_KEY", "")).strip()
    if not private_key:
        raise SystemExit("POLYMARKET_PRIVATE_KEY is empty")
    if not private_key.startswith("0x"):
        raise SystemExit("POLYMARKET_PRIVATE_KEY must start with 0x")

    client = ClobClient(
        host=config.POLYMARKET_CLOB_URL,
        chain_id=config.POLYGON_CHAIN_ID,
        key=private_key,
    )
    credentials = client.create_or_derive_api_key()

    api_key = credential_value(credentials, "apiKey", "api_key", "key")
    api_secret = credential_value(credentials, "secret", "api_secret")
    passphrase = credential_value(
        credentials, "passphrase", "api_passphrase"
    )

    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = replace_string_assignment(text, "POLYMARKET_API_KEY", api_key)
    text = replace_string_assignment(text, "POLYMARKET_API_SECRET", api_secret)
    text = replace_string_assignment(
        text, "POLYMARKET_API_PASSPHRASE", passphrase
    )

    temporary = CONFIG_PATH.with_suffix(".py.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(CONFIG_PATH)
    print("OK: Polymarket CLOB credentials were derived and saved.")
    print("Secrets were not printed. Keep api_config.py outside Git.")
    print(CONFIG_PATH)


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
