"""Consistent success/error markers for executable entry points."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


def file_marker(path: str) -> str:
    return Path(path).name


def run_sync(path: str, function: Callable[[], Any]) -> None:
    name = file_marker(path)
    try:
        function()
    except SystemExit as exc:
        marker = f"all_done{name}" if exc.code in (0, None) else f"error_in_{name}"
        print(marker, flush=True)
        raise
    except KeyboardInterrupt:
        print(f"all_done{name}", flush=True)
        raise
    except Exception:
        print(f"error_in_{name}", flush=True)
        raise
    else:
        print(f"all_done{name}", flush=True)


def run_async(path: str, function: Callable[[], Awaitable[Any]]) -> None:
    name = file_marker(path)
    try:
        asyncio.run(function())
    except SystemExit as exc:
        marker = f"all_done{name}" if exc.code in (0, None) else f"error_in_{name}"
        print(marker, flush=True)
        raise
    except KeyboardInterrupt:
        print(f"all_done{name}", flush=True)
        raise
    except Exception:
        print(f"error_in_{name}", flush=True)
        raise
    else:
        print(f"all_done{name}", flush=True)
