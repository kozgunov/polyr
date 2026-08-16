"""Единая точка запуска локального Polybot: сборщик, торговый движок и дашборд.

Обычный запуск: ``python запустить_проект.py``.
Полный перезапуск: ``python запустить_проект.py restart``.
Остановка: ``python запустить_проект.py stop``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
REQUIREMENTS = PROJECT_ROOT / "настройка_проекта" / "requirements.txt"
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
LOG_DIR = PROJECT_ROOT / "журналы"
PID_FILE = LOG_DIR / "project_processes.json"
DASHBOARD_URL = "http://127.0.0.1:8765"

SERVICES = {
    "collector": PROJECT_ROOT / "запуск" / "continuous_btc_collector.py",
    "trading": PROJECT_ROOT / "запустить_демо.py",
    "dashboard": PROJECT_ROOT / "запуск" / "run_dashboard.py",
}


def ensure_environment(force_install: bool) -> None:
    created = False
    if not VENV_PYTHON.exists():
        print("Создаю локальное окружение .venv…", flush=True)
        venv.EnvBuilder(with_pip=True).create(PROJECT_ROOT / ".venv")
        force_install = True
        created = True
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        forwarded = list(sys.argv[1:])
        if created and "--install" not in forwarded:
            forwarded.append("--install")
        arguments = [str(VENV_PYTHON), str(Path(__file__).resolve()), *forwarded, "--inside-venv"]
        os.execv(str(VENV_PYTHON), arguments)
    if force_install:
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            cwd=PROJECT_ROOT, check=True,
        )


def load_manifest() -> dict[str, dict[str, object]]:
    try:
        return json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_manifest(value: dict[str, dict[str, object]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = PID_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(PID_FILE)


def project_process(entry: dict[str, object]):
    import psutil

    try:
        process = psutil.Process(int(entry["pid"]))
        expected_created = float(entry.get("created_at", 0))
        command = " ".join(process.cmdline()).lower()
        if expected_created and abs(process.create_time() - expected_created) > 2:
            return None
        if str(PROJECT_ROOT).lower() not in command:
            return None
        return process
    except (psutil.Error, KeyError, TypeError, ValueError):
        return None


def stop_services() -> None:
    import psutil

    manifest = load_manifest()
    processes = [process for entry in manifest.values() if (process := project_process(entry))]
    # Подхватываем только точные сервисы проекта, если старый запуск не успел записать PID-файл.
    service_names = {path.name.lower() for path in SERVICES.values()}
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            if str(PROJECT_ROOT).lower() in command and any(name in command for name in service_names):
                if all(existing.pid != process.pid for existing in processes):
                    processes.append(process)
        except psutil.Error:
            continue
    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=8)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    save_manifest({})


def start_services() -> dict[str, dict[str, object]]:
    import psutil

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), str(PROJECT_ROOT)))
    # Журналы служб должны обновляться сразу: это важно для контроля задержки
    # данных и отличия текущей ошибки от старой записи после восстановления.
    environment["PYTHONUNBUFFERED"] = "1"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for name, script in SERVICES.items():
        existing = project_process(manifest.get(name, {})) if name in manifest else None
        if existing is None:
            expected_script = str(script).lower()
            for candidate in psutil.process_iter(["pid", "cmdline", "create_time"]):
                try:
                    command = " ".join(candidate.info.get("cmdline") or []).lower()
                    if expected_script in command:
                        existing = candidate
                        manifest[name] = {
                            "pid": candidate.pid,
                            "created_at": float(candidate.info.get("create_time") or candidate.create_time()),
                            "script": str(script), "log": str(LOG_DIR / f"{name}.log"),
                        }
                        break
                except psutil.Error:
                    continue
        if existing:
            continue
        log_path = LOG_DIR / f"{name}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [str(VENV_PYTHON), str(script)], cwd=PROJECT_ROOT, env=environment,
                stdout=log_handle, stderr=subprocess.STDOUT, creationflags=creation_flags,
            )
        finally:
            log_handle.close()
        created_at = psutil.Process(process.pid).create_time()
        manifest[name] = {
            "pid": process.pid, "created_at": created_at, "script": str(script), "log": str(log_path),
        }
    save_manifest(manifest)
    return manifest


def dashboard_ready(timeout: float = 35.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{DASHBOARD_URL}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(0.7)
    return False


def show_status() -> None:
    manifest = load_manifest()
    for name in SERVICES:
        process = project_process(manifest.get(name, {})) if name in manifest else None
        print(f"{name}: {'работает' if process else 'остановлен'}")


def request_trading_mode(mode: str) -> dict[str, object] | None:
    """Выбирает режим через штатный API дашборда, не обходя LIVE-проверки."""
    if mode == "saved":
        return None
    payload = json.dumps({"mode": mode, "canary": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{DASHBOARD_URL}/api/trading-mode", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Не удалось включить режим {mode}: HTTP {error.code}: {detail}") from error


def warm_dashboard_overview() -> None:
    """Один раз строит тяжёлую сводку до открытия браузера; дальше работает минутный кэш."""
    with urllib.request.urlopen(f"{DASHBOARD_URL}/api/overview", timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"Сводка дашборда не готова: HTTP {response.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Единый запуск Polybot")
    parser.add_argument("action", nargs="?", choices=("start", "restart", "stop", "status"), default="start")
    parser.add_argument("--install", action="store_true", help="Переустановить зависимости из requirements.txt")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать локальный дашборд")
    parser.add_argument(
        "--mode", choices=("saved", "paper", "live"), default="saved",
        help="saved — сохранить режим; paper — демо; live — запросить LIVE через штатные проверки",
    )
    parser.add_argument("--inside-venv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    ensure_environment(args.install)
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    if args.action == "status":
        show_status()
        return
    if args.action in {"stop", "restart"}:
        stop_services()
    if args.action == "stop":
        print("Проект остановлен")
        return
    start_services()
    if not dashboard_ready():
        show_status()
        raise RuntimeError(f"Дашборд не ответил; проверьте журналы в {LOG_DIR}")
    mode_result = request_trading_mode(args.mode)
    if mode_result is not None:
        print(f"Торговый режим: {json.dumps(mode_result, ensure_ascii=False)}", flush=True)
    warm_dashboard_overview()
    print(f"Проект запущен: {DASHBOARD_URL}", flush=True)
    if not args.no_browser:
        webbrowser.open(DASHBOARD_URL)


if __name__ == "__main__":
    try:
        main()
        print(f"all_done{Path(__file__).name}", flush=True)
    except Exception:
        print(f"error_in_{Path(__file__).name}", flush=True)
        raise
