"""Единая подготовка нового GPU-компьютера; по умолчанию только показывает план."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import app_config as settings
from polybot.tools.migration_audit import verify_existing_manifest


def hardware_audit() -> dict:
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
        cpu = platform.processor() or "unknown"
    except ImportError:
        ram_gb, cpu = None, platform.processor() or "unknown"
    gpu = {"available": False, "name": None, "vram_mb": None, "driver": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, vram, driver = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
            gpu = {"available": True, "name": name, "vram_mb": int(vram), "driver": driver}
    return {
        "os": platform.platform(), "python": platform.python_version(), "cpu": cpu,
        "ram_gb": ram_gb, "gpu": gpu,
        "recommended": {"python": "3.12", "ram_gb": 32, "gpu": "NVIDIA RTX 5070", "vram_gb_min": 8},
    }


def command_plan(cuda_wheel: str = "cu128") -> list[list[str]]:
    project = Path(settings.PROJECT_ROOT)
    python = project / ".venv-gpu" / "Scripts" / "python.exe"
    return [
        ["py", "-3.12", "-m", "venv", str(project / ".venv-gpu")],
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(python), "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{cuda_wheel}"],
        [str(python), "-m", "pip", "install", "-r", str(project / "настройка_проекта" / "requirements.txt")],
        [str(python), "-m", "pip", "install", "-r", str(settings.GPU_REQUIREMENTS_PATH)],
    ]


def verify_gpu(python: Path) -> None:
    code = (
        "import torch; assert torch.cuda.is_available(), 'CUDA недоступна'; "
        "print({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)})"
    )
    subprocess.run([str(python), "-c", code], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подготовка проекта к NVIDIA GPU")
    parser.add_argument("--execute", action="store_true", help="действительно установить окружение")
    parser.add_argument("--build-data", action="store_true", help="после установки пересобрать sequence dataset")
    parser.add_argument("--cuda-wheel", default="cu128")
    args = parser.parse_args(argv)
    commands = command_plan(args.cuda_wheel)
    migration = verify_existing_manifest()
    report = {"mode": "execute" if args.execute else "dry-run", "hardware": hardware_audit(), "migration": migration, "commands": commands}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Ничего не установлено. На новом компьютере добавьте --execute.")
        return 0
    if platform.system() != "Windows":
        raise SystemExit("Этот bootstrap подготовлен для Windows.")
    if settings.MIGRATION_MANIFEST_PATH.exists() and not migration.get("valid"):
        raise SystemExit("MIGRATION_BLOCKED: база или checksum не совпадают с migration_manifest.json")
    for command in commands:
        subprocess.run(command, cwd=settings.PROJECT_ROOT, check=True)
    python = Path(settings.PROJECT_ROOT) / ".venv-gpu" / "Scripts" / "python.exe"
    verify_gpu(python)
    if args.build_data:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(settings.PROJECT_ROOT) / "исходный_код")
        subprocess.run(
            [str(python), "-m", "polybot.models.sequence_dataset"],
            cwd=settings.PROJECT_ROOT, env=environment, check=True,
        )
    print("GPU-окружение готово; обучение намеренно не запущено автоматически.")
    return 0
