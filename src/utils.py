
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Загружает YAML-конфигурацию эксперимента и проаверяет ее параметры"""
    config_path = Path(path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Конфигурация должна содержать словарь параметров YAML!"
        )

    required = {
        "model",
        "data",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "seed",
        "project",
        "name",
    }
    missing = required - config.keys()

    if missing:
        raise ValueError(
            "В конфигурации отсутствуют обязательные параметры: "
            f"{sorted(missing)}"
        )

    # проверка строковых параметров
    for key in ("model", "data", "device", "project", "name"):
        value = config[key]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Параметр '{key}' должен быть непустой строкой."
            )

    # используется type(value) is int (так как bool является подклассом int в Python)
    integer_rules = {
        "epochs": (1, None),
        "imgsz": (1, None),
        "batch": (1, None),
        "workers": (0, None),
        "seed": (0, None),
    }

    for key, (minimum, maximum) in integer_rules.items():
        value = config[key]

        if type(value) is not int:
            raise ValueError(
                f"Параметр '{key}' должен быть целым число!"
            )

        if value < minimum or (
            maximum is not None and value > maximum
        ):
            raise ValueError(
                f"Параметр '{key}' должен быть не меньше {minimum}!"
            )

    # необязательные логические параметры конфигурации:
    boolean_keys = (
        "deterministic",
        "pretrained",
        "amp",
        "val",
        "plots",
        "save",
        "exist_ok",
    )

    for key in boolean_keys:
        if key in config and type(config[key]) is not bool:
            raise ValueError(
                f"Параметр '{key}' должен иметь значение true или false"
            )

    # необязательные числовые параметры:
    if "patience" in config:
        value = config["patience"]
        if type(value) is not int or value < 0:
            raise ValueError(
                "Параметр 'patience' должен быть целым числом >= 0"
            )

    if "save_period" in config:
        value = config["save_period"]
        if type(value) is not int or value < -1:
            raise ValueError(
                "Параметр 'save_period' должен быть целым числом >= -1"
            )

    if "optimizer" in config:
        value = config["optimizer"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Параметр 'optimizer' должен быть непустой строкой!"
            )

    return config


def sha256_file(path: str | Path) -> str:
    """Вычисляет SHA-256 файла"""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Файл для вычисления SHA-256 не найден: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit(project_root: str | Path) -> str | None:
    """Получает хеш текущего Git-коммита или None, если он недоступен"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(package: str) -> str | None:
    """Получает версию установленного пакета или None, если его нет"""
    try:
        return importlib.metadata.version(package)

    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Собирает версии Python, основных библиотек и сведения об ОС"""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics_version": package_version("ultralytics"),
        "torch_version": package_version("torch"),
        "tensorboard_version": package_version("tensorboard"),
        "pyyaml_version": package_version("PyYAML"),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Сохраняет данные в JSON, предварительно создав родительские папки"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def utc_timestamp() -> str:
    """Получает  текущие дату и время в формате ISO 8601 по UTC"""
    return datetime.now(timezone.utc).isoformat()
