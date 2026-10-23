from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.tensorboard_logger import TensorBoardLogger
from src.utils import (
    collect_environment,
    get_git_commit,
    load_config,
    sha256_file,
    utc_timestamp,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SOURCE_FILES = (
    "src/run_experiment.py",
    "src/utils.py",
    "src/tensorboard_logger.py",
)


def calculate_source_hashes() -> dict[str, str]:
    """Вычисляет SHA-256 ключевых файлов исходного кода"""
    hashes = {}

    for relative_path in SOURCE_FILES:
        file_path = ROOT / relative_path
        hashes[relative_path] = sha256_file(file_path)

    return hashes

def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки"""
    parser = argparse.ArgumentParser(
        description="Воспроизводимый эксперимент детекции объектов YOLO."
    )
    parser.add_argument(
        "--config",
        default="configs/yolo.yaml",
        help="Путь к YAML-конфигурации относительно корня проекта.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Путь к dataset.yaml; при указании переопределяет значение из конфигурации.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить данные и загрузку модели без обучения.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Переопределить число эпох для короткого тестового обучения.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    """Разрешает относительный путь относительно корня проекта"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def set_random_seed(seed: int, deterministic: bool) -> None:
    """Устанавливает seed для генераторов случайных чисел"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def validate_dataset_yaml(data_yaml: Path) -> dict[str, Any]:
    """Проверяет YAML датасета и наличие файлов каждого раздела"""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Файл датасета не найден: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        dataset = yaml.safe_load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Конфигурация датасета должна быть YAML-словарем")

    names = dataset.get("names")
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError("В dataset.yaml отсутствует непустой список классов names")

    dataset_root_value = dataset.get("path", ".")
    dataset_root = Path(dataset_root_value)

    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    dataset_root = dataset_root.resolve()
    split_files: dict[str, list[Path]] = {}

    for split in ("train", "val", "test"):
        split_value = dataset.get(split)

        if not split_value:
            raise ValueError(f"В dataset.yaml отсутствует путь для раздела {split!r}.")

        entries = split_value if isinstance(split_value, list) else [split_value]
        images: list[Path] = []

        for entry in entries:
            split_path = Path(entry)
            if not split_path.is_absolute():
                split_path = dataset_root / split_path

            if not split_path.exists():
                raise FileNotFoundError(
                    f"Путь раздела {split!r} не найден: {split_path}"
                )

            if split_path.is_dir():
                images.extend(
                    path
                    for path in split_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(split_path)
            else:
                raise ValueError(
                    f"Путь раздела {split!r} не является каталогом "
                    f"или файлом изображения: {split_path}"
                )

        images = sorted(set(path.resolve() for path in images))

        if not images:
            raise ValueError(f"В разделе {split!r} не найдены изображения.")

        split_files[split] = images

    return {
        "config": dataset,
        "root": dataset_root,
        "split_files": split_files,
        "class_count": len(names),
    }


def create_ultralytics_dataset_yaml(dataset_info: dict[str, Any], output_path: Path,) -> Path:
    """Создает YAML для Ultralytics с абсолютным корнем датасета"""
    dataset_config = dict(dataset_info["config"])
    dataset_config["path"] = str(dataset_info["root"].resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            dataset_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    return output_path.resolve()


def calculate_dataset_hash(data_yaml: Path, dataset_info: dict[str, Any],) -> tuple[str, int]:
    """Вычисляет отпечаток YAML и изображений с соответствующими метками"""
    files: set[Path] = {data_yaml.resolve()}

    # учитываются изображения и соответствующие каталоги YOLO-разметки
    for images in dataset_info["split_files"].values():
        for image_path in images:
            files.add(image_path)

            parts = image_path.parts
            if "images" in parts:
                index = len(parts) - 1 - parts[::-1].index("images")
                label_parts = list(parts[:index]) + [
                    "labels",
                    *parts[index + 1 :],
                ]
                label_path = Path(*label_parts).with_suffix(".txt")
                if label_path.is_file():
                    files.add(label_path.resolve())

    digest = hashlib.sha256()

    for path in sorted(files, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue

        try:
            relative_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative_path = path.name

        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")

    return digest.hexdigest(), len(files)


def create_run_directory(base_dir: Path, name: str) -> Path:
    """Создает отдельный каталог запуска с уникальным именем"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = base_dir / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def read_training_metrics(results_csv: Path, logger: TensorBoardLogger) -> dict[str, float]:
    """Читает потери и метрики по эпохам из CSV Ultralytics"""
    if not results_csv.is_file():
        raise FileNotFoundError(
            f"Ultralytics не создал файл с результатами: {results_csv}"
        )

    last_metrics: dict[str, float] = {}

    loss_components = {
        "train": {
            "box_loss": "train/box_loss",
            "cls_loss": "train/cls_loss",
            "dfl_loss": "train/dfl_loss",
        },
        "val": {
            "box_loss": "val/box_loss",
            "cls_loss": "val/cls_loss",
            "dfl_loss": "val/dfl_loss",
        },
    }

    metric_names = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
    }

    with results_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                step = int(float(row["epoch"])) + 1
            except (KeyError, TypeError, ValueError):
                continue

            numeric_values: dict[str, float] = {}

            for key, raw_value in row.items():
                if raw_value is None or not raw_value.strip():
                    continue

                try:
                    numeric_values[key.strip()] = float(raw_value)
                except ValueError:
                    continue

            # отдельные компоненты потерь и их сумма для train и val
            for split, components in loss_components.items():
                component_values: list[float] = []

                for component_name, csv_column in components.items():
                    if csv_column not in numeric_values:
                        continue

                    value = numeric_values[csv_column]
                    logger.log_metric(
                        component_name,
                        value,
                        step,
                        prefix=split,
                    )
                    last_metrics[f"{split}/{component_name}"] = value
                    component_values.append(value)

                # суммирование только при наличии всех трех компонентов
                if len(component_values) == len(components):
                    total_loss = sum(component_values)
                    logger.log_metric(
                        "loss",
                        total_loss,
                        step,
                        prefix=split,
                    )
                    last_metrics[f"{split}/loss"] = total_loss

            # Метрики качества детекции
            for source, target in metric_names.items():
                if source not in numeric_values:
                    continue

                value = numeric_values[source]
                logger.log_metric(
                    target,
                    value,
                    step,
                    prefix="metrics",
                )
                last_metrics[target] = value

    return last_metrics

def log_saved_images(run_dir: Path, logger: TensorBoardLogger) -> list[str]:
    """Добавляет доступные изображения результатов в TensorBoard"""
    artifacts_dir = run_dir / "ultralytics"
    candidates = (
        ("artifacts/confusion_matrix", artifacts_dir / "confusion_matrix.png"),
        ("artifacts/confusion_matrix_normalized",
         artifacts_dir / "confusion_matrix_normalized.png"),
        ("predictions/validation_batch_0",
         artifacts_dir / "val_batch0_pred.jpg"),
        ("predictions/validation_batch_1",
         artifacts_dir / "val_batch1_pred.jpg"),
    )

    logged: list[str] = []

    for tag, image_path in candidates:
        if image_path.is_file():
            logger.log_image(tag, image_path)
            logged.append(str(image_path.relative_to(ROOT)))

    return logged


def main() -> int:
    """Проверяет окружение и запускает dry-run либо обучение"""
    args = parse_args()

    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs должен быть положительным целым числом.")

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)

    if args.data is not None:
        config["data"] = args.data

    data_yaml = resolve_project_path(config["data"])
    dataset_info = validate_dataset_yaml(data_yaml)
    dataset_hash, dataset_file_count = calculate_dataset_hash(
        data_yaml, dataset_info
    )

    epochs = args.epochs if args.epochs is not None else config["epochs"]
    set_random_seed(config["seed"], config.get("deterministic", True))

    project_dir = resolve_project_path(config["project"])
    run_dir = create_run_directory(project_dir, config["name"])
    tensorboard_dir = run_dir / "tensorboard"
    ultralytics_dir = run_dir / "ultralytics"

    # сохранить точную конфигурацию, использованную в этом запуске!
    run_config = dict(config)
    run_config["data"] = str(data_yaml.relative_to(ROOT))
    run_config["epochs"] = epochs
    shutil.copy2(config_path, run_dir / "source_config.yaml")
    write_json(run_dir / "effective_config.json", run_config)

    metadata: dict[str, Any] = {
        "run_name": run_dir.name,
        "status": "started",
        "started_at_utc": utc_timestamp(),
        "finished_at_utc": None,
        "dry_run": args.dry_run,
        "training_started": False,
        "project_revision": get_git_commit(ROOT),
        "source_files_sha256": calculate_source_hashes(),
        "config_sha256": sha256_file(config_path),
        "dataset": {
            "yaml": str(data_yaml.relative_to(ROOT)),
            "sha256": dataset_hash,
            "file_count": dataset_file_count,
            "class_count": dataset_info["class_count"],
            "images_per_split": {
                split: len(paths)
                for split, paths in dataset_info["split_files"].items()
            },
        },
        "environment": collect_environment(),
        "parameters": run_config,
    }

    write_json(run_dir / "experiment.json", metadata)

    print(f"Папка эксперимента: {run_dir.relative_to(ROOT)}")
    print(f"Изображения: {metadata['dataset']['images_per_split']}")
    print(f"SHA-256 датасета: {dataset_hash}")

    try:
        with TensorBoardLogger(tensorboard_dir) as logger:
            logger.log_parameters(metadata)
            logger.log_text(
                "experiment/status",
                "Проверка эксперимента начата."
                if args.dry_run
                else "Обучение YOLO запущено.",
            )

            model_spec = str(config["model"])
            local_model_path = resolve_project_path(model_spec)

            # Если весов нет в проекте, Ultralytics может загрузить их по имени
            model_source = (
                str(local_model_path)
                if local_model_path.is_file()
                else model_spec
            )
            model = YOLO(model_source)

            if args.dry_run:
                metadata["status"] = "dry_run_passed"
                metadata["training_started"] = False
                metadata["finished_at_utc"] = utc_timestamp()
                metadata["model_loaded"] = True
                write_json(run_dir / "experiment.json", metadata)

                logger.log_text(
                    "experiment/status",
                    "Dry-run успешно завершен: данные проверены, модель загружена, обучение не запускалось.",
                )
                logger.flush()

                print("Dry-run успешно завершен. Обучение не запускалось.")
                print(f"TensorBoard: {tensorboard_dir.relative_to(ROOT)}")
                return 0

            ultralytics_data_yaml = create_ultralytics_dataset_yaml(
                dataset_info,
                run_dir / "ultralytics_dataset.yaml",
            )

            train_args: dict[str, Any] = {
                "data": str(ultralytics_data_yaml),
                "epochs": epochs,
                "patience": config.get("patience", 10),
                "imgsz": config["imgsz"],
                "batch": config["batch"],
                "device": config["device"],
                "workers": config["workers"],
                "seed": config["seed"],
                "deterministic": config.get("deterministic", True),
                "pretrained": config.get("pretrained", True),
                "optimizer": config.get("optimizer", "auto"),
                "amp": config.get("amp", False),
                "val": config.get("val", True),
                "plots": config.get("plots", True),
                "save": config.get("save", True),
                "save_period": config.get("save_period", -1),
                "project": str(run_dir),
                "name": "ultralytics",
                "exist_ok": True,
                "verbose": True,
            }

            metadata["training_started"] = True
            metadata["training_arguments"] = train_args
            write_json(run_dir / "experiment.json", metadata)

            model.train(**train_args)

            results_csv = ultralytics_dir / "results.csv"
            final_metrics = read_training_metrics(results_csv, logger)
            logged_images = log_saved_images(run_dir, logger)

            metadata["status"] = "completed"
            metadata["finished_at_utc"] = utc_timestamp()
            metadata["results_csv"] = str(results_csv.relative_to(ROOT))
            metadata["final_metrics"] = final_metrics
            metadata["tensorboard_images"] = logged_images
            metadata["checkpoints"] = [
                str(path.relative_to(ROOT))
                for path in (ultralytics_dir / "weights" / "best.pt",
                             ultralytics_dir / "weights" / "last.pt")
                if path.is_file()
            ]
            write_json(run_dir / "experiment.json", metadata)

            logger.log_text(
                "experiment/status",
                "Обучение завершено. Итоговые метрики и артефакты сохранены.",
            )
            logger.log_parameters(metadata)
            logger.flush()

        print("Эксперимент успешно завершен.")
        print(f"Результаты: {run_dir.relative_to(ROOT)}")
        print(f"TensorBoard: {tensorboard_dir.relative_to(ROOT)}")
        return 0

    except Exception as error:
        metadata["status"] = "failed"
        metadata["finished_at_utc"] = utc_timestamp()
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(run_dir / "experiment.json", metadata)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Ошибка эксперимента: {error}", file=sys.stderr)
        raise SystemExit(1)
