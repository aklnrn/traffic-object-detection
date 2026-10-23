
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

from .utils import sha256_file
from .validate_labels import read_classes, validate_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Изображения, которые должны оставаться в одной выборке
# Остальные изображения считаются независимыми группами
RELATED_GROUPS = {
    "G1": {
        "0 - 2025-04-16 16-59-56-449_00009570.jpg",
        "0 - 2025-04-16 16-59-56-449_00026790.jpg",
    },
    "G2": {
        "1_170709082023.png",
        "2_170709082023.png",
        "11_170709082023.png",
        "12_170709082023.png",
    },
    "G3": {
        "01-04-2023 183852.png",
        "05-04-2023 220421.png",
        "05-04-2023 220626.png",
        "05-04-2023 220818.png",
        "05-04-2023 220839.png",
    },
    "G4": {
        "10_9aug202317009.png",
        "102_9aug202317009.png",
    },
    "G5_winter": {
        "Снимок экрана от 2023-06-07 14-47-51.png",
        "Снимок экрана от 2023-06-07 14-48-31.png",
    },
    "G6": {
        "Снимок экрана от 2023-06-08 11-38-38.png",
        "Снимок экрана от 2023-06-08 11-38-52.png",
    },
}


def collect_images(images_dir: Path) -> list[Path]:
    """Собирает изображения и проверяет уникальность их имен"""

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Папка изображений не найдена: {images_dir}"
        )

    images = sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )

    if not images:
        raise ValueError(f"Изображения не найдены: {images_dir}")

    by_stem: dict[str, Path] = {}

    for image_path in images:
        key = image_path.stem.lower()

        if key in by_stem:
            raise ValueError(
                "Найдены изображения с одинаковым именем без расширения: "
                f"{by_stem[key]} и {image_path}. "
                "Сначала нужно устранить неоднозначность имен"
            )

        by_stem[key] = image_path

    return images


def split_images(images: list[Path], image_classes: dict[str, list[int]], seed: int) -> dict[str, list[Path]]:
    """
    Делит датасет на train/val/test с размерами 28/6/6.
    Связанные изображения остаются в одной выборке.
    Среди допустимых вариантов выбирается вариант с более равномерным распределением классов между выборками.
    """

    n = len(images)

    if n != 40:
        raise ValueError(
            f"Ожидалось 40 изображений, найдено {n}. "
            "Нужно проверить входной датасет перед разбиением"
        )

    rng = random.Random(seed)
    targets = {"train": 28, "val": 6, "test": 6}

    image_by_name = {image.name: image for image in images}

    if len(image_by_name) != len(images):
        raise ValueError(
            "Обнаружены изображения с одинаковыми именами!"
        )

    grouped_names: set[str] = set()
    groups: list[tuple[str, list[Path]]] = []

    for group_name, filenames in RELATED_GROUPS.items():
        missing = set(filenames) - image_by_name.keys()

        if missing:
            raise FileNotFoundError(
                f"Для группы {group_name} не найдены файлы: "
                f"{sorted(missing)}"
            )

        overlap = grouped_names & filenames

        if overlap:
            raise ValueError(
                "Изображения входят в несколько групп: "
                f"{sorted(overlap)}"
            )

        grouped_names.update(filenames)
        groups.append(
            (
                group_name,
                [image_by_name[name] for name in sorted(filenames)],
            )
        )

    # Каждое изображение, не входящее в связанную группу, образует отдельную группу из одного элемента
    for image in images:
        if image.name not in grouped_names:
            groups.append((f"single:{image.name}", [image]))

    classes_by_image = {
        image.name: set(image_classes[image.stem.lower()])
        for image in images
    }

    # посчитать, в скольких изображениях встречается каждый класс
    total_class_images: Counter[int] = Counter()

    for class_set in classes_by_image.values():
        total_class_images.update(class_set)

    def choose_subset(available_groups: list[tuple[str, list[Path]]], target_size: int,
    ) -> tuple[list[tuple[str, list[Path]]], list[tuple[str, list[Path]]]] | None:
        """Ищет подмножество заданного размера, не разделяя группы"""

        ordered = list(available_groups)
        rng.shuffle(ordered)

        # динамич. программирование: число изображений -> индексы выбранных групп
        dp: dict[int, list[int]] = {0: []}

        for index, (_, group_images) in enumerate(ordered):
            size = len(group_images)

            for current_size in sorted(list(dp.keys()), reverse=True):
                new_size = current_size + size

                if new_size > target_size or new_size in dp:
                    continue

                dp[new_size] = dp[current_size] + [index]

            if target_size in dp:
                break

        if target_size not in dp:
            return None

        selected_indices = set(dp[target_size])

        selected = [
            group
            for index, group in enumerate(ordered)
            if index in selected_indices
        ]
        remaining = [
            group
            for index, group in enumerate(ordered)
            if index not in selected_indices
        ]

        return selected, remaining

    def flatten(groups_to_flatten: list[tuple[str, list[Path]]],) -> list[Path]:
        """Преобразует список групп в обычный список изображений"""

        return [
            image
            for _, group_images in groups_to_flatten
            for image in group_images
        ]

    def score_split(
        candidate_splits: dict[str, list[Path]],
    ) -> float:
        """Оценивает отклонение распределения классов от ожидаемого"""

        score = 0.0

        for split_name, split_paths in candidate_splits.items():
            target = targets[split_name]
            counts: Counter[int] = Counter()

            for image in split_paths:
                counts.update(classes_by_image[image.name])

            for class_id, total in total_class_images.items():
                expected = total * target / n
                actual = counts[class_id]

                score += (((actual - expected) / max(expected, 1.0)) ** 2 / (total ** 0.5))

        return score

    best_splits: dict[str, list[Path]] | None = None
    best_score = float("inf")

    # перебор вариантов разбиения, чтобы найти подходящий по размерам и с относительно сбалансированными классами
    for _ in range(5000):
        val_result = choose_subset(groups, targets["val"])

        if val_result is None:
            continue

        val_groups, remaining_groups = val_result

        test_result = choose_subset(
            remaining_groups, targets["test"],
        )

        if test_result is None:
            continue

        test_groups, train_groups = test_result

        candidate = {
            "train": flatten(train_groups),
            "val": flatten(val_groups),
            "test": flatten(test_groups),
        }

        if any(
            len(candidate[name]) != targets[name]
            for name in targets):
            continue

        candidate_score = score_split(candidate)

        if candidate_score < best_score:
            best_score = candidate_score
            best_splits = {
                name: sorted(
                    paths, key=lambda image: image.name.lower(),
                )
                for name, paths in candidate.items()
            }

    if best_splits is None:
        raise RuntimeError(
            "Не удалось найти разбиение 28/6/6 без разделения связанных групп."
        )

    image_to_split = {
        image.name: split_name
        for split_name, paths in best_splits.items()
        for image in paths
    }

    # Дополнительная проверка: ни одна связанная группа не должна оказаться в нескольких выборках!
    for group_name, filenames in RELATED_GROUPS.items():
        assigned = {image_to_split[name] for name in filenames}

        if len(assigned) != 1:
            raise RuntimeError(
                f"Связанная группа {group_name} разделена между выборками."
            )

    return best_splits


def save_visualizations(images: list[Path], labels_by_stem: dict[str, Path], classes: list[str], output_dir: Path) -> int:
    """
    Создает изображения с нарисованными bbox для визуальной проверки.
    Исходные изображения и аннотации не изменяются.
    Результаты сохраняются отдельно в output_dir.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Не удалось прочитать изображение: {image_path}"
            )

        label_path = labels_by_stem[image_path.stem.lower()]
        height, width = image.shape[:2]

        lines = label_path.read_text(encoding="utf-8-sig").splitlines()

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            # аннотации уже проверены через validate_label()
            class_id = int(float(parts[0]))
            x_center, y_center, box_width, box_height = map(
                float, parts[1:]
            )

            x1 = round((x_center - box_width / 2) * width)
            y1 = round((y_center - box_height / 2) * height)
            x2 = round((x_center + box_width / 2) * width)
            y2 = round((y_center + box_height / 2) * height)

            # ограничегие координат границами изображения:
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            class_name = classes[class_id]
            label_text = f"{class_id}: {class_name}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_y = max(20, y1 - 6)

            cv2.putText(image, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        destination = output_dir / f"{image_path.stem}.jpg"

        if not cv2.imwrite(str(destination), image):
            raise OSError(
                f"Не удалось сохранить визуализацию: {destination}"
            )

        saved_count += 1

    return saved_count


def main() -> None:
    """Проверяет входные данные и подготавливает датасет YOLO"""

    parser = argparse.ArgumentParser(
        description="Проверка и разбиение датасета для YOLO"
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/raw/traffic_mini_corrected/images"),
        help="Папка с исправленными изображениями",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/raw/traffic_mini_corrected/labels"),
        help="Папка с исправленными YOLO-аннотациями",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("data/raw/traffic_mini_corrected/classes.txt"),
        help="Файл с именами классов в порядке их ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Папка для подготовленного датасета",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для воспроизводимого разбиения",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/dataset_report.md"),
        help="Путь к отчету о подготовке датасета",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/manifest.sha256"),
        help="Путь к манифесту SHA-256",
    )
    parser.add_argument(
        "--visualizations",
        type=Path,
        default=Path("reports/samples/visualizations"),
        help="Папка для визуализаций bbox",
    )

    args = parser.parse_args()

    # Все относительные пути разрешаются от текущей рабочей директории:
    images_dir = args.images.resolve()
    labels_dir = args.labels.resolve()
    classes_path = args.classes.resolve()
    output_dir = args.output.resolve()
    report_path = args.report.resolve()
    manifest_path = args.manifest.resolve()
    visualizations_dir = args.visualizations.resolve()

    # прочитать классы и собрать изображения
    classes = read_classes(classes_path)
    images = collect_images(images_dir)

    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Папка аннотаций не найдена: {labels_dir}"
        )

    image_by_stem = {
        image.stem.lower(): image
        for image in images
    }

    labels_by_stem: dict[str, Path] = {}

    # найти файлы разметки без учета регистра расширения и имени!
    for label_path in sorted(
        labels_dir.rglob("*"),
        key=lambda path: str(path).lower(),
    ):
        if not label_path.is_file():
            continue

        if label_path.suffix.lower() != ".txt":
            continue

        key = label_path.stem.lower()

        if key in labels_by_stem:
            raise ValueError(
                f"Повторяющееся имя аннотации без учета регистра: "
                f"{label_path.name}"
            )

        labels_by_stem[key] = label_path

    missing_labels = set(image_by_stem) - set(labels_by_stem)
    orphan_labels = set(labels_by_stem) - set(image_by_stem)

    if missing_labels or orphan_labels:
        raise ValueError(
            "Имена изображений и аннотаций не совпадают!\n"
            f"Изображения без аннотаций: {sorted(missing_labels)}\n"
            f"Аннотации без изображений: {sorted(orphan_labels)}"
        )

    # проверить каждую пару и собрать классы для балансировки
    image_classes: dict[str, list[int]] = {}
    object_counts: Counter[int] = Counter()
    total_objects = 0

    for key, image_path in image_by_stem.items():
        class_ids = validate_label(
            labels_by_stem[key],
            image_path,
            classes,
        )

        image_classes[key] = class_ids
        object_counts.update(class_ids)
        total_objects += len(class_ids)

    # Рассчитать разбиение до изменения выходных файлов
    splits = split_images(images, image_classes, args.seed)

    # защитить уже подготовленный датасет от перезаписи!
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Папка уже существует и не пустая: {output_dir}\n"
            "Чтобы не перезаписывать данные, нужно выбрать новый --output или отдельно проверить содержимое существующей папки"
        )

    # создание визуализации только после проверки выходного каталога!
    visualization_count = save_visualizations(
        images=images,
        labels_by_stem=labels_by_stem,
        classes=classes,
        output_dir=visualizations_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[str] = []
    split_object_counts: dict[str, Counter[int]] = {}

    # Копирование изображения и аннотации в соответствующие выборки!
    for split_name, split_images_list in splits.items():
        image_output_dir = output_dir / "images" / split_name
        label_output_dir = output_dir / "labels" / split_name

        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)

        split_counter: Counter[int] = Counter()

        for image_path in split_images_list:
            key = image_path.stem.lower()
            label_path = labels_by_stem[key]

            dest_image = image_output_dir / image_path.name
            dest_label = label_output_dir / label_path.name

            shutil.copy2(image_path, dest_image)
            shutil.copy2(label_path, dest_label)

            manifest_entries.append(
                f"{sha256_file(dest_image)}  "
                f"{dest_image.relative_to(output_dir)}"
            )
            manifest_entries.append(
                f"{sha256_file(dest_label)}  "
                f"{dest_label.relative_to(output_dir)}"
            )

            split_counter.update(image_classes[key])

        split_object_counts[split_name] = split_counter

    # создание конфигурации датасета для Ultralytics YOLO
    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {name}"
        for class_id, name in enumerate(classes)
    )

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    manifest_entries.append(f"{sha256_file(dataset_yaml)}  dataset.yaml")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(sorted(manifest_entries)) + "\n", encoding="utf-8")

    # формир-е отчета о подготовке датасета
    report_lines = [
        "# Отчёт подготовки датасета",
        "",
        f"- Исходных изображений: {len(images)}",
        f"- Всего корректных объектов: {total_objects}",
        f"- Классов: {len(classes)}",
        f"- Seed: {args.seed}",
        "- Разбиение: 70% / 15% / 15% (для 40 изображений: 28 / 6 / 6)",
        f"- Визуализаций bbox создано: {visualization_count}",
        f"- Папка визуализаций: `{visualizations_dir}`",
        "",
        "## Размеры выборок",
        "",
        "| Выборка | Изображений | Объектов |",
        "|---|---:|---:|",
    ]

    for split_name in ("train", "val", "test"):
        report_lines.append(
            f"| {split_name} | {len(splits[split_name])} | "
            f"{sum(split_object_counts[split_name].values())} |"
        )

    report_lines.extend(
        [
            "",
            "## Распределение объектов по классам",
            "",
            "| ID | Класс | Всего | Train | Val | Test |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )

    for class_id, class_name in enumerate(classes):
        report_lines.append(
            f"| {class_id} | {class_name} | "
            f"{object_counts[class_id]} | "
            f"{split_object_counts['train'][class_id]} | "
            f"{split_object_counts['val'][class_id]} | "
            f"{split_object_counts['test'][class_id]} |"
        )

    report_lines.extend(
        [
            "",
            "## Контрольные проверки",
            "",
            "- Каждому изображению соответствует файл аннотации.",
            "- Аннотации проверены по формату, ID классов и координатам.",
            "- Связанные изображения не разделены между выборками.",
            "- Исходные изображения и аннотации не изменялись.",
            "- SHA-256 контрольные суммы записаны в `manifest.sha256`.",
            "- Визуализации bbox сохранены отдельно от исходных изображений.",
            "",
            "## Ограничения",
            "",
            "Датасет содержит мало изображений, поэтому редкие классы могут отсутствовать в одной или нескольких выборках. "
            "Балансировка по классам не гарантирует одинакового распределения условий съемки или сложности сцен.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Подготовка датасета завершена")
    print(f"Изображений: {len(images)}")
    print(f"Корректных объектов: {total_objects}")

    for split_name in ("train", "val", "test"):
        print(
            f"{split_name}: {len(splits[split_name])} изображений, "
            f"{sum(split_object_counts[split_name].values())} объектов"
        )

    print(f"YOLO YAML: {dataset_yaml}")
    print(f"Отчет: {report_path}")
    print(f"Манифест SHA-256: {manifest_path}")
    print(f"Визуализации bbox: {visualization_count}")
    print(f"Визуализации: {visualizations_dir}")


if __name__ == "__main__":
    main()
