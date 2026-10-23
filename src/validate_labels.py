
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


# форматы изображений, которые поддерживает проверка
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Допуск для погрешностей при проверке границ рамок
TOLERANCE = 1e-6


def read_classes(path: Path) -> list[str]:
    """Читает названия классов в порядке их ID в YOLO"""

    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден файл со списком классов: {path}"
        )

    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not names:
        raise ValueError(f"Список классов пуст: {path}")

    if len(names) != len(set(names)):
        raise ValueError(
            "В списке классов есть повторяющиеся названия"
        )

    return names


def _validate_label_lines(lines: list[str], label_name: str, class_names: list[str],) -> tuple[list[int], list[str]]:
    """
    Проверяет строки аннотации YOLO.
    Возвращает список ID классов для корректных объектов и список описаний всех найденных ошибок.
    Пустые строки игнорируются. Некорректная строка не мешает проверить остальные строки файла.
    """

    valid_class_ids: list[int] = []
    issues: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        # так как пустая строка не описывает объект
        if not line:
            continue

        parts = line.split()

        # формат YOLO Detection: class_id x_center y_center width height
        if len(parts) != 5:
            issues.append(
                f"{label_name}, строка {line_number}: "
                f"ожидалось 5 значений, получено {len(parts)}"
            )
            continue

        # Преобразование ID класса и координаты в числа
        try:
            class_value = float(parts[0])

            if not math.isfinite(class_value) or not class_value.is_integer():
                raise ValueError

            class_id = int(class_value)
            x_center, y_center, width, height = map(
                float, parts[1:]
            )

        except ValueError:
            issues.append(
                f"{label_name}, строка {line_number}: "
                "ID класса должен быть целым числом, а координаты — числами"
            )
            continue

        # проверка того, что ID существует в списке классов
        if not 0 <= class_id < len(class_names):
            issues.append(
                f"{label_name}, строка {line_number}: "
                f"неизвестный ID класса {class_id}; "
                f"допустимый диапазон: 0–{len(class_names) - 1}"
            )
            continue

        coords = (x_center, y_center, width, height)

        # NaN и бесконечность не являются допустимыми координатами
        if not all(math.isfinite(value) for value in coords):
            issues.append(
                f"{label_name}, строка {line_number}: "
                "координаты содержат NaN или бесконечность"
            )
            continue

        # Координаты центра должны быть в [0, 1], ширина и высота — в интервале (0, 1]
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            issues.append(
                f"{label_name}, строка {line_number}: "
                "координаты или размеры рамки выходят за допустимые значения"
            )
            continue

        # выисление грани рамки по центру и размерам
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        # проверка того, что рамка целиком помещается в изображение
        if (
            x_min < -TOLERANCE
            or y_min < -TOLERANCE
            or x_max > 1 + TOLERANCE
            or y_max > 1 + TOLERANCE
        ):
            issues.append(
                f"{label_name}, строка {line_number}: рамка выходит за границы изображения"
            )
            continue

        # В список попадают только объекты, прошедшие все проверки!
        valid_class_ids.append(class_id)

    return valid_class_ids, issues


def validate_label(label_path: Path, image_path: Path, classes: list[str],) -> list[int]:
    """
    Проверяет одну пару «изображение + YOLO-аннотация».
    Если изображение или аннотация некорректны, то будет исключение.
    Если все правильно, то возвращает ID классов найденных объектов.
    """

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Не найдено изображение: {image_path}"
        )

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Не найден файл разметки: {label_path}"
        )

    # проверка того, что изображение действительно открывается!
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(
            f"Не удалось прочитать изображение {image_path.name}: {exc}"
        ) from exc

    # чтение файла аннотации
    try:
        lines = label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Не удалось прочитать файл разметки "
            f"{label_path.name}: {exc}"
        ) from exc

    class_ids, issues = _validate_label_lines(
        lines=lines,
        label_name=label_path.name,
        class_names=classes,
    )

    if issues:
        raise ValueError(issues[0])

    return class_ids


def validate_dataset(root: Path) -> tuple[list[str], dict]:
    """Проверяет изображения и YOLO-разметку во всем датасете"""

    image_dir = root / "images"
    label_dir = root / "labels"
    class_file = root / "classes.txt"

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка с изображениями: {image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка с разметкой: {label_dir}"
        )

    class_names = read_classes(class_file)

    # СБор изображений и файлов аннотаций
    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )

    label_files = sorted(
        (
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ),
        key=lambda path: path.name.lower(),
    )

    # Нормализация имен без расширения (регистр не имеет значения)
    image_by_stem: dict[str, Path] = {}
    duplicate_image_stems: list[str] = []

    for image_path in images:
        stem = image_path.stem.lower()

        if stem in image_by_stem:
            duplicate_image_stems.append(image_path.name)
        else:
            image_by_stem[stem] = image_path

    label_by_stem: dict[str, Path] = {}
    duplicate_label_stems: list[str] = []

    for label_path in label_files:
        stem = label_path.stem.lower()

        if stem in label_by_stem:
            duplicate_label_stems.append(label_path.name)
        else:
            label_by_stem[stem] = label_path

    # поиск изображений без аннотаций и аннотации без изображений
    missing_labels = sorted(
        set(image_by_stem) - set(label_by_stem)
    )
    orphan_labels = sorted(
        set(label_by_stem) - set(image_by_stem)
    )

    issues: list[str] = []
    class_counts: Counter[int] = Counter()
    total_objects = 0
    empty_labels = 0
    valid_objects = 0
    readable_images = 0

    # Проверка читаемости каждого изображения
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()

            readable_images += 1

        except (OSError, ValueError, UnidentifiedImageError) as exc:
            issues.append(
                f"Не удалось прочитать изображение "
                f"{image_path.name}: {exc}"
            )

    # проверить каждую аннотацию общим валидатором строк!
    for stem, label_path in sorted(label_by_stem.items()):
        try:
            lines = label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
        except (OSError, UnicodeError) as exc:
            issues.append(
                f"Не удалось прочитать файл "
                f"{label_path.name}: {exc}"
            )
            continue

        # Пустой файл допустим, т. к. на изображении может не быть объектов
        if not any(line.strip() for line in lines):
            empty_labels += 1
            continue

        class_ids, label_issues = _validate_label_lines(
            lines=lines,
            label_name=label_path.name,
            class_names=class_names,
        )

        issues.extend(label_issues)

        # учет только объектов, прошедших все проверки
        for class_id in class_ids:
            class_counts[class_id] += 1
            total_objects += 1
            valid_objects += 1

    # фиксирование повторяющихся имен файлов
    if duplicate_image_stems:
        issues.append(
            "Повторяющиеся имена изображений без учета регистра: "
            + ", ".join(duplicate_image_stems)
        )

    if duplicate_label_stems:
        issues.append(
            "Повторяющиеся имена файлов разметки без учета регистра: "
            + ", ".join(duplicate_label_stems)
        )

    # формир-е сводкм для отчета и консольного вывода
    summary = {
        "image_count": len(images),
        "label_count": len(label_files),
        "readable_images": readable_images,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "empty_labels": empty_labels,
        "total_objects": total_objects,
        "valid_objects": valid_objects,
        "class_counts": class_counts,
        "issues": issues,
    }

    return class_names, summary


def main() -> int:
    """Запускает проверку датасета из командной строки"""

    parser = argparse.ArgumentParser(
        description=(
            "Проверка изображений и разметки объектов в формате YOLO"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/traffic_mini_corrected"),
        help=(
            "Папка с подпапками images/, labels/ и файлом classes.txt"
        ),
    )

    args = parser.parse_args()

    try:
        class_names, summary = validate_dataset(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ОШИБКА: {exc}")
        return 2

    print("Результаты проверки датасета")
    print(f"Папка с данными: {args.input.resolve()}")
    print(f"Количество классов: {len(class_names)}")
    print(f"Изображений: {summary['image_count']}")
    print(
        "Изображений удалось прочитать: "
        f"{summary['readable_images']}"
    )
    print(f"Файлов разметки: {summary['label_count']}")
    print(
        "Изображений без разметки: "
        f"{len(summary['missing_labels'])}"
    )
    print(
        "Файлов разметки без изображения: "
        f"{len(summary['orphan_labels'])}"
    )
    print(f"Пустых файлов разметки: {summary['empty_labels']}")
    print(f"Корректных объектов: {summary['valid_objects']}")

    print("\nКоличество объектов по классам:")
    for class_id, class_name in enumerate(class_names):
        print(
            f"  {class_id:>2} {class_name:<12} "
            f"{summary['class_counts'][class_id]}"
        )

    if summary["missing_labels"]:
        print("\nИзображения без соответствующих файлов разметки:")
        for name in summary["missing_labels"]:
            print(f"  {name}")

    if summary["orphan_labels"]:
        print("\nФайлы разметки без соответствующих изображений:")
        for name in summary["orphan_labels"]:
            print(f"  {name}")

    if summary["issues"]:
        print(
            f"\nНайдено технических проблем: "
            f"{len(summary['issues'])}"
        )
        for issue in summary["issues"]:
            print(f"  - {issue}")

    if (
        summary["issues"]
        or summary["missing_labels"]
        or summary["orphan_labels"]
    ):
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА: требуется проверить данные!")
        return 1

    print(
        "\nПРОВЕРКА ПРОЙДЕНА УСПЕШНо: технических проблем не найдено!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
