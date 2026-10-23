from pathlib import Path

import pytest
from PIL import Image

from src.validate_labels import (
    read_classes,
    validate_dataset,
    validate_label,
)


def create_image(path: Path, color: str = "white") -> None:
    """Создает небольшое тестовое изображение размером 100 на 100 пикселей"""
    Image.new("RGB", (100, 100), color=color).save(path)


def create_dataset(root: Path) -> tuple[Path, Path]:
    """Создает минимальную структуру тестового набора данных.
    Создаются каталоги с изображениями и разметкой, а также файл classes.txt с названиями двух классов объектов.
    """
    image_dir = root / "images"
    label_dir = root / "labels"

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # названия классов записываются в порядке, определяющем их идентификаторы
    (root / "classes.txt").write_text(
        "автомобиль\nпешеход\n", encoding="utf-8",
    )

    return image_dir, label_dir


def test_read_classes_preserves_order(tmp_path: Path) -> None:
    """Проверяет, что порядок классов соответствует порядку строк в файле"""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text(
        "автомобиль\nпешеход\nвелосипед\n", encoding="utf-8",
    )

    assert read_classes(classes_file) == [
        "автомобиль",
        "пешеход",
        "велосипед",
    ]


def test_read_classes_rejects_duplicate_names(tmp_path: Path) -> None:
    """Проверяет, что файл с повторяющимися названиями классов отклоняется"""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text(
        "автомобиль\nпешеход\nавтомобиль\n", encoding="utf-8",
    )

    with pytest.raises(ValueError, match="повторяющиеся"):
        read_classes(classes_file)


def test_read_classes_rejects_missing_file(tmp_path: Path) -> None:
    """Проверяет обработку ситуации, когда файл классов отсутствует"""
    with pytest.raises(FileNotFoundError):
        read_classes(tmp_path / "несуществующие_классы.txt")


def test_validate_label_returns_class_ids(tmp_path: Path) -> None:
    """Проверяет, что проверка разметки возвращает идентификаторы классов"""
    image_path = tmp_path / "изображение.jpg"
    label_path = tmp_path / "изображение.txt"

    create_image(image_path)

    # каждая строка содержит: класс, центр рамки X/Y, ширину и высоту
    label_path.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.3 0.4 0.1 0.1\n",
        encoding="utf-8",
    )

    result = validate_label(
        label_path, image_path, ["автомобиль", "пешеход"],
    )

    assert result == [0, 1]


def test_validate_label_accepts_empty_annotation(tmp_path: Path,) -> None:
    """Проверяет, что пустая разметка допустима для изображения без объектов"""
    image_path = tmp_path / "изображение.jpg"
    label_path = tmp_path / "изображение.txt"

    create_image(image_path)
    label_path.write_text("", encoding="utf-8")

    assert validate_label(
        label_path, image_path, ["автомобиль"],
    ) == []


def test_validate_label_rejects_unknown_class(tmp_path: Path) -> None:
    """Проверяет, что разметка с неизвестным идентификатором класса отклоняется"""
    image_path = tmp_path / "изображение.jpg"
    label_path = tmp_path / "изображение.txt"

    create_image(image_path)

    # id 1 недопустим, так как объявлен только один класс
    label_path.write_text(
        "1 0.5 0.5 0.2 0.2\n", encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_label(label_path, image_path, ["автомобиль"])


def test_validate_label_rejects_invalid_coordinates(tmp_path: Path,) -> None:
    """Проверяет, что координаты рамки за пределами допустимого диапазона отклоняются"""
    image_path = tmp_path / "изображение.jpg"
    label_path = tmp_path / "изображение.txt"

    create_image(image_path)

    # координата центра X равна 1.2, хотя должна находиться в диапазоне от 0 до 1
    label_path.write_text(
        "0 1.2 0.5 0.2 0.2\n", encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_label(label_path, image_path, ["автомобиль"])


def test_validate_label_rejects_missing_image(tmp_path: Path) -> None:
    """Проверяет, что отсутствие изображения вызывает ошибку"""
    label_path = tmp_path / "изображение.txt"
    label_path.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        validate_label(
            label_path, tmp_path / "несуществующее_изображение.jpg", ["автомобиль"],
        )


def test_validate_label_rejects_corrupt_image(tmp_path: Path) -> None:
    """ПРоверяет, что поврежденный файл изображения не проходит проверку"""
    image_path = tmp_path / "изображение.jpg"
    label_path = tmp_path / "изображение.txt"

    # зпписать обычный текст вместо настоящего изображения!
    image_path.write_text(
        "это не изображение", encoding="utf-8",
    )
    label_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Не удалось прочитать"):
        validate_label(label_path, image_path, ["автомобиль"])


def test_validate_dataset_counts_objects_and_empty_labels(tmp_path: Path,) -> None:
    """Проверяет подсчет изображений, объектов и пустых файлов разметки"""
    image_dir, label_dir = create_dataset(tmp_path)

    create_image(image_dir / "изображение_001.jpg")
    create_image(image_dir / "изображение_002.jpg")

    (label_dir / "изображение_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8",
    )
    (label_dir / "изображение_002.txt").write_text(
        "\n", encoding="utf-8",
    )

    classes, summary = validate_dataset(tmp_path)

    # проверить правильность подсчетов и отсутствие ошибок в тестовом наборе:
    assert classes == ["автомобиль", "пешеход"]
    assert summary["image_count"] == 2
    assert summary["label_count"] == 2
    assert summary["readable_images"] == 2
    assert summary["empty_labels"] == 1
    assert summary["total_objects"] == 1
    assert summary["valid_objects"] == 1
    assert summary["class_counts"][0] == 1
    assert summary["issues"] == []
    assert summary["missing_labels"] == []
    assert summary["orphan_labels"] == []


def test_validate_dataset_reports_missing_label(tmp_path: Path) -> None:
    """Проверяет обнаружение изображения, для которого отсутствует разметка"""
    image_dir, label_dir = create_dataset(tmp_path)

    # создание и изображения, и файла разметки
    create_image(image_dir / "изображение_001.jpg")
    (label_dir / "изображение_001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    #  файл разметки намеренно не создавать
    create_image(image_dir / "изображение_002.jpg")

    _, summary = validate_dataset(tmp_path)

    assert summary["image_count"] == 2
    assert summary["label_count"] == 1
    assert summary["missing_labels"] == ["изображение_002"]
