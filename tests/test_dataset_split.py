from pathlib import Path

import pytest

from src.prepare_dataset import RELATED_GROUPS, split_images


def make_split_inputs() -> tuple[list[Path], dict[str, list[int]]]:
    """Подготавливает 40 путей к изображениям для тестирования разбиения.
    В набор включаются все изображения из связанных групп, которые алгоритм должен сохранять в одной выборке.
    Недостающие пути заполняются тестовыми именами.
    """
    # собрать имена всех изображений, входящих в связанные группы
    filenames = {
        filename
        for group in RELATED_GROUPS.values()
        for filename in group
    }

    # дополнить список до 40 изображений, если связанных изображений меньше
    index = 0
    while len(filenames) < 40:
        filenames.add(f"тестовое_изображение_{index:02d}.jpg")
        index += 1

    # отсортировать пути, чтобы исходный порядок был предсказуемым
    images = sorted(
        (Path(name) for name in filenames),
        key=lambda path: path.name.lower(),
    )

    # назначить каждому изображению тестовый идентификатор класса
    image_classes = {
        image.stem.lower(): [index % 3]
        for index, image in enumerate(images)
    }

    return images, image_classes


def test_split_images_has_expected_sizes() -> None:
    """Проверяет, что размеры обучающей, валидационной и тестовой выборок верны"""
    images, image_classes = make_split_inputs()

    splits = split_images(images, image_classes, seed=42)

    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 28
    assert len(splits["val"]) == 6
    assert len(splits["test"]) == 6


def test_split_images_has_no_overlap() -> None:
    """Проверяет, что одно изображение не попадает сразу в несколько выборок"""
    images, image_classes = make_split_inputs()

    splits = split_images(images, image_classes, seed=42)

    # получение множество имен изображений для каждой выборки
    train_names = {image.name for image in splits["train"]}
    val_names = {image.name for image in splits["val"]}
    test_names = {image.name for image in splits["test"]}

    # проверить, что выборки попарно не пересекаются
    assert train_names.isdisjoint(val_names)
    assert train_names.isdisjoint(test_names)
    assert val_names.isdisjoint(test_names)

    # проверить, что после разбиения сохранились все исходные изображения
    all_names = train_names | val_names | test_names
    assert all_names == {image.name for image in images}


def test_split_images_is_reproducible() -> None:
    """Проверяет, что одинаковое начальное значение дает одинаковое разбиение"""
    images, image_classes = make_split_inputs()

    first = split_images(images, image_classes, seed=42)
    second = split_images(images, image_classes, seed=42)

    # При одинаковых данных и seed состав и порядок выборок должны совпадать
    for split_name in ("train", "val", "test"):
        assert [
            image.name for image in first[split_name]
        ] == [
            image.name for image in second[split_name]
        ]


def test_related_groups_stay_in_one_split() -> None:
    """Проверяет, что связанные изображения целиком попадают в одну выборку"""
    images, image_classes = make_split_inputs()

    splits = split_images(images, image_classes, seed=42)

    # Для каждого изображения определяется название выборки, в которую оно попало
    image_to_split = {
        image.name: split_name
        for split_name, split_images_list in splits.items()
        for image in split_images_list
    }

    # все изображения одной связанной группы должны относиться к одной выборке!
    for group_name, filenames in RELATED_GROUPS.items():
        assigned_splits = {
            image_to_split[filename]
            for filename in filenames
        }

        assert len(assigned_splits) == 1, (
            f"Связанная группа {group_name} разделена между выборками"
        )


def test_split_images_rejects_wrong_number_of_images() -> None:
    """Проверет, что алгоритм отклоняет набор неправильного размера"""
    images, image_classes = make_split_inputs()

    # алгоритм ожидает ровно 40 изображений, поэтому передается только 39
    with pytest.raises(ValueError, match="Ожидалось 40 изображений"):
        split_images(images[:39], image_classes, seed=42)



def test_split_images_rejects_missing_related_image() -> None:
    """Проверяет обнаружение отсутствующего изображения из связанной группы"""
    images, image_classes = make_split_inputs()

    # найти имя изображения, которое входит в обязательную группу G1!
    missing_filename = next(iter(RELATED_GROUPS["G1"]))

    # удалить нужное изображение и добавить другое, не входящее в связанные группы
    images = [
        image
        for image in images
        if image.name != missing_filename
    ]
    images.append(Path("дополнительное_тестовое_изображение.jpg"))

    # проверить, что количество файлов осталось равным 40!
    assert len(images) == 40

    with pytest.raises(FileNotFoundError, match="не найдены файлы"):
        split_images(images, image_classes, seed=42)
