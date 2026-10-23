import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Сохраняет результаты эксперимента в локальные журналы TensorBoard.

    Args:
        log_dir: Путь к каталогу, в котором будут храниться журналы.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Создает логгер и подготавливает каталог для файлов TensorBoard"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # SummaryWriter записывает события в формат, который читает TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.closed = False

    def log_metric(self, name: str, value: int | float, step: int, prefix: str = "metrics",) -> None:
        """Записывает одну числовую метрику на указанном шаге.

        Args:
            name: Название метрики, например precision или mAP50.
            value: Числовое значение метрики.
            step: Номер эпохи или другого шага эксперимента.
            prefix: Группа, к которой относится метрика.
        """
        if self.closed:
            raise RuntimeError("Нельзя записывать данные: логгер уже закрыт.")

        # логировать только конечные числовые значения, исключая bool и NaN
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Значение метрики «{name}» должно быть числом!"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Значение метрики «{name}» должно быть конечным числом!"
            )

        if step < 0:
            raise ValueError("Номер шага не может быть отрицательным!")

        # Группировка по тегам позволяет удобно находить метрики в TensorBoard
        tag = f"{prefix}/{name}" if prefix else name
        self.writer.add_scalar(tag, float(value), global_step=step)

    def log_metrics(self, metrics: Mapping[str, Any], step: int, prefix: str = "metrics") -> None:
        """Записывает набор числовых метрик на одном шаге эксперимента.
        Нечисловые значения пропускаются, так как словарь результатов модели может содержать дополнительные служебные данные.
        """
        if self.closed:
            raise RuntimeError("Нельзя записывать данные: логгер уже закрыт.")

        # передать каждую числовую метрику в общий метод записи
        for name, value in metrics.items():
            if isinstance(value, bool):
                continue

            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue

            # Не записывать NaN и бесконечные значения!
            if not math.isfinite(float(value)):
                continue

            self.log_metric(
                name=name,
                value=float(value),
                step=step,
                prefix=prefix,
            )

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Сохраняет параметры эксперимента в виде  JSON-текста"""
        if self.closed:
            raise RuntimeError("Нельзя записывать данные: логгер уже закрыт.")

        parameters_json = json.dumps(
            dict(parameters),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        self.writer.add_text(
            "experiment/parameters",
            f"```json\n{parameters_json}\n```",
            global_step=0,
        )

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Сохраняет текстовую заметку или сведения об эксперименте"""
        if self.closed:
            raise RuntimeError("Нельзя записывать данные: логгер уже закрыт.")

        if step < 0:
            raise ValueError("Номер шага не может быть отрицательным.")

        self.writer.add_text(tag, text, global_step=step)

    def log_image(self, tag: str, image_path: str | Path, step: int = 0) -> None:
        """Добавляет изображение (например матрицу ошибок или предсказание)"""
        if self.closed:
            raise RuntimeError("Нельзя записывать данные: логгер уже закрыт.")

        if step < 0:
            raise ValueError("Номер шага не может быть отрицательным.")

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Файл изображения не найден: {image_path}"
            )

        # привести изображение к RGB, чтобы обеспечить единый формат записи
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            image_array = np.asarray(image_rgb)

        self.writer.add_image(
            tag,
            image_array,
            global_step=step,
            dataformats="HWC",
        )

    def flush(self) -> None:
        """Записывает накопленные события на диск"""
        if not self.closed:
            self.writer.flush()

    def close(self) -> None:
        """Завершает запись и освобождает ресурсы логгера"""
        if not self.closed:
            self.writer.close()
            self.closed = True

    def __enter__(self) -> "TensorBoardLogger":
        """Разрешает использование логгера через оператор with"""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any,) -> None:
        """Закрывает логгер при выходе из блока with"""
        self.close()
