import argparse
import platform
import timeit
from pathlib import Path
from statistics import median

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Полный квадратный структурирующий элемент 3×3.
KERNEL = np.ones((3, 3), dtype=np.uint8)


def erode_opencv(image):
    """Эрозия средствами OpenCV."""
    return cv2.erode(
        image,
        KERNEL,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def erode_python(image):
    """
    Нативная эрозия: только циклы и сравнения Python.

    NumPy используется для преобразования входного и выходного
    изображений, но не для вычисления эрозии.
    """
    height, width = image.shape
    rows = image.tolist()

    # Добавляем чёрную рамку толщиной один пиксель.
    padded = [[0] * (width + 2)]
    padded += [[0] + row + [0] for row in rows]
    padded += [[0] * (width + 2)]

    result = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            minimum = 255

            # Ищем минимум среди девяти пикселей окна.
            for dy in range(3):
                for dx in range(3):
                    value = padded[y + dy][x + dx]

                    if value < minimum:
                        minimum = value

            result[y][x] = minimum

    return np.array(result, dtype=np.uint8)


def check_result(image, expected):
    """Проверяем обе реализации по ожидаемому результату."""
    for function in (erode_python, erode_opencv):
        actual = function(image)

        if not np.array_equal(actual, expected):
            raise AssertionError(
                f"Неверный результат: {function.__name__}"
            )


def run_tests():
    """Контрольные примеры и полный перебор бинарных окон."""
    cases = []

    # 1. Чёрное изображение остаётся чёрным.
    black = np.zeros((7, 7), dtype=np.uint8)
    cases.append((black, black.copy()))

    # 2. У белого изображения появляется чёрная рамка.
    white = np.full((7, 7), 255, dtype=np.uint8)
    expected = black.copy()
    expected[1:-1, 1:-1] = 255
    cases.append((white, expected))

    # 3. Квадрат 3×3 превращается в одну точку.
    square = black.copy()
    square[2:5, 2:5] = 255

    point = black.copy()
    point[3, 3] = 255

    cases.append((square, point))

    # 4. Одиночная точка исчезает.
    cases.append((point, black))

    # 5. Линия толщиной один пиксель исчезает.
    line = black.copy()
    line[3, :] = 255
    cases.append((line, black))

    # 6–9. Маленькие изображения становятся чёрными.
    for shape in ((1, 1), (1, 7), (7, 1), (2, 2)):
        image = np.full(shape, 255, dtype=np.uint8)
        expected = np.zeros(shape, dtype=np.uint8)
        cases.append((image, expected))

    for image, expected in cases:
        check_result(image, expected)

    # В окне 3×3 девять бинарных пикселей:
    # всего возможно 2**9 = 512 сочетаний.
    for mask in range(512):
        values = [
            255 if mask & (1 << bit) else 0
            for bit in range(9)
        ]

        image = np.array(
            values, dtype=np.uint8
        ).reshape(3, 3)

        expected = np.zeros((3, 3), dtype=np.uint8)

        # Центр белый только у полностью белого окна.
        if mask == 511:
            expected[1, 1] = 255

        check_result(image, expected)

    return "OK: 9 контрольных случаев и все 512 бинарных окон 3x3."


def create_demo_image():
    """Тестовая картинка: фигуры, отверстие, линия и точки."""
    image = np.zeros((160, 240), dtype=np.uint8)

    # Два прямоугольника, соединённые тонкой перемычкой.
    cv2.rectangle(image, (20, 20), (75, 75), 255, -1)
    cv2.rectangle(image, (95, 20), (150, 75), 255, -1)
    cv2.line(image, (76, 48), (94, 48), 255, 1)

    # Кольцо.
    cv2.circle(image, (195, 48), 25, 255, -1)
    cv2.circle(image, (195, 48), 8, 0, -1)

    # Тонкая линия.
    cv2.line(image, (20, 100), (110, 100), 255, 1)

    # Квадрат 3×3 и одиночная точка.
    cv2.rectangle(image, (140, 100), (142, 102), 255, -1)
    image[115, 190] = 255

    cv2.putText(
        image,
        "EROSION",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
    )

    return image


def save_comparison(original, binary, native, library, folder):
    """Сохраняем отдельные картинки и общую иллюстрацию."""
    difference = cv2.absdiff(native, library)

    cv2.imwrite(str(folder / "original.png"), original)

    panels = [
        (binary, "Binary"),
        (native, "Python"),
        (library, "OpenCV"),
        (difference, "Difference"),
    ]

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))

    for axis, (image, name) in zip(axes, panels):
        cv2.imwrite(
            str(folder / f"{name.lower()}.png"),
            image,
        )

        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axis.set_title(name)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(folder / "comparison.png", dpi=160)
    plt.close(figure)

    return int(np.count_nonzero(difference))


def benchmark(folder):
    """Сравниваем время выполнения на четырёх размерах."""
    generator = np.random.default_rng(42)
    results = []

    raw = ["size,implementation,series,calls,ms_per_call"]

    for size in (128, 256, 512, 1024):
        image = (
            generator.random((size, size)) < 0.7
        ).astype(np.uint8) * 255

        # Проверка корректности вне измерения времени.
        check_result(image, erode_opencv(image))

        measured = []

        implementations = [
            ("Python", erode_python, 1),
            ("OpenCV", erode_opencv, 1000),
        ]

        for name, function, calls in implementations:
            function(image)  # Прогрев.

            # Пять серий измерений.
            samples = timeit.repeat(
                lambda: function(image),
                repeat=5,
                number=calls,
            )

            # Время одного вызова в миллисекундах.
            samples = [
                seconds * 1000 / calls
                for seconds in samples
            ]

            measured.append(median(samples))

            for number, milliseconds in enumerate(samples, 1):
                raw.append(
                    f"{size},{name},{number},{calls},"
                    f"{milliseconds:.9f}"
                )

        python_ms, opencv_ms = measured
        speedup = python_ms / opencv_ms

        results.append(
            (size, python_ms, opencv_ms, speedup)
        )

        print(
            f"{size}x{size}: "
            f"Python={python_ms:.3f} ms; "
            f"OpenCV={opencv_ms:.4f} ms; "
            f"ускорение={speedup:.1f}x"
        )

    (folder / "timings.csv").write_text(
        "\n".join(raw),
        encoding="utf-8",
    )

    pixels = [row[0] ** 2 for row in results]

    plt.figure(figsize=(8, 5))

    plt.loglog(
        pixels,
        [row[1] for row in results],
        "o-",
        label="Python",
    )

    plt.loglog(
        pixels,
        [row[2] for row in results],
        "o-",
        label="OpenCV",
    )

    plt.xlabel("Number of pixels (log scale)")
    plt.ylabel("Median time, ms (log scale)")
    plt.title("3x3 erosion: execution time")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "benchmark.png", dpi=160)
    plt.close()

    return results


def main():
    print("Программа запущена.", flush=True)

    parser = argparse.ArgumentParser(
        description="Эрозия 3×3: Python и OpenCV"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Путь к своему изображению",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Для чёрных объектов на белом фоне",
    )
    args = parser.parse_args()

    folder = Path("results")
    folder.mkdir(exist_ok=True)

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    # 1. Проверка корректности.
    print("Выполняются тесты...", flush=True)

    test_log = run_tests()
    print(test_log, flush=True)

    (folder / "tests.txt").write_text(
        test_log,
        encoding="utf-8",
    )

    # 2. Загрузка своей картинки или создание тестовой.
    if args.input:
        data = np.fromfile(args.input, dtype=np.uint8)
        original = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if original is None:
            raise ValueError("Не удалось прочитать изображение")
    else:
        original = create_demo_image()

    # 3. Бинаризация.
    mode = (
        cv2.THRESH_BINARY_INV
        if args.invert
        else cv2.THRESH_BINARY
    )

    _, binary = cv2.threshold(original, 127, 255, mode)

    # 4. Эрозия двумя способами.
    native = erode_python(binary)
    library = erode_opencv(binary)

    check_result(binary, library)

    # 5. Сохранение изображений.
    mismatches = save_comparison(
        original,
        binary,
        native,
        library,
        folder,
    )

    print(f"Различающихся пикселей: {mismatches}")
    print(f"Белых пикселей до эрозии: {np.count_nonzero(binary)}")
    print(f"Белых пикселей после эрозии: {np.count_nonzero(native)}")

    # 6. Измерение скорости и сохранение графика.
    print("Измеряется быстродействие...", flush=True)
    benchmark(folder)

    print("Готово.")
    print(f"Результаты сохранены в: {folder.resolve()}")


if __name__ == "__main__":
    main()
