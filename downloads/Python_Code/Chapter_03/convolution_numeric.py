"""Ручная кросс-корреляция и проверка через PyTorch."""

import torch
from torch.nn import functional as F


def example():
    image = [[1, 2, 0, 1], [3, 1, 2, 2], [0, 2, 1, 0], [1, 1, 3, 2]]
    kernel = [[1, 0], [-1, 1]]
    return image, kernel


def two_channel_example():
    image, kernel = example()
    other = [[2, 1, 0, 1], [0, 1, 1, 2], [1, 0, 2, 1], [2, 1, 0, 0]]
    other_kernel = [[0, 1], [1, 0]]
    return [image, other], [kernel, other_kernel], 0.5


def channels_first(values):
    """Превращаем H×W в 1×H×W, C×H×W сохраняем."""
    if not values or not values[0]:
        raise ValueError("Нужна непустая матрица или набор матриц")
    if isinstance(values[0][0], (list, tuple)):
        return values
    return [values]


def spatial_shape(volume):
    if not volume[0] or not volume[0][0]:
        raise ValueError("Размеры входа и ядра должны быть положительными")
    height, width = len(volume[0]), len(volume[0][0])
    for plane in volume:
        if len(plane) != height or any(len(row) != width for row in plane):
            raise ValueError("Все каналы должны иметь одинаковую прямоугольную форму")
    return height, width


def cross_correlation(image, kernel, *, bias=0, stride=1, padding=0, dilation=1):
    """Одна выходная карта из вложенных списков H×W или C×H×W."""
    if stride < 1 or dilation < 1 or padding < 0:
        raise ValueError("Нужны stride ≥ 1, dilation ≥ 1 и padding ≥ 0")

    inputs, weights = channels_first(image), channels_first(kernel)
    if len(inputs) != len(weights):
        raise ValueError("Число каналов входа и ядра должно совпасть")
    height, width = spatial_shape(inputs)
    kernel_height, kernel_width = spatial_shape(weights)
    padded_height, padded_width = height + 2 * padding, width + 2 * padding
    effective_height = dilation * (kernel_height - 1) + 1
    effective_width = dilation * (kernel_width - 1) + 1
    out_height = (padded_height - effective_height) // stride + 1
    out_width = (padded_width - effective_width) // stride + 1
    if out_height < 1 or out_width < 1:
        raise ValueError("Ядро выходит за размеры дополненного входа")

    padded = []
    for plane in inputs:
        matrix = [[0] * padded_width for _ in range(padded_height)]
        for i, row in enumerate(plane):
            matrix[padding + i][padding:padding + width] = row
        padded.append(matrix)

    result = []
    for i in range(out_height):
        row = []
        for j in range(out_width):
            # Начало окна задаёт stride, точки внутри окна задаёт dilation.
            total = bias
            for plane, channel_weights in zip(padded, weights):
                for u in range(kernel_height):
                    for v in range(kernel_width):
                        y = i * stride + u * dilation
                        x = j * stride + v * dilation
                        total += plane[y][x] * channel_weights[u][v]
            row.append(total)
        result.append(row)
    return result


def pytorch_cross_correlation(
    image, kernel, *, bias=0, stride=1, padding=0, dilation=1, device="cpu"
):
    options = dict(dtype=torch.float64, device=device)
    # Добавляем ось батча входу и ось выходных каналов ядру.
    x = torch.tensor(channels_first(image), **options).unsqueeze(0)
    weights = torch.tensor(channels_first(kernel), **options).unsqueeze(0)
    offset = torch.tensor([bias], **options)
    result = F.conv2d(
        x, weights, offset, stride=stride, padding=padding, dilation=dilation
    )
    return result[0, 0].tolist()


def verify_examples():
    image, kernel = example()
    flipped = [row[::-1] for row in kernel[::-1]]
    padded_map = [
        [1, 1, -2, 1, -1], [3, -1, 3, 0, -1], [0, 5, 0, 1, 2],
        [1, 0, 4, 0, -2], [0, 1, 1, 3, 2],
    ]
    variants = [
        ("s=1, p=0, d=1", kernel, {}, [[-1, 3, 0], [5, 0, 1], [0, 4, 0]]),
        ("stride=2", kernel, {"stride": 2}, [[-1, 0], [0, 0]]),
        ("padding=1", kernel, {"padding": 1}, padded_map),
        ("dilation=2", kernel, {"dilation": 2}, [[2, 0], [5, 2]]),
        ("поворот ядра", flipped, {}, [[0, 4, 1], [4, 0, 0], [-1, 4, 3]]),
    ]
    results = {}
    for name, weights, options, expected in variants:
        manual = cross_correlation(image, weights, **options)
        reference = pytorch_cross_correlation(image, weights, **options)
        assert manual == expected == reference, name
        results[name] = manual

    inputs, weights, bias = two_channel_example()
    manual = cross_correlation(inputs, weights, bias=bias)
    reference = pytorch_cross_correlation(inputs, weights, bias=bias)
    expected = [[0.5, 4.5, 2.5], [7.5, 1.5, 5.5], [2.5, 7.5, 1.5]]
    assert manual == expected == reference
    results["два канала и смещение"] = manual
    return results


if __name__ == "__main__":
    for title, matrix in verify_examples().items():
        print(title)
        for row in matrix:
            print(row)
    print("Все шесть карт совпали с результатом PyTorch.")
