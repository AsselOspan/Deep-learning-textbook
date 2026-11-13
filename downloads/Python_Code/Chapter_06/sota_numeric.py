"""Числовые примеры главы 6. Запуск: python sota_numeric.py."""

import math

import torch
from torch.nn import functional as F

DTYPE = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def check(actual, expected):
    torch.testing.assert_close(actual, tensor(expected), atol=1e-12, rtol=0)


def inception_example():
    x = tensor([[
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    ]])
    short = F.conv2d(x, tensor([0.5, 1.0]).reshape(1, 2, 1, 1))
    reduced = F.conv2d(x, tensor([0.5, 0.5]).reshape(1, 2, 1, 1))
    edge = tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    spatial = F.conv2d(reduced, edge.reshape(1, 1, 3, 3), padding=1)
    output = torch.cat((short, spatial), dim=1)
    assert output.shape == (1, 2, 3, 3)
    check(reduced[0, 0], [[1, 1, 2], [2, 3, 3], [4, 4, 5]])
    check(output[0, :, 1, 1], [3.5, 3])
    plain = 3 * 3 * 192 * 64
    projected = 192 * 32 + 3 * 3 * 32 * 64
    assert (plain, projected) == (110592, 24576)
    assert (plain * 28**2, projected * 28**2) == (86704128, 19267584)
    print(f"inception_center={output[0, :, 1, 1].tolist()}")
    print(f"dense_vs_bottleneck_parameters={plain},{projected}")


def residual_example():
    x = tensor([2, -1]).requires_grad_()
    weight = tensor([[0.2, 0.4], [-0.3, 0.1]]).requires_grad_()
    target = tensor([1, -1])
    prediction = x + weight @ x
    error = (prediction - target).square().sum() / 2
    error.backward()
    check(prediction, [2, -1.7])
    check(error, 0.745)
    check(weight.grad, [[2, -1], [-1.4, 0.7]])
    check(x.grad, [1.41, -0.37])
    with torch.no_grad():
        weight -= 0.1 * weight.grad
        check(weight, [[0, 0.5], [-0.16, 0.03]])
        updated = x + weight @ x
        check(updated, [1.5, -1.35])
        after = (updated - target).square().sum() / 2
        check(after, 0.18625)
    print(f"residual_loss={error.item():.6f}->{after.item():.6f}")
    print(f"residual_input_gradient={x.grad.tolist()}")


def efficient_layers_example():
    dense = torch.nn.Conv2d(16, 32, 3, padding=1, bias=False)
    depthwise = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
    pointwise = torch.nn.Conv2d(16, 32, 1, bias=False)
    x = torch.zeros(1, 16, 28, 28)
    assert dense(x).shape == pointwise(depthwise(x)).shape == (1, 32, 28, 28)
    plain = dense.weight.numel()
    separated = depthwise.weight.numel() + pointwise.weight.numel()
    assert (plain, separated) == (4608, 656)
    print(f"dense_vs_depthwise_parameters={plain},{separated}")


def attention_example():
    query = tensor([math.sqrt(2), 0])
    key = tensor([[1, 0], [0, 1], [1, 1]])
    value = tensor([[1, 0], [0, 2], [1, 1]])
    score = key @ query / math.sqrt(key.shape[1])
    weight = score.softmax(dim=0)
    result = weight @ value
    normalizer = 2 * math.e + 1
    check(score, [1, 0, 1])
    check(result, [2 * math.e / normalizer, (2 + math.e) / normalizer])
    print(f"attention_output={result.tolist()}")


def classification_example():
    for label, rows, expected in (
        ("A", [[59, 1], [9, 1]], [60 / 70, (118 / 128 + 2 / 12) / 2]),
        ("B", [[51, 9], [3, 7]], [58 / 70, (102 / 114 + 14 / 26) / 2]),
    ):
        matrix = tensor(rows)
        correct = matrix.diag()
        f1 = 2 * correct / (matrix.sum(dim=0) + matrix.sum(dim=1))
        accuracy = correct.sum() / matrix.sum()
        macro_f1 = f1.mean()
        check(torch.stack((accuracy, macro_f1)), expected)
        print(f"model_{label}_accuracy={accuracy:.6f} macro_f1={macro_f1:.6f}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    inception_example()
    residual_example()
    efficient_layers_example()
    attention_example()
    classification_example()

