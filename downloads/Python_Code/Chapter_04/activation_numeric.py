"""Ручные производные активаций и проверка в PyTorch."""

import math

import torch
from torch.nn import functional as F


FUNCTIONS = ("sigmoid", "tanh", "relu", "leaky_relu", "elu", "softplus")
POINTS = (-6, -2, 0, 2, 6)
ALPHA = 0.1


def value(name, z):
    if name == "sigmoid":
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)
    if name == "tanh":
        return math.tanh(z)
    if name == "relu":
        return max(0.0, z)
    if name == "leaky_relu":
        return z if z >= 0 else ALPHA * z
    if name == "elu":
        return z if z >= 0 else math.expm1(z)
    if name == "softplus":
        # Устойчивая форма log(1 + exp(z)).
        return max(0.0, z) + math.log1p(math.exp(-abs(z)))
    raise ValueError(f"Неизвестная активация: {name}")


def derivative(name, z):
    """Классическая производная. None обозначает излом."""
    if name == "sigmoid":
        s = value(name, z)
        return s * (1 - s)
    if name == "tanh":
        return 1 - value(name, z) ** 2
    if name in ("relu", "leaky_relu"):
        if z == 0:
            return None
        if z > 0:
            return 1.0
        return 0.0 if name == "relu" else ALPHA
    if name == "elu":
        return 1.0 if z >= 0 else math.exp(z)
    if name == "softplus":
        return value("sigmoid", z)
    raise ValueError(f"Неизвестная активация: {name}")


def torch_value(name, tensor):
    if name == "leaky_relu":
        return F.leaky_relu(tensor, negative_slope=ALPHA)
    if name == "elu":
        return F.elu(tensor, alpha=1.0)
    if name == "softplus":
        return F.softplus(tensor, beta=1.0)
    functions = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}
    return functions[name](tensor)


def autograd_derivative(name, z):
    x = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(torch_value(name, x), x)
    return gradient.item()


def central_difference(name, z, epsilon=1e-5):
    if epsilon <= 0:
        raise ValueError("epsilon должен быть положительным")
    return (value(name, z + epsilon) - value(name, z - epsilon)) / (2 * epsilon)


def one_sided(name, z, epsilon=1e-5):
    if epsilon <= 0:
        raise ValueError("epsilon должен быть положительным")
    center = value(name, z)
    left = (center - value(name, z - epsilon)) / epsilon
    right = (value(name, z + epsilon) - center) / epsilon
    return left, right


def activation_chain(name, depth=4, start=None):
    """Предактивации, смещение после первого слоя, ручная производная и autograd."""
    if depth < 1:
        raise ValueError("Число слоёв должно быть положительным")
    if start is None:
        start = 1.0 if name == "relu" else 0.0
    bias = start - value(name, start)
    x = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    preactivations = []
    current = x
    for layer in range(depth):
        z = current if layer == 0 else current + bias
        preactivations.append(z.item())
        current = torch_value(name, z)
    slopes = [derivative(name, z) for z in preactivations]
    manual = None if None in slopes else math.prod(slopes)
    (gradient,) = torch.autograd.grad(current, x)
    return preactivations, bias, manual, gradient.item()


def verify_examples():
    rows = []
    for name in FUNCTIONS:
        for z in POINTS:
            expected = derivative(name, z)
            auto = autograd_derivative(name, z)
            central = central_difference(name, z)
            actual = value(name, z)
            torch_actual = torch_value(name, torch.tensor(z, dtype=torch.float64)).item()
            assert math.isclose(actual, torch_actual, abs_tol=1e-12)
            if expected is not None:
                # При alpha=1 ELU имеет непрерывную первую производную в нуле.
                # На стыке у односторонних вторых производных разные значения.
                tolerance = 3e-6 if name == "elu" and z == 0 else 1e-9
                assert math.isclose(expected, auto, abs_tol=1e-12), (name, z)
                assert math.isclose(expected, central, abs_tol=tolerance), (name, z)
            rows.append((name, z, actual, expected, auto, central))
    assert one_sided("relu", 0) == (0.0, 1.0)
    assert math.isclose(central_difference("relu", 0), 0.5)
    assert autograd_derivative("relu", 0) == 0
    assert math.isclose(central_difference("leaky_relu", 0), 0.55)
    assert autograd_derivative("leaky_relu", 0) == ALPHA
    return rows


if __name__ == "__main__":
    print("Активация   z         f(z)        d/dz       autograd    центр. разн.")
    for name, z, actual, expected, auto, central in verify_examples():
        manual = "излом" if expected is None else f"{expected:.9f}"
        print(f"{name:11} {z:+2d} {actual:12.9f} {manual:>12} {auto:12.9f} {central:12.9f}")
    for name in ("relu", "leaky_relu"):
        print(f"{name}: односторонние наклоны {one_sided(name, 0)}")
    for name, start in (("sigmoid", 0), ("tanh", 0), ("relu", 0), ("relu", 1)):
        zs, bias, manual, auto = activation_chain(name, start=start)
        print(f"{name}: z={zs}, b={bias:g}, ручная={manual}, autograd={auto:g}")
