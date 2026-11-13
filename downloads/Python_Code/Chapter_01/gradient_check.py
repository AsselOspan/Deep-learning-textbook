"""Числовая проверка градиента и одного шага для сети 2–2–1."""
import math

import torch


x = torch.tensor([2., 1.], dtype=torch.float64)
target = 1.
w11 = torch.tensor(0.5, dtype=x.dtype, requires_grad=True)
v1 = torch.tensor(2., dtype=x.dtype, requires_grad=True)


def prediction():
    z1 = w11 * x[0] - x[1] + 0.5
    z2 = -0.5 * x[0] + x[1] - 0.5
    return v1 * z1.relu() - z2.relu() - 0.5


before = prediction()
loss = (before - target).square() / 2
loss.backward()
assert math.isclose(w11.grad.item(), -2.0)
assert math.isclose(v1.grad.item(), -0.25)

with torch.no_grad():
    w11 -= 0.05 * w11.grad
    v1 -= 0.05 * v1.grad
    after = prediction()
    new_loss = (after - target).square() / 2
    assert math.isclose(after.item(), 0.90875)
    print(f'Логит: {before.item():.5f} → {after.item():.5f}')
    print(f'Потери: {loss.item():.6f} → {new_loss.item():.6f}')
