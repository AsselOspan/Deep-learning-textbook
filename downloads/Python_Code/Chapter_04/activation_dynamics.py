"""Два числовых опыта: градиент логита и шаг для отрицательной ветви."""
import torch
from torch.nn import functional as F


def verify_dynamics():
    z = torch.tensor(6., dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(z)
    # Целевой класс равен нулю, поэтому большие положительные логиты ошибочны.
    bce = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
    square = p.square() / 2
    bce_grad, = torch.autograd.grad(bce, z)
    square_grad, = torch.autograd.grad(square, z)
    torch.testing.assert_close(bce_grad, p)
    torch.testing.assert_close(square_grad, p.square() * (1 - p))

    result = {"bce_grad": bce_grad.item(), "square_grad": square_grad.item()}
    for name, activation, reference in (
        ("relu", F.relu, 0.0),
        ("leaky", lambda z: F.leaky_relu(z, negative_slope=0.1), -0.24),
    ):
        w = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        loss = (activation(2 * w - 4) - 1).square() / 2
        gradient, = torch.autograd.grad(loss, w)
        torch.testing.assert_close(gradient, torch.full_like(w, reference))
        with torch.no_grad():
            after = w - 0.1 * gradient
            next_loss = (activation(2 * after - 4) - 1).square() / 2
        result[f"{name}_step"] = after.item()
        result[f"{name}_loss_before"] = loss.item()
        result[f"{name}_loss_after"] = next_loss.item()
    return result


if __name__ == '__main__':
    for name, number in verify_dynamics().items():
        print(f'{name}: {number:.12f}')
