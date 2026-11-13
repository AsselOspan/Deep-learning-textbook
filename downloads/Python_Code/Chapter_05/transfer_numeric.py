"""Два этапа переноса и состояние BatchNorm. Запуск: python transfer_numeric.py."""

import torch
from torch import nn

DTYPE = torch.float64


def tensor(values):
    return torch.tensor(values, dtype=DTYPE)


def check(actual, expected):
    torch.testing.assert_close(actual, tensor(expected), rtol=0, atol=1e-12)


def loss(model, x, y):
    return (model(x) - y).square().sum() / 2


def run_transfer():
    lower = nn.Linear(2, 2, bias=False, dtype=DTYPE)
    upper = nn.Linear(2, 2, dtype=DTYPE)
    head = nn.Linear(2, 1, dtype=DTYPE)
    model = nn.Sequential(lower, upper, head)
    with torch.no_grad():
        lower.weight.copy_(tensor([[0.5, 0.0], [0.0, 1.0]]))
        upper.weight.copy_(tensor([[1.0, 0.5], [-0.5, 2.0]]))
        upper.bias.zero_()
        head.weight.copy_(tensor([[0.4, -0.2]]))
        head.bias.copy_(tensor([0.1]))

    x, y = tensor([[4.0, 1.0]]), tensor([[1.5]])
    lower.requires_grad_(False)
    upper.requires_grad_(False)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    before = loss(model, x, y)
    check(before, 0.18)
    model.zero_grad(set_to_none=True)
    before.backward()
    assert all(p.grad is None for p in [*lower.parameters(), *upper.parameters()])
    check(head.weight.grad, [[-1.5, -0.6]])
    check(head.bias.grad, [-0.6])
    optimizer.step()
    after_head = loss(model, x, y)
    check(after_head, 0.0055125)
    check(head.weight, [[0.55, -0.14]])
    check(head.bias, [0.16])
    check(lower.weight, [[0.5, 0.0], [0.0, 1.0]])
    check(upper.weight, [[1.0, 0.5], [-0.5, 2.0]])
    check(upper.bias, [0.0, 0.0])

    upper.requires_grad_(True)
    optimizer = torch.optim.SGD([
        {"params": upper.parameters(), "lr": 0.02},
        {"params": head.parameters(), "lr": 0.1},
    ])
    model.zero_grad(set_to_none=True)
    loss(model, x, y).backward()
    assert lower.weight.grad is None
    check(upper.weight.grad, [[-0.1155, -0.05775], [0.0294, 0.0147]])
    check(upper.bias.grad, [-0.05775, 0.0147])
    check(head.weight.grad, [[-0.2625, -0.105]])
    check(head.bias.grad, [-0.105])
    optimizer.step()

    check(lower.weight, [[0.5, 0.0], [0.0, 1.0]])
    check(upper.weight, [[1.00231, 0.501155], [-0.500588, 1.999706]])
    check(upper.bias, [0.001155, -0.000294])
    check(head.weight, [[0.57625, -0.1295]])
    check(head.bias, [0.1705])
    after_fine_tune = loss(model, x, y)
    check(model(x), [[1.4858468505]])
    check(after_fine_tune, 0.00010015582038467223)
    head_only_loss = (head(tensor([[2.5, 1.0]])) - y).square().sum() / 2
    check(head_only_loss, 0.0001688203125)

    print(f"start_loss={before.item():.12f}")
    print(f"head_loss={after_head.item():.12f}")
    print(f"selective_loss={after_fine_tune.item():.12f}")
    print(f"head_only_continuation_loss={head_only_loss.item():.12f}")


def run_batch_norm():
    layer = nn.BatchNorm1d(2, momentum=0.1, dtype=DTYPE)
    layer.requires_grad_(False)
    batch = tensor([[1.0, 2.0], [3.0, 6.0]])

    layer.train()
    layer(batch)
    check(layer.running_mean, [0.2, 0.4])
    check(layer.running_var, [1.1, 1.7])
    assert layer.num_batches_tracked.item() == 1

    layer.eval()
    layer(batch)
    check(layer.running_mean, [0.2, 0.4])
    check(layer.running_var, [1.1, 1.7])
    assert layer.num_batches_tracked.item() == 1
    check(layer.weight, [1.0, 1.0])
    check(layer.bias, [0.0, 0.0])
    print("batch_norm_checks=passed")


if __name__ == "__main__":
    run_transfer()
    run_batch_norm()

