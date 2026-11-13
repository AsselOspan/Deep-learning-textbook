"""Проверка обратного распространения на сети 2–2–2."""
import argparse

import torch


PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


def make_example(device="cpu"):
    options = dict(dtype=torch.float64, device=device)
    parameters = {
        "W1": torch.tensor([[0.20, -0.10], [0.30, 0.05]], **options),
        "b1": torch.tensor([0.20, -0.30], **options),
        "W2": torch.tensor([[0.40, -0.20], [-0.10, 0.30]], **options),
        "b2": torch.tensor([0.10, -0.10], **options),
    }
    x = torch.tensor([1.0, 2.0], **options)
    target = torch.tensor([0.20, 0.80], **options)
    return parameters, x, target


def forward(parameters, x):
    z1 = parameters["W1"] @ x + parameters["b1"]
    h = torch.sigmoid(z1)
    z2 = parameters["W2"] @ h + parameters["b2"]
    prediction = torch.sigmoid(z2)
    return z1, h, z2, prediction


def loss_value(parameters, x, target):
    _, _, _, prediction = forward(parameters, x)
    return (prediction - target).square().sum() / 2


@torch.no_grad()
def manual_gradients(parameters, x, target):
    _, h, _, prediction = forward(parameters, x)
    delta2 = (prediction - target) * prediction * (1 - prediction)
    delta1 = (parameters["W2"].T @ delta2) * h * (1 - h)
    return {
        "W1": torch.outer(delta1, x),
        "b1": delta1,
        "W2": torch.outer(delta2, h),
        "b2": delta2,
    }


def autograd_gradients(parameters, x, target):
    leaves = {
        name: value.detach().clone().requires_grad_()
        for name, value in parameters.items()
    }
    loss_value(leaves, x, target).backward()
    return {name: value.grad for name, value in leaves.items()}


@torch.no_grad()
def central_gradients(parameters, x, target, epsilon=1e-5):
    if epsilon <= 0:
        raise ValueError("epsilon должен быть положительным")
    gradients = {}
    for name, value in parameters.items():
        gradient = torch.empty_like(value, memory_format=torch.contiguous_format)
        for index in range(value.numel()):
            offset = torch.zeros_like(value, memory_format=torch.contiguous_format)
            offset.view(-1)[index] = epsilon
            plus = dict(parameters, **{name: value + offset})
            minus = dict(parameters, **{name: value - offset})
            numerator = loss_value(plus, x, target) - loss_value(minus, x, target)
            gradient.view(-1)[index] = numerator / (2 * epsilon)
        gradients[name] = gradient
    return gradients


@torch.no_grad()
def gradient_step(parameters, gradients, eta=0.25):
    return {
        name: value - eta * gradients[name]
        for name, value in parameters.items()
    }


def flatten(parameters):
    return torch.cat([parameters[name].reshape(-1) for name in PARAMETER_NAMES])


def parameter_labels(parameters):
    labels = []
    for name in PARAMETER_NAMES:
        shape = parameters[name].shape
        if len(shape) == 2:
            labels.extend(f"{name}[{i + 1},{j + 1}]"
                          for i in range(shape[0]) for j in range(shape[1]))
        else:
            labels.extend(f"{name}[{i + 1}]" for i in range(shape[0]))
    return labels


def experiment(device="cpu"):
    parameters, x, target = make_example(device)
    manual = manual_gradients(parameters, x, target)
    automatic = autograd_gradients(parameters, x, target)
    numerical = central_gradients(parameters, x, target)
    torch.testing.assert_close(flatten(manual), flatten(automatic),
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(flatten(manual), flatten(numerical),
                               atol=1e-9, rtol=0)
    return parameters, x, target, manual, automatic, numerical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    parameters, x, target, manual, automatic, numerical = experiment(args.device)
    updated = gradient_step(parameters, manual)
    before = loss_value(parameters, x, target).item()
    after = loss_value(updated, x, target).item()
    assert after < before

    print(f"PyTorch {torch.__version__}, {args.device}, float64")
    print(f"{'Параметр':12s} {'Вручную':>15s} {'Autograd':>15s} {'Центр. разность':>15s}")
    for label, hand, auto, num in zip(parameter_labels(parameters),
                                      flatten(manual).tolist(),
                                      flatten(automatic).tolist(),
                                      flatten(numerical).tolist()):
        print(f"{label:12s} {hand:+15.10f} {auto:+15.10f} {num:+15.10f}")
    manual_flat = flatten(manual)
    auto_error = (manual_flat - flatten(automatic)).abs().max().item()
    central_error = (manual_flat - flatten(numerical)).abs().max().item()
    print(f"Макс. отклонение от autograd: {auto_error:.3e}")
    print(f"Макс. отклонение от центральной разности: {central_error:.3e}")
    print(f"E до шага = {before:.12f}")
    print(f"E после шага = {after:.12f}")
    print("ŷ после шага =", forward(updated, x)[-1].tolist())


if __name__ == "__main__":
    main()


