"""Числовые проверки baseline, BCE-градиента и CNN-декодера."""
import argparse
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def check_numeric():
    train = torch.tensor([[.2, .4], [.4, .8], [.6, .6]], dtype=torch.float64)
    test = torch.tensor([[.5, .5], [.9, .1]], dtype=torch.float64)
    average = train.mean(dim=0)
    torch.testing.assert_close(average, torch.tensor([.4, .6], dtype=torch.float64))
    train_mse = (train - average).square().mean()
    test_each = (test - average).square().mean(dim=1)
    torch.testing.assert_close(test_each, torch.tensor([.01, .25], dtype=torch.float64))
    assert abs(train_mse.item() - .02666666666666667) < 1e-12
    print(f"toy mean={average.tolist()} train_mse={train_mse:.9f} test_mse={test_each.mean():.9f}")

    target = torch.tensor(.8, dtype=torch.float64)
    logit = torch.tensor(.25, dtype=torch.float64).log().requires_grad_()
    probability = logit.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logit, target)
    mse = (probability - target).square()
    gb, = torch.autograd.grad(bce, logit, retain_graph=True)
    gm, = torch.autograd.grad(mse, logit)
    torch.testing.assert_close(gb, probability - target)
    torch.testing.assert_close(gm, 2 * (probability-target) * probability * (1-probability))
    for name, value, gradient in (("BCE", bce, gb), ("MSE", mse, gm)):
        updated = (logit.detach() - .1 * gradient).sigmoid()
        print(f"{name} loss={value.item():.9f} grad_logit={gradient.item():.9f} p_after={updated.item():.9f}")


class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(16, 32*8*8)
        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.stem(z)).reshape(-1, 32, 8, 8)
        h = F.relu(self.up1(h))
        return self.up2(h)  # Логиты, без дополнительной sigmoid перед BCE.


def check_decoder():
    torch.manual_seed(2026)
    model = CNNDecoder()
    count = sum(p.numel() for p in model.parameters())
    assert count == 43281
    z = torch.randn(4, 16)
    target = torch.linspace(.1, .9, 32*32).reshape(1, 1, 32, 32).repeat(4, 1, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.0001)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    output = model(z)
    assert output.shape == (4, 1, 32, 32)
    loss = F.binary_cross_entropy_with_logits(output, target, reduction="none").flatten(1).sum(1).mean()
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    changed = sum(not torch.equal(before[name], p.detach()) for name,p in model.named_parameters())
    with torch.no_grad():
        after = F.binary_cross_entropy_with_logits(model(z), target, reduction="none").flatten(1).sum(1).mean()
    assert changed == len(before) and after < loss
    print(f"CNN params={count} shape={tuple(output.shape)} changed_tensors={changed} loss={loss.item():.9f}->{after.item():.9f}")
    extension = nn.Sequential(nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
                              nn.ConvTranspose2d(8, 1, 4, 2, 1))
    with torch.no_grad():
        hidden = F.relu(model.up1(F.relu(model.stem(z)).reshape(-1, 32, 8, 8)))
        enlarged = extension(hidden)
    count64 = sum(p.numel() for layer in (model.stem, model.up1, extension)
                  for p in layer.parameters())
    assert enlarged.shape == (4, 1, 64, 64) and count64 == 45209
    print(f"CNN64 shape-only check params={count64} shape={tuple(enlarged.shape)}; not trained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfw-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    check_numeric()
    check_decoder()
    if args.lfw_dir:
        from lfw_generation import read_lfw
        data, _ = read_lfw(args.lfw_dir, seed=2026)
        mean_train = data["train"].mean(dim=0)
        for split, values in data.items():
            error = (values-mean_train).square().mean()
            print(f"LFW train-mean baseline {split} mse={error.item():.9f}")
    print("PASS: numerical, gradients, shapes and one CNN update")


if __name__ == "__main__":
    main()
