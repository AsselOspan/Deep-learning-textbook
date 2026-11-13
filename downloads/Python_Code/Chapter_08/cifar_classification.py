"""Контролируемое сравнение аугментации на CIFAR-10."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import gettempdir

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

PROFILES = {"full": (800, 200, 200, 4), "short": (80, 20, 20, 1)}
CLASSES = 10


def select(labels, per_class, rng, offset=0):
    return torch.cat([
        ids[torch.randperm(len(ids), generator=rng)][offset : offset + per_class]
        for label in range(CLASSES)
        for ids in [(labels == label).nonzero().flatten()]
    ])


def subset(source, indices):
    images = torch.as_tensor(source.data[indices.numpy()]).permute(0, 3, 1, 2)
    labels = torch.as_tensor(source.targets)[indices]
    return images, labels


def prepare(folder, profile, seed, download):
    train_n, val_n, test_n, _ = PROFILES[profile]
    original_train = CIFAR10(folder, train=True, download=download)
    original_test = CIFAR10(folder, train=False, download=download)
    labels = torch.as_tensor(original_train.targets)
    selected = select(labels, train_n + val_n, torch.Generator().manual_seed(seed))
    selected = selected.view(CLASSES, train_n + val_n)
    train_ids = selected[:, :train_n].flatten()
    val_ids = selected[:, train_n:].flatten()
    assert not set(train_ids.tolist()) & set(val_ids.tolist())
    test_ids = select(torch.as_tensor(original_test.targets), test_n,
                      torch.Generator().manual_seed(seed + 1))
    pieces = {
        "train": subset(original_train, train_ids),
        "val": subset(original_train, val_ids),
        "test": subset(original_test, test_ids),
    }
    training_pixels = pieces["train"][0].float() / 255
    mean = training_pixels.mean(dim=(0, 2, 3)).view(1, 3, 1, 1)
    std = training_pixels.std(dim=(0, 2, 3), unbiased=False).view(1, 3, 1, 1)
    return {
        key: TensorDataset((x.float() / 255 - mean) / std, y)
        for key, (x, y) in pieces.items()
    }


def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48 * 4 * 4, CLASSES),
    )


@torch.inference_mode()
def evaluate(model, dataset, device):
    model.eval()
    matrix = torch.zeros(CLASSES, CLASSES, dtype=torch.long)
    for x, y in DataLoader(dataset, batch_size=128):
        prediction = model(x.to(device)).argmax(dim=1).cpu()
        matrix += torch.bincount(CLASSES * y + prediction, minlength=CLASSES**2).view(CLASSES, CLASSES)
    diagonal = matrix.diag()
    f1 = 2 * diagonal / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    return {"accuracy": (diagonal.sum() / matrix.sum()).item(),
            "macro_f1": f1.mean().item(), "matrix": matrix.tolist()}


def fit(dataset, validation, initial, flip, epochs, seed, device):
    model = make_model().to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    batches = DataLoader(dataset, batch_size=128, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    augment = torch.Generator().manual_seed(seed + 2)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in batches:
            if flip:
                mask = torch.rand(len(x), generator=augment) < 0.5
                x[mask] = x[mask].flip(dims=(-1,))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        val = evaluate(model, validation, device)
        print(f"epoch={epoch} train_ce={loss_sum / len(dataset):.6f} "
              f"val_accuracy={val['accuracy']:.6f} val_macro_f1={val['macro_f1']:.6f}")
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            best_epoch = epoch
            state = deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model, best_epoch, best


def numeric():
    logits = torch.tensor([[2., 1., 0.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1])
    loss = F.cross_entropy(logits, target)
    loss.backward()
    probability = logits.detach().softmax(dim=1)
    torch.testing.assert_close(logits.grad, probability - F.one_hot(target, 3))
    torch.testing.assert_close(loss, torch.logsumexp(logits.detach(), dim=1)[0] - 1)
    after = F.cross_entropy(logits.detach() - 0.1 * logits.grad, target)
    assert after < loss
    print(f"numeric_ce={loss.item():.9f}->{after.item():.9f} gradient={logits.grad.tolist()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(gettempdir()) / "cifar10-textbook")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    numeric()
    data = prepare(args.data_dir, args.profile, args.seed, args.download)
    sizes = " ".join(f"{key}={len(value)}" for key, value in data.items())
    print(f"torch={torch.__version__} profile={args.profile} {sizes}")
    template = make_model()
    initial = deepcopy(template.state_dict())
    reports = {}
    for name, flip in (("plain", False), ("horizontal_flip", True)):
        print(name)
        model, epoch, val = fit(data["train"], data["val"], initial, flip,
                                PROFILES[args.profile][-1], args.seed, args.device)
        result = evaluate(model, data["test"], args.device)
        result.update(best_epoch=epoch, val_macro_f1=val)
        print(f"best_epoch={epoch} val_macro_f1={val:.6f} test_accuracy={result['accuracy']:.6f} "
              f"test_macro_f1={result['macro_f1']:.6f}")
        for row in result["matrix"]:
            print(" ".join(map(str, row)))
        reports[name] = result
    return reports


if __name__ == "__main__":
    main()

