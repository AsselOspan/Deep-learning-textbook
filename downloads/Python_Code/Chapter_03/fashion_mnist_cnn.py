"""Свёрточная классификация Fashion-MNIST на PyTorch."""

import argparse
from pathlib import Path
import sys
from tempfile import gettempdir

import torch
from torch import nn
from torch.utils.data import DataLoader

# Используем проверенный протокол главы 2 при запуске из любой папки.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Chapter_02"))
from fashion_mnist_mlp import CHECKSUMS, download_data, evaluate, fit, load_data


def build_model(architecture="cnn2"):
    channels = {"cnn2": (16, 32), "cnn3": (16, 32, 64)}[architecture]
    layers = [nn.Unflatten(1, (1, 28, 28))]
    in_channels, side = 1, 28
    for out_channels in channels:
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ])
        in_channels, side = out_channels, side // 2
    layers.extend([nn.Flatten(), nn.Linear(in_channels * side * side, 10)])
    return nn.Sequential(*layers)


@torch.no_grad()
def show_shapes(model):
    x = torch.zeros((1, 784))
    for layer in model:
        x = layer(x)
        count = sum(p.numel() for p in layer.parameters())
        print(f"{type(layer).__name__:11s} shape={tuple(x.shape)!s:16s} parameters={count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(gettempdir()) / "fashion-mnist")
    parser.add_argument("--download", action="store_true", help="Скачать файлы из репозитория Zalando")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=("cnn2", "cnn3"), default="cnn2")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs и batch-size должны быть положительными")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.download:
        download_data(args.data_dir)
    missing = [name for name in CHECKSUMS if not (args.data_dir / name).is_file()]
    if missing:
        parser.error(f"Не найдены {', '.join(missing)}. Укажите --data-dir или --download.")

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    train, val, test = load_data(args.data_dir, args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model = build_model(args.architecture)
    print(f"torch={torch.__version__} device={device} seed={args.seed} "
          f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"{args.architecture}: parameters={sum(p.numel() for p in model.parameters())}")
    show_shapes(model)
    model.to(device)
    best_epoch, best_accuracy = fit(model, train_loader, val_loader, device, args.epochs)
    accuracy, f1, matrix = evaluate(model, test_loader, device)
    print(f"best_epoch={best_epoch} best_val_accuracy={best_accuracy:.4f} "
          f"test_accuracy={accuracy:.4f} test_macro_f1={f1:.4f}")
    print("confusion_matrix (rows=true, columns=predicted; classes=0..9)")
    for row in matrix.tolist():
        print(" ".join(f"{value:4d}" for value in row))


if __name__ == "__main__":
    main()
