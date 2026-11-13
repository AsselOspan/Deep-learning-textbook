"""Контролируемое сравнение ReLU и sigmoid на Fashion-MNIST."""

import argparse
from pathlib import Path
import sys
from tempfile import gettempdir

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Chapter_02"))
from fashion_mnist_mlp import CHECKSUMS, download_data, evaluate, fit, load_data


def build_models():
    def build(activation):
        layers = [nn.Linear(784, 128), activation()]
        for _ in range(3):
            layers.extend([nn.Linear(128, 128), activation()])
        return nn.Sequential(*layers, nn.Linear(128, 10))

    relu, sigmoid = build(nn.ReLU), build(nn.Sigmoid)
    sigmoid.load_state_dict(relu.state_dict())
    return {"ReLU": relu, "Sigmoid": sigmoid}


def probe_gradients(model, first_batch, device):
    """Вычисляет нормы градиентов весов до первого обновления."""
    x, y = (tensor.to(device) for tensor in first_batch)
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    affine = [layer for layer in model if isinstance(layer, nn.Linear)]
    first = affine[0].weight.grad.norm().item()
    last = affine[-2].weight.grad.norm().item()
    model.zero_grad(set_to_none=True)
    return loss.item(), first, last


def make_train_loader(dataset, batch_size, seed):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(gettempdir()) / "fashion-mnist")
    parser.add_argument("--download", action="store_true", help="Скачать файлы из репозитория Zalando")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
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

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    train, val, test = load_data(args.data_dir, args.seed)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    first_batch = next(iter(make_train_loader(train, args.batch_size, args.seed)))
    print(f"torch={torch.__version__} device={device} seed={args.seed} "
          f"train={len(train)} val={len(val)} test={len(test)}")

    for name, model in build_models().items():
        print(f"\n{name}: parameters={sum(p.numel() for p in model.parameters())}")
        model.to(device)
        loss, first, last = probe_gradients(model, first_batch, device)
        print(f"initial_loss={loss:.6f} grad_first={first:.6e} grad_last_hidden={last:.6e}")
        train_loader = make_train_loader(train, args.batch_size, args.seed)
        best_epoch, best_accuracy = fit(model, train_loader, val_loader, device, args.epochs)
        accuracy, f1, matrix = evaluate(model, test_loader, device)
        print(f"best_epoch={best_epoch} best_val_accuracy={best_accuracy:.4f} "
              f"test_accuracy={accuracy:.4f} test_macro_f1={f1:.4f}")
        print("confusion_matrix (rows=true, columns=predicted; classes=0..9)")
        for row in matrix.tolist():
            print(" ".join(f"{value:4d}" for value in row))


if __name__ == "__main__":
    main()
