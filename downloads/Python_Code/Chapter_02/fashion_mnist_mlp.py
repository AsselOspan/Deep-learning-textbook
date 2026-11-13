"""Сравнение Linear и MLP на официальном Fashion-MNIST."""

import argparse
from copy import deepcopy
import gzip
from hashlib import md5
from pathlib import Path
import struct
from tempfile import gettempdir
from urllib.request import urlopen

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


SOURCE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def download_data(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in CHECKSUMS.items():
        target = folder / name
        if target.is_file() and md5(target.read_bytes()).hexdigest() == digest:
            continue
        with urlopen(SOURCE + name, timeout=60) as response:
            data = response.read()
        if md5(data).hexdigest() != digest:
            raise ValueError(f"Контрольная сумма не совпала: {name}")
        target.write_bytes(data)


def read_split(folder, prefix, count):
    with gzip.open(folder / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
        image_header = struct.unpack(">IIII", f.read(16))
        images = bytearray(f.read())
    with gzip.open(folder / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
        label_header = struct.unpack(">II", f.read(8))
        labels = bytearray(f.read())
    if (image_header != (2051, count, 28, 28) or label_header != (2049, count)
            or len(images) != count * 784 or len(labels) != count):
        raise ValueError(f"Некорректный формат или размер IDX: {prefix}")

    x = torch.frombuffer(images, dtype=torch.uint8).reshape(count, 784)
    y = torch.frombuffer(labels, dtype=torch.uint8).long()
    if y.max().item() > 9:
        raise ValueError(f"Некорректные метки классов: {prefix}")
    return TensorDataset(x.float().div_(255), y)


def load_data(folder, seed):
    training = read_split(folder, "train", 60000)
    test = read_split(folder, "t10k", 10000)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed))
    return Subset(training, order[:55000]), Subset(training, order[55000:]), test


def models():
    return {
        "Linear": nn.Linear(784, 10),
        "MLP": nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 10),
        ),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros((10, 10), dtype=torch.long)
    for x, y in loader:
        predicted = model(x.to(device)).argmax(dim=1).cpu()
        confusion += torch.bincount(10 * y + predicted, minlength=100).reshape(10, 10)

    correct = confusion.diag()
    accuracy = correct.sum().item() / confusion.sum().item()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    macro_f1 = (2 * correct / denominator.clamp_min(1)).mean().item()
    return accuracy, macro_f1, confusion


def fit(model, train_loader, val_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy, best_epoch = -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        accuracy, f1, _ = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={total_loss / len(train_loader.dataset):.4f} "
              f"val_accuracy={accuracy:.4f} val_macro_f1={f1:.4f}")
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_weights = deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return best_epoch, best_accuracy


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
    print(f"torch={torch.__version__} device={device} seed={args.seed} "
          f"train={len(train)} val={len(val)} test={len(test)}")

    for name, model in models().items():
        print(f"\n{name}: parameters={sum(p.numel() for p in model.parameters())}")
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
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
