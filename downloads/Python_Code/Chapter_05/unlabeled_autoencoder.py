"""Автоэнкодер Fashion-MNIST и поиск соседей без загрузки меток."""

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
from torch.utils.data import DataLoader


SOURCE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
}


def download_images(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKSUMS.items():
        target = folder / name
        if target.is_file() and md5(target.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(SOURCE + name, timeout=60) as response:
            contents = response.read()
        if md5(contents).hexdigest() != expected:
            raise ValueError(f"Контрольная сумма не совпала: {name}")
        target.write_bytes(contents)


def read_images(folder, prefix, count):
    name = f"{prefix}-images-idx3-ubyte.gz"
    with gzip.open(folder / name, "rb") as stream:
        header = struct.unpack(">IIII", stream.read(16))
        pixels = bytearray(stream.read())
    if header != (2051, count, 28, 28) or len(pixels) != count * 784:
        raise ValueError(f"Некорректный формат или размер IDX: {name}")
    images = torch.frombuffer(pixels, dtype=torch.uint8).reshape(count, 784)
    return images.float().div_(255)


def load_images(folder, seed):
    source = read_images(folder, "train", 60000)
    order = torch.randperm(len(source), generator=torch.Generator().manual_seed(seed))
    test = read_images(folder, "t10k", 10000)
    return source[order[:55000]], source[order[55000:]], test


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@torch.no_grad()
def pixel_mse(model, loader, device):
    model.eval()
    squares, count = 0.0, 0
    for images in loader:
        images = images.to(device)
        squares += (model(images) - images).square().sum().item()
        count += images.numel()
    return squares / count


@torch.no_grad()
def mean_image_mse(train, loader, device):
    mean_image = train.mean(dim=0).to(device)
    squares, count = 0.0, 0
    for images in loader:
        images = images.to(device)
        squares += (mean_image - images).square().sum().item()
        count += images.numel()
    return squares / count


def fit(model, train_loader, val_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_epoch, best_mse = 0, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        squares, count = 0.0, 0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = (model(images) - images).square().mean()
            loss.backward()
            optimizer.step()
            squares += loss.item() * images.numel()
            count += images.numel()
        val_mse = pixel_mse(model, val_loader, device)
        print(f"epoch={epoch} train_mse={squares / count:.6f} val_mse={val_mse:.6f}")
        if val_mse < best_mse:
            best_epoch, best_mse = epoch, val_mse
            best_weights = deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    return best_epoch, best_mse


@torch.no_grad()
def encode(model, loader, device):
    model.eval()
    return torch.cat([model.encoder(images.to(device)).cpu() for images in loader])


def nearest_neighbors(embeddings, query_index, k=5):
    if not 0 <= query_index < len(embeddings) or not 0 < k < len(embeddings):
        raise ValueError("Нужны индекс из галереи и 1 ≤ k < числа объектов")
    squared = (embeddings - embeddings[query_index]).square().sum(dim=1)
    # Убираем из поиска объект с тем же индексом. Остальные строки остаются.
    squared[query_index] = float("inf")
    order = torch.argsort(squared, stable=True)[:k]
    return order, squared[order].sqrt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(gettempdir()) / "fashion-mnist")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.latent_dim) < 1:
        parser.error("epochs, batch-size и latent-dim должны быть положительными")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.download:
        download_images(args.data_dir)
    if any(not (args.data_dir / name).is_file() for name in CHECKSUMS):
        parser.error("Укажите папку с изображениями Fashion-MNIST или добавьте --download")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    train, val, test = load_images(args.data_dir, args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    print(f"PyTorch {torch.__version__} | {device} | seed={args.seed}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"latent={args.latent_dim}, parameters={sum(p.numel() for p in model.parameters())}")
    best_epoch, best_mse = fit(model, train_loader, val_loader, device, args.epochs)
    print(f"best_epoch={best_epoch} val_mse={best_mse:.6f}")
    print(f"mean_image_val_mse={mean_image_mse(train, val_loader, device):.6f}")
    print(f"test_mse={pixel_mse(model, test_loader, device):.6f}")
    print(f"mean_image_test_mse={mean_image_mse(train, test_loader, device):.6f}")

    gallery = DataLoader(val[:1024], batch_size=args.batch_size)
    embeddings = encode(model, gallery, device)
    neighbors, distances = nearest_neighbors(embeddings, query_index=0)
    print("nearest_indices:", neighbors.tolist())
    print("distances:", [round(v, 6) for v in distances.tolist()])


if __name__ == "__main__":
    main()
