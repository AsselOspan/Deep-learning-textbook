"""Перенос ResNet18 на шесть классов из локального архива Oxford Flowers 102."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from random import Random
from tarfile import open as open_tar
from struct import unpack, unpack_from
from tempfile import gettempdir
from urllib.request import urlopen
from zlib import decompress

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

OXFORD = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def mat_elements(data):
    """Блоки MATLAB v5 little-endian в двух официальных файлах Oxford."""
    offset = 0
    while offset < len(data):
        tag, size = unpack_from("<II", data, offset)
        if tag >> 16:
            yield tag & 0xFFFF, data[offset + 4 : offset + 4 + (tag >> 16)]
            offset += 8
        else:
            yield tag, data[offset + 8 : offset + 8 + size]
            # Сжатые элементы MATLAB следуют без выравнивания.
            offset += 8 + (size if tag == 15 else (size + 7) // 8 * 8)


def oxford_arrays(data):
    if data[124:128] != bytes((0, 1, 73, 77)):
        raise ValueError("Ожидались метаданные MATLAB v5 little-endian")
    arrays = {}
    for kind, body in mat_elements(data[128:]):
        if kind != 15:
            raise ValueError("Ожидался сжатый блок официальных метаданных")
        (matrix_kind, matrix), = mat_elements(decompress(body))
        if matrix_kind != 14:
            raise ValueError("Ожидалась матрица официальных метаданных")
        _, (_, dims), (_, name), (dtype, raw) = mat_elements(matrix)
        rows, cols = unpack("<ii", dims)
        if dtype == 2 and len(raw) == rows * cols:
            values = tuple(raw)
        elif dtype == 4 and len(raw) == 2 * rows * cols:
            values = unpack(f"<{rows * cols}H", raw)
        else:
            raise ValueError("Неподдерживаемый тип массива метаданных")
        arrays[name.decode("ascii")] = values
    return arrays


def metadata(cache):
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {}
    hashes = {
        "imagelabels.mat": "4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d",
        "setid.mat": "46b8678f91fd95d3c8f4feab80d271a6c834a1dd896fe29fd3e6ad9ce5c8dccd",
    }
    for name, expected in hashes.items():
        path = cache / name
        if path.is_file():
            data = path.read_bytes()
        else:
            with urlopen(OXFORD + name, timeout=30) as response:
                data = response.read()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Контрольная сумма {name} не совпала с официальным файлом")
        arrays.update(oxford_arrays(data))
        if not path.is_file():
            path.write_bytes(data)
    labels = arrays.pop("labels")
    assert len(labels) == 8189 and set(labels) == set(range(1, 103))
    assert tuple(map(len, arrays.values())) == (1020, 1020, 6149)
    assert set(i for split in arrays.values() for i in split) == set(range(1, 8190))
    return labels, arrays


def select_ids(labels, splits, classes, limit, seed):
    """Oxford использует 1-индексные идентификаторы изображений и классов."""
    rng = Random(seed)
    picked = {}
    for split in ("trnid", "valid", "tstid"):
        ids = [int(i) for i in splits[split]]
        per_class = []
        for cls in classes:
            members = sorted(i for i in ids if labels[i - 1] == cls)
            if split == "tstid" and len(members) > limit:
                members = sorted(rng.sample(members, limit))
            per_class.extend(members)
        picked[split] = per_class
    all_ids = [i for ids in picked.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)), "Разделения должны быть непересекающимися"
    return picked


def load_data(archive, ids, labels, classes):
    """Читаем только нужные JPEG из архива без распаковки на диск."""
    preprocess = WEIGHTS.transforms()
    target_index = {cls: i for i, cls in enumerate(classes)}
    names = {f"image_{i:05d}.jpg": i for split in ids.values() for i in split}
    images = {}
    with open_tar(archive, mode="r|gz") as tar:
        for entry in tar:
            name = Path(entry.name).name
            if not entry.isfile() or name not in names:
                continue
            with tar.extractfile(entry) as stream:
                with Image.open(BytesIO(stream.read())) as image:
                    images[names.pop(name)] = preprocess(image.convert("RGB"))
            if not names:
                break
    if names:
        raise ValueError(f"В архиве не хватает {len(names)} выбранных изображений")
    return {
        key: TensorDataset(
            torch.stack([images[i] for i in split]),
            torch.tensor([target_index[int(labels[i - 1])] for i in split]),
        )
        for key, split in ids.items()
    }


def frozen_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def fit(model, train, validation, optimizer, epochs, device, seed, batch_size, label):
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    objective = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        frozen_batchnorm(model)
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = objective(model(x), y)
            value.backward()
            optimizer.step()
            total += value.item() * len(y)
        val_loss, matrix = evaluate(model, validation, device)
        print(f"{label} epoch={epoch} train_loss={total / len(train):.6f} "
              f"val_loss={val_loss:.6f} val_correct={matrix.diag().sum().item()}/{matrix.sum().item()}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    count = model.fc.out_features
    matrix = torch.zeros((count, count), dtype=torch.long)
    total = 0.0
    for x, y in loader:
        y = y.to(device)
        logits = model(x.to(device))
        total += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        codes = (y * count + logits.argmax(dim=1)).cpu()
        matrix += torch.bincount(codes, minlength=count * count).reshape(count, count)
    return total / matrix.sum().item(), matrix


def snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_changes(model, initial, late):
    changed = {name for name, current in snapshot(model).items()
               if not torch.equal(initial[name], current)}
    assert any(name.startswith("fc.") for name in changed)
    assert any(name.startswith("layer4.") for name in changed) == late
    assert all(name.startswith(("fc.", "layer4.") if late else ("fc.",)) for name in changed)
    buffer_names = {name for name, _ in model.named_buffers()}
    assert changed.isdisjoint(buffer_names), changed & buffer_names
    print(f"changed_tensors={len(changed)} changed_buffers=0 trainable={trainable(model)}")


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    labels, splits = metadata(args.cache_dir)
    ids = select_ids(labels, splits, args.classes, args.max_test_per_class, args.seed)
    sets = load_data(args.archive, ids, labels, args.classes)
    train = sets["trnid"]
    validation = DataLoader(sets["valid"], batch_size=args.batch_size)
    test = DataLoader(sets["tstid"], batch_size=args.batch_size)
    print(f"torch={torch.__version__} device={args.device} seed={args.seed}")
    print(f"classes={args.classes} train={len(train)} val={len(validation.dataset)} "
          f"test={len(test.dataset)} batch={args.batch_size}")
    model = resnet18(weights=WEIGHTS)
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(args.classes))
    model.to(args.device)
    initial = snapshot(model)
    head_lr, layer_lr = 0.01, 0.001
    warmup = torch.optim.SGD(model.fc.parameters(), lr=head_lr)
    fit(model, train, validation, warmup, args.head_epochs, args.device,
        args.seed, args.batch_size, "warmup")
    check_changes(model, initial, late=False)
    common = snapshot(model)

    test_results = []
    for label, late in (("head_only", False), ("fine_tune", True)):
        model.load_state_dict(common)
        model.layer4.requires_grad_(late)
        model.zero_grad(set_to_none=True)
        groups = [{"params": model.fc.parameters(), "lr": head_lr}]
        if late:
            groups.append({"params": model.layer4.parameters(), "lr": layer_lr})
        optimizer = torch.optim.SGD(groups)
        fit(model, train, validation, optimizer, args.continuation_epochs, args.device,
            args.seed + 1, args.batch_size, label)
        check_changes(model, common, late=late)
        test_results.append((label, *evaluate(model, test, args.device)))

    for label, test_loss, matrix in test_results:
        correct = matrix.diag().sum().item()
        count = matrix.sum().item()
        print(f"{label} test_loss={test_loss:.6f} test_correct={correct}/{count} "
              f"test_accuracy={correct / count:.6f}")
        print(f"{label} confusion_rows=true columns=prediction")
        for row in matrix.tolist():
            print(" ".join(map(str, row)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(gettempdir()) / "oxford-flowers-102")
    parser.add_argument("--classes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--max-test-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--continuation-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if len(set(args.classes)) != len(args.classes) or len(args.classes) < 2:
        parser.error("Нужны хотя бы два различных класса")
    if any(not 1 <= cls <= 102 for cls in args.classes):
        parser.error("Номера категорий должны быть от 1 до 102")
    if min(args.max_test_per_class, args.batch_size, args.head_epochs,
           args.continuation_epochs, args.threads) < 1:
        parser.error("Размеры и число эпох должны быть положительными")
    if not args.archive.is_file():
        parser.error("Укажите доступный локальный архив 102flowers.tgz")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    run(args)


if __name__ == "__main__":
    main()

