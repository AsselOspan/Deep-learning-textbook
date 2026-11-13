"""Сравнение замороженных ImageNet-экстракторов на Ford vs Ferrari."""

import argparse
from copy import deepcopy
import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import random
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, gettempdir
from time import perf_counter
from zipfile import ZipFile

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import __version__ as vision_version
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights
from torchvision.models import mobilenet_v3_small, resnet18


CLASSES = 10
SPLITS = ("train", "val", "test")


def cache_training_zip(source, folder):
    folder.mkdir(parents=True, exist_ok=True)
    meta = source.stat()
    target = folder / f"train-{meta.st_size:x}-{meta.st_mtime_ns:x}.zip"
    with ZipFile(source) as outer:
        rows = list(csv.DictReader(outer.read("train.csv").decode("utf-8-sig").splitlines()))
        if not target.is_file() or target.stat().st_size != outer.getinfo("train.zip").file_size:
            with NamedTemporaryFile(dir=folder, delete=False) as output:
                temporary = Path(output.name)
            try:
                with outer.open("train.zip") as stream, temporary.open("wb") as output:
                    copyfileobj(stream, output, length=1024 * 1024)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return rows, target


def open_rgb(raw):
    with Image.open(BytesIO(raw)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def difference_hash(image):
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    gray = small.tobytes()
    signature = 0
    for row in range(8):
        for col in range(8):
            index = row * 9 + col
            signature = (signature << 1) | (gray[index] > gray[index + 1])
    return signature


def duplicate_groups(items):
    """Связные компоненты по совпадению SHA256 или близости dHash."""
    parent = list(range(len(items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, (_, _, sha, dhash) in enumerate(items):
        for j in range(i):
            if items[j][2] == sha or (items[j][3] ^ dhash).bit_count() <= 3:
                parent[find(i)] = find(j)

    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)
    return list(groups.values())


def create_split(inner, rows, limit, split_counts, seed):
    paths = {(PurePosixPath(name).parent.name, PurePosixPath(name).name): name
             for name in inner.namelist() if not name.endswith("/")}
    rng = random.Random(seed)
    items = []
    unreadable = 0
    for label in range(CLASSES):
        candidates = sorted({row["Id"] for row in rows if int(row["Category"]) == label})
        if len(candidates) < limit:
            raise ValueError(f"Класс {label}: доступно {len(candidates)} кандидатов, нужно {limit}")
        for image_id in rng.sample(candidates, limit):
            member = paths.get((str(label), image_id))
            if member is None:
                unreadable += 1
                continue
            try:
                raw = inner.read(member)
                dhash = difference_hash(open_rgb(raw))
            except (OSError, UnidentifiedImageError):
                unreadable += 1
                continue
            items.append((label, member, sha256(raw).hexdigest(), dhash))

    groups = duplicate_groups(items)
    representatives = [[] for _ in range(CLASSES)]
    conflicts = 0
    for group in groups:
        if len({item[0] for item in group}) != 1:
            conflicts += 1
            continue
        label, name, _, _ = group[0]
        representatives[label].append((name, label))
    print(f"readable={len(items)} unreadable={unreadable} groups={len(groups)} "
          f"multi_item_groups={sum(len(g) > 1 for g in groups)} conflicting_groups={conflicts} "
          f"usable_per_class={[len(group) for group in representatives]}")

    split = {part: [] for part in SPLITS}
    for label, choices in enumerate(representatives):
        rng.shuffle(choices)
        if len(choices) < sum(split_counts):
            raise ValueError(f"Класс {label}: доступно {len(choices)} групп. "
                             "Увеличьте число кандидатов или уменьшите размер выборок.")
        start = 0
        for part, count in zip(SPLITS, split_counts):
            split[part].extend(choices[start:start + count])
            start += count
    return split


def model_configs():
    return (
        ("ResNet18", resnet18, ResNet18_Weights.IMAGENET1K_V1),
        ("MobileNetV3-Small", mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    )


def replace_classifier(backbone, name):
    if name == "ResNet18":
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
    return nn.Linear(features, CLASSES)


@torch.no_grad()
def encode(inner, split, backbone, transform, batch_size):
    results = {}
    for part, examples in split.items():
        batches = []
        for start in range(0, len(examples), batch_size):
            images = [transform(open_rgb(inner.read(path)))
                      for path, _ in examples[start:start + batch_size]]
            batches.append(backbone(torch.stack(images)))
        targets = torch.tensor([label for _, label in examples])
        results[part] = TensorDataset(torch.cat(batches), targets)
    return results


@torch.no_grad()
def metrics(head, dataset):
    x, y = dataset.tensors
    predictions = head(x).argmax(dim=1)
    confusion = torch.bincount(CLASSES * y + predictions, minlength=CLASSES ** 2)
    confusion = confusion.reshape(CLASSES, CLASSES)
    correct = confusion.diag()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    return {
        "accuracy": correct.sum().item() / len(y),
        "macro_f1": (2 * correct / denominator.clamp_min(1)).mean().item(),
        "confusion": confusion.tolist(),
    }


def fit_head(head, data, args):
    train = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for x, y in train:
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val = metrics(head, data["val"])
        print(f"epoch={epoch} train_loss={total / len(train.dataset):.4f} "
              f"val_accuracy={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            checkpoint = deepcopy(head.state_dict())
    head.load_state_dict(checkpoint)
    return best_epoch, best_f1


@torch.inference_mode()
def latency_ms(model, sample):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(10):
            model(sample)
        times = []
        for _ in range(30):
            start = perf_counter()
            model(sample)
            times.append(1000 * (perf_counter() - start))
    finally:
        torch.set_num_threads(previous)
    return torch.quantile(torch.tensor(times), torch.tensor([0.50, 0.95])).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(gettempdir()) / "ford-ferrari")
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--train-per-class", type=int, default=30)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--candidates-per-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    counts = (args.train_per_class, args.val_per_class, args.test_per_class)
    if min(*counts, args.epochs, args.batch_size) < 1 or sum(counts) > args.candidates_per_class:
        parser.error("Размеры должны быть положительными, кандидатов нужно не меньше суммы сплитов.")
    if args.torch_home:
        torch.hub.set_dir(str(args.torch_home / "hub"))
    torch.set_num_threads(4)
    rows, inner_path = cache_training_zip(args.archive, args.cache_dir)
    reports = {}
    with ZipFile(inner_path) as inner:
        split = create_split(inner, rows, args.candidates_per_class, counts, args.seed)
        print(f"torch={torch.__version__} torchvision={vision_version} seed={args.seed} "
              f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
        for name, factory, weights in model_configs():
            torch.manual_seed(args.seed)
            backbone = factory(weights=weights).requires_grad_(False).eval()
            head = replace_classifier(backbone, name)
            full = nn.Sequential(backbone, head).eval()
            parameters = sum(p.numel() for p in full.parameters())
            trainable = sum(p.numel() for p in head.parameters())
            print(f"\n{name}: parameters={parameters} trainable={trainable}")
            transform = weights.transforms()
            data = encode(inner, split, backbone, transform, args.batch_size)
            best_epoch, val_macro_f1 = fit_head(head, data, args)
            report = metrics(head, data["test"])
            test_path, _ = split["test"][0]
            sample = transform(open_rgb(inner.read(test_path))).unsqueeze(0)
            p50, p95 = latency_ms(full, sample)
            report.update(best_epoch=best_epoch, val_macro_f1=val_macro_f1,
                          parameters=parameters, trainable=trainable,
                          latency_p50_ms=p50, latency_p95_ms=p95)
            reports[name] = report
            print(f"best_epoch={best_epoch} val_macro_f1={val_macro_f1:.4f} "
                  f"test_accuracy={report['accuracy']:.4f} test_macro_f1={report['macro_f1']:.4f} "
                  f"latency_p50_ms={p50:.3f} latency_p95_ms={p95:.3f}")
            print("confusion_matrix (rows=true, columns=predicted; classes=0..9)")
            for row in report["confusion"]:
                print(" ".join(f"{value:3d}" for value in row))
    return reports


if __name__ == "__main__":
    main()
