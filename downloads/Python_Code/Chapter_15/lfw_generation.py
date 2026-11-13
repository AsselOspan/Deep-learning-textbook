"""Учебная генерация на локальной или зеркальной подвыборке LFW."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import re
import sys
from tempfile import gettempdir
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image
import torch

chapter14 = Path(__file__).resolve().parents[1] / "Chapter_14"
sys.path.insert(0, str(chapter14))
from mini_generative import train_models, evaluate, prior_samples

REPOSITORY = "bitmind/lfw"
REVISION = "274dfbca8b2dc9c6d21ae80f8627c911222eb347"
IMAGE_NAME = re.compile(r"^(.+)_(\d{4})\.jpg$")


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "DeepLearningTextbook/1.0"}), timeout=30) as reply:
        return reply.read()


def download_mirror(folder, max_images=600):
    """Получает первые max_images строк конкретной зеркальной копии."""
    if not 100 <= max_images <= 13233:
        raise ValueError("Нужно от 100 до 13233 изображений")
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, max_images, 100):
        query = urlencode(dict(dataset=REPOSITORY, config="default", split="train",
                               offset=start, length=min(100, max_images - start)))
        response = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))
        if response["num_rows_total"] != 13233 or response["partial"]:
            raise ValueError("Структура зеркала отличается от проверенного набора")
        records.extend(entry["row"] for entry in response["rows"])

    def save(record):
        name = record["filename"]
        match = IMAGE_NAME.fullmatch(name)
        if not match or Path(name).name != name or "/" in match[1] or "\\" in match[1]:
            raise ValueError("Некорректное имя изображения")
        url = record["image"]["src"]
        if f"/{REVISION}/" not in url:
            raise ValueError("Изменилась ревизия зеркала. Проверьте источник.")
        destination = folder / match[1] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            raw = destination.read_bytes()
        else:
            raw = fetch(url)
            with Image.open(BytesIO(raw)) as image:
                if image.size != (250, 250):
                    raise ValueError("Непредвиденный размер изображения")
                image.verify()
            destination.write_bytes(raw)
        return dict(identity=match[1], filename=name, sha256=sha256(raw).hexdigest(),
                    source=f"https://huggingface.co/datasets/{REPOSITORY}/tree/{REVISION}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = list(pool.map(save, records))
    (folder / "source_manifest.json").write_text(json.dumps(
        dict(repository=REPOSITORY, revision=REVISION, subset="first rows", rows=manifest),
        ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def read_lfw(folder, seed=2026, max_per_identity=3):
    grouped = {}
    hashes = set()
    for file in sorted(folder.glob("*/*.jpg")):
        match = IMAGE_NAME.fullmatch(file.name)
        if match is None or match[1] != file.parent.name:
            continue
        raw = file.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        grouped.setdefault(match[1], []).append(file)
    identities = sorted(grouped)
    if len(identities) < 30:
        raise ValueError("Для группового разбиения нужно хотя бы 30 разных людей")
    rng = random.Random(seed)
    rng.shuffle(identities)
    cuts = (int(0.7 * len(identities)), int(0.85 * len(identities)))
    identity_splits = {"train": identities[:cuts[0]], "val": identities[cuts[0]:cuts[1]],
                       "test": identities[cuts[1]:]}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(identity_splits[left]).isdisjoint(identity_splits[right])
    data = {}
    for key, names in identity_splits.items():
        values = []
        for name in names:
            for file in grouped[name][:max_per_identity]:
                with Image.open(file) as source:
                    image = source.convert("L")
                    width, height = image.size
                    side = int(min(width, height) * 0.6)
                    left, top = (width - side) // 2, (height - side) // 2
                    crop = image.crop((left, top, left + side, top + side))
                    pixels = crop.resize((32, 32), Image.Resampling.BILINEAR)
                    values.append(torch.tensor(list(pixels.tobytes()), dtype=torch.float32) / 255)
        data[key] = torch.stack(values)
    return data, {key: len(names) for key, names in identity_splits.items()}


@torch.no_grad()
def diagnostics(samples, train, heldout):
    count = min(64, len(train), len(heldout))
    if count < 2:
        raise ValueError("Недостаточно контрольных изображений")
    distances_train = (samples[:, None, :] - train[None, :count, :]).square().mean(dim=-1)
    distances_heldout = (samples[:, None, :] - heldout[None, :count, :]).square().mean(dim=-1)
    pairwise = torch.pdist(samples).square() / samples.shape[1]
    return dict(reference_size=count, nearest_train_mse=distances_train.min(dim=1).values.mean().item(),
                nearest_heldout_mse=distances_heldout.min(dim=1).values.mean().item(),
                sample_pair_mse=pairwise.mean().item())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfw-dir", type=Path, default=Path(gettempdir()) / "lfw-textbook")
    parser.add_argument("--download-mirror", action="store_true")
    parser.add_argument("--max-images", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.download_mirror:
        download_mirror(args.lfw_dir, args.max_images)
    data, people = read_lfw(args.lfw_dir, args.seed)
    print(f"torch={torch.__version__} cpu seed={args.seed} crop=central60% grayscale32x32 epochs={args.epochs}")
    print("identities " + " ".join(f"{key}={value}" for key,value in people.items()))
    print("images " + " ".join(f"{key}={len(value)}" for key,value in data.items()))
    _, vae, gen, history = train_models(data["train"], epochs=args.epochs,
                                       seed=args.seed, latent=16, hidden=128, include_ae=False)
    for i,row in enumerate(history,1):
        print(f"epoch={i} " + " ".join(f"{key}={value:.6f}" for key,value in row.items()))
    val = evaluate(None,vae,data["val"])
    test = evaluate(None,vae,data["test"])
    print("validation " + " ".join(f"{key}={value:.6f}" for key,value in val.items()))
    print("test " + " ".join(f"{key}={value:.6f}" for key,value in test.items()))
    vs, gs = prior_samples(vae,gen,n=16,latent=16,seed=55)
    results = {"VAE":diagnostics(vs,data["train"],data["test"]),
               "GAN":diagnostics(gs,data["train"],data["test"])}
    for name,result in results.items():
        print(name + " " + " ".join(f"{key}={value:.6f}" for key,value in result.items()))
    if args.samples:
        torch.save(dict(vae=vs,gan=gs),args.samples)
    return dict(vae=vs,gan=gs,validation=val,test=test,diagnostics=results,identities=people,
                sizes={key:len(value) for key,value in data.items()})


if __name__ == "__main__":
    main()
