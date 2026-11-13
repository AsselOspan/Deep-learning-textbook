"""Предсказание жанра фильма по описанию из локального учебного архива."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
import random
import re
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ("comedy", "documentary", "drama", "horror")
PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)
MAX_LEN = 100
PROFILES = {"full": (700, 150, 150, 5), "short": (80, 20, 20, 1)}


def tokenize(text):
    return PATTERN.findall(text.lower())[:MAX_LEN]


def prepare(archive, profile, seed):
    train_n, val_n, test_n, _ = PROFILES[profile]
    groups = {key: [] for key in CLASSES}
    titles, plots = set(), set()
    duplicates = 0
    with ZipFile(archive) as source, source.open("train.csv") as raw:
        for row in csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline="")):
            if row["genre"] not in groups:
                continue
            title = " ".join(row["name"].casefold().split())
            plot = " ".join(row["text"].casefold().split())
            words = tokenize(row["text"])
            if not words:
                continue
            digest = sha256(plot.encode()).hexdigest()
            if title in titles or digest in plots:
                duplicates += 1
                continue
            titles.add(title)
            plots.add(digest)
            groups[row["genre"]].append(words)
    rng = random.Random(seed)
    rows = {"train": [], "val": [], "test": []}
    for label, name in enumerate(CLASSES):
        values = groups[name]
        rng.shuffle(values)
        if len(values) < train_n + val_n + test_n:
            raise ValueError(f"Слишком мало описаний класса {name}")
        rows["train"].extend((words, label) for words in values[:train_n])
        rows["val"].extend((words, label) for words in values[train_n : train_n + val_n])
        rows["test"].extend((words, label) for words in values[train_n + val_n : train_n + val_n + test_n])

    counts = Counter(word for words, _ in rows["train"] for word in words)
    ordered = sorted(counts, key=lambda word: (-counts[word], word))[:4000]
    vocab = {word: index for index, word in enumerate(ordered, start=2)}
    datasets = {}
    for name, data in rows.items():
        token_ids = torch.zeros((len(data), MAX_LEN), dtype=torch.long)
        lengths, labels = [], []
        for index, (words, label) in enumerate(data):
            token_ids[index, : len(words)] = torch.tensor([vocab.get(w, 1) for w in words])
            lengths.append(len(words))
            labels.append(label)
        datasets[name] = TensorDataset(token_ids, torch.tensor(lengths), torch.tensor(labels))
    print(f"duplicates_removed={duplicates} vocabulary={len(vocab) + 2}")
    return datasets, len(vocab) + 2


class GenreGRU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 48, padding_idx=0)
        self.encoder = nn.GRU(48, 48, batch_first=True)
        self.head = nn.Linear(48, len(CLASSES))

    def forward(self, tokens, lengths):
        sequence = pack_padded_sequence(self.embedding(tokens), lengths, batch_first=True,
                                        enforce_sorted=False)
        _, hidden = self.encoder(sequence)
        return self.head(hidden[-1])


def naive_bayes(train, vocab_size):
    tokens, _, y = train.tensors
    counts = torch.ones(len(CLASSES), vocab_size)
    for k in range(len(CLASSES)):
        counts[k] += torch.bincount(tokens[y == k].flatten(), minlength=vocab_size)
    counts[:, 0] = 0
    log_prob = (counts / counts.sum(dim=1, keepdim=True)).clamp_min(1e-30).log()
    log_prob[:, 0] = 0
    log_prior = (torch.bincount(y, minlength=len(CLASSES)) / len(y)).log()

    def predict(dataset):
        ids = dataset.tensors[0]
        return (log_prob[:, ids].sum(dim=2).T + log_prior).argmax(dim=1)

    return predict


def scores(prediction, dataset):
    y = dataset.tensors[-1]
    n = len(CLASSES)
    matrix = torch.bincount(n * y + prediction, minlength=n * n).view(n, n)
    correct = matrix.diag()
    f1 = 2 * correct / (matrix.sum(dim=0) + matrix.sum(dim=1)).clamp_min(1)
    return {"accuracy": (correct.sum() / len(y)).item(),
            "macro_f1": f1.mean().item(), "matrix": matrix.tolist()}


@torch.inference_mode()
def gru_predict(model, data, device):
    model.eval()
    return torch.cat([model(tokens.to(device), lengths).argmax(dim=1).cpu()
                      for tokens, lengths, _ in DataLoader(data, batch_size=128)])


def fit(model, data, epochs, seed, device):
    batches = DataLoader(data["train"], batch_size=64, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, lengths, y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(tokens.to(device), lengths), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        score = scores(gru_predict(model, data["val"], device), data["val"])["macro_f1"]
        print(f"epoch={epoch} val_macro_f1={score:.6f}")
        if score > best:
            best, best_epoch = score, epoch
            state = deepcopy(model.state_dict())
    model.load_state_dict(state)
    return best_epoch


def numeric():
    logits = torch.tensor([1., 0., 3.], dtype=torch.float64)
    available = torch.tensor([True, True, False])
    weights = logits.masked_fill(~available, float("-inf")).softmax(dim=0)
    values = torch.tensor([[2., 0.], [0., 4.], [9., 9.]], dtype=torch.float64)
    context = weights @ values
    torch.testing.assert_close(context, torch.tensor([1.4621171572600098, 1.0757656854799804],
                                                     dtype=torch.float64))
    print(f"masked_attention={weights.tolist()} context={context.tolist()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    numeric()
    data, vocab_size = prepare(args.archive, args.profile, args.seed)
    print(f"profile={args.profile} " + " ".join(f"{k}={len(v)}" for k, v in data.items()))
    print(f"classes={','.join(CLASSES)}")
    model = GenreGRU(vocab_size).to(args.device)
    best_epoch = fit(model, data, PROFILES[args.profile][-1], args.seed, args.device)
    print(f"GRU_parameters={sum(p.numel() for p in model.parameters())} best_epoch={best_epoch}")
    bayes = naive_bayes(data["train"], vocab_size)
    reports = {}
    for name, predict in (("NaiveBayes", bayes),
                          ("GRU", lambda d: gru_predict(model, d, args.device))):
        val = scores(predict(data["val"]), data["val"])
        test = scores(predict(data["test"]), data["test"])
        print(f"{name} val_f1={val['macro_f1']:.6f} test_accuracy={test['accuracy']:.6f} "
              f"test_macro_f1={test['macro_f1']:.6f}")
        for row in test["matrix"]:
            print(" ".join(map(str, row)))
        reports[name] = test
    return reports


if __name__ == "__main__":
    main()

