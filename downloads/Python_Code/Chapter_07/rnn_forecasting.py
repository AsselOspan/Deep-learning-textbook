"""Прямой прогноз ряда на горизонтах 1, 3 и 6 с RNN, GRU и LSTM."""

import argparse
from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HORIZONS = (1, 3, 6)
HISTORY = 48
HIDDEN = 32
BATCH_SIZE = 128
PROFILES = {"short": (1200, 2), "full": (6000, 8)}


def synthetic_series(length, seed):
    t = torch.arange(length, dtype=torch.float32)
    noise = torch.randn(length, generator=torch.Generator().manual_seed(seed))
    return (torch.sin(2 * math.pi * t / 60)
            + 0.35 * torch.sin(2 * math.pi * t / 17 + 0.4)
            + 0.12 * noise)


def make_splits(series):
    train_end, val_end = 3 * len(series) // 5, 4 * len(series) // 5
    train = series[:train_end]
    mean, std = train.mean(), train.std(unbiased=False).clamp_min(1e-8)
    pieces = {"train": train, "val": series[train_end:val_end], "test": series[val_end:]}
    horizons = torch.tensor(HORIZONS)
    splits = {}
    for name, piece in pieces.items():
        z = (piece - mean) / std
        count = len(z) - HISTORY - max(HORIZONS) + 1
        if count < 1:
            raise ValueError(f"Часть {name} слишком коротка для окна и горизонтов")
        last_seen = torch.arange(count) + HISTORY - 1
        history = z.unfold(0, HISTORY, 1)[:count].unsqueeze(-1).contiguous()
        target = z[last_seen[:, None] + horizons]
        splits[name] = history, target
    return splits, mean, std


class DirectForecaster(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.recurrent = cell(1, HIDDEN, batch_first=True)
        self.readout = nn.Linear(HIDDEN, len(HORIZONS))

    def forward(self, history):
        states, _ = self.recurrent(history)
        return self.readout(states[:, -1, :])


def build_model(name):
    cells = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
    return DirectForecaster(cells[name])


def metrics(prediction, truth):
    error = prediction - truth
    return {
        "mae": error.abs().mean(dim=0).tolist(),
        "rmse": error.square().mean(dim=0).sqrt().tolist(),
    }


@torch.inference_mode()
def predict(model, history, device):
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in history.split(BATCH_SIZE)])


def fit(model, splits, std, epochs, seed, device):
    train = DataLoader(
        TensorDataset(*splits["train"]), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x, val_y = splits["val"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    best_rmse = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        max_norm, clipped = 0.0, 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            total += loss.item() * len(y)
            max_norm = max(max_norm, norm)
            clipped += int(norm > 1.0)
        val_error = (predict(model, val_x, device) - val_y) * std
        val_rmse = val_error.square().mean().sqrt().item()
        print(f"epoch={epoch} train_mse_scaled={total / len(train.dataset):.6f} "
              f"val_rmse={val_rmse:.6f} grad_max_before_clip={max_norm:.4f} "
              f"clipped_batches={clipped}/{len(train)}")
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            checkpoint = deepcopy(model.state_dict())
    model.load_state_dict(checkpoint)
    return best_epoch, best_rmse


def persistence(history):
    return history[:, -1, :].expand(-1, len(HORIZONS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(2)
    length, epochs = PROFILES[args.profile]
    series = synthetic_series(length, args.seed)
    splits, mean, std = make_splits(series)
    test_x, test_y = splits["test"]
    val_x, val_y = splits["val"]
    truth = test_y * std + mean
    print(f"torch={torch.__version__} device={device} seed={args.seed} "
          f"profile={args.profile} observations={length} history={HISTORY} horizons={HORIZONS}")
    print(f"train_windows={len(splits['train'][0])} val_windows={len(val_x)} "
          f"test_windows={len(test_x)} train_mean={mean:.6f} train_std={std:.6f}")

    reports = {}
    for name in ("Persistence", "RNN", "GRU", "LSTM"):
        print(f"\n{name}")
        if name == "Persistence":
            parameters, best_epoch = 0, None
            val_error = (persistence(val_x) - val_y) * std
            val_rmse = val_error.square().mean().sqrt().item()
            prediction = persistence(test_x)
        else:
            torch.manual_seed(args.seed)
            model = build_model(name).to(device)
            parameters = sum(p.numel() for p in model.parameters())
            best_epoch, val_rmse = fit(model, splits, std, epochs, args.seed, device)
            prediction = predict(model, test_x, device)
        prediction = prediction * std + mean
        stats = metrics(prediction, truth)
        reports[name] = dict(stats, parameters=parameters, best_epoch=best_epoch,
                             val_rmse=val_rmse, truth=truth, prediction=prediction)
        print(f"parameters={parameters} best_epoch={best_epoch} val_rmse={val_rmse:.6f}")
        for h, mae, rmse in zip(HORIZONS, stats["mae"], stats["rmse"]):
            print(f"horizon={h} test_mae={mae:.6f} test_rmse={rmse:.6f}")
    return reports


if __name__ == "__main__":
    main()
