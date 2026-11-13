"""Сопоставление GRU, LSTM и прогноза последним наблюдением."""
import argparse
import csv
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_series(length=9000, seed=42):
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    return torch.sin(0.04 * t) + 0.1 * torch.randn(length, generator=generator)


def windows(series, back=50, horizon=10):
    frames = series.unfold(0, back + horizon, 1)
    return frames[:, :back, None].contiguous(), frames[:, back:, None].contiguous()


class Forecaster(nn.Module):
    def __init__(self, name, hidden=64, horizon=10):
        super().__init__()
        cell = {'GRU': nn.GRU, 'LSTM': nn.LSTM}[name]
        self.cell = cell(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        sequence, _ = self.cell(x)
        return self.head(sequence[:, -1, :]).unsqueeze(-1)


def persistence(x, horizon=10):
    return x[:, -1:, :].expand(-1, horizon, -1)


def evaluate(prediction, target):
    residual = prediction - target
    return residual.square().mean().sqrt().item(), residual.abs().mean().item()


@torch.inference_mode()
def forecast(model, x):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(part.to(device)).cpu() for part in x.split(256)])


def fit(name, batch, epochs, train, val, device):
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(TensorDataset(*train), batch_size=batch,
                        shuffle=True, generator=generator)
    model = Forecaster(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, best_state = float('inf'), 0, None
    times = []
    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = perf_counter()
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(perf_counter() - start)
        val_rmse, _ = evaluate(forecast(model, val[0]), val[1])
        if val_rmse < best_loss:
            best_loss, best_epoch = val_rmse, epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    info = dict(model=name, batch=batch, best_epoch=best_epoch,
                val_rmse=best_loss, seconds_per_epoch=mean(times),
                parameters=sum(p.numel() for p in model.parameters()))
    return model, info


def experiment(epochs=10, batches=(16, 64, 256), device=None):
    if epochs < 1 or any(batch < 1 for batch in batches) or not batches:
        raise ValueError('Укажите положительное число эпох и размеры батча.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cpu':
        torch.set_num_threads(2)
    series = make_series()
    train, val, test = (windows(part) for part in
                        (series[:6000], series[6000:7500], series[7500:]))
    runs = [fit(name, batch, epochs, train, val, device)
            for name in ('GRU', 'LSTM') for batch in batches]
    winner = min((info for _, info in runs), key=lambda info: info['val_rmse'])
    baseline_rmse, baseline_mae = evaluate(persistence(test[0]), test[1])
    for model, info in runs:
        info['selected_by_val'] = info is winner
        predictions = forecast(model, test[0])
        info['test_rmse'], info['test_mae'] = evaluate(predictions, test[1])
        if info is winner:
            trace = (test[1], predictions, persistence(test[0]))
    print(f'{device}: persistence RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}')
    return [info for _, info in runs], trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batches', type=int, nargs='+', default=[16, 64, 256])
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    rows, _ = experiment(args.epochs, args.batches, args.device)
    for row in rows:
        print(row)
    if args.csv:
        with args.csv.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
