"""Двоичная классификация зашумлённых растров 2×2."""
import torch
from torch import nn


def make_patterns(per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.tensor([[1., 0., 1., 0.], [1., 1., 0., 0.]])
    labels = torch.arange(2).repeat_interleave(per_class)
    noise = 0.25 * torch.randn(len(labels), 4, generator=generator)
    return (prototypes[labels] + noise).clamp(0, 1), labels


def reference_mlp():
    x = torch.tensor([2., 1.])
    weight = torch.tensor([[0.5, -1.], [-0.5, 1.]])
    bias = torch.tensor([0.5, -0.5])
    hidden = torch.relu(weight @ x + bias)
    return (torch.tensor([2., -1.]) @ hidden - 0.5).item()


def run():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train, y_train = (t.to(device) for t in make_patterns(96, 10))
    x_val, y_val = (t.to(device) for t in make_patterns(48, 20))
    x_test, y_test = (t.to(device) for t in make_patterns(48, 30))
    loss_fn = nn.CrossEntropyLoss()
    models = {
        'linear': nn.Linear(4, 2),
        'mlp': nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
    }
    scores = {}
    for name, model in models.items():
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
        model.train()
        for _ in range(60):
            optimizer.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x_val)
            accuracy = (logits.argmax(1) == y_val).float().mean().item()
            scores[name] = loss_fn(logits, y_val).item()
        print(f'{name}: val loss={scores[name]:.4f}, accuracy={accuracy:.3f}')

    winner = min(scores, key=scores.get)
    with torch.inference_mode():
        prediction = models[winner](x_test).argmax(1)
        matrix = torch.bincount(2 * y_test + prediction, minlength=4)
        accuracy = (prediction == y_test).float().mean().item()
    print(f'{winner}: test accuracy={accuracy:.3f}')
    print('Confusion matrix, rows=true, cols=predicted:')
    print(matrix.reshape(2, 2).cpu())


if __name__ == '__main__':
    print(f'Ручной пример MLP: {reference_mlp():.2f}')
    run()
