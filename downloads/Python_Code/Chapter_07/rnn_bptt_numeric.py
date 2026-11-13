"""Три шага RNN, ручной BPTT и независимые проверки PyTorch."""

import math

import torch
from torch import nn

INPUTS = (1.0, -1.0, 0.5)
TARGETS = (0.3, -0.2, 0.4)
INITIAL = (0.8, 0.5, 0.1, 1.2, -0.2)  # w, u, b, v, c
DTYPE = torch.float64


def tensor(values):
    return torch.tensor(values, dtype=DTYPE)


def check(actual, expected, atol=1e-12):
    torch.testing.assert_close(tensor(actual), tensor(expected), atol=atol, rtol=0)


def forward_manual(params):
    w, u, b, v, c = params
    states = [0.0]
    errors = []
    for x, target in zip(INPUTS, TARGETS):
        states.append(math.tanh(w * x + u * states[-1] + b))
        errors.append(v * states[-1] + c - target)
    return states, errors


def loss_manual(params):
    _, errors = forward_manual(params)
    return 0.5 * sum(error**2 for error in errors)


def backward_manual(params, states, errors, cut=False):
    _, u, _, v, _ = params
    delta = [0.0] * len(INPUTS)
    future = 0.0
    for t in reversed(range(len(INPUTS))):
        if cut and t == 1:  # связь h2 → h3 отсоединена
            future = 0.0
        delta[t] = (v * errors[t] + u * future) * (1 - states[t + 1] ** 2)
        future = delta[t]
    return delta, [
        sum(d * x for d, x in zip(delta, INPUTS)),
        sum(d * h for d, h in zip(delta, states[:-1])),
        sum(delta),
        sum(error * h for error, h in zip(errors, states[1:])),
        sum(errors),
    ]


def autograd_check(params, cut=False):
    w, u, b, v, c = params.unbind()
    h = tensor(0.0)
    errors, preactivations = [], []
    for t, (x, target) in enumerate(zip(INPUTS, TARGETS)):
        if cut and t == 2:
            h = h.detach()
        a = w * x + u * h + b
        a.retain_grad()
        preactivations.append(a)
        h = a.tanh()
        errors.append(v * h + c - target)
    loss = torch.stack(errors).square().sum() / 2
    loss.backward()
    return loss.item(), [a.grad.item() for a in preactivations]


def finite_differences(params, eps=1e-6):
    values = []
    for i in range(len(params)):
        plus, minus = list(params), list(params)
        plus[i] += eps
        minus[i] -= eps
        values.append((loss_manual(plus) - loss_manual(minus)) / (2 * eps))
    return values


@torch.no_grad()
def library_check(states):
    w, u, b, _, _ = INITIAL
    rnn = nn.RNN(1, 1, batch_first=True, dtype=DTYPE)
    rnn.weight_ih_l0.fill_(w)
    rnn.weight_hh_l0.fill_(u)
    rnn.bias_ih_l0.fill_(b)
    rnn.bias_hh_l0.zero_()
    outputs, final = rnn(tensor(INPUTS).reshape(1, 3, 1))
    torch.testing.assert_close(outputs.flatten(), tensor(states[1:]))
    torch.testing.assert_close(final.flatten(), tensor([states[-1]]))


@torch.no_grad()
def gated_examples():
    # LSTM: порядок блоков смещений i, f, g, o.
    lstm = nn.LSTMCell(1, 1, dtype=DTYPE)
    for p in lstm.parameters():
        p.zero_()
    lstm.bias_ih.copy_(tensor([
        math.log(0.25 / 0.75), math.log(0.8 / 0.2),
        math.atanh(-0.4), math.log(0.6 / 0.4),
    ]))
    h, c = lstm(tensor([1.0]), (tensor([0.0]), tensor([0.7])))
    torch.testing.assert_close(c, tensor([0.46]))
    torch.testing.assert_close(h, tensor([0.6 * math.tanh(0.46)]))

    # GRU: порядок блоков r, z, n. r умножает скрытую аффинную ветвь n.
    gru = nn.GRUCell(1, 1, dtype=DTYPE)
    for p in gru.parameters():
        p.zero_()
    gru.weight_hh[2, 0] = 0.8
    gru.bias_hh[2] = 0.1
    gru.bias_ih.copy_(tensor([
        math.log(0.25 / 0.75), math.log(0.7 / 0.3),
        math.atanh(-0.2) - 0.25 * (0.8 * 0.6 + 0.1),
    ]))
    updated = gru(tensor([1.0]), tensor([0.6]))
    torch.testing.assert_close(updated, tensor([0.36]))
    print(f"lstm_c={c.item():.6f} lstm_h={h.item():.6f} gru_h={updated.item():.6f}")


def main():
    states, errors = forward_manual(INITIAL)
    manual_delta, gradient = backward_manual(INITIAL, states, errors)
    params = tensor(INITIAL).requires_grad_()
    loss, autodiff_delta = autograd_check(params)
    check(manual_delta, autodiff_delta)
    check(gradient, params.grad.tolist())
    check(gradient, finite_differences(INITIAL), atol=1e-8)
    check(loss, 0.16509137042712307)
    check(gradient, [0.4927578241703158, -0.30048484000298153,
                     -0.6693536676524958, 0.319005696245494, -0.2473216698643103])
    updated = [p - 0.1 * g for p, g in zip(INITIAL, gradient)]
    after = loss_manual(updated)
    check(after, 0.09447774288257743)
    cut_params = tensor(INITIAL).requires_grad_()
    cut_loss, cut_delta = autograd_check(cut_params, cut=True)
    expected_delta, expected_gradient = backward_manual(INITIAL, states, errors, cut=True)
    check(cut_loss, loss)
    check(cut_delta, expected_delta)
    check(cut_params.grad.tolist(), expected_gradient)
    check(expected_gradient[1], -0.22776443843425997)
    library_check(states)
    gated_examples()
    print(f"states={[round(h, 9) for h in states[1:]]}")
    print(f"deltas={[round(d, 9) for d in manual_delta]}")
    print(f"gradients={[round(g, 9) for g in gradient]}")
    print(f"loss={loss:.12f}->{after:.12f} truncated_du={expected_gradient[1]:.12f}")
    print("manual_autograd_finite_difference_checks=passed")


if __name__ == "__main__":
    main()

