"""Small independent numerical checks for academic supplements 1 through 5."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

D = torch.float64


def chapter_01():
    x = torch.tensor([[1., 0.], [1., 1.], [1., 2.]], dtype=D)
    y = torch.tensor([1., 2., 2.], dtype=D)
    beta = torch.linalg.solve(x.T @ x, x.T @ y)
    ridge = torch.linalg.solve(x.T @ x + torch.diag(torch.tensor([0., 1.], dtype=D)), x.T @ y)
    torch.testing.assert_close(beta, torch.tensor([7/6, .5], dtype=D))
    torch.testing.assert_close(x.T @ (x @ beta-y), torch.zeros(2, dtype=D), atol=1e-12, rtol=0)
    torch.testing.assert_close(ridge, torch.tensor([4/3, 1/3], dtype=D))
    rank = int(torch.linalg.matrix_rank(torch.column_stack((x, 2*x[:, 1]))))
    assert rank == 2
    risks = []
    for predictions in ([1.5, 2., 2.5], [1.8, 1.9, 2.]):
        f = torch.tensor(predictions, dtype=D)
        bias2 = float((f.mean()-2).square())
        variance = float(f.var(unbiased=False))
        exact = sum((p-target)**2 for p in predictions for target in (1.5, 2.5))/6
        assert abs(exact-(bias2+variance+.25)) < 1e-12
        risks.append(dict(bias2=bias2, variance=variance, noise=.25, risk=exact))
    return dict(beta=beta.tolist(), ridge=ridge.tolist(), rank=rank,
                mse=float((x @ beta-y).square().mean()), ridge_mse=float((x @ ridge-y).square().mean()), risks=risks)


def chapter_02():
    x = torch.tensor([1., -2.], dtype=D, requires_grad=True)
    w = torch.tensor([[.5, -1.], [1., .25]], dtype=D, requires_grad=True)
    b = torch.tensor([.1, -.2], dtype=D, requires_grad=True)
    target = torch.tensor([.2, -.1], dtype=D)
    z = w @ x+b
    h = z.tanh()
    loss = .5*(h-target).square().sum()
    gw, gb, gx = torch.autograd.grad(loss, (w, b, x))
    dz = (h-target)*(1-h.square())
    jac = torch.diag(1-h.square()) @ w
    torch.testing.assert_close(gw, torch.outer(dz, x))
    torch.testing.assert_close(gb, dz)
    torch.testing.assert_close(gx, (h-target) @ jac)
    theta = torch.cat((w.detach().flatten(), b.detach())).requires_grad_()

    def objective(p):
        prediction = (p[:4].reshape(2, 2) @ x.detach()+p[4:]).tanh()
        return .5*(prediction-target).square().sum()

    grad = torch.autograd.grad(objective(theta), theta)[0]
    estimates = []
    for j in range(6):
        delta = torch.zeros_like(theta)
        delta[j] = 1e-6
        estimates.append(float(((objective(theta+delta)-objective(theta-delta))/(2e-6)).detach()))
    torch.testing.assert_close(grad, torch.tensor(estimates, dtype=D), rtol=1e-7, atol=1e-8)
    inputs = torch.tensor([[1., -2.], [.5, 1.]], dtype=D)
    targets = torch.tensor([[.2, -.1], [.3, .7]], dtype=D)
    model = lambda p, batch: (batch @ p[:4].reshape(2, 2).T+p[4:]).tanh()
    (.5*(model(theta, inputs)-targets).square().sum(1).mean()).backward()
    whole = theta.grad.clone()
    theta.grad = None
    for xx, yy in zip(inputs, targets):
        (.5*(model(theta, xx)-yy).square().sum()/2).backward()
    torch.testing.assert_close(theta.grad, whole, atol=1e-12, rtol=1e-12)
    sgd = nn.Parameter(torch.tensor(1., dtype=D))
    adam = nn.Parameter(torch.tensor(1., dtype=D))
    sgd_opt = torch.optim.SGD([sgd], lr=.1, momentum=.9)
    adam_opt = torch.optim.Adam([adam], lr=.01)
    m = v = 0.
    manual = 1.
    for step, g in enumerate((.4, .2), 1):
        sgd.grad = torch.tensor(g, dtype=D)
        adam.grad = torch.tensor(g, dtype=D)
        sgd_opt.step(); adam_opt.step()
        m = .9*m+.1*g
        v = .999*v+.001*g*g
        manual -= .01*(m/(1-.9**step))/(math.sqrt(v/(1-.999**step))+1e-8)
    assert abs(float(sgd.detach())-.904) < 1e-12
    assert abs(float(adam.detach())-manual) < 1e-12
    return dict(z=z.detach().tolist(), h=h.detach().tolist(), dz=dz.detach().tolist(),
                jacobian=jac.detach().tolist(), grad_x=gx.tolist(), grad_theta=grad.tolist(),
                max_fd_error=float((grad-torch.tensor(estimates, dtype=D)).abs().max()),
                microbatch_error=float((theta.grad-whole).abs().max()), sgd=float(sgd.detach()), adam=float(adam.detach()))


def chapter_03():
    image = torch.arange(1., 26., dtype=D).reshape(1, 1, 5, 5)
    kernel = torch.tensor([[1., 0., -1.]]*3, dtype=D).reshape(1, 1, 3, 3)
    result = F.conv2d(image, kernel, stride=2, padding=1)
    padded = F.pad(image, (1, 1, 1, 1))
    manual = torch.empty(3, 3, dtype=D)
    for r in range(3):
        for c in range(3):
            manual[r, c] = (padded[0, 0, 2*r:2*r+3, 2*c:2*c+3]*kernel[0, 0]).sum()
    torch.testing.assert_close(result[0, 0], manual)
    dilated = float(F.conv2d(image, kernel, dilation=2).item())
    assert dilated == -12
    receptive, jump = 1, 1
    for size, stride, dilation in ((3, 1, 1), (2, 2, 1), (3, 1, 2)):
        receptive += (size-1)*dilation*jump
        jump *= stride
    assert (receptive, jump) == (12, 2)
    a = torch.tensor([[1., 2., 0.], [0., 1., 2.]], dtype=D)
    x, u = torch.tensor([1., 2., 3.], dtype=D), torch.tensor([1., -1.], dtype=D)
    adjoint = F.conv_transpose1d(u.reshape(1, 1, 2), torch.tensor([[[1., 2.]]], dtype=D)).flatten()
    torch.testing.assert_close(adjoint, a.T @ u)
    assert float((a @ x) @ u) == float(x @ adjoint) == -3
    torch.testing.assert_close(a @ torch.tensor([4., -2., 1.], dtype=D), torch.zeros(2, dtype=D))
    overlap = F.conv_transpose1d(torch.ones(1, 1, 4, dtype=D), torch.ones(1, 1, 3, dtype=D), stride=2)
    return dict(strided=result[0, 0].tolist(), dilated=dilated, receptive=receptive, jump=jump,
                Ax=(a @ x).tolist(), adjoint=adjoint.tolist(), dot_product=-3., overlap=overlap.flatten().tolist())


def chapter_04():
    x = torch.tensor([-1., 0., 1.], dtype=D, requires_grad=True)
    gelu = F.gelu(x, approximate='none')
    derivative = torch.autograd.grad(gelu.sum(), x)[0]
    phi = torch.exp(-x.square()/2)/math.sqrt(2*math.pi)
    cdf = .5*(1+torch.erf(x/math.sqrt(2)))
    torch.testing.assert_close(derivative, cdf+x*phi)
    endpoints = torch.tensor([-1000., 0., 1000.], dtype=D)
    stable = torch.clamp_min(endpoints, 0)+torch.log1p(torch.exp(-endpoints.abs()))
    torch.testing.assert_close(stable, F.softplus(endpoints))
    torch.manual_seed(91)
    z = math.sqrt(2)*torch.randn(100_000, dtype=D)
    moment = float(z.relu().square().mean())
    assert abs(moment-1) < .03
    return dict(he_std=math.sqrt(2/64), xavier_std=math.sqrt(1/64),
                he_relu_second_moment=1., he_relu_variance=1-1/math.pi,
                empirical_second_moment=moment, gelu=gelu.detach().tolist(), gelu_derivative=derivative.tolist(),
                softplus=stable.tolist(), sigmoid_chain=.25**10,
                singular_factors=[1.2**5, .8**5], softplus_error_beta5=math.log(2)/5)


def chapter_05():
    source = torch.tensor([.8, .2], dtype=D)
    target = torch.tensor([.3, .7], dtype=D)
    losses = torch.tensor([.1, .9], dtype=D)
    weights = target/source
    assert abs(float((source*weights*losses).sum())-.66) < 1e-12
    sample_weights = torch.tensor([float(weights[0])]*8+[float(weights[1])]*2, dtype=D)
    ess = float(sample_weights.sum().square()/sample_weights.square().sum())
    probabilities = torch.tensor([.8, .6], dtype=D)
    logits = torch.logit(probabilities)+math.log(.2/.8)
    adjusted = logits.sigmoid()
    torch.testing.assert_close(adjusted, torch.tensor([.5, 3/11], dtype=D))
    backbone = nn.Parameter(torch.tensor([3., 4.], dtype=D))
    head = nn.Parameter(torch.tensor([.3, .4], dtype=D))
    before = [backbone.detach().clone(), head.detach().clone()]
    opt = torch.optim.SGD([{'params':[backbone], 'lr':.01}, {'params':[head], 'lr':.1}])
    backbone.grad = torch.tensor([.6, .8], dtype=D)
    head.grad = torch.tensor([.6, .8], dtype=D)
    opt.step()
    relative = [float((p.detach()-old).norm()/old.norm()) for p, old in zip((backbone, head), before)]
    torch.testing.assert_close(torch.tensor(relative, dtype=D), torch.tensor([.002, .2], dtype=D))
    bn = nn.BatchNorm1d(1, momentum=.1, dtype=D).requires_grad_(False).train()
    running = []
    for k in range(1, 11):
        bn(torch.tensor([[9.], [11.]], dtype=D))
        value = float(bn.running_mean)
        assert abs(value-10*(1-.9**k)) < 1e-12
        running.append(value)
    orders = []
    for means in ((0., 10.), (10., 0.)):
        layer = nn.BatchNorm1d(1, momentum=.1, dtype=D).train()
        for mean in means:layer(torch.tensor([[mean-1], [mean+1]], dtype=D))
        orders.append(float(layer.running_mean))
    return dict(source_risk=float((source*losses).sum()), target_risk=float((target*losses).sum()),
                weights=weights.tolist(), effective_sample_size=ess, adjusted_posterior=adjusted.tolist(),
                relative_updates=relative, bn_mean=running, bn_orders=orders)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapter', choices=['1', '2', '3', '4', '5', 'all'], default='all')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    requested = range(1, 6) if args.chapter == 'all' else [int(args.chapter)]
    result = {str(n): globals()[f'chapter_{n:02}']() for n in requested}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:args.json.write_text(text+'\n', encoding='utf-8')
    print('PASS: all requested numerical checks')


if __name__ == '__main__':
    main()
