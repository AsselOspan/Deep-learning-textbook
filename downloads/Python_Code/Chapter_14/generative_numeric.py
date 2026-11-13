"""Проверка KL, репараметризации, GAN и triplet в малых числовых задачах."""
import math
import torch
from torch.nn import functional as F
from mini_generative import kl_per_image


def check_examples():
    dt = torch.float64
    mu = torch.tensor([[1.0, -0.5]], dtype=dt, requires_grad=True)
    logvar = torch.tensor([[math.log(0.25), math.log(4.0)]], dtype=dt, requires_grad=True)
    eps = torch.tensor([[2.0, -0.25]], dtype=dt)
    z = mu + (0.5 * logvar).exp() * eps
    kl = kl_per_image(mu, logvar)
    dz_dmu, dz_dlogvar = torch.autograd.grad(z.sum(), (mu, logvar))
    torch.testing.assert_close(z, torch.tensor([[2.0, -1.0]], dtype=dt))
    torch.testing.assert_close(kl, torch.tensor([1.75], dtype=dt))
    torch.testing.assert_close(dz_dmu, torch.ones_like(mu))
    torch.testing.assert_close(dz_dlogvar, torch.tensor([[0.5, -0.25]], dtype=dt))

    a, p, n = [torch.tensor(v, dtype=dt, requires_grad=True) for v in
               ([1.0, 0.0], [0.8, 0.2], [0.6, 0.1])]
    dpos = (a - p).square().sum()
    dneg = (a - n).square().sum()
    triplet = (dpos - dneg + 0.25).clamp_min(0)
    grads = torch.autograd.grad(triplet, (a, p, n))
    torch.testing.assert_close(triplet, torch.tensor(0.16, dtype=dt))
    for grad, expected in zip(grads, ([-0.4, -0.2], [-0.4, 0.4], [0.8, -0.2])):
        torch.testing.assert_close(grad, torch.tensor(expected, dtype=dt))

    real, fake = torch.logit(torch.tensor([0.8, 0.25], dtype=dt)).unbind()
    fake = fake.detach().requires_grad_()
    dloss = (F.softplus(-real) + F.softplus(fake)) / 2
    gloss = F.softplus(-fake)
    grad = torch.autograd.grad(gloss, fake)[0]
    torch.testing.assert_close(grad, torch.tensor(-0.75, dtype=dt))
    print(f"z={z.detach().tolist()} KL={kl.item():.6f} triplet={triplet.item():.6f}")
    print(f"D={dloss.item():.6f} G={gloss.item():.6f} dG/dlogit={grad.item():.6f}")


if __name__ == "__main__":
    check_examples()
