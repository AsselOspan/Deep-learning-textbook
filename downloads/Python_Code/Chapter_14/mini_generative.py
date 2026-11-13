"""Небольшие AE, VAE и GAN для Fashion-MNIST и учебных данных главы 15."""
import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def decoder(latent, pixels, hidden):
    return nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, pixels))


class Autoencoder(nn.Module):
    def __init__(self, pixels, latent, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(pixels, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = decoder(latent, pixels, hidden)

    def forward(self, x):
        return self.decoder(self.encoder(x)).sigmoid()


class VAE(nn.Module):
    def __init__(self, pixels, latent, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(pixels, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = decoder(latent, pixels, hidden)

    def posterior(self, x):
        hidden = self.encoder(x)
        return self.mean(hidden), self.logvar(hidden)

    def forward(self, x):
        mu, logvar = self.posterior(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


class Generator(nn.Module):
    def __init__(self, pixels, latent, hidden):
        super().__init__()
        self.decoder = decoder(latent, pixels, hidden)

    def forward(self, noise):
        return self.decoder(noise).sigmoid()


def kl_per_image(mu, logvar):
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1)


def vae_terms(logits, x, mu, logvar):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
    return recon.mean(), kl_per_image(mu, logvar).mean()


def noise(batch, latent, x):
    return torch.randn(batch, latent, device=x.device, dtype=x.dtype)


def discriminator_step(generator, discriminator, optimizer, real, latent):
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = generator(noise(len(real), latent, real))
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)
    loss = (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    ) / 2
    loss.backward()
    optimizer.step()
    return loss.detach()


def generator_step(generator, discriminator, optimizer, real, latent):
    optimizer.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    try:
        fake_logits = discriminator(generator(noise(len(real), latent, real)))
        loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        loss.backward()
        optimizer.step()
    finally:
        discriminator.requires_grad_(True)
    return loss.detach()


def train_models(train, *, epochs, seed, latent=12, hidden=128,
                 batch_size=128, beta=1.0, device="cpu", include_ae=True):
    if len(train) < 1 or train.ndim != 2 or epochs < 1 or beta < 0:
        raise ValueError("Нужны непустая матрица изображений, положительное число эпох и beta ≥ 0")
    torch.manual_seed(seed)
    pixels = train.shape[1]
    ae = Autoencoder(pixels, latent, hidden).to(device) if include_ae else None
    vae = VAE(pixels, latent, hidden).to(device)
    gen = Generator(pixels, latent, hidden).to(device)
    disc = nn.Sequential(nn.Linear(pixels, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1)).to(device)
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=0.001) if ae is not None else None
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0005, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0005, betas=(0.5, 0.999))
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    for epoch in range(epochs):
        totals = dict(recon_bce=0.0, kl=0.0, discriminator=0.0, generator=0.0)
        if ae is not None:
            totals["ae_mse"] = 0.0
        for x in loader:
            x = x.to(device)
            if ae is not None:
                ae_optimizer.zero_grad(set_to_none=True)
                mse = F.mse_loss(ae(x), x)
                mse.backward()
                ae_optimizer.step()
                totals["ae_mse"] += mse.detach().item() * len(x)
            vae_optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = vae(x)
            recon, kl = vae_terms(logits, x, mu, logvar)
            (recon + beta * kl).backward()
            vae_optimizer.step()
            d = discriminator_step(gen, disc, disc_optimizer, x, latent)
            g = generator_step(gen, disc, gen_optimizer, x, latent)
            for name, value in zip(("recon_bce", "kl", "discriminator", "generator"),
                                   (recon.detach(), kl.detach(), d, g)):
                totals[name] += value.item() * len(x)
        history.append({name: total / len(train) for name, total in totals.items()})
    return ae, vae, gen, history


@torch.no_grad()
def evaluate(ae, vae, test, device="cpu"):
    x = test.to(device)
    mu, logvar = vae.posterior(x)
    logits = vae.decoder(mu)
    bce, kl = vae_terms(logits, x, mu, logvar)
    result = dict(vae_mu_mse=F.mse_loss(logits.sigmoid(), x).item(),
                  vae_mu_bce=bce.item(), vae_kl=kl.item())
    if ae is not None:
        result["ae_mse"] = F.mse_loss(ae(x), x).item()
    return result


@torch.no_grad()
def prior_samples(vae, generator, *, n, latent, seed, device="cpu"):
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(n, latent, generator=rng).to(device)
    return vae.decoder(z).sigmoid().cpu(), generator(z).cpu()


def fashion_data(folder, n_train=6000, n_test=512, seed=2026, download=False):
    chapter2 = Path(__file__).resolve().parents[1] / "Chapter_02"
    if str(chapter2) not in sys.path:
        sys.path.insert(0, str(chapter2))
    from fashion_mnist_mlp import download_data, read_split

    if not 1 <= n_train <= 60000 or not 1 <= n_test <= 10000:
        raise ValueError("Размер подвыборки выходит за пределы Fashion-MNIST")
    if download:
        download_data(folder)
    train = read_split(folder, "train", 60000).tensors[0]
    test = read_split(folder, "t10k", 10000).tensors[0]
    rng = torch.Generator().manual_seed(seed)
    return train[torch.randperm(len(train), generator=rng)[:n_train]], (
        test[torch.randperm(len(test), generator=rng)[:n_test]])


def report(history, metrics):
    for epoch, row in enumerate(history, 1):
        print(f"epoch={epoch} " + " ".join(f"{k}={v:.6f}" for k, v in row.items()))
    print("held_out " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=Path, help="Сохранить только сгенерированные тензоры")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    torch.set_num_threads(2)
    train, test = fashion_data(args.data_dir, args.n_train, args.n_test, args.seed, args.download)
    ae, vae, gen, history = train_models(train, epochs=args.epochs, seed=args.seed, device=args.device)
    print(f"torch={torch.__version__} device={args.device} seed={args.seed} train={len(train)} test={len(test)}")
    report(history, evaluate(ae, vae, test, args.device))
    vae_samples, gan_samples = prior_samples(vae, gen, n=8, latent=12, seed=55, device=args.device)
    assert vae_samples.shape == gan_samples.shape == (8, 784)
    assert torch.isfinite(vae_samples).all() and torch.isfinite(gan_samples).all()
    if args.samples:
        torch.save({"vae": vae_samples, "gan": gan_samples}, args.samples)


if __name__ == "__main__":
    main()
