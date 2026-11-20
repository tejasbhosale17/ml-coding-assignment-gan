"""Adversarial training loop for the baseline (and label-smoothing) GAN."""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torchvision.utils import save_image

from .data import get_loader
from .models import Discriminator, Generator
from .utils import SEED, get_device, set_seed

DEFAULTS = dict(
    learning_rate=0.0002,
    beta1=0.5,
    batch_size=128,
    epochs=20,
    latent_dim=100,
    real_label=1.0,  # < 1.0 enables one-sided label smoothing (Part 6)
)


def save_grid(images, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(images, path, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)


def train_gan(out_dir, seed=SEED, num_fixed=64, verbose=True, **overrides):
    """Train G and D; write per-epoch sample grids to ``out_dir/samples``.

    Returns (generator, discriminator, history). History holds per-iteration
    losses and D outputs plus per-epoch means, and is also saved as JSON.
    """
    cfg = {**DEFAULTS, **overrides}
    device = get_device()
    set_seed(seed)

    loader_gen = torch.Generator().manual_seed(seed)
    loader = get_loader(cfg["batch_size"], generator=loader_gen)

    G = Generator(cfg["latent_dim"]).to(device)
    D = Discriminator().to(device)
    criterion = nn.BCELoss()
    betas = (cfg["beta1"], 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate"], betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate"], betas=betas)

    # Fixed noise reused every epoch so samples are comparable across epochs.
    fixed_z = torch.randn(num_fixed, cfg["latent_dim"], generator=torch.Generator().manual_seed(seed)).to(device)

    hist = dict(config=cfg, g_loss=[], d_loss=[], d_real=[], d_fake=[],
                epoch_g_loss=[], epoch_d_loss=[], epoch_time=[])
    sample_dir = os.path.join(out_dir, "samples")

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        e_g, e_d = [], []
        for real, _ in loader:
            real = real.to(device)
            n = real.size(0)
            real_t = torch.full((n, 1), cfg["real_label"], device=device)
            fake_t = torch.zeros(n, 1, device=device)

            # Steps 1-3: train D on real images, then on fake images.
            opt_d.zero_grad()
            out_real = D(real)
            loss_real = criterion(out_real, real_t)
            z = torch.randn(n, cfg["latent_dim"], device=device)
            fake = G(z)
            out_fake = D(fake.detach())
            loss_fake = criterion(out_fake, fake_t)
            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_d.step()

            # Step 4: train G so D labels generated images as real (target 1).
            opt_g.zero_grad()
            g_loss = criterion(D(fake), torch.ones(n, 1, device=device))
            g_loss.backward()
            opt_g.step()

            # Step 5: record losses.
            hist["g_loss"].append(g_loss.item())
            hist["d_loss"].append(d_loss.item())
            hist["d_real"].append(out_real.mean().item())
            hist["d_fake"].append(out_fake.mean().item())
            e_g.append(g_loss.item())
            e_d.append(d_loss.item())

        hist["epoch_g_loss"].append(sum(e_g) / len(e_g))
        hist["epoch_d_loss"].append(sum(e_d) / len(e_d))
        hist["epoch_time"].append(time.time() - start)

        G.eval()
        with torch.no_grad():
            save_grid(G(fixed_z).cpu(), os.path.join(sample_dir, f"epoch_{epoch:02d}.png"))
        G.train()

        if verbose:
            print(f"[{os.path.basename(out_dir)}] epoch {epoch:2d}/{cfg['epochs']} "
                  f"D_loss={hist['epoch_d_loss'][-1]:.3f} G_loss={hist['epoch_g_loss'][-1]:.3f} "
                  f"({hist['epoch_time'][-1]:.1f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "discriminator.pt"))
    torch.save(fixed_z.cpu(), os.path.join(out_dir, "fixed_z.pt"))
    return G, D, hist


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the baseline GAN on MNIST")
    p.add_argument("--out", default="results/baseline")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--real-label", type=float, default=DEFAULTS["real_label"])
    a = p.parse_args()
    train_gan(a.out, epochs=a.epochs, real_label=a.real_label)
