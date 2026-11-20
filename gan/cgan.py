"""Optional bonus: Conditional GAN (noise + digit label) on MNIST."""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .data import get_loader
from .evaluate import classify, get_classifier
from .train import DEFAULTS, save_grid
from .utils import SEED, get_device, set_seed
from .viz import _smooth, tile_images

NUM_CLASSES = 10


class ConditionalGenerator(nn.Module):
    """G(z, y): the label embedding is concatenated with the noise vector."""

    def __init__(self, latent_dim=100, embed_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(NUM_CLASSES, embed_dim)

        def block(n_in, n_out):
            return [nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.ReLU(inplace=True)]

        self.net = nn.Sequential(
            *block(latent_dim + embed_dim, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.embed(labels)], dim=1)).view(-1, 1, 28, 28)


class ConditionalDiscriminator(nn.Module):
    """D(x, y): the label embedding is concatenated with the flattened image."""

    def __init__(self, embed_dim=10, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(28 * 28 + embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, labels):
        return self.net(torch.cat([image.flatten(1), self.embed(labels)], dim=1))


def train_cgan(out_dir, seed=SEED, epochs=DEFAULTS["epochs"], **overrides):
    cfg = {**DEFAULTS, "epochs": epochs, **overrides}
    device = get_device()
    set_seed(seed)
    loader = get_loader(cfg["batch_size"], generator=torch.Generator().manual_seed(seed))

    G = ConditionalGenerator(cfg["latent_dim"]).to(device)
    D = ConditionalDiscriminator().to(device)
    criterion = nn.BCELoss()
    betas = (cfg["beta1"], 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate"], betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate"], betas=betas)

    # Fixed noise: 10 columns x 10 digits, so each row is one digit.
    fixed_z = torch.randn(100, cfg["latent_dim"], generator=torch.Generator().manual_seed(seed)).to(device)
    fixed_y = torch.arange(10, device=device).repeat_interleave(10)

    hist = dict(config=cfg, g_loss=[], d_loss=[], epoch_g_loss=[], epoch_d_loss=[])
    for epoch in range(1, cfg["epochs"] + 1):
        e_g, e_d, start = [], [], time.time()
        for real, y in loader:
            real, y = real.to(device), y.to(device)
            n = real.size(0)
            ones, zeros = torch.ones(n, 1, device=device), torch.zeros(n, 1, device=device)

            opt_d.zero_grad()
            z = torch.randn(n, cfg["latent_dim"], device=device)
            fake_y = torch.randint(0, NUM_CLASSES, (n,), device=device)
            fake = G(z, fake_y)
            d_loss = criterion(D(real, y), ones) + criterion(D(fake.detach(), fake_y), zeros)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            g_loss = criterion(D(fake, fake_y), ones)
            g_loss.backward()
            opt_g.step()

            hist["g_loss"].append(g_loss.item())
            hist["d_loss"].append(d_loss.item())
            e_g.append(g_loss.item())
            e_d.append(d_loss.item())

        hist["epoch_g_loss"].append(sum(e_g) / len(e_g))
        hist["epoch_d_loss"].append(sum(e_d) / len(e_d))
        G.eval()
        with torch.no_grad():
            save_grid(G(fixed_z, fixed_y).cpu(), os.path.join(out_dir, "samples", f"epoch_{epoch:02d}.png"), nrow=10)
        G.train()
        print(f"[cgan] epoch {epoch:2d}/{cfg['epochs']} D_loss={hist['epoch_d_loss'][-1]:.3f} "
              f"G_loss={hist['epoch_g_loss'][-1]:.3f} ({time.time() - start:.1f}s)", flush=True)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
    return G, hist


@torch.no_grad()
def demo(G, out_dir, device, digit=7, n_eval=5000):
    """Ten examples of one digit, a digit-by-digit grid, and label-conditioning accuracy."""
    G.eval()
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(SEED + 3)
    latent = G.latent_dim

    z = torch.randn(10, latent, generator=gen).to(device)
    sevens = G(z, torch.full((10,), digit, device=device)).cpu()
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.imshow(tile_images(sevens, 10), cmap="gray")
    ax.set_title(f"Conditional GAN: ten different generated examples of digit {digit}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"digit_{digit}_examples.png"), dpi=150)
    plt.close(fig)
    save_grid(sevens, os.path.join(out_dir, "samples", f"digit_{digit}_examples.png"), nrow=10)

    z = torch.randn(100, latent, generator=gen).to(device)
    y = torch.arange(10, device=device).repeat_interleave(10)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(tile_images(G(z, y).cpu(), 10), cmap="gray")
    ax.set_title("Conditional GAN: each row is one requested digit (0-9)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "all_digits_grid.png"), dpi=150)
    plt.close(fig)

    # Does the generated image match the requested label? Check with the evaluator CNN.
    clf = get_classifier(device=device)
    z = torch.randn(n_eval, latent, generator=gen).to(device)
    y = torch.arange(n_eval, device=device) % NUM_CLASSES
    imgs = torch.cat([G(z[i:i + 1000], y[i:i + 1000]).cpu() for i in range(0, n_eval, 1000)])
    pred, _, _ = classify(clf, imgs, device)
    acc = float((pred == y.cpu().numpy()).mean())
    per_digit = [float((pred[y.cpu().numpy() == d] == d).mean()) for d in range(NUM_CLASSES)]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(dict(label_match_accuracy=acc, per_digit_label_match=per_digit, n_eval=n_eval), f, indent=2)
    print(f"label-match accuracy (classifier agrees with requested digit): {acc:.3f}")
    return acc


def plot_losses(out_dir, window=100):
    with open(os.path.join(out_dir, "history.json")) as f:
        h = json.load(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(_smooth(h["g_loss"], window), label="Generator")
    ax.plot(_smooth(h["d_loss"], window), label="Discriminator")
    ax.set(xlabel="Iteration", ylabel="BCE loss", title=f"Conditional GAN losses (moving avg, {window})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "loss_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/cgan")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G, _ = train_cgan(a.out, epochs=a.epochs)
    demo(G, a.out, get_device())
    plot_losses(a.out)
