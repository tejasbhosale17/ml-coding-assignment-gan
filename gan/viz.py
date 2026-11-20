"""Figures: epoch progression, final sample grid, and loss curves."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import Generator


def load_history(run_dir):
    with open(os.path.join(run_dir, "history.json")) as f:
        return json.load(f)


def load_generator(run_dir, device="cpu"):
    hist = load_history(run_dir)
    G = Generator(hist["config"]["latent_dim"])
    G.load_state_dict(torch.load(os.path.join(run_dir, "generator.pt"), map_location=device))
    return G.to(device).eval()


def _smooth(x, window):
    x = np.asarray(x)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def tile_images(images, nrow):
    """images: (N,1,28,28) in [-1,1] -> single 2D array in [0,1]."""
    imgs = (images.squeeze(1).cpu().numpy() + 1) / 2
    n = (len(imgs) // nrow) * nrow
    rows = [np.concatenate(imgs[i:i + nrow], axis=1) for i in range(0, n, nrow)]
    return np.concatenate(rows, axis=0)


def epoch_progress_figure(run_dir, out_path, epochs=None, title="Generated samples (fixed noise)"):
    """Part 4: samples from epochs 1, 5, 10, 15 and the final epoch."""
    n_epochs = load_history(run_dir)["config"]["epochs"]
    epochs = epochs or sorted({e for e in (1, 5, 10, 15, n_epochs) if e <= n_epochs})
    fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 3.6))
    for ax, e in zip(np.atleast_1d(axes), epochs):
        ax.imshow(plt.imread(os.path.join(run_dir, "samples", f"epoch_{e:02d}.png")), cmap="gray")
        ax.set_title(f"Epoch {e}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sample_grid(images, out_path, nrow=10, title=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(tile_images(images, nrow), cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def loss_curves(hist, out_path, title="Training losses", window=100):
    """Part 5B: per-iteration (smoothed) and per-epoch losses, plus D outputs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(hist["g_loss"], color="tab:blue", alpha=0.15)
    ax.plot(hist["d_loss"], color="tab:orange", alpha=0.15)
    ax.plot(np.arange(window - 1, len(hist["g_loss"])), _smooth(hist["g_loss"], window),
            color="tab:blue", label="Generator")
    ax.plot(np.arange(window - 1, len(hist["d_loss"])), _smooth(hist["d_loss"], window),
            color="tab:orange", label="Discriminator")
    ax.set(xlabel="Iteration", ylabel="BCE loss", title=f"Per-iteration (moving avg, {window})")
    ax.legend()

    ax = axes[1]
    epochs = np.arange(1, len(hist["epoch_g_loss"]) + 1)
    ax.plot(epochs, hist["epoch_g_loss"], "o-", label="Generator")
    ax.plot(epochs, hist["epoch_d_loss"], "o-", label="Discriminator")
    ax.set(xlabel="Epoch", ylabel="Mean BCE loss", title="Per-epoch mean")
    ax.legend()

    ax = axes[2]
    ax.plot(_smooth(hist["d_real"], window), label="D(real)")
    ax.plot(_smooth(hist["d_fake"], window), label="D(fake)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Iteration", ylabel="Mean D output", title="Discriminator outputs", ylim=(0, 1))
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
