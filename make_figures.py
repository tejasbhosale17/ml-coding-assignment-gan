"""Generate Part 4 / Part 5A / Part 5B figures for a trained run."""
import argparse
import os

import torch

from gan.utils import SEED, get_device
from gan.viz import epoch_progress_figure, load_generator, load_history, loss_curves, sample_grid


def make_figures(run_dir, name):
    device = get_device()
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    hist = load_history(run_dir)

    epoch_progress_figure(run_dir, os.path.join(fig_dir, "epoch_progress.png"),
                          title=f"{name}: samples from a fixed noise set")

    G = load_generator(run_dir, device)
    z = torch.randn(100, hist["config"]["latent_dim"], generator=torch.Generator().manual_seed(SEED + 1))
    with torch.no_grad():
        imgs = G(z.to(device)).cpu()
    sample_grid(imgs, os.path.join(fig_dir, "final_grid.png"), nrow=10,
                title=f"{name}: 100 final generated images")

    loss_curves(hist, os.path.join(fig_dir, "loss_curves.png"), title=f"{name}: training losses")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/baseline")
    p.add_argument("--name", default="Baseline")
    a = p.parse_args()
    make_figures(a.run, a.name)
