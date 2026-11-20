"""Quantitative evaluation: classifier-based class distribution, Frechet distance,
and nearest-neighbour diversity / memorisation checks."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import get_loader, get_mnist
from .utils import SEED, get_device, set_seed
from .viz import load_generator, load_history, tile_images


class MNISTClassifier(nn.Module):
    """Small CNN used only as an evaluator (not part of the GAN)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(x).flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.features(x))


def get_classifier(path="results/mnist_classifier.pt", device="cpu", epochs=3):
    clf = MNISTClassifier().to(device)
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    set_seed(SEED)
    loader = get_loader(128, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        clf.train()
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(clf(x.to(device)), y.to(device)).backward()
            opt.step()
    clf.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf


@torch.no_grad()
def classify(clf, images, device, bs=1000):
    preds, confs, feats = [], [], []
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(device)
        f = clf.features(x)
        p = F.softmax(clf.fc2(f), dim=1)
        conf, pred = p.max(1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
        feats.append(f.cpu())
    return torch.cat(preds).numpy(), torch.cat(confs).numpy(), torch.cat(feats).numpy()


def frechet_distance(f1, f2):
    """Frechet distance between Gaussians fitted to two feature sets (FID-style)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    eig = np.linalg.eigvals(s1 @ s2)
    tr_sqrt = np.sqrt(np.clip(eig.real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)


def nn_dist(a, b, same=False):
    """Mean distance from each row of a to its nearest row in b (flattened, [0,1] pixels)."""
    a = a.flatten(1)
    b = b.flatten(1)
    d = torch.cdist(a, b)
    if same:
        d.fill_diagonal_(float("inf"))
    return d.min(1).values


def evaluate_run(run_dir, name, n_gen=10000, device=None):
    device = device or get_device()
    G = load_generator(run_dir, device)
    latent = load_history(run_dir)["config"]["latent_dim"]
    clf = get_classifier(device=device)
    with torch.no_grad():
        return _evaluate(run_dir, name, n_gen, device, G, latent, clf)


def _evaluate(run_dir, name, n_gen, device, G, latent, clf):
    z = torch.randn(n_gen, latent, generator=torch.Generator().manual_seed(SEED + 2))
    gen = torch.cat([G(z[i:i + 1000].to(device)).cpu() for i in range(0, n_gen, 1000)])

    train, test = get_mnist(train=True), get_mnist(train=False)
    real_test = torch.stack([test[i][0] for i in range(n_gen)])
    real_test_y = np.array([test[i][1] for i in range(n_gen)])
    real_train = torch.stack([train[i][0] for i in range(20000)])

    gp, gc, gf = classify(clf, gen, device)
    rp, rc, rf = classify(clf, real_test, device)
    acc = float((rp == real_test_y).mean())

    g_dist = np.bincount(gp, minlength=10) / len(gp)
    r_dist = np.bincount(real_test_y, minlength=10) / len(real_test_y)

    # Pixel-space diversity checks on 1,000-image subsets scaled to [0, 1].
    g1k = (gen[:1000] + 1) / 2
    r1k = (real_test[:1000] + 1) / 2
    rtrain = (real_train + 1) / 2
    g_nn_self = nn_dist(g1k, g1k, same=True)
    r_nn_self = nn_dist(r1k, r1k, same=True)
    g_nn_train = nn_dist(g1k, rtrain)
    r_nn_train = nn_dist(r1k, rtrain)

    metrics = dict(
        name=name,
        n_generated=int(n_gen),
        classifier_test_accuracy=acc,
        frechet_distance=frechet_distance(gf, rf),
        gen_class_dist=g_dist.tolist(),
        real_class_dist=r_dist.tolist(),
        total_variation_distance=float(0.5 * np.abs(g_dist - r_dist).sum()),
        gen_class_entropy_bits=float(-(g_dist[g_dist > 0] * np.log2(g_dist[g_dist > 0])).sum()),
        real_class_entropy_bits=float(-(r_dist * np.log2(r_dist)).sum()),
        gen_mean_confidence=float(gc.mean()),
        real_mean_confidence=float(rc.mean()),
        gen_low_confidence_frac=float((gc < 0.9).mean()),
        real_low_confidence_frac=float((rc < 0.9).mean()),
        gen_nn_self_dist=float(g_nn_self.mean()),
        real_nn_self_dist=float(r_nn_self.mean()),
        gen_nn_to_train_dist=float(g_nn_train.mean()),
        real_nn_to_train_dist=float(r_nn_train.mean()),
    )

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Class distribution (approximate, via classifier).
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(10)
    ax.bar(x - 0.2, r_dist, 0.4, label="Real MNIST (labels)")
    ax.bar(x + 0.2, g_dist, 0.4, label="Generated (classifier)")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Digit", ylabel="Fraction", title=f"{name}: class distribution ({n_gen} images)",
           xticks=x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # Generated images grouped by predicted digit, vs real images of the same digit.
    fig, axes = plt.subplots(2, 1, figsize=(10, 2.6))
    for ax, imgs, preds, t in ((axes[0], gen, gp, "Generated"), (axes[1], real_test, real_test_y, "Real")):
        cols = [imgs[np.where(preds == d)[0][:1]] for d in range(10)]
        cols = [c if len(c) else torch.full((1, 1, 28, 28), -1.0) for c in cols]
        ax.imshow(tile_images(torch.cat(cols), 10), cmap="gray")
        ax.set_title(f"{t} (digits 0-9, one example each)", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_digit_examples.png"), dpi=150)
    plt.close(fig)

    # Generated image next to its closest real training image.
    idx = torch.argsort(g_nn_train)[:8].tolist() + torch.argsort(g_nn_train)[-8:].tolist()
    pairs = []
    for i in idx:
        j = torch.cdist(g1k[i].flatten()[None], rtrain.flatten(1)).argmin().item()
        pairs += [gen[i:i + 1], real_train[j:j + 1]]
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.imshow(tile_images(torch.cat(pairs), 16), cmap="gray")
    ax.set_title("Generated image followed by its nearest real training image "
                 "(top row: 8 closest pairs, bottom row: 8 farthest)", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nearest_real_neighbors.png"), dpi=150)
    plt.close(fig)

    # 1,000-image montage (generated vs real) for visual comparison.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, imgs, t in ((axes[0], gen[:400], "Generated (400 shown of 10,000)"),
                        (axes[1], real_test[:400], "Real MNIST (400 shown)")):
        ax.imshow(tile_images(imgs, 20), cmap="gray")
        ax.set_title(t)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "gen_vs_real_montage.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
