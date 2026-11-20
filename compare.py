"""Part 6: compare the label-smoothing experiment against the baseline."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gan.viz import _smooth, load_history

RUNS = [("Baseline", "results/baseline"), ("Label smoothing (0.9)", "results/label_smoothing")]
OUT = "results/comparison"


def side_by_side():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, (name, run) in zip(axes, RUNS):
        ax.imshow(plt.imread(os.path.join(run, "samples", "epoch_20.png")), cmap="gray")
        ax.set_title(f"{name} (epoch 20, same fixed noise)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "samples_side_by_side.png"), dpi=150)
    plt.close(fig)


def loss_overlay(window=100):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for (name, run), color in zip(RUNS, ("tab:blue", "tab:red")):
        h = load_history(run)
        axes[0].plot(_smooth(h["g_loss"], window), color=color, label=name)
        axes[1].plot(_smooth(h["d_loss"], window), color=color, label=name)
        axes[2].plot(_smooth(h["d_real"], window), color=color, label=f"{name}: D(real)")
        axes[2].plot(_smooth(h["d_fake"], window), color=color, linestyle="--", label=f"{name}: D(fake)")
    axes[0].set(title="Generator loss (moving avg)", xlabel="Iteration", ylabel="BCE loss")
    axes[1].set(title="Discriminator loss (moving avg)", xlabel="Iteration", ylabel="BCE loss")
    axes[2].set(title="Discriminator outputs", xlabel="Iteration", ylim=(0, 1))
    axes[2].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_overlay.png"), dpi=150)
    plt.close(fig)


def class_distribution_overlay():
    ms = [json.load(open(os.path.join(run, "metrics.json"))) for _, run in RUNS]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.27, ms[0]["real_class_dist"], 0.27, label="Real MNIST", color="gray")
    ax.bar(x, ms[0]["gen_class_dist"], 0.27, label=RUNS[0][0])
    ax.bar(x + 0.27, ms[1]["gen_class_dist"], 0.27, label=RUNS[1][0])
    ax.set(xlabel="Digit", ylabel="Fraction", xticks=x, title="Class distribution (10,000 generated images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_distribution.png"), dpi=150)
    plt.close(fig)
    return ms


def metrics_table(ms):
    rows = [
        ("Frechet distance (lower is better)", "frechet_distance", "{:.1f}"),
        ("Class-distribution TV distance vs real (lower is better)", "total_variation_distance", "{:.3f}"),
        ("Class entropy, bits (real: 3.319)", "gen_class_entropy_bits", "{:.3f}"),
        ("Mean classifier confidence (real: 0.988)", "gen_mean_confidence", "{:.3f}"),
        ("Fraction with confidence < 0.9 (real: 0.033)", "gen_low_confidence_frac", "{:.3f}"),
        ("Mean NN distance within generated set", "gen_nn_self_dist", "{:.2f}"),
        ("Mean NN distance to training set", "gen_nn_to_train_dist", "{:.2f}"),
    ]
    lines = ["| Metric | " + " | ".join(n for n, _ in RUNS) + " |", "|---|---|---|"]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt.format(m[key]) for m in ms) + " |")
    hs = [load_history(run) for _, run in RUNS]
    lines.append("| Final-epoch G loss | " + " | ".join(f"{h['epoch_g_loss'][-1]:.3f}" for h in hs) + " |")
    lines.append("| Final-epoch D loss | " + " | ".join(f"{h['epoch_d_loss'][-1]:.3f}" for h in hs) + " |")
    with open(os.path.join(OUT, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    side_by_side()
    loss_overlay()
    metrics_table(class_distribution_overlay())
