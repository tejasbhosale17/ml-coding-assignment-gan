"""Reproduce every result: baseline, label-smoothing experiment, conditional GAN, figures, metrics.

Usage: python run_all.py
"""
import os
import shutil

import compare
from gan.cgan import demo, plot_losses, train_cgan
from gan.evaluate import evaluate_run
from gan.train import train_gan
from gan.utils import get_device
from make_figures import make_figures

BASELINE = "results/baseline"
SMOOTHING = "results/label_smoothing"
CGAN = "results/cgan"


def main():
    # Part 1-4: baseline GAN, per-epoch samples, figures.
    train_gan(BASELINE)
    make_figures(BASELINE, "Baseline")

    # Part 5C: quantitative / diversity evaluation.
    evaluate_run(BASELINE, "Baseline")

    # Part 6: one change only -> one-sided label smoothing (real target 0.9).
    train_gan(SMOOTHING, real_label=0.9)
    make_figures(SMOOTHING, "Label smoothing (0.9)")
    evaluate_run(SMOOTHING, "Label smoothing (0.9)")
    os.makedirs(compare.OUT, exist_ok=True)
    compare.side_by_side()
    compare.loss_overlay()
    compare.metrics_table(compare.class_distribution_overlay())

    # Bonus: conditional GAN.
    os.makedirs(CGAN, exist_ok=True)
    G, _ = train_cgan(CGAN)
    demo(G, CGAN, get_device())
    plot_losses(CGAN)

    # Collect the headline sample images in one folder.
    os.makedirs("samples", exist_ok=True)
    for src, dst in [
        (f"{BASELINE}/figures/epoch_progress.png", "baseline_epoch_progress.png"),
        (f"{BASELINE}/figures/final_grid.png", "baseline_final_grid_100.png"),
        (f"{BASELINE}/samples/epoch_20.png", "baseline_epoch_20_fixed_noise.png"),
        (f"{SMOOTHING}/samples/epoch_20.png", "label_smoothing_epoch_20_fixed_noise.png"),
        (f"{CGAN}/figures/digit_7_examples.png", "cgan_digit_7_examples.png"),
        (f"{CGAN}/figures/all_digits_grid.png", "cgan_all_digits_grid.png"),
    ]:
        shutil.copy(src, os.path.join("samples", dst))
    print("Done. See results/ and samples/.")


if __name__ == "__main__":
    main()
