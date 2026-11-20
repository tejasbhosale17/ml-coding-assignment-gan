# GAN on MNIST (PyTorch)

A Generative Adversarial Network that learns to generate MNIST handwritten digits, with training-behaviour analysis,
a quantitative diversity evaluation, one additional experiment (label smoothing) and the optional Conditional GAN bonus.

The write-up is in [report.md](report.md).

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

A GPU is optional; the code uses CUDA automatically if available. On CPU the full pipeline takes roughly 25 minutes
(about 20 s per epoch per model). MNIST is downloaded automatically to `data/` on first run.

## Run everything

```bash
python run_all.py
```

This runs, in order, with fixed seeds (42):

1. Baseline GAN training (20 epochs) with fixed-noise samples saved every epoch.
2. Figures: epoch progression, 100-image final grid, loss curves.
3. Quantitative evaluation on 10,000 generated images.
4. The additional experiment (label smoothing) and its comparison with the baseline.
5. The Conditional GAN bonus (ten generated 7s, plus a digit-by-digit grid).
6. Copies the headline images into `samples/`.

## Run individual steps

```bash
python -m gan.train --out results/baseline                          # baseline training
python -m gan.train --out results/label_smoothing --real-label 0.9  # experiment
python make_figures.py --run results/baseline --name Baseline       # Part 4 / 5A / 5B figures
python evaluate.py --run results/baseline --name Baseline           # Part 5C metrics + figures
python compare.py                                                   # baseline vs experiment
python -m gan.cgan                                                  # Conditional GAN
```

## Project layout

| Path | Purpose |
|---|---|
| `gan/models.py` | `Generator` and `Discriminator` |
| `gan/data.py` | MNIST loading, normalisation to [-1, 1] |
| `gan/train.py` | Adversarial training loop (BCE loss, Adam) |
| `gan/viz.py` | Epoch-progress figure, sample grids, loss curves |
| `gan/evaluate.py` | CNN evaluator, class distribution, Frechet distance, nearest-neighbour checks |
| `gan/cgan.py` | Conditional GAN (bonus) |
| `run_all.py` | End-to-end reproduction |
| `make_figures.py`, `evaluate.py`, `compare.py` | CLI wrappers for individual steps |
| `samples/` | Headline generated images |
| `results/` | Per-run outputs: `samples/epoch_XX.png`, `figures/`, `history.json`, `metrics.json` |
| `report.md` | Report |

## Notes on code provenance

All code was written for this assignment following the standard GAN formulation (Goodfellow et al., 2014).
The generator/discriminator layer sizes follow common PyTorch MLP-GAN conventions, and BatchNorm / dropout choices are
explained in the report. No code was copied from a specific tutorial.
