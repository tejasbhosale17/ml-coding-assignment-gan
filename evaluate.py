"""Quantitative / diversity evaluation (Part 5C) for a trained run."""
import argparse
import json

from gan.evaluate import evaluate_run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/baseline")
    p.add_argument("--name", default="Baseline")
    p.add_argument("--n-gen", type=int, default=10000)
    a = p.parse_args()
    print(json.dumps(evaluate_run(a.run, a.name, a.n_gen), indent=2))
