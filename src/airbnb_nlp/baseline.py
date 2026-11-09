"""VADER baseline on the test split.

Two versions:
  default - VADER's standard rule: compound >= 0.05 means positive (label 1)
  tuned   - threshold chosen on the VALIDATION split for macro F1, applied once to test

Run:  python -m airbnb_nlp.baseline
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from airbnb_nlp.metrics import classification_metrics

DEFAULT_THRESHOLD = 0.05


def vader_scorer() -> Callable[[str], float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    return lambda text: sia.polarity_scores(text)["compound"]


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the compound-score cut-off that maximises macro F1."""
    grid = np.round(np.arange(-0.5, 1.0, 0.01), 2)
    f1s = [f1_score(labels, (scores >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(f1s))])


def run(processed_dir: str | Path, out_path: str | Path, scorer: Callable[[str], float]) -> dict:
    processed_dir = Path(processed_dir)
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    val_scores = val["text"].map(scorer).to_numpy()
    test_scores = test["text"].map(scorer).to_numpy()

    tuned = tune_threshold(val_scores, val["label"].to_numpy())
    results = {}
    for name, t in [("vader_default", DEFAULT_THRESHOLD), ("vader_tuned", tuned)]:
        preds = (test_scores >= t).astype(int)
        results[name] = {"threshold": t, **classification_metrics(test["label"], preds)}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="reports/vader_metrics.json")
    args = parser.parse_args()
    results = run(args.processed_dir, args.out, vader_scorer())
    for name, m in results.items():
        print(f"{name:14s} threshold={m['threshold']:+.2f}  accuracy={m['accuracy']:.4f}  "
              f"f1_macro={m['f1_macro']:.4f}  f1_low={m['f1_low']:.4f}  cm={m['confusion_matrix']}")


if __name__ == "__main__":
    main()
