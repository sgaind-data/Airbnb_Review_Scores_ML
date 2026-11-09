"""Evaluate the fine-tuned model on the test split and compare it with VADER.

Writes to --reports-dir:
  distilbert_metrics.json   test metrics (same function as the VADER baseline)
  comparison.md             results table: VADER default / VADER tuned / DistilBERT
  confusion_matrix.png      DistilBERT test confusion matrix
  error_analysis.csv        10 most confident mistakes (5 each way) for manual review
  test_predictions.csv      every test prediction with its probability

Run:  python -m airbnb_nlp.evaluate --model-dir models/distilbert/final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from airbnb_nlp.metrics import classification_metrics


def predict_proba(texts: list[str], model_dir: Path, batch_size: int = 64, max_length: int = 256) -> np.ndarray:
    """Probability of label 1 (top_tier) for each text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            probs.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def top_errors(df: pd.DataFrame, n_each: int = 5) -> pd.DataFrame:
    """Most confident wrong predictions in each direction."""
    wrong = df[df["pred"] != df["label"]].copy()
    wrong["confidence"] = np.where(wrong["pred"] == 1, wrong["prob_top_tier"], 1 - wrong["prob_top_tier"])
    fp = wrong[wrong["pred"] == 1].nlargest(n_each, "confidence")   # said top-tier, was below-par
    fn = wrong[wrong["pred"] == 0].nlargest(n_each, "confidence")   # said below-par, was top-tier
    out = pd.concat([fp, fn])
    out["error_type"] = np.where(out["pred"] == 1, "false_top_tier", "false_below_par")
    out["your_notes"] = ""
    cols = ["error_type", "confidence", "rating", "listing_id", "text", "your_notes"]
    return out[cols].round({"confidence": 3}).reset_index(drop=True)


def comparison_table(vader: dict, bert: dict) -> str:
    rows = [("VADER (default 0.05)", vader["vader_default"]),
            (f"VADER (tuned {vader['vader_tuned']['threshold']:.2f})", vader["vader_tuned"]),
            ("DistilBERT (fine-tuned)", bert)]
    lines = ["| Model | Accuracy | Macro F1 | F1 below-par | F1 top-tier |",
             "| --- | --- | --- | --- | --- |"]
    lines += [f"| {name} | {m['accuracy']:.3f} | {m['f1_macro']:.3f} | {m['f1_low']:.3f} | {m['f1_high']:.3f} |"
              for name, m in rows]
    return "\n".join(lines) + f"\n\nTest set: {bert['n']:,} reviews from unseen listings.\n"


def plot_confusion(cm: list[list[int]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["below_par", "top_tier"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("DistilBERT — test set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--model-dir", default="models/distilbert/final", type=Path)
    parser.add_argument("--vader", default="reports/vader_metrics.json", type=Path)
    parser.add_argument("--reports-dir", default="reports", type=Path)
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(args.data_dir / "test.csv")
    test["prob_top_tier"] = predict_proba(test["text"].tolist(), args.model_dir)
    test["pred"] = (test["prob_top_tier"] >= 0.5).astype(int)

    bert = classification_metrics(test["label"], test["pred"])
    vader = json.loads(args.vader.read_text())
    (args.reports_dir / "distilbert_metrics.json").write_text(json.dumps(bert, indent=2))
    table = comparison_table(vader, bert)
    (args.reports_dir / "comparison.md").write_text(table)
    plot_confusion(bert["confusion_matrix"], args.reports_dir / "confusion_matrix.png")
    top_errors(test).to_csv(args.reports_dir / "error_analysis.csv", index=False)
    test.drop(columns=["text"]).to_csv(args.reports_dir / "test_predictions.csv", index=False)
    print(table)
    print("Confusion matrix (rows=actual, cols=predicted):", bert["confusion_matrix"])


if __name__ == "__main__":
    main()
