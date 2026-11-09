"""Production monitoring for the API.

- PredictionMonitor: keeps a rolling window of recent predictions for GET /stats and appends each
  prediction to a JSON-lines log. Raw review text is never logged, only its length.
- psi(): Population Stability Index, used to detect drift between live predictions and the test set.

Drift check:  python -m airbnb_nlp.monitoring --log logs/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


class PredictionMonitor:
    def __init__(self, log_path: str | Path | None = None, window: int = 10_000):
        self._recent: deque[dict] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total = 0
        self.started = time.time()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, prob_top_tier: float, n_chars: int, latency_ms: float, model_version: str) -> None:
        event = {"ts": round(time.time(), 3), "prob_top_tier": round(prob_top_tier, 5),
                 "label": int(prob_top_tier >= 0.5), "n_chars": n_chars,
                 "latency_ms": round(latency_ms, 2), "model_version": model_version}
        with self._lock:
            self._recent.append(event)
            self._total += 1
            if self.log_path:
                with self.log_path.open("a") as f:
                    f.write(json.dumps(event) + "\n")

    def summary(self) -> dict:
        with self._lock:
            recent = list(self._recent)
            total = self._total
        out = {"predictions_total": total, "window_size": len(recent),
               "uptime_s": round(time.time() - self.started, 1)}
        if recent:
            probs = np.array([e["prob_top_tier"] for e in recent])
            lat = np.array([e["latency_ms"] for e in recent])
            out.update({
                "share_top_tier": round(float((probs >= 0.5).mean()), 4),
                "mean_prob_top_tier": round(float(probs.mean()), 4),
                "mean_chars": round(float(np.mean([e["n_chars"] for e in recent])), 1),
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            })
        return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of probabilities in [0, 1]. <0.1 stable, 0.1-0.25 moderate, >0.25 drift."""
    edges = np.linspace(0, 1, bins + 1)
    e = np.histogram(np.clip(expected, 0, 1), edges)[0] / max(len(expected), 1)
    a = np.histogram(np.clip(actual, 0, 1), edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate drift" if value < 0.25 else "significant drift"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/predictions.jsonl", type=Path)
    parser.add_argument("--reference", default="reports/test_predictions.csv", type=Path)
    args = parser.parse_args()

    live = pd.read_json(args.log, lines=True)
    ref = pd.read_csv(args.reference)
    value = psi(ref["prob_top_tier"].to_numpy(), live["prob_top_tier"].to_numpy())
    report = {
        "live_predictions": len(live),
        "reference_predictions": len(ref),
        "share_top_tier_live": round(float(live["label"].mean()), 4),
        "share_top_tier_reference": round(float(ref["pred"].mean()), 4),
        "psi": round(value, 4),
        "status": drift_status(value),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
