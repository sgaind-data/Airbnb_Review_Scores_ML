import json

import numpy as np

from airbnb_nlp.monitoring import PredictionMonitor, drift_status, psi


def test_monitor_summary_and_log(tmp_path):
    log = tmp_path / "logs" / "p.jsonl"
    m = PredictionMonitor(log, window=3)
    for prob in (0.9, 0.2, 0.8, 0.7):
        m.record(prob, n_chars=100, latency_ms=10.0, model_version="test")
    s = m.summary()
    assert s["predictions_total"] == 4 and s["window_size"] == 3      # rolling window keeps last 3
    assert s["share_top_tier"] == round(2 / 3, 4)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 4 and "text" not in lines[0]                  # raw text never logged


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, 5000)
    assert psi(ref, rng.uniform(0, 1, 5000)) < 0.1
    shifted = rng.beta(8, 2, 5000)
    assert psi(ref, shifted) > 0.25
    assert drift_status(0.05) == "stable" and drift_status(0.3) == "significant drift"
