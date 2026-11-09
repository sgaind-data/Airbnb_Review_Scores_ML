import json

import numpy as np
import pandas as pd

from airbnb_nlp.baseline import run, tune_threshold
from airbnb_nlp.metrics import classification_metrics


def test_metrics_perfect_and_confusion_layout():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]


def test_tune_threshold_finds_the_separating_cut():
    scores = np.array([-0.9, -0.2, 0.1, 0.3, 0.6, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = tune_threshold(scores, labels)
    assert 0.1 < t <= 0.3


def test_run_writes_both_baselines(tmp_path):
    words = {"bad": -0.8, "meh": 0.2, "good": 0.9}
    df = pd.DataFrame({"text": ["bad", "meh", "good", "good"], "label": [0, 0, 1, 1]})
    df.to_csv(tmp_path / "val.csv", index=False)
    df.to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "reports" / "vader.json"

    results = run(tmp_path, out, scorer=words.get)

    assert set(results) == {"vader_default", "vader_tuned"}
    assert results["vader_tuned"]["accuracy"] == 1.0
    assert json.loads(out.read_text()) == results
