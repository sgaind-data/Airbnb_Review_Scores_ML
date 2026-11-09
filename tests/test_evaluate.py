import numpy as np
import pandas as pd

from airbnb_nlp.evaluate import comparison_table, top_errors
from airbnb_nlp.metrics import metrics_from_logits


def test_metrics_from_logits_uses_argmax():
    logits = np.array([[2.0, -1.0], [0.1, 0.9], [3.0, 0.0], [-1.0, 1.0]])
    m = metrics_from_logits(logits, np.array([0, 1, 1, 1]))
    assert m["accuracy"] == 0.75
    assert set(m) == {"accuracy", "f1_macro", "f1_low", "f1_high"}


def test_top_errors_picks_most_confident_each_way():
    df = pd.DataFrame({
        "label":         [0,    0,    1,    1,    0],
        "pred":          [1,    1,    0,    1,    0],
        "prob_top_tier": [0.99, 0.6,  0.05, 0.9,  0.1],
        "rating": 4.5, "listing_id": 1, "text": list("abcde"),
    })
    out = top_errors(df, n_each=1)
    assert out["text"].tolist() == ["a", "c"]
    assert out["error_type"].tolist() == ["false_top_tier", "false_below_par"]


def test_comparison_table_has_three_models():
    m = {"accuracy": 0.6, "f1_macro": 0.6, "f1_low": 0.6, "f1_high": 0.6, "n": 100}
    vader = {"vader_default": m, "vader_tuned": {**m, "threshold": 0.88}}
    table = comparison_table(vader, m)
    assert table.count("\n| ") == 4 and "tuned 0.88" in table
