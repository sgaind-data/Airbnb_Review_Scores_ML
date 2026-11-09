from collections import Counter

import pandas as pd

from airbnb_nlp.ner import compare_amenities, entity_summary, is_negated, load_amenities, ruler_patterns


def test_amenity_config_loads_and_builds_patterns():
    amenities = load_amenities("configs/amenities.yaml")
    assert {"wifi", "parking", "pool"} <= set(amenities)
    patterns = ruler_patterns(amenities)
    assert {"label": "AMENITY", "pattern": "wi-fi", "id": "wifi"} in patterns


def test_is_negated():
    assert is_negated(["there", "was", "no"])
    assert is_negated(["the", "wifi", "didn't"]) is False
    assert is_negated(["did", "n't"])
    assert not is_negated(["great", "fast"])


def _row(amenities=(), negated=(), ents=None):
    return {"amenities": set(amenities), "negated": set(negated), "entities": Counter(ents or {})}


def test_compare_amenities_shares_and_negation():
    labels = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    rows = [_row(["parking"], ["parking"]), _row(["parking"]), _row(), _row(),
            _row(["pool"]), _row(), _row(), _row()]
    out = compare_amenities(labels, rows, ["parking", "pool", "wifi"]).set_index("amenity")
    assert out.loc["parking", "pct_below_par"] == 50.0
    assert out.loc["parking", "pct_top_tier"] == 0.0
    assert out.loc["parking", "negated_pct_below_par"] == 50.0
    assert out.loc["pool", "diff_pp"] == -25.0
    assert out.loc["wifi", "mentions"] == 0 and out.loc["wifi", "p_value"] == 1.0
    assert out.index[0] == "parking"  # sorted by how much more below-par reviews mention it


def test_entity_summary_averages_per_class():
    labels = pd.Series([0, 1])
    rows = [_row(ents={"GPE": 2}), _row(ents={"GPE": 0, "PERSON": 1})]
    out = entity_summary(labels, rows)
    assert out.loc["GPE", "below_par"] == 2 and out.loc["PERSON", "top_tier"] == 1
