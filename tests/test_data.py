import pandas as pd
import pytest

from airbnb_nlp.config import DataConfig
from airbnb_nlp.data import (
    DataValidationError,
    assign_labels,
    clean,
    clean_text,
    load_raw,
    looks_english,
    sample_balanced,
    split_by_listing,
)


def test_clean_text_removes_br_and_keeps_case_and_punctuation():
    assert clean_text("Great stay!<br/>Not  clean,<BR>though.") == "Great stay! Not clean, though."


@pytest.mark.parametrize("text,expected", [
    ("The place was clean and we loved it", True),
    ("Muy buena ubicación, todo excelente y limpio", False),
])
def test_looks_english(text, expected):
    assert looks_english(text) is expected


def test_clean_drops_empty_short_automated_and_duplicates():
    df = pd.DataFrame({
        "listing_id": [1, 1, 1, 1, 2],
        "rating": [4.9] * 5,
        "text": [
            "The place was great and very clean",
            "The place was great and very clean",            # duplicate in same listing
            "Nice",                                            # too short
            "The host canceled this reservation 3 days before arrival. This is an automated posting.",
            None,                                              # empty
        ],
    })
    out = clean(df, min_words=5)
    assert out["text"].tolist() == ["The place was great and very clean"]


def test_assign_labels_drops_the_gap():
    df = pd.DataFrame({"rating": [4.5, 4.6, 4.7, 4.89, 4.9, 5.0]})
    out = assign_labels(df, low_max=4.6, high_min=4.9)
    assert out["rating"].tolist() == [4.5, 4.6, 4.9, 5.0]
    assert out["label"].tolist() == [0, 0, 1, 1]


def test_sample_is_balanced_and_capped(labelled_df):
    out = sample_balanced(labelled_df, per_class=10_000, max_per_listing=3, seed=1)
    assert out["label"].value_counts().nunique() == 1          # same count per class
    assert out.groupby("listing_id").size().max() <= 3


def test_splits_have_no_shared_listings_and_keep_balance(labelled_df):
    splits = split_by_listing(labelled_df, test_size=0.2, val_size=0.1, seed=42)
    ids = [set(d["listing_id"]) for d in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(d) for d in splits.values()) == len(labelled_df)
    for d in splits.values():
        assert 0.35 <= d["label"].mean() <= 0.65
    assert 0.15 <= len(splits["test"]) / len(labelled_df) <= 0.25


def test_load_raw_rejects_bad_dates(tmp_path):
    p = tmp_path / "raw.csv"
    pd.DataFrame({
        "id": [1], "listing_id": [10], "date": ["2023-01-06"],
        "comments": ["ok"], "review_scores_rating": [4.8],
    }).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="dates"):
        load_raw(p, "%m/%d/%y")


def test_load_raw_rejects_missing_columns(tmp_path):
    p = tmp_path / "raw.csv"
    pd.DataFrame({"id": [1], "comments": ["ok"]}).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="Missing"):
        load_raw(p, "%m/%d/%y")


def test_config_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        DataConfig("a", "b", "%m/%d/%y", 4.9, 4.6, 5, 25, 100, 0.2, 0.1, 42)
