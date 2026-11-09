"""Data pipeline: load -> clean -> label -> balanced sample -> listing-grouped splits.

Run:  python -m airbnb_nlp.data            (uses configs/data.yaml)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from airbnb_nlp.config import DataConfig, load_config

log = logging.getLogger(__name__)

RAW_COLUMNS = {
    "id": "review_id",
    "listing_id": "listing_id",
    "date": "date",
    "comments": "text",
    "review_scores_rating": "rating",
}
OUTPUT_COLUMNS = ["review_id", "listing_id", "date", "rating", "label", "text"]
AUTOMATED_NOTE = "this is an automated posting"
_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WS = re.compile(r"\s+")
_ENGLISH_HINTS = {
    "the", "and", "was", "is", "to", "we", "very", "it", "in", "for", "of",
    "great", "place", "stay", "with", "you", "this", "our", "would", "clean",
}


class DataValidationError(ValueError):
    """Raised when the raw file breaks the data contract (docs/DATA.md)."""


def load_raw(path: str | Path, date_format: str) -> pd.DataFrame:
    """Read only the needed columns, rename them, and enforce the data contract."""
    header = pd.read_csv(path, nrows=0).columns
    missing = set(RAW_COLUMNS) - set(header)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    df = pd.read_csv(path, usecols=list(RAW_COLUMNS), dtype={"comments": "string"})
    df = df.rename(columns=RAW_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")

    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        raise DataValidationError(f"{bad_dates} dates do not match format {date_format!r}")
    if df["review_id"].duplicated().any():
        raise DataValidationError("Duplicate review IDs found")
    if not df["rating"].dropna().between(0, 5).all():
        raise DataValidationError("Ratings outside the 0-5 range")
    log.info("Loaded %d reviews from %d listings", len(df), df["listing_id"].nunique())
    return df


def clean_text(text: str) -> str:
    """Remove HTML line breaks and collapse whitespace. Keep case and punctuation (VADER uses them)."""
    return _WS.sub(" ", _BR.sub(" ", text)).strip()


def looks_english(text: str, min_hits: int = 2) -> bool:
    """Cheap language filter: English reviews almost always contain common English words."""
    words = re.findall(r"[a-z']+", text.lower())
    return sum(w in _ENGLISH_HINTS for w in words) >= min_hits


def clean(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    """Drop empty, automated, non-English, very short and duplicate reviews."""
    n0 = len(df)
    df = df.dropna(subset=["text", "rating"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[~df["text"].str.lower().str.contains(AUTOMATED_NOTE, regex=False)]
    df = df[df["text"].str.split().str.len() >= min_words]
    df = df[df["text"].map(looks_english)]
    df = df.drop_duplicates(subset=["listing_id", "text"])
    log.info("Cleaning kept %d of %d reviews", len(df), n0)
    return df.reset_index(drop=True)


def assign_labels(df: pd.DataFrame, low_max: float, high_min: float) -> pd.DataFrame:
    """Label 0 if rating <= low_max, 1 if rating >= high_min; drop the gap in between."""
    keep = (df["rating"] <= low_max) | (df["rating"] >= high_min)
    out = df[keep].copy()
    out["label"] = (out["rating"] >= high_min).astype(int)
    return out.reset_index(drop=True)


def sample_balanced(df: pd.DataFrame, per_class: int, max_per_listing: int, seed: int) -> pd.DataFrame:
    """Cap reviews per listing, then take the same number of reviews from each label."""
    shuffled = df.sample(frac=1, random_state=seed)
    capped = shuffled.groupby("listing_id", sort=False).head(max_per_listing)
    n = min(per_class, capped["label"].value_counts().min())
    out = capped.groupby("label", group_keys=False).sample(n=n, random_state=seed)
    log.info("Balanced sample: %d per class, %d listings", n, out["listing_id"].nunique())
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def _group_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(df, df["label"], groups=df["listing_id"]))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


def split_by_listing(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> dict[str, pd.DataFrame]:
    """Stratified splits where every listing appears in exactly one split (no leakage)."""
    trainval, test = _group_split(df, test_size, seed)
    train, val = _group_split(trainval, val_size / (1 - test_size), seed)
    splits = {"train": train, "val": val, "test": test}
    assert_no_listing_overlap(splits)
    return splits


def assert_no_listing_overlap(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["listing_id"]) & set(splits[b]["listing_id"])
            if shared:
                raise AssertionError(f"{len(shared)} listings appear in both {a} and {b}")


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Run the full pipeline and write splits plus a manifest to cfg.processed_dir."""
    raw = load_raw(cfg.raw_path, cfg.date_format)
    labelled = assign_labels(clean(raw, cfg.min_words), cfg.low_max, cfg.high_min)
    sample = sample_balanced(labelled, cfg.per_class, cfg.max_per_listing, cfg.seed)
    splits = split_by_listing(sample, cfg.test_size, cfg.val_size, cfg.seed)

    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"raw_sha256": _sha256(cfg.raw_path), "config": cfg.__dict__, "splits": {}}
    for name, d in splits.items():
        d[OUTPUT_COLUMNS].to_csv(out_dir / f"{name}.csv", index=False)
        manifest["splits"][name] = {
            "rows": len(d),
            "listings": int(d["listing_id"].nunique()),
            "share_high": round(float(d["label"].mean()), 4),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = build_dataset(load_config(args.config))
    for name, d in splits.items():
        print(f"{name:5s} rows={len(d):6,} listings={d['listing_id'].nunique():5,} "
              f"share_high={d['label'].mean():.3f}")


if __name__ == "__main__":
    main()
