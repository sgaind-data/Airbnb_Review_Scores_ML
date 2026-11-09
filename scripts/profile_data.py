"""Profile the raw Airbnb reviews file before building the pipeline.

Usage: python scripts/profile_data.py [path/to/Dallas_Airbnb.csv]
"""
import os
import sys

import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Dallas_Airbnb.csv"
REQUIRED = ["listing_id", "id", "date", "comments", "review_scores_rating"]

print(f"File: {PATH}  ({os.path.getsize(PATH) / 1e6:,.1f} MB)")
all_cols = pd.read_csv(PATH, nrows=0).columns.tolist()
missing = [c for c in REQUIRED if c not in all_cols]
print(f"Columns: {len(all_cols)}  | missing required: {missing or 'none'}")
if missing:
    sys.exit("Stop: required columns are missing. Send this output to Claude.")

df = pd.read_csv(PATH, usecols=REQUIRED, low_memory=False)
print(f"Rows (reviews): {len(df):,}")
print(f"Listings: {df.listing_id.nunique():,}")
print(f"Duplicate review ids: {df.id.duplicated().sum():,}")
print(f"Empty comments: {df.comments.isna().sum():,}")
print(f"Missing rating: {df.review_scores_rating.isna().sum():,}")
dates = pd.to_datetime(df.date, format="%m/%d/%y", errors="coerce")
print(f"Unparseable dates: {dates.isna().sum():,}")
print(f"Review dates: {dates.min().date()} to {dates.max().date()}")

s = df.review_scores_rating
print("\nReview-level rating quantiles:")
print(s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75]).round(2).to_string())

per_listing = df.drop_duplicates("listing_id")
print("\nCandidate 'low' cut-offs:")
for t in [4.0, 4.25, 4.5, 4.6, 4.7]:
    print(f"  rating <= {t}: {(s <= t).sum():>8,} reviews from {(per_listing.review_scores_rating <= t).sum():>6,} listings")
print("Candidate 'high' cut-offs:")
for t in [4.85, 4.9, 4.95]:
    print(f"  rating >= {t}: {(s >= t).sum():>8,} reviews from {(per_listing.review_scores_rating >= t).sum():>6,} listings")
