import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def labelled_df() -> pd.DataFrame:
    """200 listings x 10 reviews, half low-rated and half high-rated."""
    rng = np.random.default_rng(0)
    rows = []
    for listing in range(200):
        rating = 4.3 if listing % 2 == 0 else 4.95
        for r in range(10):
            rows.append({
                "review_id": listing * 100 + r,
                "listing_id": listing,
                "date": pd.Timestamp("2023-01-01"),
                "rating": rating,
                "label": int(rating >= 4.9),
                "text": f"The place was great and we loved it {rng.integers(1e9)}",
            })
    return pd.DataFrame(rows)
