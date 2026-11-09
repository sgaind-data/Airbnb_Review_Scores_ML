"""Airbnb review NLP: classify below-par vs top-tier listings from review text."""

__version__ = "1.1.0"
DEFAULT_MODEL_ID = "sgaind-dataai/distilbert-airbnb-listing-quality"
LABELS = {0: "below_par", 1: "top_tier"}
