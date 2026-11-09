---
license: apache-2.0
language: en
base_model: distilbert-base-uncased
pipeline_tag: text-classification
tags: [airbnb, reviews, text-classification, distilbert]
---
# DistilBERT: Airbnb listing quality from review text

Fine-tuned `distilbert-base-uncased` that reads one guest review and predicts whether it comes from a
**below-par** (overall rating <= 4.6) or **top-tier** (rating >= 4.9) Dallas Airbnb listing.

Code, data pipeline, tests and API: https://github.com/sgaind-data/Airbnb_Review_Scores_ML
Live demo: https://huggingface.co/spaces/sgaind-dataai/airbnb-review-quality

## Results (test set, listings never seen in training)
{{COMPARISON}}
## Labels
`0 = below_par`, `1 = top_tier`. Labels come from the listing's overall rating, not the individual review.

## Training data
Inside Airbnb, Dallas TX, June 2024 scrape (CC BY 4.0, insideairbnb.com). 15,694 English reviews,
balanced 50/50, at most 25 reviews per listing, split 70/10/20 **by listing** so no listing appears in two
splits. 3 epochs, learning rate 2e-5, batch 32, max length 256, free Colab T4 (about 3 minutes).

## Intended use
Portfolio and research demo: flagging reviews that read like those of below-par listings, and studying what
guests complain about. Not intended for decisions about hosts or guests.

## Limitations
- Labels are listing-level: a happy guest at a 4.5 listing counts as below-par, which caps accuracy near 70%.
- Dallas only, English only, reviews from 2010 to 2024.
- The most confident errors are reviews whose sentiment disagrees with the listing average; the model mainly
  detects complaints (cleanliness, price and fees).

## Usage
```python
from transformers import pipeline
clf = pipeline("text-classification", model="sgaind-dataai/distilbert-airbnb-listing-quality")
clf("Great location but the sheets were dirty and parking was expensive.")
```
