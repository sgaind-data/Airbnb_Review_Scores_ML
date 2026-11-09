# Data contract

## Source
- Inside Airbnb (insideairbnb.com), Dallas, TX — scrape of June 2024 (scrape_id 20240619201441)
- Licensed CC BY 4.0 by Inside Airbnb; attribution required
- File: `Dallas_Airbnb.csv` (reviews pre-joined to listings, 81 columns, ~633 MB)
- Not stored in Git. Download from the [project's Google Drive folder](https://drive.google.com/drive/folders/13pYovFwyjd87VZBGFgblgFNI_ABXZNvN?usp=drive_link) into `data/raw/`

## Required columns
| Column | Type | Meaning |
|---|---|---|
| `id` | int | Review ID (unique) |
| `listing_id` | int | Listing the review belongs to; used to group splits |
| `date` | str, `%m/%d/%y` (e.g. `1/6/24`) | Review date |
| `comments` | str | Review text (model input) |
| `review_scores_rating` | float, 0–5 | Listing's overall rating at scrape time (label source) |

## Profile (Nov 2026)
- 201,901 reviews, 4,160 listings, review dates 2010-12-29 to 2024-06-19
- 0 duplicate review IDs, 0 missing ratings, 79 empty comments, 0 unparseable dates
- Rating quantiles: p1 4.28, p5 4.50, p10 4.61, p25 4.75, p50 4.86, p75 4.94

## Known limitations
- Rating is per listing, not per review: every review inherits its listing's score, so labels are noisy.
- Rating is a 2024 snapshot; older reviews may describe a listing that has since changed.
- Ratings are heavily skewed toward 5; "low" means below-par for Dallas, not bad.
- Dates use 2-digit years, so the file was likely re-saved by a spreadsheet tool after download.
  Always parse with an explicit format; never rely on inference.
