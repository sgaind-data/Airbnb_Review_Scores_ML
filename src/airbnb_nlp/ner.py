"""Named-entity recognition: spaCy en_core_web_sm + an EntityRuler for amenities.

Compares how often each amenity is mentioned (and mentioned negatively, e.g. "no wifi")
in reviews of below-par vs top-tier listings.

Run:  python -m airbnb_nlp.ner
Needs: pip install -e ".[ner]" && python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import chi2_contingency

log = logging.getLogger(__name__)

NEGATIONS = {"no", "not", "never", "without", "n't", "broken", "lack", "lacked", "missing", "dirty"}
NEGATION_WINDOW = 3          # tokens before the amenity that are checked for a negation
STANDARD_LABELS = ("GPE", "LOC", "FAC", "ORG", "PERSON", "MONEY", "TIME", "DATE")


def load_amenities(path: str | Path = "configs/amenities.yaml") -> dict[str, list[str]]:
    return yaml.safe_load(Path(path).read_text())


def ruler_patterns(amenities: dict[str, list[str]]) -> list[dict]:
    """EntityRuler phrase patterns; the category is stored as the entity id."""
    return [{"label": "AMENITY", "pattern": phrase, "id": category}
            for category, phrases in amenities.items() for phrase in phrases]


def is_negated(preceding_tokens: Iterable[str]) -> bool:
    """True if any of the few tokens before a mention is a negation or complaint word."""
    return any(t.lower() in NEGATIONS for t in preceding_tokens)


def build_nlp(amenities: dict[str, list[str]], model: str = "en_core_web_sm"):
    import spacy

    nlp = spacy.load(model, disable=["lemmatizer"])
    ruler = nlp.add_pipe("entity_ruler", before="ner",
                         config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True})
    ruler.add_patterns(ruler_patterns(amenities))
    return nlp


def extract(texts: list[str], nlp, batch_size: int = 256) -> list[dict]:
    """Per review: amenity categories mentioned, those mentioned negatively, standard entity counts."""
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        mentioned, negated, ents = set(), set(), Counter()
        for ent in doc.ents:
            if ent.label_ == "AMENITY":
                mentioned.add(ent.ent_id_)
                start = max(0, ent.start - NEGATION_WINDOW)
                if is_negated(t.text for t in doc[start:ent.start]):
                    negated.add(ent.ent_id_)
            elif ent.label_ in STANDARD_LABELS:
                ents[ent.label_] += 1
        rows.append({"amenities": mentioned, "negated": negated, "entities": ents})
    return rows


def compare_amenities(labels: pd.Series, rows: list[dict], categories: list[str]) -> pd.DataFrame:
    """Share of below-par vs top-tier reviews mentioning each amenity, with a chi-square test."""
    labels = labels.reset_index(drop=True)
    is_low = labels == 0
    n_low, n_high = int(is_low.sum()), int((~is_low).sum())
    out = []
    for cat in categories:
        mention = pd.Series([cat in r["amenities"] for r in rows])
        neg = pd.Series([cat in r["negated"] for r in rows])
        low, high = int(mention[is_low].sum()), int(mention[~is_low].sum())
        table = [[low, n_low - low], [high, n_high - high]]
        p = chi2_contingency(table)[1] if low + high > 0 else 1.0
        out.append({
            "amenity": cat,
            "pct_below_par": round(100 * low / n_low, 2),
            "pct_top_tier": round(100 * high / n_high, 2),
            "diff_pp": round(100 * (low / n_low - high / n_high), 2),
            "negated_pct_below_par": round(100 * neg[is_low].sum() / max(low, 1), 1),
            "negated_pct_top_tier": round(100 * neg[~is_low].sum() / max(high, 1), 1),
            "mentions": low + high,
            "p_value": float(p),
        })
    return pd.DataFrame(out).sort_values("diff_pp", ascending=False).reset_index(drop=True)


def entity_summary(labels: pd.Series, rows: list[dict]) -> pd.DataFrame:
    """Average count per review of each standard spaCy entity type, by class."""
    df = pd.DataFrame([{lab: r["entities"].get(lab, 0) for lab in STANDARD_LABELS} for r in rows])
    df["label"] = labels.reset_index(drop=True).map({0: "below_par", 1: "top_tier"})
    return df.groupby("label").mean().T.round(3)


def plot_comparison(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("pct_below_par")
    y = range(len(t))
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.2))
    ax.barh([i + 0.2 for i in y], t["pct_below_par"], height=0.4, label="below-par listings", color="#c2410c")
    ax.barh([i - 0.2 for i in y], t["pct_top_tier"], height=0.4, label="top-tier listings", color="#1d4ed8")
    ax.set_yticks(list(y), t["amenity"])
    ax.set_xlabel("% of reviews mentioning the amenity")
    ax.set_title("Amenity mentions by listing quality")
    ax.legend(loc="lower right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--amenities", default="configs/amenities.yaml")
    parser.add_argument("--reports-dir", default="reports", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.concat([pd.read_csv(args.data_dir / f"{s}.csv") for s in ("train", "val", "test")],
                   ignore_index=True)
    amenities = load_amenities(args.amenities)
    log.info("Running spaCy on %d reviews (a few minutes)...", len(df))
    rows = extract(df["text"].tolist(), build_nlp(amenities))

    table = compare_amenities(df["label"], rows, list(amenities))
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.reports_dir / "amenity_comparison.csv", index=False)
    entity_summary(df["label"], rows).to_csv(args.reports_dir / "ner_entities.csv")
    plot_comparison(table, args.reports_dir / "amenity_mentions.png")

    pd.set_option("display.width", 140)
    print(table.drop(columns="p_value").to_string(index=False))
    print("\nStandard entities per review:\n", entity_summary(df["label"], rows))


if __name__ == "__main__":
    main()
