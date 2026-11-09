"""Fine-tune DistilBERT on the listing-grouped splits.

Checkpoints are written every epoch to --output-dir (point it at Google Drive in Colab).
If a run is interrupted, re-running the same command resumes from the last checkpoint.

Run:  python -m airbnb_nlp.train --data-dir data/processed --output-dir models/distilbert
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from airbnb_nlp.metrics import metrics_from_logits

log = logging.getLogger(__name__)


def load_split(data_dir: Path, name: str):
    from datasets import Dataset

    df = pd.read_csv(data_dir / f"{name}.csv", usecols=["text", "label"])
    return Dataset.from_pandas(df, preserve_index=False)


def train(data_dir: Path, output_dir: Path, cfg: dict) -> dict:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(cfg["seed"])
    id2label = {int(k): v for k, v in cfg["labels"].items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize(batch):
        # Truncate here; pad per batch in the collator (dynamic padding is faster than padding to 256).
        return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

    ds = {name: load_split(data_dir, name).map(tokenize, batched=True, remove_columns=["text"])
          for name in ("train", "val")}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2, id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: metrics_from_logits(p.predictions, p.label_ids),
    )

    last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
    if last:
        log.info("Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    history = [
        {"epoch": round(e["epoch"]), **{k.removeprefix("eval_"): e[k] for k in e
                                         if k in ("eval_loss", "eval_accuracy", "eval_f1_macro")}}
        for e in trainer.state.log_history if "eval_f1_macro" in e
    ]
    summary = {"config": cfg, "best_checkpoint": trainer.state.best_model_checkpoint,
               "val_per_epoch": history}
    (output_dir / "train_history.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--output-dir", default="models/distilbert", type=Path)
    parser.add_argument("--epochs", type=int, help="override config (e.g. 1 for a smoke test)")
    parser.add_argument("--limit", type=int, help="use only N training rows (smoke test)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["epochs"] = args.epochs
    data_dir = args.data_dir
    if args.limit:  # write a small copy of the splits for a quick end-to-end check
        small = args.output_dir / "smoke_data"
        small.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val"):
            pd.read_csv(data_dir / f"{name}.csv").head(args.limit).to_csv(small / f"{name}.csv", index=False)
        data_dir = small
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = train(data_dir, args.output_dir, cfg)
    print("\nValidation per epoch:")
    for row in summary["val_per_epoch"]:
        print(f"  epoch {row['epoch']}: f1_macro={row['f1_macro']:.4f} "
              f"accuracy={row['accuracy']:.4f} loss={row['loss']:.4f}")


if __name__ == "__main__":
    main()
