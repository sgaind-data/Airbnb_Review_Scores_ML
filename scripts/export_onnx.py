"""Export the fine-tuned model to ONNX, quantize it to int8, and measure what quantization costs.

Writes models/onnx/ (model.onnx, model_quantized.onnx + tokenizer/config files) and reports/onnx_metrics.json.
With --upload, pushes the ONNX files to the model's Hugging Face repo under onnx/, which is where
Transformers.js looks for them (the free browser demo loads onnx/model_quantized.onnx).

Usage:  python scripts/export_onnx.py            # export + evaluate on data/processed/test.csv
        python scripts/export_onnx.py --upload   # ...and upload (run `hf auth login` first)
Needs:  pip install -e ".[inference,onnx]"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from airbnb_nlp import DEFAULT_MODEL_ID
from airbnb_nlp.metrics import classification_metrics

MAX_LENGTH = 256


def export(model_id: str, out_dir: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class LogitsOnly(torch.nn.Module):
        """Expose a plain (input_ids, attention_mask) -> logits graph for ONNX."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_id, attn_implementation="eager").eval()
    model = LogitsOnly(hf_model).eval()
    sample = tokenizer(["A sample review to trace the graph.", "Short one."], padding=True, return_tensors="pt")
    args = (sample["input_ids"], sample["attention_mask"])
    path = out_dir / "onnx" / "model.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    common = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "logits": {0: "batch"}},
    )
    with torch.no_grad():
        try:  # classic TorchScript-based exporter
            torch.onnx.export(model, args, str(path), opset_version=17, dynamo=False, **common)
        except Exception as err:  # newer torch: use the dynamo exporter (needs onnxscript)
            print(f"TorchScript exporter unavailable ({type(err).__name__}); using the dynamo exporter")
            torch.onnx.export(model, args, str(path), dynamo=True, external_data=False, **common)

    # Sanity check: ONNX output must match PyTorch on the sample
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"input_ids": args[0].numpy(), "attention_mask": args[1].numpy()})[0]
    with torch.no_grad():
        torch_logits = model(*args).numpy()
    diff = float(np.abs(onnx_logits - torch_logits).max())
    print(f"Max |ONNX - PyTorch| logit difference: {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch; stopping.")

    tokenizer.save_pretrained(out_dir)
    hf_model.config.save_pretrained(out_dir)


def quantize(out_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(out_dir / "onnx" / "model.onnx"), str(out_dir / "onnx" / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8)


def evaluate(onnx_path: Path, tokenizer, texts: list[str], labels: np.ndarray) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probs = []
    for i in range(0, len(texts), 32):
        enc = tokenizer(texts[i:i + 32], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="np")
        logits = session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                          "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs.append(e[:, 1] / e.sum(axis=1))
    probs = np.concatenate(probs)

    # single-review latency, the way the demo uses it
    timings = []
    for text in texts[:100]:
        enc = tokenizer([text], truncation=True, max_length=MAX_LENGTH, return_tensors="np")
        start = time.perf_counter()
        session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                 "attention_mask": enc["attention_mask"].astype(np.int64)})
        timings.append((time.perf_counter() - start) * 1000)

    m = classification_metrics(labels, (probs >= 0.5).astype(int))
    return {"size_mb": round(onnx_path.stat().st_size / 1e6, 1),
            "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
            "latency_ms_p50_single": round(float(np.percentile(timings, 50)), 1),
            "probs": probs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", default="models/onnx", type=Path)
    parser.add_argument("--test", default="data/processed/test.csv", type=Path)
    parser.add_argument("--report", default="reports/onnx_metrics.json", type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("Exporting to ONNX...")
    export(args.model_id, args.out)
    print("Quantizing to int8...")
    quantize(args.out)

    tokenizer = AutoTokenizer.from_pretrained(args.out)
    test = pd.read_csv(args.test)
    texts, labels = test["text"].tolist(), test["label"].to_numpy()
    print(f"Evaluating both models on {len(texts):,} test reviews (a few minutes)...")
    fp32 = evaluate(args.out / "onnx" / "model.onnx", tokenizer, texts, labels)
    int8 = evaluate(args.out / "onnx" / "model_quantized.onnx", tokenizer, texts, labels)
    agreement = float(((fp32.pop("probs") >= 0.5) == (int8.pop("probs") >= 0.5)).mean())

    report = {"model_id": args.model_id, "test_reviews": len(texts), "onnx_fp32": fp32, "onnx_int8": int8,
              "int8_vs_fp32_prediction_agreement": round(agreement, 4)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(folder_path=str(args.out / "onnx"), path_in_repo="onnx", repo_id=args.model_id,
                              commit_message="Add ONNX export (fp32 + int8) for Transformers.js")
        print(f"Uploaded to https://huggingface.co/{args.model_id}/tree/main/onnx")


if __name__ == "__main__":
    main()
