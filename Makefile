# Shortcuts. Run `make help` to list them.
.PHONY: help setup test lint data baseline ner api load-test drift onnx

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:     ## install everything for local work (inside .venv)
	python -m pip install -e ".[baseline,ner,inference,api,dev]" && python -m spacy download en_core_web_sm

test:      ## lint + unit tests
	python -m ruff check src tests scripts && python -m pytest

data:      ## build train/val/test splits from data/raw
	python -m airbnb_nlp.data

baseline:  ## VADER baseline on the test split
	python -m airbnb_nlp.baseline

ner:       ## spaCy amenity analysis
	python -m airbnb_nlp.ner

api:       ## run the API on http://localhost:8000
	python -m uvicorn airbnb_nlp.api:app --port 8000

load-test: ## load test a running API
	python scripts/load_test.py --url http://localhost:8000

drift:     ## compare live predictions with the test set (PSI)
	python -m airbnb_nlp.monitoring

onnx:      ## export to ONNX, quantize to int8, evaluate (add ARGS=--upload to publish)
	python scripts/export_onnx.py $(ARGS)
