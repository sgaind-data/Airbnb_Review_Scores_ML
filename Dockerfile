# Production image for the review classifier API (CPU only).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    MODEL_ID=sgaind-dataai/distilbert-airbnb-listing-quality \
    PREDICTION_LOG=/app/logs/predictions.jsonl \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[api,inference]"

# Non-root user (also required by Hugging Face Spaces).
RUN useradd -m -u 1000 app && mkdir -p /app/logs /app/.cache && chown -R app /app
USER app

# Download the model at build time so containers start fast and work offline.
RUN python -c "from airbnb_nlp.inference import Predictor; Predictor()"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn airbnb_nlp.api:app --host 0.0.0.0 --port ${PORT}"]
