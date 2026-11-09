# Hugging Face Space: installs the released package (tag v1.0.0) straight from GitHub.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.cache/huggingface \
    MODEL_ID=sgaind-dataai/distilbert-airbnb-listing-quality \
    PREDICTION_LOG=/home/user/logs/predictions.jsonl PORT=7860

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install "airbnb-nlp[api,inference] @ https://github.com/sgaind-data/Airbnb_Review_Scores_ML/archive/refs/tags/v1.0.0.zip"

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user
RUN python -c "from airbnb_nlp.inference import Predictor; Predictor()"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn airbnb_nlp.api:app --host 0.0.0.0 --port ${PORT}"]
