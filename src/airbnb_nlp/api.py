"""REST API for the review classifier.

Endpoints
  GET  /          small demo page
  GET  /health    liveness + model info
  POST /predict   {"texts": ["..."]} -> label + probability per text
  GET  /stats     rolling monitoring summary (volume, class share, latency)
  GET  /docs      interactive OpenAPI docs

Run locally:  uvicorn airbnb_nlp.api:app --port 8000
Env vars:     MODEL_ID (Hugging Face model), PREDICTION_LOG (JSON-lines path, empty to disable)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, StringConstraints

from airbnb_nlp import DEFAULT_MODEL_ID, LABELS, __version__
from airbnb_nlp.monitoring import PredictionMonitor

log = logging.getLogger("airbnb_nlp.api")
MAX_TEXTS = 32
MAX_CHARS = 5000

ReviewText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHARS)]


class PredictorLike(Protocol):
    model_version: str

    def predict(self, texts: list[str]) -> list[float]: ...


class PredictRequest(BaseModel):
    texts: list[ReviewText] = Field(min_length=1, max_length=MAX_TEXTS,
                                    examples=[["Spotless place, wonderful host!",
                                               "The room was dusty and parking cost extra."]])


class Prediction(BaseModel):
    label: str
    prob_top_tier: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    latency_ms: float


def create_app(predictor: PredictorLike | None = None, monitor: PredictionMonitor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            from airbnb_nlp.inference import Predictor

            app.state.predictor = Predictor(os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))
        log.info("Model ready: %s", app.state.predictor.model_version)
        yield

    app = FastAPI(title="Airbnb Review Quality API", version=__version__, lifespan=lifespan,
                  description="Predicts whether a review comes from a below-par or top-tier listing.")
    app.state.predictor = predictor
    app.state.monitor = monitor or PredictionMonitor(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
                                                     or None)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        log.info("request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
                 request_id, request.method, request.url.path, response.status_code, ms)
        return response

    @app.get("/health")
    def health() -> dict:
        p = app.state.predictor
        return {"status": "ok" if p else "loading", "model_loaded": p is not None,
                "model_version": p.model_version if p else None, "api_version": __version__}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        p = app.state.predictor
        if p is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        start = time.perf_counter()
        probs = p.predict(req.texts)
        ms = (time.perf_counter() - start) * 1000
        for text, prob in zip(req.texts, probs, strict=True):
            app.state.monitor.record(prob, len(text), ms / len(req.texts), p.model_version)
        return PredictResponse(
            predictions=[Prediction(label=LABELS[int(prob >= 0.5)], prob_top_tier=round(prob, 4))
                         for prob in probs],
            model_version=p.model_version,
            latency_ms=round(ms, 2),
        )

    @app.get("/stats")
    def stats() -> dict:
        return app.state.monitor.summary()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return DEMO_PAGE

    return app


DEMO_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Airbnb Review Quality</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;padding:0 16px;color:#1f2937}
 textarea{width:100%;height:120px;font:inherit;padding:10px;border:1px solid #d1d5db;border-radius:8px}
 button{margin-top:10px;padding:10px 18px;border:0;border-radius:8px;background:#1d4ed8;color:#fff;font:inherit}
 #out{margin-top:18px;font-size:1.1rem} small{color:#6b7280} a{color:#1d4ed8}
</style></head><body>
<h1>Airbnb review quality</h1>
<p>Paste a guest review. A fine-tuned DistilBERT predicts whether it comes from a
<b>below-par</b> (rating &le; 4.6) or <b>top-tier</b> (&ge; 4.9) Dallas listing.</p>
<textarea id="t">Great location, but the sheets were dirty and we had to pay extra for parking.</textarea>
<button onclick="go()">Predict</button><div id="out"></div>
<p><small><a href="/docs">API docs</a> &middot; <a href="/stats">monitoring stats</a> &middot;
<a href="https://github.com/sgaind-data/Airbnb_Review_Scores_ML">source code</a></small></p>
<script>
async function go(){
 const out=document.getElementById('out'); out.textContent='Scoring...';
 const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({texts:[document.getElementById('t').value]})});
 if(!r.ok){out.textContent='Error '+r.status;return}
 const p=(await r.json()).predictions[0];
 out.innerHTML='<b>'+p.label.replace('_','-')+'</b> &middot; P(top-tier) = '+p.prob_top_tier.toFixed(2);
}
</script></body></html>"""

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
