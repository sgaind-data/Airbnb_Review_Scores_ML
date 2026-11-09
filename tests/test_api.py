import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from airbnb_nlp.api import MAX_TEXTS, create_app  # noqa: E402
from airbnb_nlp.monitoring import PredictionMonitor  # noqa: E402


class FakePredictor:
    """Stands in for DistilBERT so API tests run in seconds without downloading a model."""

    model_version = "fake@test"

    def predict(self, texts):
        return [0.1 if "dirty" in t else 0.9 for t in texts]


@pytest.fixture
def client(tmp_path):
    app = create_app(FakePredictor(), PredictionMonitor(tmp_path / "p.jsonl"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "fake@test"


def test_predict_labels_and_probabilities(client):
    r = client.post("/predict", json={"texts": ["Lovely stay!", "The sheets were dirty."]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["label"] for p in preds] == ["top_tier", "below_par"]
    assert r.headers["x-request-id"]


@pytest.mark.parametrize("payload", [
    {"texts": []},
    {"texts": ["   "]},
    {"texts": ["x" * 5001]},
    {"texts": ["ok"] * (MAX_TEXTS + 1)},
    {"wrong": ["ok"]},
])
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_stats_track_predictions(client):
    client.post("/predict", json={"texts": ["Lovely", "dirty room", "Great host"]})
    s = client.get("/stats").json()
    assert s["predictions_total"] == 3
    assert s["share_top_tier"] == round(2 / 3, 4)


def test_demo_page(client):
    assert "Airbnb review quality" in client.get("/").text
