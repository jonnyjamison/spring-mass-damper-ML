from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert "ready" in r.json()


def test_predict_valid(client):
    payload = {"m": 1.0, "c": 0.2, "k": 2.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["num_points"] > 10
    assert len(data["displacement"]) == data["num_points"]