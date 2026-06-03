from fastapi.testclient import TestClient

from main import app


def test_openapi_keeps_core_routes() -> None:
    schema = app.openapi()
    paths = set(schema["paths"])

    required_paths = {
        "/health",
        "/ready",
        "/ready/deep",
        "/models",
        "/model-configs",
        "/model-package",
        "/predict",
        "/infer/persons",
        "/infer/person-embeddings",
        "/infer/person-tracks",
        "/infer/video/person-tracks",
        "/infer/stream/person-tracks",
        "/vision/infer",
        "/vision/batch-infer",
        "/debug/model-output",
        "/rollout/aliases",
        "/rollout/aliases/preview",
        "/rollout/aliases/switch",
        "/rollout/aliases/weighted",
        "/rollout/aliases/rollback",
    }

    assert required_paths <= paths


def test_health_endpoint_is_public() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "available_providers" in payload
