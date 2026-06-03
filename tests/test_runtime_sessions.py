import pytest

from app import runtime_sessions


def test_session_providers_rejects_tensorrt_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(runtime_sessions, "ENABLE_TENSORRT", False)
    monkeypatch.setattr(runtime_sessions, "model_config", lambda key: {"runtime": "tensorrt"})

    with pytest.raises(RuntimeError, match="ENABLE_TENSORRT"):
        runtime_sessions.session_providers("project/model.onnx")


def test_session_providers_defaults_to_cuda(monkeypatch) -> None:
    monkeypatch.setattr(runtime_sessions, "model_config", lambda key: {"runtime": "onnxruntime"})

    providers = runtime_sessions.session_providers("project/model.onnx")

    assert providers[0][0] == "CUDAExecutionProvider"
