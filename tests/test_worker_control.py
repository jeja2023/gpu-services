import pytest

from tools.worker_control import split_model_id


def test_split_model_id_accepts_project_model() -> None:
    assert split_model_id("project/model.onnx") == {
        "project_name": "project",
        "model_name": "model.onnx",
    }


def test_split_model_id_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        split_model_id("model.onnx")
