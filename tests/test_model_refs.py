import pytest
from fastapi import HTTPException

from app.model_refs import cache_key, split_cache_key, validate_path_name


def test_validate_path_name_rejects_path_segments() -> None:
    for value in ["..", ".", "../model.onnx", "nested/model.onnx", "nested\\model.onnx"]:
        with pytest.raises(ValueError):
            validate_path_name(value)


def test_split_cache_key_validates_format() -> None:
    assert split_cache_key("project/model.onnx") == ("project", "model.onnx")
    assert cache_key("project", "model.onnx") == "project/model.onnx"

    with pytest.raises(HTTPException):
        split_cache_key("project/nested/model.onnx")
