from typing import Any

import onnxruntime as ort

from app.model_config_state import MODEL_CONFIGS
from app.schemas import ModelConfig


def model_config(cache_key_value: str, default_type: str | None = None) -> ModelConfig:
    config: ModelConfig = dict(MODEL_CONFIGS.get(cache_key_value, {}))
    if default_type and "type" not in config:
        config["type"] = default_type
    if default_type and "task" not in config:
        if default_type == "yolo":
            config["task"] = "detection"
        elif default_type in {"classification", "classifier"}:
            config["task"] = "classification"
        else:
            config["task"] = default_type
    return config


def config_section(config: ModelConfig, name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def config_value(config: ModelConfig, section: str, key: str, default: Any = None) -> Any:
    section_value = config_section(config, section).get(key)
    if section_value is not None:
        return section_value
    return config.get(key, default)


def model_task(config: ModelConfig, default: str = "detection") -> str:
    task = str(config.get("task") or config.get("type") or default).strip().lower()
    if task in {"yolo", "yolov8", "detector"}:
        return "detection"
    if task in {"classifier", "image_classification"}:
        return "classification"
    if task in {"embedding", "embeddings"}:
        return "reid"
    return task


def configured_sha256(config: ModelConfig) -> str | None:
    raw = config_section(config, "artifact").get("sha256") or config.get("sha256")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return None


def parse_image_size(session: ort.InferenceSession, default: tuple[int, int] = (640, 640)) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    height = shape[2] if len(shape) > 2 else None
    width = shape[3] if len(shape) > 3 else None
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return default


def configured_input_size(
    cache_key_value: str,
    session: ort.InferenceSession,
    default: tuple[int, int],
) -> tuple[int, int]:
    config = model_config(cache_key_value)
    raw_size = config_value(config, "input", "size", config.get("input_size"))
    if isinstance(raw_size, list) and len(raw_size) == 2:
        height, width = raw_size
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return height, width
    return parse_image_size(session, default=default)
