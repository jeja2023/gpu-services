from typing import Any

import yaml

from app.observability import logger
from app.schemas import ModelConfig
from app.settings import MODEL_CONFIG_PATH


def load_model_config_document() -> tuple[dict[str, ModelConfig], dict[str, Any]]:
    if not MODEL_CONFIG_PATH.is_file():
        logger.info("model config file not found, using built-in defaults: %s", MODEL_CONFIG_PATH)
        return {}, {}
    try:
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception:
        logger.exception("failed to read model config file: %s", MODEL_CONFIG_PATH)
        return {}, {}

    if not isinstance(raw, dict):
        logger.warning("model config file root must be a mapping: %s", MODEL_CONFIG_PATH)
        return {}, {}

    models = raw.get("models", raw)
    if not isinstance(models, dict):
        logger.warning("model config file has no models mapping: %s", MODEL_CONFIG_PATH)
        models = {}
    aliases = raw.get("aliases", {})
    if not isinstance(aliases, dict):
        logger.warning("model config aliases must be a mapping: %s", MODEL_CONFIG_PATH)
        aliases = {}
    return (
        {str(key): normalize_model_config(str(key), value) for key, value in models.items() if isinstance(value, dict)},
        {str(key): value for key, value in aliases.items()},
    )


def normalize_model_config(cache_key_value: str, raw_config: dict[str, Any]) -> ModelConfig:
    config: ModelConfig = dict(raw_config)
    model_type = str(config.get("type") or config.get("task") or "").strip().lower()
    if model_type in {"yolo", "yolov8", "detector"}:
        config.setdefault("task", "detection")
        config.setdefault("type", "yolo")
    elif model_type in {"classification", "classifier", "image_classification"}:
        config.setdefault("task", "classification")
        config.setdefault("type", "classification")
    elif model_type in {"reid", "embedding", "embeddings"}:
        config.setdefault("task", "reid")
        config.setdefault("type", "reid")

    input_section = config.get("input")
    if not isinstance(input_section, dict):
        input_section = {}
        config["input"] = input_section
    output_section = config.get("output")
    if not isinstance(output_section, dict):
        output_section = {}
        config["output"] = output_section

    if "input_size" in config and "size" not in input_section:
        input_section["size"] = config["input_size"]
    if "confidence" in config and "confidence" not in output_section:
        output_section["confidence"] = config["confidence"]
    if "iou" in config and "iou" not in output_section:
        output_section["iou"] = config["iou"]
    if "classes" in config and "classes" not in output_section:
        output_section["classes"] = config["classes"]
    if "normalize" in config and "normalize" not in input_section:
        input_section["normalize"] = config["normalize"]

    artifact_section = config.get("artifact")
    if not isinstance(artifact_section, dict):
        config["artifact"] = {}
    rollout_section = config.get("rollout")
    if not isinstance(rollout_section, dict):
        config["rollout"] = {}

    if not config.get("task"):
        logger.warning("model config has no task/type and will need explicit task routing: %s", cache_key_value)
    return config
