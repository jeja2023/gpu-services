import hashlib
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, status

from app.constants import COCO_CLASSES
from app.model_config import config_section, config_value, configured_sha256, model_config, model_task
from app.observability import logger
from app.schemas import ModelConfig
from app.settings import MODELS_ROOT


def get_model_path(project_name: str, model_name: str) -> Path:
    model_path = (MODELS_ROOT / project_name / model_name).resolve()
    try:
        model_path.relative_to(MODELS_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model path must stay inside the shared models directory",
        ) from exc

    if not model_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model '{model_name}' was not found under project '{project_name}'",
        )
    return model_path


def model_hash(model_path: Path) -> str:
    digest = hashlib.sha256()
    with model_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_sidecar_path(model_path: Path, relative_path: str) -> Path:
    sidecar_path = (model_path.parent / relative_path).resolve()
    try:
        sidecar_path.relative_to(model_path.parent)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model sidecar path must stay inside the model project directory",
        ) from exc
    return sidecar_path


def load_yaml_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception:
        logger.exception("failed to read model sidecar yaml: %s", path)
        return {}
    return raw if isinstance(raw, dict) else {}


def load_text_labels(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
    except Exception:
        logger.exception("failed to read model labels: %s", path)
        return []


def labels_from_config(config: ModelConfig, model_path: Path | None = None) -> list[str]:
    raw_classes = config_value(config, "output", "classes")
    if isinstance(raw_classes, list):
        return [str(item) for item in raw_classes]
    if isinstance(raw_classes, str) and raw_classes.lower() == "coco":
        return COCO_CLASSES

    labels_path = config_section(config, "artifact").get("labels")
    if model_path is not None and isinstance(labels_path, str) and labels_path.strip():
        return load_text_labels(safe_sidecar_path(model_path, labels_path.strip()))
    if model_path is not None:
        default_labels_path = model_path.with_suffix(".labels.txt")
        labels = load_text_labels(default_labels_path)
        if labels:
            return labels
    return []


def class_name(class_id: int, labels: list[str] | None = None) -> str:
    if labels and 0 <= class_id < len(labels):
        return labels[class_id]
    return str(class_id)


def parse_class_filter(raw_filter: Any, labels: list[str]) -> set[int] | None:
    if raw_filter is None or raw_filter == "":
        return None
    if not isinstance(raw_filter, list):
        raw_filter = [raw_filter]

    label_to_id = {label.lower(): index for index, label in enumerate(labels)}
    class_ids: set[int] = set()
    for item in raw_filter:
        if isinstance(item, int):
            class_ids.add(item)
        elif isinstance(item, str):
            value = item.strip()
            if value.isdigit():
                class_ids.add(int(value))
            elif value.lower() in label_to_id:
                class_ids.add(label_to_id[value.lower()])
            elif value.lower() == "person":
                class_ids.add(0)
    return class_ids or None


def model_card_for_path(config: ModelConfig, model_path: Path) -> dict[str, Any]:
    card_path = config_section(config, "artifact").get("model_card")
    if isinstance(card_path, str) and card_path.strip():
        return load_yaml_sidecar(safe_sidecar_path(model_path, card_path.strip()))
    default_card_path = model_path.with_suffix(".model-card.yml")
    return load_yaml_sidecar(default_card_path)


def model_package_info(cache_key_value: str, model_path: Path, digest: str | None = None) -> dict[str, Any]:
    config = model_config(cache_key_value)
    labels = labels_from_config(config, model_path)
    card = model_card_for_path(config, model_path)
    expected_sha256 = configured_sha256(config)
    artifact = config_section(config, "artifact")
    return {
        "model": cache_key_value,
        "task": model_task(config),
        "type": config.get("type"),
        "runtime": config.get("runtime", "onnxruntime"),
        "version": config.get("version") or config_section(card, "model").get("version"),
        "precision": config.get("precision") or config_section(card, "model").get("precision"),
        "artifact": {
            "model_card": artifact.get("model_card"),
            "labels": artifact.get("labels"),
            "sha256": expected_sha256,
            "sha256_match": None if not expected_sha256 or not digest else expected_sha256 == digest.lower(),
        },
        "labels": {
            "count": len(labels),
            "items": labels,
        },
        "model_card": card,
    }


def validate_model_hash(cache_key_value: str, digest: str) -> None:
    expected = configured_sha256(model_config(cache_key_value))
    if expected and expected != digest.lower():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "model sha256 does not match configured artifact hash",
                "model": cache_key_value,
                "expected_sha256": expected,
                "actual_sha256": digest,
            },
        )

