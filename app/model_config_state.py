from typing import Any

from app.model_config_loader import load_model_config_document
from app.schemas import ModelConfig


MODEL_CONFIGS, MODEL_ALIASES = load_model_config_document()


def reload_model_config_state() -> tuple[dict[str, ModelConfig], dict[str, Any]]:
    models, aliases = load_model_config_document()
    MODEL_CONFIGS.clear()
    MODEL_CONFIGS.update(models)
    MODEL_ALIASES.clear()
    MODEL_ALIASES.update(aliases)
    return MODEL_CONFIGS, MODEL_ALIASES
