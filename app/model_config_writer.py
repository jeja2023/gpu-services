from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, status

from app.model_config_resolver import alias_target
from app.model_refs import split_cache_key, validate_path_name
from app.rollout_audit import write_rollout_audit
from app.settings import MODEL_CONFIG_PATH


def load_raw_model_config() -> dict[str, Any]:
    if not MODEL_CONFIG_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model config file not found: {MODEL_CONFIG_PATH}",
        )
    try:
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to read model config file: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model config root must be a mapping")
    return raw


def write_raw_model_config(raw: dict[str, Any]) -> None:
    try:
        MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_CONFIG_PATH.open("w", encoding="utf-8") as file:
            yaml.safe_dump(raw, file, allow_unicode=True, sort_keys=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to write model config file: {exc}",
        ) from exc


def models_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    models = raw.get("models", raw)
    if not isinstance(models, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="models must be a mapping")
    return models


def aliases_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = raw.get("aliases")
    if aliases is None:
        aliases = {}
        raw["aliases"] = aliases
    if not isinstance(aliases, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="aliases must be a mapping")
    return aliases


def current_alias_target(alias_name: str, alias_config: Any) -> str:
    try:
        return alias_target(alias_name, alias_config)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failed to resolve alias {alias_name}: {exc}",
        ) from exc


def validate_configured_target(target_model_id: str, models: dict[str, Any]) -> None:
    split_cache_key(target_model_id)
    if target_model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"target model is not configured in models.yml: {target_model_id}",
        )


def switch_alias_target(
    alias_name: str,
    target_model_id: str,
    expected_current_target: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    validate_path_name(alias_name)
    raw = load_raw_model_config()
    models = models_mapping(raw)
    aliases = aliases_mapping(raw)
    validate_configured_target(target_model_id, models)

    old_config = aliases.get(alias_name)
    old_target = current_alias_target(alias_name, old_config) if old_config is not None else None
    if expected_current_target is not None and old_target != expected_current_target:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "alias current target does not match expected_current_target",
                "alias": alias_name,
                "expected_current_target": expected_current_target,
                "actual_current_target": old_target,
            },
        )

    if isinstance(old_config, dict):
        next_config = dict(old_config)
    else:
        next_config = {}
    next_config["target"] = target_model_id
    if old_target and old_target != target_model_id:
        next_config["previous_target"] = old_target
    aliases[alias_name] = next_config

    result = {
        "alias": alias_name,
        "old_target": old_target,
        "new_target": target_model_id,
        "dry_run": dry_run,
        "config_path": str(MODEL_CONFIG_PATH),
        "would_write": dry_run,
        "written": not dry_run,
    }

    if not dry_run:
        write_raw_model_config(raw)
        write_rollout_audit("alias_switch", result)

    return result


def configure_weighted_alias_rollout(
    alias_name: str,
    targets: list[dict[str, Any]],
    expected_current_target: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    validate_path_name(alias_name)
    raw = load_raw_model_config()
    models = models_mapping(raw)
    aliases = aliases_mapping(raw)
    if not targets:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="targets must not be empty")

    rollout_targets = []
    total_weight = 0
    for item in targets:
        target_model_id = str(item.get("target_model_id") or item.get("target") or "").strip()
        weight = int(item.get("weight", 0))
        if weight < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target weights must be >= 0")
        validate_configured_target(target_model_id, models)
        total_weight += weight
        rollout_item: dict[str, Any] = {"target": target_model_id, "weight": weight}
        if item.get("status"):
            rollout_item["status"] = item["status"]
        rollout_targets.append(rollout_item)
    if total_weight <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="total rollout weight must be > 0")

    old_config = aliases.get(alias_name)
    old_target = current_alias_target(alias_name, old_config) if old_config is not None else None
    if expected_current_target is not None and old_target != expected_current_target:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "alias current target does not match expected_current_target",
                "alias": alias_name,
                "expected_current_target": expected_current_target,
                "actual_current_target": old_target,
            },
        )

    next_config = dict(old_config) if isinstance(old_config, dict) else {}
    next_config.pop("target", None)
    next_config["rollout"] = rollout_targets
    if old_target:
        next_config["previous_target"] = old_target
    aliases[alias_name] = next_config

    result = {
        "alias": alias_name,
        "old_target": old_target,
        "rollout": rollout_targets,
        "total_weight": total_weight,
        "dry_run": dry_run,
        "config_path": str(MODEL_CONFIG_PATH),
        "would_write": dry_run,
        "written": not dry_run,
    }

    if not dry_run:
        write_raw_model_config(raw)
        write_rollout_audit("alias_weighted_rollout", result)

    return result


def rollback_alias_target(alias_name: str, dry_run: bool = False) -> dict[str, Any]:
    validate_path_name(alias_name)
    raw = load_raw_model_config()
    models = models_mapping(raw)
    aliases = aliases_mapping(raw)
    if alias_name not in aliases:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"alias not found: {alias_name}")
    alias_config = aliases[alias_name]
    current_target = current_alias_target(alias_name, alias_config)
    if not isinstance(alias_config, dict) or not isinstance(alias_config.get("previous_target"), str):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"alias has no previous_target: {alias_name}")

    rollback_target = alias_config["previous_target"].strip()
    validate_configured_target(rollback_target, models)
    alias_config["target"] = rollback_target
    alias_config["previous_target"] = current_target
    aliases[alias_name] = alias_config

    result = {
        "alias": alias_name,
        "old_target": current_target,
        "new_target": rollback_target,
        "dry_run": dry_run,
        "config_path": str(Path(MODEL_CONFIG_PATH)),
        "would_write": dry_run,
        "written": not dry_run,
    }

    if not dry_run:
        write_raw_model_config(raw)
        write_rollout_audit("alias_rollback", result)

    return result
