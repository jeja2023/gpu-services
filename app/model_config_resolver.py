import hashlib
from typing import Any

from fastapi import HTTPException, status

from app.model_config_state import MODEL_ALIASES
from app.model_refs import cache_key, split_cache_key, validate_path_name


def rollout_candidates(alias_config: dict[str, Any]) -> list[dict[str, Any]]:
    rollout = alias_config.get("rollout")
    if isinstance(rollout, dict):
        rollout = rollout.get("targets") or rollout.get("candidates")
    if not isinstance(rollout, list):
        return []

    candidates: list[dict[str, Any]] = []
    for item in rollout:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        if not isinstance(target, str) or not target.strip():
            continue
        weight = item.get("weight", 0)
        try:
            weight_value = max(0, int(weight))
        except (TypeError, ValueError):
            weight_value = 0
        candidates.append(
            {
                "target": target.strip(),
                "weight": weight_value,
                "status": item.get("status"),
            }
        )
    if candidates and not any(item["weight"] > 0 for item in candidates):
        for item in candidates:
            item["weight"] = 1
    return candidates


def weighted_rollout_target(
    alias_name: str,
    candidates: list[dict[str, Any]],
    traffic_key: str,
) -> tuple[str, int, int]:
    total_weight = sum(int(item["weight"]) for item in candidates if int(item["weight"]) > 0)
    if total_weight <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"alias rollout has no positive weights: {alias_name}")
    digest = hashlib.sha256(f"{alias_name}:{traffic_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % total_weight
    cursor = 0
    for item in candidates:
        weight = int(item["weight"])
        if weight <= 0:
            continue
        cursor += weight
        if bucket < cursor:
            return str(item["target"]), bucket, total_weight
    return str(candidates[-1]["target"]), bucket, total_weight


def alias_resolution(alias_name: str, alias_config: Any, traffic_key: str | None = None) -> dict[str, Any]:
    if isinstance(alias_config, str):
        return {"alias": alias_name, "target": alias_config, "strategy": "static", "traffic_key": traffic_key}
    if not isinstance(alias_config, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid alias config: {alias_name}")

    target = alias_config.get("target")
    if isinstance(target, str) and target.strip():
        return {"alias": alias_name, "target": target.strip(), "strategy": "static", "traffic_key": traffic_key}

    project_name = alias_config.get("project_name")
    model_name = alias_config.get("model_name")
    if isinstance(project_name, str) and isinstance(model_name, str):
        target = cache_key(validate_path_name(project_name), validate_path_name(model_name))
        return {"alias": alias_name, "target": target, "strategy": "static", "traffic_key": traffic_key}

    candidates = rollout_candidates(alias_config)
    if candidates:
        if traffic_key:
            target, bucket, total_weight = weighted_rollout_target(alias_name, candidates, traffic_key)
            return {
                "alias": alias_name,
                "target": target,
                "strategy": "weighted",
                "traffic_key": traffic_key,
                "bucket": bucket,
                "total_weight": total_weight,
                "candidates": candidates,
            }
        selected = max(candidates, key=lambda item: int(item["weight"]))
        return {
            "alias": alias_name,
            "target": str(selected["target"]),
            "strategy": "highest_weight",
            "traffic_key": traffic_key,
            "candidates": candidates,
        }

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"alias has no target: {alias_name}")


def alias_target(alias_name: str, alias_config: Any, traffic_key: str | None = None) -> str:
    return str(alias_resolution(alias_name, alias_config, traffic_key=traffic_key)["target"])


def resolve_model_reference(
    model_id: str | None = None,
    project_name: str | None = None,
    model_name: str | None = None,
    traffic_key: str | None = None,
) -> tuple[str, str, str, str | None]:
    if project_name and model_name:
        try:
            project = validate_path_name(project_name)
            model = validate_path_name(model_name)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return project, model, cache_key(project, model), model_id

    if not model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="provide model_id or both project_name and model_name",
        )

    model_ref = model_id.strip()
    alias_name: str | None = None
    if model_ref in MODEL_ALIASES:
        alias_name = model_ref
        model_ref = alias_target(model_ref, MODEL_ALIASES[model_ref], traffic_key=traffic_key)

    project, model = split_cache_key(model_ref)
    return project, model, cache_key(project, model), alias_name
