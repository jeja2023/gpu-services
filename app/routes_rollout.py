from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core import *
from app.model_config_writer import configure_weighted_alias_rollout, rollback_alias_target, switch_alias_target


router = APIRouter()


def alias_view(alias_name: str, alias_config: Any) -> dict[str, Any]:
    try:
        resolution = alias_resolution(alias_name, alias_config)
        target = resolution["target"]
        ok = True
        error = None
    except Exception as exc:
        resolution = None
        target = None
        ok = False
        error = str(exc)
    return {
        "alias": alias_name,
        "target": target,
        "previous_target": alias_config.get("previous_target") if isinstance(alias_config, dict) else None,
        "resolution": resolution,
        "config": alias_config,
        "ok": ok,
        "error": error,
    }


@router.get("/rollout/aliases", dependencies=[Depends(require_api_token)])
async def rollout_aliases() -> dict[str, Any]:
    aliases = [alias_view(name, config) for name, config in sorted(MODEL_ALIASES.items())]
    return {
        "status": "success",
        "config_path": str(MODEL_CONFIG_PATH),
        "aliases": aliases,
        "count": len(aliases),
    }


@router.get("/rollout/aliases/preview", dependencies=[Depends(require_api_token)])
async def rollout_alias_preview(
    alias_name: str = Query(..., min_length=1, max_length=128),
    traffic_key: str = Query(..., min_length=1, max_length=256),
) -> dict[str, Any]:
    validate_path_name(alias_name)
    alias_config = MODEL_ALIASES.get(alias_name)
    if alias_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"alias not found: {alias_name}")
    resolution = alias_resolution(alias_name, alias_config, traffic_key=traffic_key)
    project, model = split_cache_key(str(resolution["target"]))
    return {
        "status": "success",
        "alias": alias_name,
        "traffic_key": traffic_key,
        "target": resolution["target"],
        "project_name": project,
        "model_name": model,
        "resolution": resolution,
    }


@router.post("/rollout/aliases/switch", dependencies=[Depends(require_api_token)])
async def rollout_alias_switch(req: AliasSwitchRequest) -> dict[str, Any]:
    project, model = split_cache_key(req.target_model_id)
    get_model_path(project, model)
    result = switch_alias_target(
        alias_name=req.alias_name,
        target_model_id=req.target_model_id,
        expected_current_target=req.expected_current_target,
        dry_run=req.dry_run,
    )
    if not req.dry_run:
        reload_model_config_state()
    return {
        "status": "success",
        **result,
        "aliases": [alias_view(name, config) for name, config in sorted(MODEL_ALIASES.items())],
    }


@router.post("/rollout/aliases/weighted", dependencies=[Depends(require_api_token)])
async def rollout_alias_weighted(req: AliasWeightedRolloutRequest) -> dict[str, Any]:
    for item in req.targets:
        project, model = split_cache_key(item.target_model_id)
        get_model_path(project, model)
    result = configure_weighted_alias_rollout(
        alias_name=req.alias_name,
        targets=[item.model_dump() for item in req.targets],
        expected_current_target=req.expected_current_target,
        dry_run=req.dry_run,
    )
    if not req.dry_run:
        reload_model_config_state()
    return {
        "status": "success",
        **result,
        "aliases": [alias_view(name, config) for name, config in sorted(MODEL_ALIASES.items())],
    }


@router.post("/rollout/aliases/rollback", dependencies=[Depends(require_api_token)])
async def rollout_alias_rollback(req: AliasRollbackRequest) -> dict[str, Any]:
    dry_result = rollback_alias_target(alias_name=req.alias_name, dry_run=True)
    project, model = split_cache_key(dry_result["new_target"])
    get_model_path(project, model)
    result = dry_result if req.dry_run else rollback_alias_target(alias_name=req.alias_name, dry_run=False)
    if not req.dry_run:
        reload_model_config_state()
    return {
        "status": "success",
        **result,
        "aliases": [alias_view(name, config) for name, config in sorted(MODEL_ALIASES.items())],
    }
