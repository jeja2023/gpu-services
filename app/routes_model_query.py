import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core import *


router = APIRouter()


@router.get("/models", dependencies=[Depends(require_api_token)])
async def models() -> dict[str, Any]:
    return {
        "loaded_models": [
            bundle_info(key, bundle) for key, bundle in MODEL_REGISTRY.items()
        ],
        "count": len(MODEL_REGISTRY),
        "max_loaded_models": MAX_LOADED_MODELS,
    }


@router.get("/model-configs", dependencies=[Depends(require_api_token)])
async def model_configs() -> dict[str, Any]:
    return {
        "config_path": str(MODEL_CONFIG_PATH),
        "models": MODEL_CONFIGS,
        "aliases": MODEL_ALIASES,
        "count": len(MODEL_CONFIGS),
        "alias_count": len(MODEL_ALIASES),
    }


@router.post("/reload-config", dependencies=[Depends(require_api_token)])
async def reload_config() -> dict[str, Any]:
    model_configs, model_aliases = reload_model_config_state()
    return {
        "status": "success",
        "config_path": str(MODEL_CONFIG_PATH),
        "models": model_configs,
        "aliases": model_aliases,
        "count": len(model_configs),
        "alias_count": len(model_aliases),
    }


@router.get("/model-info", dependencies=[Depends(require_api_token)])
async def model_info(
    project_name: str = Query(..., min_length=1, max_length=128),
    model_name: str = Query(..., min_length=1, max_length=256),
) -> dict[str, Any]:
    try:
        project_name = validate_path_name(project_name)
        model_name = validate_path_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    req = ModelRequest(project_name=project_name, model_name=model_name)
    key = cache_key(req.project_name, req.model_name)
    model_path = get_model_path(req.project_name, req.model_name)
    bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
    return {
        **bundle_info(key, bundle),
        "cold_loaded": cold_loaded,
        "load_seconds": load_seconds,
        "config": model_config(key),
        "package": model_package_info(key, model_path, bundle["model_hash"]),
    }


@router.get("/model-package", dependencies=[Depends(require_api_token)])
async def model_package(
    model_id: str | None = Query(None),
    project_name: str | None = Query(None, min_length=1, max_length=128),
    model_name: str | None = Query(None, min_length=1, max_length=256),
    traffic_key: str | None = Query(None, min_length=1, max_length=256),
) -> dict[str, Any]:
    project, model, key, alias_name = resolve_model_reference(model_id, project_name, model_name, traffic_key=traffic_key)
    model_path = get_model_path(project, model)
    digest = await asyncio.to_thread(model_hash, model_path)
    return {
        "status": "success",
        "model": {
            "id": alias_name or model_id or key,
            "alias": alias_name,
            "traffic_key": traffic_key if alias_name else None,
            "project_name": project,
            "model_name": model,
            "key": key,
            "path": str(model_path),
            "hash": digest,
        },
        "config": model_config(key),
        "package": model_package_info(key, model_path, digest),
    }
