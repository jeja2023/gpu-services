from typing import Any

from fastapi import APIRouter, Depends

from app.core import *


router = APIRouter()


@router.post("/warmup", dependencies=[Depends(require_api_token)])
async def warmup(req: WarmupRequest) -> dict[str, Any]:
    results = []
    for model in req.models:
        key = cache_key(model.project_name, model.model_name)
        model_path = get_model_path(model.project_name, model.model_name)
        bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
        results.append(
            {
                "model": key,
                "cold_loaded": cold_loaded,
                "load_seconds": load_seconds,
                "model_hash": bundle["model_hash"],
            }
        )
    return {"status": "success", "results": results}


@router.post("/unload", dependencies=[Depends(require_api_token)])
async def unload(req: ModelRequest) -> dict[str, Any]:
    key = cache_key(req.project_name, req.model_name)
    unloaded = await unload_model_by_key(key)
    return {"status": "success", "model": key, "unloaded": unloaded}


@router.post("/reload", dependencies=[Depends(require_api_token)])
async def reload_model(req: ModelRequest) -> dict[str, Any]:
    key = cache_key(req.project_name, req.model_name)
    await unload_model_by_key(key)
    model_path = get_model_path(req.project_name, req.model_name)
    bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
    return {
        "status": "success",
        "model": key,
        "cold_loaded": cold_loaded,
        "load_seconds": load_seconds,
        "model_hash": bundle["model_hash"],
    }
