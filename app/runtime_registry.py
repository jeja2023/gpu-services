import asyncio
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.metrics import observe
from app.model_config import config_value, model_config
from app.model_package import model_hash, validate_model_hash
from app.observability import logger, now, wall_time
from app.runtime_sessions import create_session, io_meta
from app.runtime_state import MODEL_LOAD_LOCKS, MODEL_REGISTRY, REGISTRY_LOCK
from app.schemas import ModelBundle
from app.settings import MAX_LOADED_MODELS, MODEL_CONCURRENCY_LIMIT, MODEL_QUEUE_TIMEOUT_SECONDS


def bundle_info(cache_key_value: str, bundle: ModelBundle) -> dict[str, Any]:
    session = bundle["session"]
    return {
        "model": cache_key_value,
        "path": bundle["path"],
        "model_hash": bundle["model_hash"],
        "file_size": bundle["file_size"],
        "loaded_at": bundle["loaded_at"],
        "last_used_at": bundle["last_used_at"],
        "load_count": bundle["load_count"],
        "inference_count": bundle["inference_count"],
        "max_concurrency": bundle.get("max_concurrency", 1),
        "queue_timeout_seconds": bundle.get("queue_timeout_seconds", 0),
        "providers": session.get_providers(),
        **io_meta(session),
    }


def model_runtime_limits(cache_key_value: str) -> tuple[int, float]:
    config = model_config(cache_key_value)
    raw_concurrency = config_value(config, "runtime", "max_concurrency", config.get("max_concurrency", MODEL_CONCURRENCY_LIMIT))
    raw_timeout = config_value(config, "runtime", "queue_timeout_seconds", config.get("queue_timeout_seconds", MODEL_QUEUE_TIMEOUT_SECONDS))
    try:
        max_concurrency = max(1, int(raw_concurrency))
    except (TypeError, ValueError):
        max_concurrency = max(1, MODEL_CONCURRENCY_LIMIT)
    try:
        queue_timeout = max(0.0, float(raw_timeout))
    except (TypeError, ValueError):
        queue_timeout = max(0.0, MODEL_QUEUE_TIMEOUT_SECONDS)
    return max_concurrency, queue_timeout
async def get_model_load_lock(cache_key_value: str) -> asyncio.Lock:
    async with REGISTRY_LOCK:
        lock = MODEL_LOAD_LOCKS.get(cache_key_value)
        if lock is None:
            lock = asyncio.Lock()
            MODEL_LOAD_LOCKS[cache_key_value] = lock
        return lock
async def evict_lru_if_needed(except_key: str | None = None) -> None:
    if MAX_LOADED_MODELS <= 0:
        return

    async with REGISTRY_LOCK:
        while len(MODEL_REGISTRY) > MAX_LOADED_MODELS:
            evict_key = next((key for key in MODEL_REGISTRY if key != except_key), None)
            if evict_key is None:
                return
            MODEL_REGISTRY.pop(evict_key, None)
            MODEL_LOAD_LOCKS.pop(evict_key, None)
            observe("model_unloads_total")
            logger.info("evicted model from cache: %s", evict_key)


async def unload_model_by_key(cache_key_value: str) -> bool:
    async with REGISTRY_LOCK:
        removed = MODEL_REGISTRY.pop(cache_key_value, None)
        MODEL_LOAD_LOCKS.pop(cache_key_value, None)
    if removed is not None:
        observe("model_unloads_total")
        logger.info("unloaded model: %s", cache_key_value)
        return True
    return False


async def touch_model(cache_key_value: str, bundle: ModelBundle) -> None:
    bundle["last_used_at"] = wall_time()
    async with REGISTRY_LOCK:
        if cache_key_value in MODEL_REGISTRY:
            MODEL_REGISTRY.move_to_end(cache_key_value)


async def get_or_load_model(
    cache_key_value: str,
    model_path: Path,
) -> tuple[ModelBundle, bool, float]:
    bundle = MODEL_REGISTRY.get(cache_key_value)
    if bundle is not None:
        observe("cache_hits_total")
        await touch_model(cache_key_value, bundle)
        return bundle, False, 0

    observe("cache_misses_total")
    load_lock = await get_model_load_lock(cache_key_value)
    async with load_lock:
        bundle = MODEL_REGISTRY.get(cache_key_value)
        if bundle is not None:
            observe("cache_hits_total")
            await touch_model(cache_key_value, bundle)
            return bundle, False, 0

        start = now()
        logger.info("loading model: %s from %s", cache_key_value, model_path)
        try:
            digest = await asyncio.to_thread(model_hash, model_path)
            validate_model_hash(cache_key_value, digest)
            session = await asyncio.to_thread(create_session, model_path, cache_key_value)
        except HTTPException:
            observe("model_load_errors_total")
            raise
        except Exception as exc:
            observe("model_load_errors_total")
            logger.exception("failed to load model: %s", cache_key_value)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"failed to load model into GPU: {exc}",
            ) from exc

        load_seconds = now() - start
        stat = model_path.stat()
        max_concurrency, queue_timeout = model_runtime_limits(cache_key_value)
        bundle = {
            "session": session,
            "lock": asyncio.Lock(),
            "semaphore": asyncio.Semaphore(max_concurrency),
            "path": str(model_path),
            "model_hash": digest,
            "file_size": stat.st_size,
            "loaded_at": wall_time(),
            "last_used_at": wall_time(),
            "load_count": 1,
            "inference_count": 0,
            "max_concurrency": max_concurrency,
            "queue_timeout_seconds": queue_timeout,
        }
        MODEL_REGISTRY[cache_key_value] = bundle
        await touch_model(cache_key_value, bundle)
        observe("model_loads_total")
        observe("model_load_seconds_sum", load_seconds)
        await evict_lru_if_needed(except_key=cache_key_value)
        logger.info(
            "model loaded on CUDA: %s load_seconds=%.6f hash=%s",
            cache_key_value,
            load_seconds,
            digest,
        )
        return bundle, True, load_seconds
