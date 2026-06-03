import logging
from typing import Any

from fastapi import FastAPI, Request, Response

from app.core import (
    WARMUP_MODELS,
    cache_key,
    get_model_path,
    get_or_load_model,
    log_json,
    logger,
    now,
    observe,
    request_id_from_headers,
    split_cache_key,
)
from app.routes import router
from app.settings import APP_VERSION


app = FastAPI(title="Global GPU Inference Service", version=APP_VERSION)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> Response:
    request_id = request_id_from_headers(request)
    start = now()
    observe("requests_total")
    try:
        response = await call_next(request)
    except Exception:
        duration = now() - start
        log_json(
            logging.ERROR,
            "http_request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_seconds=round(duration, 6),
        )
        raise
    duration = now() - start
    response.headers["X-Request-ID"] = request_id
    log_json(
        logging.INFO,
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_seconds=round(duration, 6),
    )
    return response


@app.on_event("startup")
async def startup_warmup() -> None:
    if not WARMUP_MODELS:
        return

    for item in WARMUP_MODELS:
        project_name, model_name = split_cache_key(item)
        model_path = get_model_path(project_name, model_name)
        key = cache_key(project_name, model_name)
        await get_or_load_model(key, model_path)
        logger.info("startup warmup completed: %s", key)


app.include_router(router)
