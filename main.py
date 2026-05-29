import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field, field_validator


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("gpu-worker")

PROJECTS_ROOT = Path(os.getenv("PROJECTS_ROOT", "/workspace/projects")).resolve()
MAX_TENSOR_ITEMS = parse_int_env("MAX_TENSOR_ITEMS", 12_582_912)
MAX_LOADED_MODELS = parse_int_env("MAX_LOADED_MODELS", 0)
GPU_QUEUE_LIMIT = parse_int_env("GPU_QUEUE_LIMIT", 1)
WARMUP_MODELS = [
    item.strip()
    for item in os.getenv("WARMUP_MODELS", "").split(",")
    if item.strip()
]
API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI(title="Global GPU Inference Service", version="1.1.0")


class ModelBundle(TypedDict):
    session: ort.InferenceSession
    lock: asyncio.Lock
    path: str
    model_hash: str
    file_size: int
    loaded_at: float
    last_used_at: float
    load_count: int
    inference_count: int


class InferenceRequest(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=128)
    model_name: str = Field(..., min_length=1, max_length=256)
    tensor_data: list[Any] = Field(..., min_length=1)

    @field_validator("project_name", "model_name")
    @classmethod
    def reject_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class ModelRequest(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=128)
    model_name: str = Field(..., min_length=1, max_length=256)

    @field_validator("project_name", "model_name")
    @classmethod
    def reject_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class WarmupRequest(BaseModel):
    models: list[ModelRequest] = Field(..., min_length=1)


MODEL_REGISTRY: "OrderedDict[str, ModelBundle]" = OrderedDict()
MODEL_LOAD_LOCKS: dict[str, asyncio.Lock] = {}
REGISTRY_LOCK = asyncio.Lock()
GPU_SEMAPHORE = asyncio.Semaphore(max(1, GPU_QUEUE_LIMIT))

METRICS: dict[str, float] = {
    "requests_total": 0,
    "predict_requests_total": 0,
    "predict_errors_total": 0,
    "model_loads_total": 0,
    "model_load_errors_total": 0,
    "cache_hits_total": 0,
    "cache_misses_total": 0,
    "model_unloads_total": 0,
    "inference_seconds_sum": 0,
    "queue_seconds_sum": 0,
    "model_load_seconds_sum": 0,
}

CUDA_PROVIDERS: list[Any] = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    ),
    "CPUExecutionProvider",
]


def now() -> float:
    return time.perf_counter()


def wall_time() -> float:
    return time.time()


def observe(metric: str, value: float = 1) -> None:
    METRICS[metric] = METRICS.get(metric, 0) + value


def cache_key(project_name: str, model_name: str) -> str:
    return f"{project_name}/{model_name}"


def validate_path_name(value: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("path separators and relative path segments are not allowed")
    return value


def split_cache_key(value: str) -> tuple[str, str]:
    parts = value.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model must use 'project_name/model_name' format",
        )
    try:
        return validate_path_name(parts[0]), validate_path_name(parts[1])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def get_model_path(project_name: str, model_name: str) -> Path:
    model_path = (PROJECTS_ROOT / project_name / "models" / model_name).resolve()
    try:
        model_path.relative_to(PROJECTS_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model path must stay inside the shared projects directory",
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


def io_meta(session: ort.InferenceSession) -> dict[str, Any]:
    return {
        "inputs": [
            {
                "name": item.name,
                "type": item.type,
                "shape": list(item.shape),
            }
            for item in session.get_inputs()
        ],
        "outputs": [
            {
                "name": item.name,
                "type": item.type,
                "shape": list(item.shape),
            }
            for item in session.get_outputs()
        ],
    }


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
        "providers": session.get_providers(),
        **io_meta(session),
    }


async def get_model_load_lock(cache_key_value: str) -> asyncio.Lock:
    async with REGISTRY_LOCK:
        lock = MODEL_LOAD_LOCKS.get(cache_key_value)
        if lock is None:
            lock = asyncio.Lock()
            MODEL_LOAD_LOCKS[cache_key_value] = lock
        return lock


def create_session(model_path: Path) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider is not available. available providers: {sorted(available)}"
        )

    session = ort.InferenceSession(str(model_path), providers=CUDA_PROVIDERS)
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"model session did not enable CUDA. active providers: {active}")
    return session


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
            session = await asyncio.to_thread(create_session, model_path)
            digest = await asyncio.to_thread(model_hash, model_path)
        except Exception as exc:
            observe("model_load_errors_total")
            logger.exception("failed to load model: %s", cache_key_value)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"failed to load model into GPU: {exc}",
            ) from exc

        load_seconds = now() - start
        stat = model_path.stat()
        bundle = {
            "session": session,
            "lock": asyncio.Lock(),
            "path": str(model_path),
            "model_hash": digest,
            "file_size": stat.st_size,
            "loaded_at": wall_time(),
            "last_used_at": wall_time(),
            "load_count": 1,
            "inference_count": 0,
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


def input_dtype(input_type: str) -> Any:
    if "double" in input_type:
        return np.float64
    if "float16" in input_type:
        return np.float16
    if "int64" in input_type:
        return np.int64
    if "int32" in input_type:
        return np.int32
    if "bool" in input_type:
        return np.bool_
    return np.float32


def build_input_array(tensor_data: list[Any], dtype: Any) -> np.ndarray:
    input_array = np.asarray(tensor_data, dtype=dtype)
    if input_array.size > MAX_TENSOR_ITEMS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"tensor is too large: {input_array.size} items, max {MAX_TENSOR_ITEMS}",
        )
    return input_array


async def require_api_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    if not API_TOKEN:
        return

    bearer = f"Bearer {API_TOKEN}"
    if authorization == bearer or x_api_key == API_TOKEN:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing API token",
    )


def request_id_from_headers(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def log_json(level: int, event: str, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, ensure_ascii=False))


def run_session(session: ort.InferenceSession, input_array: np.ndarray) -> list[np.ndarray]:
    input_meta = session.get_inputs()[0]
    return session.run(None, {input_meta.name: input_array})


def prometheus_metrics() -> str:
    loaded_models = len(MODEL_REGISTRY)
    lines = [
        "# HELP gpu_worker_requests_total Total HTTP requests observed by app middleware.",
        "# TYPE gpu_worker_requests_total counter",
        f"gpu_worker_requests_total {METRICS.get('requests_total', 0)}",
        "# HELP gpu_worker_predict_requests_total Total predict requests.",
        "# TYPE gpu_worker_predict_requests_total counter",
        f"gpu_worker_predict_requests_total {METRICS.get('predict_requests_total', 0)}",
        "# HELP gpu_worker_predict_errors_total Total predict errors.",
        "# TYPE gpu_worker_predict_errors_total counter",
        f"gpu_worker_predict_errors_total {METRICS.get('predict_errors_total', 0)}",
        "# HELP gpu_worker_model_loads_total Total successful model loads.",
        "# TYPE gpu_worker_model_loads_total counter",
        f"gpu_worker_model_loads_total {METRICS.get('model_loads_total', 0)}",
        "# HELP gpu_worker_model_load_errors_total Total failed model loads.",
        "# TYPE gpu_worker_model_load_errors_total counter",
        f"gpu_worker_model_load_errors_total {METRICS.get('model_load_errors_total', 0)}",
        "# HELP gpu_worker_cache_hits_total Total model cache hits.",
        "# TYPE gpu_worker_cache_hits_total counter",
        f"gpu_worker_cache_hits_total {METRICS.get('cache_hits_total', 0)}",
        "# HELP gpu_worker_cache_misses_total Total model cache misses.",
        "# TYPE gpu_worker_cache_misses_total counter",
        f"gpu_worker_cache_misses_total {METRICS.get('cache_misses_total', 0)}",
        "# HELP gpu_worker_model_unloads_total Total model unloads or evictions.",
        "# TYPE gpu_worker_model_unloads_total counter",
        f"gpu_worker_model_unloads_total {METRICS.get('model_unloads_total', 0)}",
        "# HELP gpu_worker_loaded_models Current loaded model count.",
        "# TYPE gpu_worker_loaded_models gauge",
        f"gpu_worker_loaded_models {loaded_models}",
        "# HELP gpu_worker_inference_seconds_sum Sum of inference execution seconds.",
        "# TYPE gpu_worker_inference_seconds_sum counter",
        f"gpu_worker_inference_seconds_sum {METRICS.get('inference_seconds_sum', 0)}",
        "# HELP gpu_worker_queue_seconds_sum Sum of queue wait seconds.",
        "# TYPE gpu_worker_queue_seconds_sum counter",
        f"gpu_worker_queue_seconds_sum {METRICS.get('queue_seconds_sum', 0)}",
        "# HELP gpu_worker_model_load_seconds_sum Sum of model load seconds.",
        "# TYPE gpu_worker_model_load_seconds_sum counter",
        f"gpu_worker_model_load_seconds_sum {METRICS.get('model_load_seconds_sum', 0)}",
    ]
    return "\n".join(lines) + "\n"


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


@app.get("/")
async def root() -> dict[str, Any]:
    return await health()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": app.version,
        "projects_root": str(PROJECTS_ROOT),
        "loaded_models": sorted(MODEL_REGISTRY.keys()),
        "available_providers": ort.get_available_providers(),
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "available_providers": available},
        )
    return {"status": "ready", "available_providers": available}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=prometheus_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/models", dependencies=[Depends(require_api_token)])
async def models() -> dict[str, Any]:
    return {
        "loaded_models": [
            bundle_info(key, bundle) for key, bundle in MODEL_REGISTRY.items()
        ],
        "count": len(MODEL_REGISTRY),
        "max_loaded_models": MAX_LOADED_MODELS,
    }


@app.get("/model-info", dependencies=[Depends(require_api_token)])
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
    }


@app.post("/warmup", dependencies=[Depends(require_api_token)])
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


@app.post("/unload", dependencies=[Depends(require_api_token)])
async def unload(req: ModelRequest) -> dict[str, Any]:
    key = cache_key(req.project_name, req.model_name)
    unloaded = await unload_model_by_key(key)
    return {"status": "success", "model": key, "unloaded": unloaded}


@app.post("/reload", dependencies=[Depends(require_api_token)])
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


@app.post("/predict", dependencies=[Depends(require_api_token)])
async def predict(req: InferenceRequest, request: Request) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    observe("predict_requests_total")
    total_start = now()
    key = cache_key(req.project_name, req.model_name)
    model_path = get_model_path(req.project_name, req.model_name)

    try:
        bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
        session = bundle["session"]
        input_meta = session.get_inputs()[0]
        input_array = build_input_array(req.tensor_data, input_dtype(input_meta.type))

        queue_start = now()
        async with GPU_SEMAPHORE:
            async with bundle["lock"]:
                queue_seconds = now() - queue_start
                inference_start = now()
                raw_outputs = await asyncio.to_thread(run_session, session, input_array)
                inference_seconds = now() - inference_start

        total_seconds = now() - total_start
        bundle["inference_count"] += 1
        await touch_model(key, bundle)
        observe("queue_seconds_sum", queue_seconds)
        observe("inference_seconds_sum", inference_seconds)
        log_json(
            logging.INFO,
            "predict_completed",
            request_id=request_id,
            model=key,
            input_shape=list(input_array.shape),
            input_dtype=str(input_array.dtype),
            cold_loaded=cold_loaded,
            queue_seconds=round(queue_seconds, 6),
            load_seconds=round(load_seconds, 6),
            inference_seconds=round(inference_seconds, 6),
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        observe("predict_errors_total")
        raise
    except Exception as exc:
        observe("predict_errors_total")
        logger.exception("inference failed for model: %s", key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"inference runtime error: {exc}",
        ) from exc

    return {
        "status": "success",
        "request_id": request_id,
        "model": key,
        "cold_loaded": cold_loaded,
        "timing": {
            "queue_seconds": queue_seconds,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
        },
        "outputs": [output.tolist() for output in raw_outputs],
    }
