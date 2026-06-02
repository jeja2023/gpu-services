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
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import cv2
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("gpu-worker")

MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "/models")).resolve()
MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "models.yml"))
MAX_TENSOR_ITEMS = parse_int_env("MAX_TENSOR_ITEMS", 12_582_912)
MAX_IMAGE_BYTES = parse_int_env("MAX_IMAGE_BYTES", 10 * 1024 * 1024)
MAX_PERSON_FRAMES = parse_int_env("MAX_PERSON_FRAMES", 16)
MAX_EMBEDDING_IMAGES = parse_int_env("MAX_EMBEDDING_IMAGES", 64)
MAX_PIPELINE_FRAMES = parse_int_env("MAX_PIPELINE_FRAMES", 16)
MAX_VIDEO_BYTES = parse_int_env("MAX_VIDEO_BYTES", 100 * 1024 * 1024)
VIDEO_FRAME_INTERVAL = parse_int_env("VIDEO_FRAME_INTERVAL", 15)
MAX_VIDEO_FRAMES = parse_int_env("MAX_VIDEO_FRAMES", 64)
STREAM_FRAME_INTERVAL = parse_int_env("STREAM_FRAME_INTERVAL", 15)
MAX_STREAM_FRAMES = parse_int_env("MAX_STREAM_FRAMES", 32)
STREAM_READ_TIMEOUT_SECONDS = parse_int_env("STREAM_READ_TIMEOUT_SECONDS", 10)
ALLOW_STREAM_URLS = parse_bool_env("ALLOW_STREAM_URLS", False)
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


class LetterboxMeta(TypedDict):
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_left: float
    pad_top: float


class ModelConfig(TypedDict, total=False):
    type: str
    input_size: list[int]
    person_class_id: int
    confidence: float
    iou: float
    classes: str
    normalize: str
    embedding_normalize: str
    batch_size: int


class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_name: str = Field(..., min_length=1, max_length=128)
    model_name: str = Field(..., min_length=1, max_length=256)
    tensor_data: list[Any] = Field(..., min_length=1)

    @field_validator("project_name", "model_name")
    @classmethod
    def reject_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class ModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_name: str = Field(..., min_length=1, max_length=128)
    model_name: str = Field(..., min_length=1, max_length=256)

    @field_validator("project_name", "model_name")
    @classmethod
    def reject_path_segments(cls, value: str) -> str:
        return validate_path_name(value)


class WarmupRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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
    "persons_requests_total": 0,
    "persons_errors_total": 0,
    "persons_detected_total": 0,
    "persons_frames_total": 0,
    "embeddings_requests_total": 0,
    "embeddings_errors_total": 0,
    "tracks_requests_total": 0,
    "tracks_errors_total": 0,
    "decode_seconds_sum": 0,
    "preprocess_seconds_sum": 0,
    "postprocess_seconds_sum": 0,
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


def load_model_configs() -> dict[str, ModelConfig]:
    if not MODEL_CONFIG_PATH.is_file():
        logger.info("model config file not found, using built-in defaults: %s", MODEL_CONFIG_PATH)
        return {}
    try:
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception:
        logger.exception("failed to read model config file: %s", MODEL_CONFIG_PATH)
        return {}

    models = raw.get("models", raw)
    if not isinstance(models, dict):
        logger.warning("model config file has no models mapping: %s", MODEL_CONFIG_PATH)
        return {}
    return {str(key): value for key, value in models.items() if isinstance(value, dict)}


MODEL_CONFIGS = load_model_configs()


def model_config(cache_key_value: str, default_type: str | None = None) -> ModelConfig:
    config: ModelConfig = dict(MODEL_CONFIGS.get(cache_key_value, {}))
    if default_type and "type" not in config:
        config["type"] = default_type
    return config


def configured_input_size(
    cache_key_value: str,
    session: ort.InferenceSession,
    default: tuple[int, int],
) -> tuple[int, int]:
    config = model_config(cache_key_value)
    raw_size = config.get("input_size")
    if isinstance(raw_size, list) and len(raw_size) == 2:
        height, width = raw_size
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return height, width
    return parse_image_size(session, default=default)


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


def parse_image_size(session: ort.InferenceSession, default: tuple[int, int] = (640, 640)) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    height = shape[2] if len(shape) > 2 else None
    width = shape[3] if len(shape) > 3 else None
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return default


async def read_image_file(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"uploaded file '{file.filename}' is empty",
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"uploaded file '{file.filename}' is too large: {len(data)} bytes, max {MAX_IMAGE_BYTES}",
        )
    return data


def decode_image(data: bytes, filename: str | None) -> Image.Image:
    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"uploaded file '{filename or '<unnamed>'}' is not a valid image",
        ) from exc


def letterbox_image(image: Image.Image, input_height: int, input_width: int) -> tuple[np.ndarray, LetterboxMeta]:
    original_width, original_height = image.size
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (input_width - resized_width) / 2
    pad_top = (input_height - resized_height) / 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_left - 0.1)), int(round(pad_top - 0.1))))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))
    meta: LetterboxMeta = {
        "original_width": original_width,
        "original_height": original_height,
        "input_width": input_width,
        "input_height": input_height,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }
    return tensor, meta


def resize_image_tensor(image: Image.Image, input_height: int, input_width: int, normalize: str = "none") -> np.ndarray:
    resized = image.resize((input_width, input_height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if normalize == "imagenet":
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def restore_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - meta["pad_left"]) / meta["scale"]
    restored[:, [1, 3]] = (restored[:, [1, 3]] - meta["pad_top"]) / meta["scale"]
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, meta["original_width"])
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, meta["original_height"])
    return restored


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        inter_width = np.maximum(0, xx2 - xx1)
        inter_height = np.maximum(0, yy2 - yy1)
        intersection = inter_width * inter_height
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = rest[iou <= iou_threshold]

    return keep


def yolo_person_detections(
    raw_outputs: list[np.ndarray],
    meta: LetterboxMeta,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    person_class_id: int = 0,
) -> list[dict[str, Any]]:
    if not raw_outputs:
        return []

    output = np.asarray(raw_outputs[0])
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"unsupported YOLO output shape: {list(raw_outputs[0].shape)}")

    if output.shape[0] < output.shape[1] and output.shape[0] in {5, 6, 84, 85}:
        output = output.T

    if output.shape[1] < 5:
        raise ValueError(f"unsupported YOLO output shape: {list(raw_outputs[0].shape)}")

    boxes_xywh = output[:, :4].astype(np.float32)
    if output.shape[1] == 6:
        scores = output[:, 4].astype(np.float32)
        class_ids = output[:, 5].astype(np.int64)
    elif output.shape[1] == 5:
        scores = output[:, 4].astype(np.float32)
        class_ids = np.zeros(output.shape[0], dtype=np.int64)
    elif output.shape[1] == 84:
        class_scores = output[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        objectness = output[:, 4].astype(np.float32)
        class_scores = output[:, 5:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = (class_ids == person_class_id) & (scores >= confidence_threshold)
    if not np.any(mask):
        return []

    boxes = restore_boxes(xywh_to_xyxy(boxes_xywh[mask]), meta)
    scores = scores[mask]
    class_ids = class_ids[mask]
    keep = nms(boxes, scores, iou_threshold)[:max_detections]

    detections: list[dict[str, Any]] = []
    for index in keep:
        box = boxes[index]
        detections.append(
            {
                "box": [round(float(value), 3) for value in box.tolist()],
                "score": round(float(scores[index]), 6),
                "class_id": int(class_ids[index]),
                "class_name": "person",
            }
        )
    return detections


def normalize_embeddings(output: np.ndarray, mode: str = "l2") -> np.ndarray:
    embeddings = np.asarray(output, dtype=np.float32)
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape((embeddings.shape[0], -1))
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape((1, -1))
    if mode == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings


def crop_person(image: Image.Image, box: list[float], min_size: int = 2) -> Image.Image | None:
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right - left < min_size or bottom - top < min_size:
        return None
    return image.crop((left, top, right, bottom))


async def load_images(files: list[UploadFile]) -> tuple[list[Image.Image], list[str | None], float]:
    decode_start = now()
    images: list[Image.Image] = []
    filenames: list[str | None] = []
    for file in files:
        data = await read_image_file(file)
        image = await asyncio.to_thread(decode_image, data, file.filename)
        images.append(image)
        filenames.append(file.filename)
    return images, filenames, now() - decode_start


async def read_video_file(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"uploaded video '{file.filename}' is empty",
        )
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"uploaded video '{file.filename}' is too large: {len(data)} bytes, max {MAX_VIDEO_BYTES}",
        )
    return data


def cv_frame_to_image(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def validate_stream_url(stream_url: str) -> str:
    parsed = urlparse(stream_url)
    if parsed.scheme not in {"rtsp", "rtmp", "http", "https"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream_url must use rtsp, rtmp, http, or https",
        )
    if not parsed.netloc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="stream_url must include host")
    return stream_url


def extract_video_frames_from_path(
    source: str,
    frame_interval: int,
    max_frames: int,
    read_timeout_seconds: int | None = None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError("failed to open video source")

    start = now()
    frame_interval = max(1, frame_interval)
    max_frames = max(1, max_frames)
    frames: list[Image.Image] = []
    source_frame_indexes: list[int] = []
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0

    try:
        while len(frames) < max_frames:
            if read_timeout_seconds is not None and now() - start > read_timeout_seconds:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                frames.append(cv_frame_to_image(frame))
                source_frame_indexes.append(frame_index)
            frame_index += 1
    finally:
        capture.release()

    meta = {
        "source_frame_indexes": source_frame_indexes,
        "source_frames_read": frame_index,
        "source_frame_count": int(frame_count),
        "source_width": int(width),
        "source_height": int(height),
        "extracted_frames": len(frames),
        "fps": fps,
        "frame_interval": frame_interval,
        "max_frames": max_frames,
        "decode_seconds": now() - start,
    }
    return frames, meta


async def extract_video_frames_from_upload(
    file: UploadFile,
    frame_interval: int,
    max_frames: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    data = await read_video_file(file)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name
        frames, meta = await asyncio.to_thread(
            extract_video_frames_from_path,
            temp_path,
            frame_interval,
            max_frames,
            None,
        )
        meta["filename"] = file.filename
        meta["video_bytes"] = len(data)
        return frames, meta
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("failed to remove temp video file: %s", temp_path)


async def infer_person_frames(
    bundle: ModelBundle,
    key: str,
    images: list[Image.Image],
    filenames: list[str | None],
    confidence: float,
    iou: float,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = bundle["session"]
    config = model_config(key, default_type="yolo")
    input_height, input_width = configured_input_size(key, session, default=(640, 640))
    person_class_id = int(config.get("person_class_id", 0))

    preprocess_start = now()
    image_tensors: list[np.ndarray] = []
    image_metas: list[LetterboxMeta] = []
    for image in images:
        tensor, meta = await asyncio.to_thread(letterbox_image, image, input_height, input_width)
        image_tensors.append(tensor)
        image_metas.append(meta)
    input_array = np.stack(image_tensors, axis=0).astype(np.float32)
    preprocess_seconds = now() - preprocess_start

    raw_outputs, queue_seconds, inference_seconds, inference_mode = await run_yolo_frames(bundle, input_array)

    postprocess_start = now()
    frames = []
    output_shapes = [list(output.shape) for output in raw_outputs]
    for index, meta in enumerate(image_metas):
        frame_outputs = []
        for output in raw_outputs:
            if output.ndim > 0 and output.shape[0] == len(image_metas):
                frame_outputs.append(output[index : index + 1])
            else:
                frame_outputs.append(output)
        persons = yolo_person_detections(
            frame_outputs,
            meta,
            confidence_threshold=confidence,
            iou_threshold=iou,
            max_detections=max_detections,
            person_class_id=person_class_id,
        )
        frames.append(
            {
                "frame_index": index,
                "filename": filenames[index],
                "width": meta["original_width"],
                "height": meta["original_height"],
                "persons": persons,
                "person_count": len(persons),
            }
        )
    postprocess_seconds = now() - postprocess_start

    timing = {
        "preprocess_seconds": preprocess_seconds,
        "queue_seconds": queue_seconds,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
    }
    meta = {
        "input_shape": list(input_array.shape),
        "output_shapes": output_shapes,
        "inference_mode": inference_mode,
        "timing": timing,
    }
    return frames, meta


async def infer_reid_images(
    bundle: ModelBundle,
    key: str,
    images: list[Image.Image],
) -> tuple[np.ndarray, dict[str, Any]]:
    session = bundle["session"]
    config = model_config(key, default_type="reid")
    input_height, input_width = configured_input_size(key, session, default=(256, 128))
    normalize = str(config.get("normalize", "imagenet"))
    embedding_normalize = str(config.get("embedding_normalize", "l2"))

    preprocess_start = now()
    tensors = [
        await asyncio.to_thread(resize_image_tensor, image, input_height, input_width, normalize)
        for image in images
    ]
    input_array = np.stack(tensors, axis=0).astype(np.float32)
    preprocess_seconds = now() - preprocess_start

    raw_outputs, queue_seconds, inference_seconds, inference_mode = await run_yolo_frames(bundle, input_array)

    postprocess_start = now()
    embeddings = normalize_embeddings(raw_outputs[0], mode=embedding_normalize)
    postprocess_seconds = now() - postprocess_start

    meta = {
        "input_shape": list(input_array.shape),
        "output_shapes": [list(output.shape) for output in raw_outputs],
        "inference_mode": inference_mode,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "timing": {
            "preprocess_seconds": preprocess_seconds,
            "queue_seconds": queue_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
        },
    }
    return embeddings, meta


async def infer_tracks_for_images(
    images: list[Image.Image],
    filenames: list[str | None],
    detector_project_name: str,
    detector_model_name: str,
    reid_project_name: str,
    reid_model_name: str,
    confidence: float,
    iou: float,
    max_detections: int,
    include_embeddings: bool,
) -> dict[str, Any]:
    detector_key = cache_key(detector_project_name, detector_model_name)
    reid_key = cache_key(reid_project_name, reid_model_name)

    detector_bundle, detector_cold_loaded, detector_load_seconds = await get_or_load_model(
        detector_key,
        get_model_path(detector_project_name, detector_model_name),
    )
    reid_bundle, reid_cold_loaded, reid_load_seconds = await get_or_load_model(
        reid_key,
        get_model_path(reid_project_name, reid_model_name),
    )

    frames, detector_meta = await infer_person_frames(
        detector_bundle,
        detector_key,
        images,
        filenames,
        confidence=confidence,
        iou=iou,
        max_detections=max_detections,
    )

    crops: list[Image.Image] = []
    crop_refs: list[tuple[int, int]] = []
    for frame in frames:
        image = images[frame["frame_index"]]
        for person_index, person in enumerate(frame["persons"]):
            crop = crop_person(image, person["box"])
            if crop is not None:
                crops.append(crop)
                crop_refs.append((frame["frame_index"], person_index))

    embedding_count = 0
    if crops:
        embeddings, embedding_meta = await infer_reid_images(reid_bundle, reid_key, crops)
        embedding_count = embeddings.shape[0]
        for index, (frame_index, person_index) in enumerate(crop_refs):
            person = frames[frame_index]["persons"][person_index]
            person["embedding_dim"] = int(embeddings.shape[1])
            person["embedding_index"] = index
            if include_embeddings:
                person["embedding"] = [round(float(value), 8) for value in embeddings[index].tolist()]
    else:
        embedding_meta = {
            "input_shape": [0],
            "output_shapes": [],
            "inference_mode": "none",
            "embedding_dim": 0,
            "timing": {
                "preprocess_seconds": 0,
                "queue_seconds": 0,
                "inference_seconds": 0,
                "postprocess_seconds": 0,
            },
        }

    person_count = sum(frame["person_count"] for frame in frames)
    await touch_model(detector_key, detector_bundle)
    await touch_model(reid_key, reid_bundle)
    observe("persons_detected_total", person_count)
    observe("persons_frames_total", len(frames))
    observe("preprocess_seconds_sum", detector_meta["timing"]["preprocess_seconds"] + embedding_meta["timing"]["preprocess_seconds"])
    observe("postprocess_seconds_sum", detector_meta["timing"]["postprocess_seconds"] + embedding_meta["timing"]["postprocess_seconds"])

    return {
        "detector_key": detector_key,
        "reid_key": reid_key,
        "detector_cold_loaded": detector_cold_loaded,
        "reid_cold_loaded": reid_cold_loaded,
        "detector_load_seconds": detector_load_seconds,
        "reid_load_seconds": reid_load_seconds,
        "detector_meta": detector_meta,
        "embedding_meta": embedding_meta,
        "frames": frames,
        "person_count": person_count,
        "embedding_count": embedding_count,
    }


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


async def run_model_bundle(bundle: ModelBundle, input_array: np.ndarray) -> tuple[list[np.ndarray], float, float]:
    session = bundle["session"]
    queue_start = now()
    async with GPU_SEMAPHORE:
        async with bundle["lock"]:
            queue_seconds = now() - queue_start
            inference_start = now()
            raw_outputs = await asyncio.to_thread(run_session, session, input_array)
            inference_seconds = now() - inference_start

    bundle["inference_count"] += 1
    observe("queue_seconds_sum", queue_seconds)
    observe("inference_seconds_sum", inference_seconds)
    return raw_outputs, queue_seconds, inference_seconds


def stack_outputs(output_groups: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not output_groups:
        return []

    output_count = len(output_groups[0])
    stacked: list[np.ndarray] = []
    for output_index in range(output_count):
        stacked.append(np.concatenate([group[output_index] for group in output_groups], axis=0))
    return stacked


async def run_yolo_frames(
    bundle: ModelBundle,
    input_array: np.ndarray,
) -> tuple[list[np.ndarray], float, float, str]:
    if input_array.shape[0] == 1:
        raw_outputs, queue_seconds, inference_seconds = await run_model_bundle(bundle, input_array)
        return raw_outputs, queue_seconds, inference_seconds, "single"

    try:
        raw_outputs, queue_seconds, inference_seconds = await run_model_bundle(bundle, input_array)
        return raw_outputs, queue_seconds, inference_seconds, "batch"
    except Exception as exc:
        logger.warning("batch inference failed, falling back to per-frame inference: %s", exc)

    output_groups: list[list[np.ndarray]] = []
    queue_seconds_sum = 0.0
    inference_seconds_sum = 0.0
    for index in range(input_array.shape[0]):
        raw_outputs, queue_seconds, inference_seconds = await run_model_bundle(bundle, input_array[index : index + 1])
        output_groups.append(raw_outputs)
        queue_seconds_sum += queue_seconds
        inference_seconds_sum += inference_seconds
    return stack_outputs(output_groups), queue_seconds_sum, inference_seconds_sum, "per_frame"


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
        "# HELP gpu_worker_persons_requests_total Total /infer/persons requests.",
        "# TYPE gpu_worker_persons_requests_total counter",
        f"gpu_worker_persons_requests_total {METRICS.get('persons_requests_total', 0)}",
        "# HELP gpu_worker_persons_errors_total Total /infer/persons errors.",
        "# TYPE gpu_worker_persons_errors_total counter",
        f"gpu_worker_persons_errors_total {METRICS.get('persons_errors_total', 0)}",
        "# HELP gpu_worker_persons_detected_total Total detected persons.",
        "# TYPE gpu_worker_persons_detected_total counter",
        f"gpu_worker_persons_detected_total {METRICS.get('persons_detected_total', 0)}",
        "# HELP gpu_worker_persons_frames_total Total frames processed by person detection.",
        "# TYPE gpu_worker_persons_frames_total counter",
        f"gpu_worker_persons_frames_total {METRICS.get('persons_frames_total', 0)}",
        "# HELP gpu_worker_embeddings_requests_total Total /infer/person-embeddings requests.",
        "# TYPE gpu_worker_embeddings_requests_total counter",
        f"gpu_worker_embeddings_requests_total {METRICS.get('embeddings_requests_total', 0)}",
        "# HELP gpu_worker_embeddings_errors_total Total /infer/person-embeddings errors.",
        "# TYPE gpu_worker_embeddings_errors_total counter",
        f"gpu_worker_embeddings_errors_total {METRICS.get('embeddings_errors_total', 0)}",
        "# HELP gpu_worker_tracks_requests_total Total /infer/person-tracks requests.",
        "# TYPE gpu_worker_tracks_requests_total counter",
        f"gpu_worker_tracks_requests_total {METRICS.get('tracks_requests_total', 0)}",
        "# HELP gpu_worker_tracks_errors_total Total /infer/person-tracks errors.",
        "# TYPE gpu_worker_tracks_errors_total counter",
        f"gpu_worker_tracks_errors_total {METRICS.get('tracks_errors_total', 0)}",
        "# HELP gpu_worker_decode_seconds_sum Sum of image decode seconds.",
        "# TYPE gpu_worker_decode_seconds_sum counter",
        f"gpu_worker_decode_seconds_sum {METRICS.get('decode_seconds_sum', 0)}",
        "# HELP gpu_worker_preprocess_seconds_sum Sum of preprocessing seconds.",
        "# TYPE gpu_worker_preprocess_seconds_sum counter",
        f"gpu_worker_preprocess_seconds_sum {METRICS.get('preprocess_seconds_sum', 0)}",
        "# HELP gpu_worker_postprocess_seconds_sum Sum of postprocessing seconds.",
        "# TYPE gpu_worker_postprocess_seconds_sum counter",
        f"gpu_worker_postprocess_seconds_sum {METRICS.get('postprocess_seconds_sum', 0)}",
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
        "models_root": str(MODELS_ROOT),
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


@app.get("/ready/deep", dependencies=[Depends(require_api_token)])
async def ready_deep(
    load_models: bool = Query(False),
    dummy_inference: bool = Query(False),
) -> dict[str, Any]:
    available = ort.get_available_providers()
    checks = []
    ok = "CUDAExecutionProvider" in available

    for key, config in MODEL_CONFIGS.items():
        try:
            project_name, model_name = split_cache_key(key)
            model_path = get_model_path(project_name, model_name)
            item: dict[str, Any] = {
                "model": key,
                "type": config.get("type"),
                "exists": True,
                "path": str(model_path),
            }
            if load_models or dummy_inference:
                bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
                item.update(
                    {
                        "loaded": True,
                        "cold_loaded": cold_loaded,
                        "load_seconds": load_seconds,
                        "providers": bundle["session"].get_providers(),
                    }
                )
                if dummy_inference:
                    session = bundle["session"]
                    input_meta = session.get_inputs()[0]
                    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
                    dtype = input_dtype(input_meta.type)
                    dummy = np.zeros(shape, dtype=dtype)
                    _, queue_seconds, inference_seconds = await run_model_bundle(bundle, dummy)
                    item.update(
                        {
                            "dummy_inference": True,
                            "dummy_input_shape": shape,
                            "queue_seconds": queue_seconds,
                            "inference_seconds": inference_seconds,
                        }
                    )
            checks.append(item)
        except Exception as exc:
            ok = False
            checks.append({"model": key, "ok": False, "error": str(exc)})

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "available_providers": available, "checks": checks},
        )
    return {"status": "ready", "available_providers": available, "checks": checks}


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


@app.get("/model-configs", dependencies=[Depends(require_api_token)])
async def model_configs() -> dict[str, Any]:
    return {
        "config_path": str(MODEL_CONFIG_PATH),
        "models": MODEL_CONFIGS,
        "count": len(MODEL_CONFIGS),
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
        "config": model_config(key),
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

        raw_outputs, queue_seconds, inference_seconds = await run_model_bundle(bundle, input_array)
        total_seconds = now() - total_start
        output_shapes = [list(output.shape) for output in raw_outputs]
        await touch_model(key, bundle)
        log_json(
            logging.INFO,
            "predict_completed",
            request_id=request_id,
            model=key,
            input_shape=list(input_array.shape),
            input_dtype=str(input_array.dtype),
            output_shapes=output_shapes,
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
        "input_shape": list(input_array.shape),
        "output_shapes": output_shapes,
        "outputs": [output.tolist() for output in raw_outputs],
    }


@app.post("/infer/persons", dependencies=[Depends(require_api_token)])
async def infer_persons(
    request: Request,
    files: list[UploadFile] = File(...),
    project_name: str = Form("cross_camera_tracking"),
    model_name: str = Form("yolov8n.onnx"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    max_detections: int = Form(100),
) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    observe("persons_requests_total")
    total_start = now()

    try:
        project_name = validate_path_name(project_name)
        model_name = validate_path_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="at least one image file is required")
    if len(files) > MAX_PERSON_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"too many image files: {len(files)}, max {MAX_PERSON_FRAMES}",
        )
    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="confidence must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="iou must be between 0 and 1")
    if max_detections < 1 or max_detections > 1000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="max_detections must be between 1 and 1000")

    key = cache_key(project_name, model_name)
    model_path = get_model_path(project_name, model_name)

    try:
        bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
        images, filenames, decode_seconds = await load_images(files)
        frames, infer_meta = await infer_person_frames(
            bundle,
            key,
            images,
            filenames,
            confidence=confidence,
            iou=iou,
            max_detections=max_detections,
        )

        total_seconds = now() - total_start
        await touch_model(key, bundle)
        person_count = sum(frame["person_count"] for frame in frames)
        observe("persons_detected_total", person_count)
        observe("persons_frames_total", len(frames))
        observe("decode_seconds_sum", decode_seconds)
        observe("preprocess_seconds_sum", infer_meta["timing"]["preprocess_seconds"])
        observe("postprocess_seconds_sum", infer_meta["timing"]["postprocess_seconds"])
        log_json(
            logging.INFO,
            "persons_infer_completed",
            request_id=request_id,
            model=key,
            frame_count=len(files),
            inference_mode=infer_meta["inference_mode"],
            input_shape=infer_meta["input_shape"],
            output_shapes=infer_meta["output_shapes"],
            person_count=person_count,
            cold_loaded=cold_loaded,
            decode_seconds=round(decode_seconds, 6),
            preprocess_seconds=round(infer_meta["timing"]["preprocess_seconds"], 6),
            queue_seconds=round(infer_meta["timing"]["queue_seconds"], 6),
            load_seconds=round(load_seconds, 6),
            inference_seconds=round(infer_meta["timing"]["inference_seconds"], 6),
            postprocess_seconds=round(infer_meta["timing"]["postprocess_seconds"], 6),
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        observe("persons_errors_total")
        raise
    except Exception as exc:
        observe("persons_errors_total")
        logger.exception("person inference failed for model: %s", key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"person inference runtime error: {exc}",
        ) from exc

    return {
        "status": "success",
        "request_id": request_id,
        "model": key,
        "cold_loaded": cold_loaded,
        "timing": {
            "decode_seconds": decode_seconds,
            "preprocess_seconds": infer_meta["timing"]["preprocess_seconds"],
            "queue_seconds": infer_meta["timing"]["queue_seconds"],
            "load_seconds": load_seconds,
            "inference_seconds": infer_meta["timing"]["inference_seconds"],
            "postprocess_seconds": infer_meta["timing"]["postprocess_seconds"],
            "total_seconds": total_seconds,
        },
        "input_shape": infer_meta["input_shape"],
        "output_shapes": infer_meta["output_shapes"],
        "inference_mode": infer_meta["inference_mode"],
        "frames": frames,
        "frame_count": len(frames),
        "person_count": person_count,
    }


@app.post("/infer/person-embeddings", dependencies=[Depends(require_api_token)])
async def infer_person_embeddings(
    request: Request,
    files: list[UploadFile] = File(...),
    project_name: str = Form("cross_camera_tracking"),
    model_name: str = Form("osnet_ibn_x1_0.onnx"),
    include_vectors: bool = Form(True),
) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    observe("embeddings_requests_total")
    total_start = now()

    try:
        project_name = validate_path_name(project_name)
        model_name = validate_path_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="at least one image file is required")
    if len(files) > MAX_EMBEDDING_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"too many image files: {len(files)}, max {MAX_EMBEDDING_IMAGES}",
        )

    key = cache_key(project_name, model_name)
    model_path = get_model_path(project_name, model_name)

    try:
        bundle, cold_loaded, load_seconds = await get_or_load_model(key, model_path)
        images, filenames, decode_seconds = await load_images(files)
        embeddings, infer_meta = await infer_reid_images(bundle, key, images)
        total_seconds = now() - total_start
        await touch_model(key, bundle)

        observe("decode_seconds_sum", decode_seconds)
        observe("preprocess_seconds_sum", infer_meta["timing"]["preprocess_seconds"])
        observe("postprocess_seconds_sum", infer_meta["timing"]["postprocess_seconds"])

        items = []
        for index, filename in enumerate(filenames):
            item: dict[str, Any] = {
                "index": index,
                "filename": filename,
                "embedding_dim": infer_meta["embedding_dim"],
            }
            if include_vectors:
                item["embedding"] = [round(float(value), 8) for value in embeddings[index].tolist()]
            items.append(item)

        log_json(
            logging.INFO,
            "embeddings_infer_completed",
            request_id=request_id,
            model=key,
            image_count=len(images),
            inference_mode=infer_meta["inference_mode"],
            input_shape=infer_meta["input_shape"],
            output_shapes=infer_meta["output_shapes"],
            embedding_dim=infer_meta["embedding_dim"],
            cold_loaded=cold_loaded,
            decode_seconds=round(decode_seconds, 6),
            preprocess_seconds=round(infer_meta["timing"]["preprocess_seconds"], 6),
            queue_seconds=round(infer_meta["timing"]["queue_seconds"], 6),
            load_seconds=round(load_seconds, 6),
            inference_seconds=round(infer_meta["timing"]["inference_seconds"], 6),
            postprocess_seconds=round(infer_meta["timing"]["postprocess_seconds"], 6),
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        observe("embeddings_errors_total")
        raise
    except Exception as exc:
        observe("embeddings_errors_total")
        logger.exception("embedding inference failed for model: %s", key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"embedding inference runtime error: {exc}",
        ) from exc

    return {
        "status": "success",
        "request_id": request_id,
        "model": key,
        "cold_loaded": cold_loaded,
        "timing": {
            "decode_seconds": decode_seconds,
            "preprocess_seconds": infer_meta["timing"]["preprocess_seconds"],
            "queue_seconds": infer_meta["timing"]["queue_seconds"],
            "load_seconds": load_seconds,
            "inference_seconds": infer_meta["timing"]["inference_seconds"],
            "postprocess_seconds": infer_meta["timing"]["postprocess_seconds"],
            "total_seconds": total_seconds,
        },
        "input_shape": infer_meta["input_shape"],
        "output_shapes": infer_meta["output_shapes"],
        "inference_mode": infer_meta["inference_mode"],
        "embedding_dim": infer_meta["embedding_dim"],
        "items": items,
        "count": len(items),
    }


@app.post("/infer/person-tracks", dependencies=[Depends(require_api_token)])
async def infer_person_tracks(
    request: Request,
    files: list[UploadFile] = File(...),
    detector_project_name: str = Form("cross_camera_tracking"),
    detector_model_name: str = Form("yolov8n.onnx"),
    reid_project_name: str = Form("cross_camera_tracking"),
    reid_model_name: str = Form("osnet_ibn_x1_0.onnx"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    max_detections: int = Form(100),
    include_embeddings: bool = Form(False),
) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    observe("tracks_requests_total")
    total_start = now()

    try:
        detector_project_name = validate_path_name(detector_project_name)
        detector_model_name = validate_path_name(detector_model_name)
        reid_project_name = validate_path_name(reid_project_name)
        reid_model_name = validate_path_name(reid_model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="at least one image file is required")
    if len(files) > MAX_PIPELINE_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"too many image files: {len(files)}, max {MAX_PIPELINE_FRAMES}",
        )
    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="confidence must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="iou must be between 0 and 1")
    if max_detections < 1 or max_detections > 1000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="max_detections must be between 1 and 1000")

    try:
        images, filenames, decode_seconds = await load_images(files)
        result = await infer_tracks_for_images(
            images,
            filenames,
            detector_project_name,
            detector_model_name,
            reid_project_name,
            reid_model_name,
            confidence=confidence,
            iou=iou,
            max_detections=max_detections,
            include_embeddings=include_embeddings,
        )

        total_seconds = now() - total_start
        detector_meta = result["detector_meta"]
        embedding_meta = result["embedding_meta"]
        observe("decode_seconds_sum", decode_seconds)

        log_json(
            logging.INFO,
            "person_tracks_infer_completed",
            request_id=request_id,
            detector_model=result["detector_key"],
            reid_model=result["reid_key"],
            frame_count=len(result["frames"]),
            person_count=result["person_count"],
            embedding_count=result["embedding_count"],
            detector_mode=detector_meta["inference_mode"],
            reid_mode=embedding_meta["inference_mode"],
            decode_seconds=round(decode_seconds, 6),
            detector_inference_seconds=round(detector_meta["timing"]["inference_seconds"], 6),
            reid_inference_seconds=round(embedding_meta["timing"]["inference_seconds"], 6),
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        observe("tracks_errors_total")
        raise
    except Exception as exc:
        observe("tracks_errors_total")
        logger.exception("person track inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"person track inference runtime error: {exc}",
        ) from exc

    return {
        "status": "success",
        "request_id": request_id,
        "detector_model": result["detector_key"],
        "reid_model": result["reid_key"],
        "cold_loaded": {
            "detector": result["detector_cold_loaded"],
            "reid": result["reid_cold_loaded"],
        },
        "timing": {
            "decode_seconds": decode_seconds,
            "detector_load_seconds": result["detector_load_seconds"],
            "reid_load_seconds": result["reid_load_seconds"],
            "detector": detector_meta["timing"],
            "reid": embedding_meta["timing"],
            "total_seconds": total_seconds,
        },
        "detector": {
            "input_shape": detector_meta["input_shape"],
            "output_shapes": detector_meta["output_shapes"],
            "inference_mode": detector_meta["inference_mode"],
        },
        "reid": {
            "input_shape": embedding_meta["input_shape"],
            "output_shapes": embedding_meta["output_shapes"],
            "inference_mode": embedding_meta["inference_mode"],
            "embedding_dim": embedding_meta["embedding_dim"],
            "embedding_count": embedding_count,
        },
        "frames": result["frames"],
        "frame_count": len(result["frames"]),
        "person_count": result["person_count"],
    }


@app.post("/infer/video/person-tracks", dependencies=[Depends(require_api_token)])
async def infer_video_person_tracks(
    request: Request,
    file: UploadFile = File(...),
    detector_project_name: str = Form("cross_camera_tracking"),
    detector_model_name: str = Form("yolov8n.onnx"),
    reid_project_name: str = Form("cross_camera_tracking"),
    reid_model_name: str = Form("osnet_ibn_x1_0.onnx"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    max_detections: int = Form(100),
    include_embeddings: bool = Form(False),
    frame_interval: int = Form(VIDEO_FRAME_INTERVAL),
    max_frames: int = Form(MAX_VIDEO_FRAMES),
) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    observe("tracks_requests_total")
    total_start = now()

    try:
        detector_project_name = validate_path_name(detector_project_name)
        detector_model_name = validate_path_name(detector_model_name)
        reid_project_name = validate_path_name(reid_project_name)
        reid_model_name = validate_path_name(reid_model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="confidence must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="iou must be between 0 and 1")
    if max_detections < 1 or max_detections > 1000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="max_detections must be between 1 and 1000")
    if frame_interval < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="frame_interval must be >= 1")
    if max_frames < 1 or max_frames > MAX_VIDEO_FRAMES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"max_frames must be between 1 and {MAX_VIDEO_FRAMES}")

    try:
        images, video_meta = await extract_video_frames_from_upload(file, frame_interval, max_frames)
        if not images:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no frames could be extracted from video")
        filenames = [f"{file.filename or 'video'}#frame-{frame_index}" for frame_index in video_meta["source_frame_indexes"]]
        result = await infer_tracks_for_images(
            images,
            filenames,
            detector_project_name,
            detector_model_name,
            reid_project_name,
            reid_model_name,
            confidence=confidence,
            iou=iou,
            max_detections=max_detections,
            include_embeddings=include_embeddings,
        )
        for frame, source_frame_index in zip(result["frames"], video_meta["source_frame_indexes"]):
            frame["source_frame_index"] = source_frame_index
            if video_meta.get("fps"):
                frame["source_seconds"] = round(source_frame_index / video_meta["fps"], 6)

        total_seconds = now() - total_start
        observe("decode_seconds_sum", video_meta["decode_seconds"])
        log_json(
            logging.INFO,
            "video_person_tracks_completed",
            request_id=request_id,
            detector_model=result["detector_key"],
            reid_model=result["reid_key"],
            filename=file.filename,
            extracted_frames=len(images),
            person_count=result["person_count"],
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        observe("tracks_errors_total")
        raise
    except Exception as exc:
        observe("tracks_errors_total")
        logger.exception("video person track inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"video person track inference runtime error: {exc}",
        ) from exc

    detector_meta = result["detector_meta"]
    embedding_meta = result["embedding_meta"]
    return {
        "status": "success",
        "request_id": request_id,
        "source_type": "video_upload",
        "video": video_meta,
        "detector_model": result["detector_key"],
        "reid_model": result["reid_key"],
        "timing": {
            "video_decode_seconds": video_meta["decode_seconds"],
            "detector_load_seconds": result["detector_load_seconds"],
            "reid_load_seconds": result["reid_load_seconds"],
            "detector": detector_meta["timing"],
            "reid": embedding_meta["timing"],
            "total_seconds": total_seconds,
        },
        "frames": result["frames"],
        "frame_count": len(result["frames"]),
        "person_count": result["person_count"],
    }


@app.post("/infer/stream/person-tracks", dependencies=[Depends(require_api_token)])
async def infer_stream_person_tracks(
    request: Request,
    stream_url: str = Form(...),
    detector_project_name: str = Form("cross_camera_tracking"),
    detector_model_name: str = Form("yolov8n.onnx"),
    reid_project_name: str = Form("cross_camera_tracking"),
    reid_model_name: str = Form("osnet_ibn_x1_0.onnx"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    max_detections: int = Form(100),
    include_embeddings: bool = Form(False),
    frame_interval: int = Form(STREAM_FRAME_INTERVAL),
    max_frames: int = Form(MAX_STREAM_FRAMES),
    read_timeout_seconds: int = Form(STREAM_READ_TIMEOUT_SECONDS),
) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    observe("tracks_requests_total")
    total_start = now()

    if not ALLOW_STREAM_URLS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="stream URL pulling is disabled. Set ALLOW_STREAM_URLS=true to enable it for trusted networks.",
        )

    try:
        stream_url = validate_stream_url(stream_url)
        detector_project_name = validate_path_name(detector_project_name)
        detector_model_name = validate_path_name(detector_model_name)
        reid_project_name = validate_path_name(reid_project_name)
        reid_model_name = validate_path_name(reid_model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="confidence must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="iou must be between 0 and 1")
    if max_detections < 1 or max_detections > 1000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="max_detections must be between 1 and 1000")
    if frame_interval < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="frame_interval must be >= 1")
    if max_frames < 1 or max_frames > MAX_STREAM_FRAMES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"max_frames must be between 1 and {MAX_STREAM_FRAMES}")
    if read_timeout_seconds < 1 or read_timeout_seconds > STREAM_READ_TIMEOUT_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"read_timeout_seconds must be between 1 and {STREAM_READ_TIMEOUT_SECONDS}",
        )

    try:
        images, stream_meta = await asyncio.to_thread(
            extract_video_frames_from_path,
            stream_url,
            frame_interval,
            max_frames,
            read_timeout_seconds,
        )
        if not images:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no frames could be read from stream")
        filenames = [f"stream#frame-{frame_index}" for frame_index in stream_meta["source_frame_indexes"]]
        result = await infer_tracks_for_images(
            images,
            filenames,
            detector_project_name,
            detector_model_name,
            reid_project_name,
            reid_model_name,
            confidence=confidence,
            iou=iou,
            max_detections=max_detections,
            include_embeddings=include_embeddings,
        )
        for frame, source_frame_index in zip(result["frames"], stream_meta["source_frame_indexes"]):
            frame["source_frame_index"] = source_frame_index
            if stream_meta.get("fps"):
                frame["source_seconds"] = round(source_frame_index / stream_meta["fps"], 6)

        total_seconds = now() - total_start
        observe("decode_seconds_sum", stream_meta["decode_seconds"])
        log_json(
            logging.INFO,
            "stream_person_tracks_completed",
            request_id=request_id,
            detector_model=result["detector_key"],
            reid_model=result["reid_key"],
            extracted_frames=len(images),
            person_count=result["person_count"],
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        observe("tracks_errors_total")
        raise
    except Exception as exc:
        observe("tracks_errors_total")
        logger.exception("stream person track inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"stream person track inference runtime error: {exc}",
        ) from exc

    detector_meta = result["detector_meta"]
    embedding_meta = result["embedding_meta"]
    return {
        "status": "success",
        "request_id": request_id,
        "source_type": "stream",
        "stream": {
            **stream_meta,
            "url": stream_url,
            "read_timeout_seconds": read_timeout_seconds,
        },
        "detector_model": result["detector_key"],
        "reid_model": result["reid_key"],
        "timing": {
            "stream_read_seconds": stream_meta["decode_seconds"],
            "detector_load_seconds": result["detector_load_seconds"],
            "reid_load_seconds": result["reid_load_seconds"],
            "detector": detector_meta["timing"],
            "reid": embedding_meta["timing"],
            "total_seconds": total_seconds,
        },
        "frames": result["frames"],
        "frame_count": len(result["frames"]),
        "person_count": result["person_count"],
    }


@app.post("/debug/model-output", dependencies=[Depends(require_api_token)])
async def debug_model_output(
    request: Request,
    file: UploadFile = File(...),
    project_name: str = Form(...),
    model_name: str = Form(...),
    model_type: str = Form("yolo"),
    sample_values: int = Form(12),
) -> dict[str, Any]:
    request_id = request_id_from_headers(request)
    total_start = now()

    try:
        project_name = validate_path_name(project_name)
        model_name = validate_path_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if sample_values < 0 or sample_values > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sample_values must be between 0 and 100")

    key = cache_key(project_name, model_name)
    try:
        bundle, cold_loaded, load_seconds = await get_or_load_model(key, get_model_path(project_name, model_name))
        images, _, decode_seconds = await load_images([file])
        session = bundle["session"]

        preprocess_start = now()
        if model_type == "reid":
            config = model_config(key, default_type="reid")
            input_height, input_width = configured_input_size(key, session, default=(256, 128))
            tensor = await asyncio.to_thread(
                resize_image_tensor,
                images[0],
                input_height,
                input_width,
                str(config.get("normalize", "imagenet")),
            )
        else:
            input_height, input_width = configured_input_size(key, session, default=(640, 640))
            tensor, _ = await asyncio.to_thread(letterbox_image, images[0], input_height, input_width)
        input_array = np.expand_dims(tensor, axis=0).astype(np.float32)
        preprocess_seconds = now() - preprocess_start

        raw_outputs, queue_seconds, inference_seconds = await run_model_bundle(bundle, input_array)
        outputs = []
        for index, output in enumerate(raw_outputs):
            flat = output.reshape(-1)
            outputs.append(
                {
                    "index": index,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "min": float(np.min(output)) if output.size else None,
                    "max": float(np.max(output)) if output.size else None,
                    "sample": [round(float(value), 8) for value in flat[:sample_values].tolist()],
                }
            )

        total_seconds = now() - total_start
        log_json(
            logging.INFO,
            "debug_model_output_completed",
            request_id=request_id,
            model=key,
            model_type=model_type,
            input_shape=list(input_array.shape),
            output_shapes=[item["shape"] for item in outputs],
            total_seconds=round(total_seconds, 6),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("debug model output failed for model: %s", key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"debug model output runtime error: {exc}",
        ) from exc

    return {
        "status": "success",
        "request_id": request_id,
        "model": key,
        "model_type": model_type,
        "cold_loaded": cold_loaded,
        "timing": {
            "decode_seconds": decode_seconds,
            "preprocess_seconds": preprocess_seconds,
            "queue_seconds": queue_seconds,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
        },
        "input_shape": list(input_array.shape),
        "outputs": outputs,
    }
