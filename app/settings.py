import os
from pathlib import Path
from typing import Any


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


def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


APP_VERSION = "0.0.1"
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
MAX_VISION_IMAGES = parse_int_env("MAX_VISION_IMAGES", 16)
ALLOW_STREAM_URLS = parse_bool_env("ALLOW_STREAM_URLS", False)
MAX_LOADED_MODELS = parse_int_env("MAX_LOADED_MODELS", 0)
GPU_QUEUE_LIMIT = parse_int_env("GPU_QUEUE_LIMIT", 1)
MODEL_CONCURRENCY_LIMIT = parse_int_env("MODEL_CONCURRENCY_LIMIT", 1)
MODEL_QUEUE_TIMEOUT_SECONDS = parse_float_env("MODEL_QUEUE_TIMEOUT_SECONDS", 0)
ENABLE_TENSORRT = parse_bool_env("ENABLE_TENSORRT", False)
TENSORRT_ENGINE_CACHE_ENABLE = parse_bool_env("TENSORRT_ENGINE_CACHE_ENABLE", True)
TENSORRT_ENGINE_CACHE_PATH = os.getenv("TENSORRT_ENGINE_CACHE_PATH", "/tmp/tensorrt-engine-cache")
ROLLOUT_AUDIT_PATH = Path(os.getenv("ROLLOUT_AUDIT_PATH", "rollout-audit.jsonl"))
WARMUP_MODELS = [
    item.strip()
    for item in os.getenv("WARMUP_MODELS", "").split(",")
    if item.strip()
]
API_TOKEN = os.getenv("API_TOKEN")

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
