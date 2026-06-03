from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.model_config import model_config
from app.settings import CUDA_PROVIDERS, ENABLE_TENSORRT, TENSORRT_ENGINE_CACHE_ENABLE, TENSORRT_ENGINE_CACHE_PATH


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
def session_providers(cache_key_value: str | None = None) -> list[Any]:
    if cache_key_value is None:
        return CUDA_PROVIDERS

    config = model_config(cache_key_value)
    runtime = str(config.get("runtime") or "onnxruntime").strip().lower()
    if runtime not in {"tensorrt", "onnxruntime-tensorrt", "trt"}:
        return CUDA_PROVIDERS
    if not ENABLE_TENSORRT:
        raise RuntimeError("model requested TensorRT runtime but ENABLE_TENSORRT is false")

    available = set(ort.get_available_providers())
    if "TensorrtExecutionProvider" not in available:
        raise RuntimeError(
            f"TensorrtExecutionProvider is not available. available providers: {sorted(available)}"
        )
    return [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": TENSORRT_ENGINE_CACHE_ENABLE,
                "trt_engine_cache_path": TENSORRT_ENGINE_CACHE_PATH,
            },
        ),
        *CUDA_PROVIDERS,
    ]


def create_session(model_path: Path, cache_key_value: str | None = None) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider is not available. available providers: {sorted(available)}"
        )

    requested_providers = session_providers(cache_key_value)
    session = ort.InferenceSession(str(model_path), providers=requested_providers)
    active = session.get_providers()
    if requested_providers and isinstance(requested_providers[0], tuple) and requested_providers[0][0] == "TensorrtExecutionProvider":
        if "TensorrtExecutionProvider" not in active:
            raise RuntimeError(f"model session did not enable TensorRT. active providers: {active}")
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"model session did not enable CUDA. active providers: {active}")
    return session
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
def run_session(session: ort.InferenceSession, input_array: np.ndarray) -> list[np.ndarray]:
    input_meta = session.get_inputs()[0]
    dtype = input_dtype(input_meta.type)
    if input_array.dtype != dtype:
        input_array = input_array.astype(dtype, copy=False)
    return session.run(None, {input_meta.name: input_array})
