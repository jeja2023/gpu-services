import asyncio
from typing import Any

import numpy as np
from fastapi import HTTPException, status

from app.metrics import observe
from app.observability import logger, now
from app.runtime_sessions import input_dtype, run_session
from app.runtime_state import GPU_SEMAPHORE
from app.schemas import ModelBundle
from app.settings import MAX_TENSOR_ITEMS


def build_input_array(tensor_data: list[Any], dtype: Any) -> np.ndarray:
    input_array = np.asarray(tensor_data, dtype=dtype)
    if input_array.size > MAX_TENSOR_ITEMS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"tensor is too large: {input_array.size} items, max {MAX_TENSOR_ITEMS}",
        )
    return input_array


async def acquire_with_timeout(semaphore: asyncio.Semaphore, timeout_seconds: float, detail: str) -> None:
    try:
        if timeout_seconds > 0:
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout_seconds)
        else:
            await semaphore.acquire()
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc


async def run_model_bundle(bundle: ModelBundle, input_array: np.ndarray) -> tuple[list[np.ndarray], float, float]:
    session = bundle["session"]
    model_semaphore = bundle.get("semaphore")
    queue_timeout = float(bundle.get("queue_timeout_seconds", 0) or 0)
    queue_start = now()
    model_acquired = False
    gpu_acquired = False
    try:
        if model_semaphore is not None:
            await acquire_with_timeout(
                model_semaphore,
                queue_timeout,
                "model inference queue timeout",
            )
            model_acquired = True
        else:
            await bundle["lock"].acquire()
            model_acquired = True

        elapsed = now() - queue_start
        remaining_timeout = max(0.001, queue_timeout - elapsed) if queue_timeout > 0 else 0
        await acquire_with_timeout(
            GPU_SEMAPHORE,
            remaining_timeout,
            "GPU inference queue timeout",
        )
        gpu_acquired = True

        queue_seconds = now() - queue_start
        inference_start = now()
        raw_outputs = await asyncio.to_thread(run_session, session, input_array)
        inference_seconds = now() - inference_start
    finally:
        if gpu_acquired:
            GPU_SEMAPHORE.release()
        if model_acquired:
            if model_semaphore is not None:
                model_semaphore.release()
            else:
                bundle["lock"].release()

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
