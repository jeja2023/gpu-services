import asyncio
import logging
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status

from app.core import *
from app.settings import APP_VERSION


router = APIRouter()


@router.post("/debug/model-output", dependencies=[Depends(require_api_token)])
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
                str(config_value(config, "input", "normalize", "imagenet")),
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
