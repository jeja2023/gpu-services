import asyncio
import logging
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status

from app.core import *
from app.settings import APP_VERSION


router = APIRouter()


@router.post("/infer/person-embeddings", dependencies=[Depends(require_api_token)])
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
