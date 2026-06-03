import asyncio
import logging
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status

from app.core import *
from app.settings import APP_VERSION


router = APIRouter()


@router.post("/infer/persons", dependencies=[Depends(require_api_token)])
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
