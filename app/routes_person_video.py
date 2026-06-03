import asyncio
import logging
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status

from app.core import *
from app.settings import APP_VERSION


router = APIRouter()


@router.post("/infer/video/person-tracks", dependencies=[Depends(require_api_token)])
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
        "embedding_count": result["embedding_count"],
    }
