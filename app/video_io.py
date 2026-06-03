import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image

from app.observability import logger, now
from app.settings import MAX_VIDEO_BYTES


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
