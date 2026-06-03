import asyncio
from typing import Any

import numpy as np
from PIL import Image

from app.model_config import config_value, configured_input_size, model_config
from app.observability import now
from app.runtime import run_yolo_frames
from app.schemas import ModelBundle
from app.vision import normalize_embeddings, resize_image_tensor


async def infer_reid_images(
    bundle: ModelBundle,
    key: str,
    images: list[Image.Image],
) -> tuple[np.ndarray, dict[str, Any]]:
    session = bundle["session"]
    config = model_config(key, default_type="reid")
    input_height, input_width = configured_input_size(key, session, default=(256, 128))
    normalize = str(config_value(config, "input", "normalize", "imagenet"))
    embedding_normalize = str(config_value(config, "output", "embedding_normalize", config.get("embedding_normalize", "l2")))

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
