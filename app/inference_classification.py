import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.model_config import config_section, config_value, configured_input_size, model_config
from app.model_package import labels_from_config
from app.observability import now
from app.runtime import run_yolo_frames
from app.schemas import ModelBundle
from app.vision import classification_predictions, resize_image_tensor


async def infer_classification_images(
    bundle: ModelBundle,
    key: str,
    images: list[Image.Image],
    filenames: list[str | None],
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = bundle["session"]
    config = model_config(key, default_type="classification")
    model_path = Path(bundle["path"])
    input_height, input_width = configured_input_size(key, session, default=(224, 224))
    normalize = str(config_value(config, "input", "normalize", "imagenet"))
    labels = labels_from_config(config, model_path)
    output_config = config_section(config, "output")
    multi_label = bool(output_config.get("multi_label", False))
    activation = str(output_config.get("activation") or ("sigmoid" if multi_label else "softmax"))
    threshold_raw = output_config.get("threshold")
    threshold = float(threshold_raw) if threshold_raw is not None else None
    top_k_value = int(top_k if top_k is not None else output_config.get("top_k", 5))

    preprocess_start = now()
    tensors = [
        await asyncio.to_thread(resize_image_tensor, image, input_height, input_width, normalize)
        for image in images
    ]
    input_array = np.stack(tensors, axis=0).astype(np.float32)
    preprocess_seconds = now() - preprocess_start

    raw_outputs, queue_seconds, inference_seconds, inference_mode = await run_yolo_frames(bundle, input_array)

    postprocess_start = now()
    batch_predictions, class_count = classification_predictions(
        raw_outputs,
        labels=labels,
        top_k=top_k_value,
        activation=activation,
        threshold=threshold,
    )
    items = []
    for index, predictions in enumerate(batch_predictions):
        image = images[index]
        items.append(
            {
                "image_index": index,
                "filename": filenames[index],
                "width": image.width,
                "height": image.height,
                "predictions": predictions,
                "prediction_count": len(predictions),
            }
        )
    postprocess_seconds = now() - postprocess_start

    meta = {
        "input_shape": list(input_array.shape),
        "output_shapes": [list(output.shape) for output in raw_outputs],
        "inference_mode": inference_mode,
        "class_count": class_count,
        "labels_count": len(labels),
        "timing": {
            "preprocess_seconds": preprocess_seconds,
            "queue_seconds": queue_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
        },
        "parameters": {
            "top_k": top_k_value,
            "activation": activation,
            "threshold": threshold,
        },
    }
    return items, meta
