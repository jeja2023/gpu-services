from typing import Any

from PIL import Image

from app.inference_detection import infer_person_frames
from app.inference_reid import infer_reid_images
from app.metrics import observe
from app.model_package import get_model_path
from app.model_refs import cache_key
from app.runtime import get_or_load_model, touch_model
from app.vision import crop_person


async def infer_tracks_for_images(
    images: list[Image.Image],
    filenames: list[str | None],
    detector_project_name: str,
    detector_model_name: str,
    reid_project_name: str,
    reid_model_name: str,
    confidence: float,
    iou: float,
    max_detections: int,
    include_embeddings: bool,
) -> dict[str, Any]:
    detector_key = cache_key(detector_project_name, detector_model_name)
    reid_key = cache_key(reid_project_name, reid_model_name)

    detector_bundle, detector_cold_loaded, detector_load_seconds = await get_or_load_model(
        detector_key,
        get_model_path(detector_project_name, detector_model_name),
    )
    reid_bundle, reid_cold_loaded, reid_load_seconds = await get_or_load_model(
        reid_key,
        get_model_path(reid_project_name, reid_model_name),
    )

    frames, detector_meta = await infer_person_frames(
        detector_bundle,
        detector_key,
        images,
        filenames,
        confidence=confidence,
        iou=iou,
        max_detections=max_detections,
    )

    crops: list[Image.Image] = []
    crop_refs: list[tuple[int, int]] = []
    for frame in frames:
        image = images[frame["frame_index"]]
        for person_index, person in enumerate(frame["persons"]):
            crop = crop_person(image, person["box"])
            if crop is not None:
                crops.append(crop)
                crop_refs.append((frame["frame_index"], person_index))

    embedding_count = 0
    if crops:
        embeddings, embedding_meta = await infer_reid_images(reid_bundle, reid_key, crops)
        embedding_count = embeddings.shape[0]
        for index, (frame_index, person_index) in enumerate(crop_refs):
            person = frames[frame_index]["persons"][person_index]
            person["embedding_dim"] = int(embeddings.shape[1])
            person["embedding_index"] = index
            if include_embeddings:
                person["embedding"] = [round(float(value), 8) for value in embeddings[index].tolist()]
    else:
        embedding_meta = {
            "input_shape": [0],
            "output_shapes": [],
            "inference_mode": "none",
            "embedding_dim": 0,
            "timing": {
                "preprocess_seconds": 0,
                "queue_seconds": 0,
                "inference_seconds": 0,
                "postprocess_seconds": 0,
            },
        }

    person_count = sum(frame["person_count"] for frame in frames)
    await touch_model(detector_key, detector_bundle)
    await touch_model(reid_key, reid_bundle)
    observe("persons_detected_total", person_count)
    observe("persons_frames_total", len(frames))
    observe("preprocess_seconds_sum", detector_meta["timing"]["preprocess_seconds"] + embedding_meta["timing"]["preprocess_seconds"])
    observe("postprocess_seconds_sum", detector_meta["timing"]["postprocess_seconds"] + embedding_meta["timing"]["postprocess_seconds"])

    return {
        "detector_key": detector_key,
        "reid_key": reid_key,
        "detector_cold_loaded": detector_cold_loaded,
        "reid_cold_loaded": reid_cold_loaded,
        "detector_load_seconds": detector_load_seconds,
        "reid_load_seconds": reid_load_seconds,
        "detector_meta": detector_meta,
        "embedding_meta": embedding_meta,
        "frames": frames,
        "person_count": person_count,
        "embedding_count": embedding_count,
    }
