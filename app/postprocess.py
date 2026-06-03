from typing import Any

import numpy as np

from app.constants import COCO_CLASSES
from app.geometry import nms, restore_boxes, xywh_to_xyxy
from app.model_package import class_name
from app.schemas import LetterboxMeta


def yolo_detections(
    raw_outputs: list[np.ndarray],
    meta: LetterboxMeta,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_filter_ids: set[int] | None = None,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not raw_outputs:
        return []

    output = np.asarray(raw_outputs[0])
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"unsupported YOLO output shape: {list(raw_outputs[0].shape)}")

    if output.shape[0] < output.shape[1] and output.shape[0] in {5, 6, 84, 85}:
        output = output.T

    if output.shape[1] < 5:
        raise ValueError(f"unsupported YOLO output shape: {list(raw_outputs[0].shape)}")

    boxes_xywh = output[:, :4].astype(np.float32)
    if output.shape[1] == 6:
        scores = output[:, 4].astype(np.float32)
        class_ids = output[:, 5].astype(np.int64)
    elif output.shape[1] == 5:
        scores = output[:, 4].astype(np.float32)
        class_ids = np.zeros(output.shape[0], dtype=np.int64)
    elif output.shape[1] == 84:
        class_scores = output[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        objectness = output[:, 4].astype(np.float32)
        class_scores = output[:, 5:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= confidence_threshold
    if class_filter_ids is not None:
        mask = mask & np.isin(class_ids, list(class_filter_ids))
    if not np.any(mask):
        return []

    boxes = restore_boxes(xywh_to_xyxy(boxes_xywh[mask]), meta)
    scores = scores[mask]
    class_ids = class_ids[mask]
    keep = nms(boxes, scores, iou_threshold)[:max_detections]

    detections: list[dict[str, Any]] = []
    for index in keep:
        box = boxes[index]
        detections.append(
            {
                "box": [round(float(value), 3) for value in box.tolist()],
                "score": round(float(scores[index]), 6),
                "class_id": int(class_ids[index]),
                "class_name": class_name(int(class_ids[index]), labels),
            }
        )
    return detections


def yolo_person_detections(
    raw_outputs: list[np.ndarray],
    meta: LetterboxMeta,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    person_class_id: int = 0,
) -> list[dict[str, Any]]:
    detections = yolo_detections(
        raw_outputs,
        meta,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        class_filter_ids={person_class_id},
        labels=COCO_CLASSES if person_class_id == 0 else None,
    )
    for detection in detections:
        if detection["class_id"] == person_class_id:
            detection["class_name"] = "person" if person_class_id == 0 else detection["class_name"]
    return detections


def normalize_embeddings(output: np.ndarray, mode: str = "l2") -> np.ndarray:
    embeddings = np.asarray(output, dtype=np.float32)
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape((embeddings.shape[0], -1))
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape((1, -1))
    if mode == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def classification_predictions(
    raw_outputs: list[np.ndarray],
    labels: list[str],
    top_k: int,
    activation: str,
    threshold: float | None,
) -> tuple[list[list[dict[str, Any]]], int]:
    if not raw_outputs:
        return [], 0

    scores = np.asarray(raw_outputs[0], dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape((1, -1))
    elif scores.ndim > 2:
        scores = scores.reshape((scores.shape[0], -1))

    activation = activation.lower()
    if activation == "sigmoid":
        probabilities = sigmoid(scores)
    elif activation in {"none", "raw", "identity"}:
        probabilities = scores
    else:
        probabilities = softmax(scores)

    class_count = probabilities.shape[1] if probabilities.ndim == 2 else 0
    safe_top_k = max(1, min(top_k, class_count or top_k))
    batch_predictions: list[list[dict[str, Any]]] = []
    for row in probabilities:
        ranked = row.argsort()[::-1]
        if threshold is not None:
            ranked = np.asarray([index for index in ranked if row[index] >= threshold], dtype=np.int64)
        ranked = ranked[:safe_top_k]
        predictions = [
            {
                "class_id": int(index),
                "class_name": class_name(int(index), labels),
                "score": round(float(row[index]), 8),
            }
            for index in ranked
        ]
        batch_predictions.append(predictions)
    return batch_predictions, class_count
