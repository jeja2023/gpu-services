import numpy as np
from PIL import Image

from app.schemas import LetterboxMeta


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def restore_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - meta["pad_left"]) / meta["scale"]
    restored[:, [1, 3]] = (restored[:, [1, 3]] - meta["pad_top"]) / meta["scale"]
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, meta["original_width"])
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, meta["original_height"])
    return restored


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        inter_width = np.maximum(0, xx2 - xx1)
        inter_height = np.maximum(0, yy2 - yy1)
        intersection = inter_width * inter_height
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = rest[iou <= iou_threshold]

    return keep
def crop_person(image: Image.Image, box: list[float], min_size: int = 2) -> Image.Image | None:
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right - left < min_size or bottom - top < min_size:
        return None
    return image.crop((left, top, right, bottom))
