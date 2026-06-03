import numpy as np

from app.geometry import nms
from app.postprocess import classification_predictions, normalize_embeddings, yolo_person_detections


def test_classification_predictions_returns_top_k_labels() -> None:
    predictions, class_count = classification_predictions(
        [np.array([[0.1, 2.0, 1.0]], dtype=np.float32)],
        labels=["cat", "dog", "bird"],
        top_k=2,
        activation="softmax",
        threshold=None,
    )

    assert class_count == 3
    assert [item["class_name"] for item in predictions[0]] == ["dog", "bird"]


def test_yolo_person_detections_filters_person_class() -> None:
    meta = {
        "original_width": 640,
        "original_height": 480,
        "input_width": 640,
        "input_height": 640,
        "scale": 1.0,
        "pad_left": 0.0,
        "pad_top": 0.0,
    }
    output = np.array(
        [
            [320, 240, 100, 80, 0.9, 0],
            [100, 120, 50, 30, 0.95, 2],
        ],
        dtype=np.float32,
    )

    detections = yolo_person_detections([output], meta, 0.25, 0.45, 10)

    assert len(detections) == 1
    assert detections[0]["class_name"] == "person"


def test_nms_removes_overlapping_lower_score_box() -> None:
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = nms(boxes, scores, 0.5)

    assert keep == [0, 2]


def test_normalize_embeddings_l2_normalizes_rows() -> None:
    embeddings = normalize_embeddings(np.array([[3.0, 4.0]], dtype=np.float32))

    assert np.allclose(np.linalg.norm(embeddings, axis=1), np.array([1.0]))
