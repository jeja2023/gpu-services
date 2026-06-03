from app.model_config_loader import normalize_model_config
from app.model_config_resolver import alias_resolution, alias_target


def test_normalize_model_config_maps_legacy_yolo_fields() -> None:
    config = normalize_model_config(
        "project/model.onnx",
        {
            "type": "yolo",
            "input_size": [640, 640],
            "confidence": 0.4,
            "iou": 0.5,
            "classes": "coco",
        },
    )

    assert config["task"] == "detection"
    assert config["type"] == "yolo"
    assert config["input"]["size"] == [640, 640]
    assert config["output"]["confidence"] == 0.4
    assert config["output"]["iou"] == 0.5
    assert config["output"]["classes"] == "coco"


def test_alias_target_uses_highest_weight_rollout() -> None:
    target = alias_target(
        "detector_default",
        {
            "rollout": [
                {"target": "project/old.onnx", "weight": 10},
                {"target": "project/new.onnx", "weight": 90},
            ]
        },
    )

    assert target == "project/new.onnx"


def test_alias_target_uses_weighted_rollout_with_traffic_key() -> None:
    config = {
        "rollout": [
            {"target": "project/old.onnx", "weight": 50},
            {"target": "project/new.onnx", "weight": 50},
        ]
    }

    first = alias_resolution("detector_default", config, traffic_key="customer-001")
    second = alias_resolution("detector_default", config, traffic_key="customer-001")

    assert first["target"] == second["target"]
    assert first["strategy"] == "weighted"
    assert first["total_weight"] == 100
