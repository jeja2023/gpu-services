from pathlib import Path

import yaml

from app import model_config_writer


def test_switch_and_rollback_alias_target(monkeypatch, workspace_tmp_path: Path) -> None:
    case_root = workspace_tmp_path / "rollout_case"
    case_root.mkdir(parents=True, exist_ok=True)
    config_path = case_root / "models.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aliases": {"detector_default": {"target": "project/old.onnx"}},
                "models": {
                    "project/old.onnx": {"task": "detection"},
                    "project/new.onnx": {"task": "detection"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config_writer, "MODEL_CONFIG_PATH", config_path)
    monkeypatch.setattr(model_config_writer, "write_rollout_audit", lambda event, payload: None)

    switched = model_config_writer.switch_alias_target(
        "detector_default",
        "project/new.onnx",
        expected_current_target="project/old.onnx",
    )

    assert switched["old_target"] == "project/old.onnx"
    assert switched["new_target"] == "project/new.onnx"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["aliases"]["detector_default"]["target"] == "project/new.onnx"
    assert raw["aliases"]["detector_default"]["previous_target"] == "project/old.onnx"

    rolled_back = model_config_writer.rollback_alias_target("detector_default")

    assert rolled_back["old_target"] == "project/new.onnx"
    assert rolled_back["new_target"] == "project/old.onnx"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["aliases"]["detector_default"]["target"] == "project/old.onnx"


def test_configure_weighted_alias_rollout(monkeypatch, workspace_tmp_path: Path) -> None:
    case_root = workspace_tmp_path / "weighted_rollout_case"
    case_root.mkdir(parents=True, exist_ok=True)
    config_path = case_root / "models.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aliases": {"detector_default": {"target": "project/old.onnx"}},
                "models": {
                    "project/old.onnx": {"task": "detection"},
                    "project/new.onnx": {"task": "detection"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config_writer, "MODEL_CONFIG_PATH", config_path)
    monkeypatch.setattr(model_config_writer, "write_rollout_audit", lambda event, payload: None)

    result = model_config_writer.configure_weighted_alias_rollout(
        "detector_default",
        [
            {"target_model_id": "project/old.onnx", "weight": 90, "status": "active"},
            {"target_model_id": "project/new.onnx", "weight": 10, "status": "candidate"},
        ],
        expected_current_target="project/old.onnx",
    )

    assert result["total_weight"] == 100
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "target" not in raw["aliases"]["detector_default"]
    assert raw["aliases"]["detector_default"]["rollout"][1]["target"] == "project/new.onnx"
