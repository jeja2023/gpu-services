from argparse import Namespace
from hashlib import sha256
from pathlib import Path

import yaml

from tools.validate_model_package import validate_config


def test_validate_model_package_accepts_complete_package(workspace_tmp_path: Path) -> None:
    case_root = workspace_tmp_path / "model_package_case"
    models_root = case_root / "models"
    model_dir = models_root / "project"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "classifier.onnx"
    model_bytes = b"fake onnx bytes"
    model_path.write_bytes(model_bytes)
    digest = sha256(model_bytes).hexdigest()
    (model_dir / "classifier.labels.txt").write_text("ok\nng\n", encoding="utf-8")
    (model_dir / "classifier.model-card.yml").write_text(
        yaml.safe_dump(
            {
                "model": {"version": "1.0.0", "precision": "fp32"},
                "evaluation": {"accuracy": 0.99},
            }
        ),
        encoding="utf-8",
    )
    config_path = case_root / "models.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aliases": {"classifier_default": {"target": "project/classifier.onnx"}},
                "models": {
                    "project/classifier.onnx": {
                        "task": "classification",
                        "runtime": "onnxruntime",
                        "input": {"size": [224, 224]},
                        "output": {"format": "classification"},
                        "artifact": {
                            "model_card": "classifier.model-card.yml",
                            "labels": "classifier.labels.txt",
                            "sha256": digest,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = validate_config(
        Namespace(
            config=str(config_path),
            models_root=str(models_root),
            model_id=None,
            strict_hash=True,
            strict_sidecars=True,
            json=True,
        )
    )

    assert report["ok"] is True
    assert report["model_count"] == 1
    assert report["errors"] == []
