"""Validate configured model packages before loading them in the service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_yaml(path: Path, result: CheckResult) -> dict[str, Any]:
    if not path.is_file():
        result.error(f"config file not found: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except Exception as exc:
        result.error(f"failed to read yaml {path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        result.error(f"yaml root must be a mapping: {path}")
        return {}
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_model_key(key: str, result: CheckResult) -> tuple[str, str] | None:
    if "/" not in key:
        result.error(f"model key must be project/model.onnx: {key}")
        return None
    project, model = key.split("/", 1)
    if not project or not model:
        result.error(f"model key must include project and model name: {key}")
        return None
    if any(part in {".", ".."} for part in Path(project).parts + Path(model).parts):
        result.error(f"model key must not contain relative path segments: {key}")
        return None
    if Path(project).is_absolute() or Path(model).is_absolute():
        result.error(f"model key must be relative: {key}")
        return None
    return project, model


def safe_sidecar(model_path: Path, relative_path: str, result: CheckResult) -> Path | None:
    sidecar = (model_path.parent / relative_path).resolve()
    try:
        sidecar.relative_to(model_path.parent.resolve())
    except ValueError:
        result.error(f"sidecar path escapes model directory: {relative_path}")
        return None
    return sidecar


def alias_target(alias_name: str, alias_config: Any, result: CheckResult) -> str | None:
    if isinstance(alias_config, str):
        return alias_config.strip()
    if not isinstance(alias_config, dict):
        result.error(f"alias config must be string or mapping: {alias_name}")
        return None

    target = alias_config.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip()

    project_name = alias_config.get("project_name")
    model_name = alias_config.get("model_name")
    if isinstance(project_name, str) and isinstance(model_name, str):
        return f"{project_name.strip()}/{model_name.strip()}"

    rollout = alias_config.get("rollout")
    if isinstance(rollout, list) and rollout:
        candidates = [item for item in rollout if isinstance(item, dict) and isinstance(item.get("target"), str)]
        if candidates:
            candidates.sort(key=lambda item: int(item.get("weight", 0) or 0), reverse=True)
            return str(candidates[0]["target"]).strip()

    result.error(f"alias has no target: {alias_name}")
    return None


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def list_labels(labels_path: Path) -> list[str]:
    with labels_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]


def validate_sidecars(
    key: str,
    config: dict[str, Any],
    model_path: Path,
    result: CheckResult,
    strict_sidecars: bool,
) -> dict[str, Any]:
    artifact = section(config, "artifact")
    output = section(config, "output")
    sidecar_info: dict[str, Any] = {"model_card": None, "labels": None}

    card_name = artifact.get("model_card")
    card_path = safe_sidecar(model_path, card_name.strip(), result) if isinstance(card_name, str) and card_name.strip() else model_path.with_suffix(".model-card.yml")
    if card_path and card_path.is_file():
        sidecar_info["model_card"] = str(card_path)
        card = load_yaml(card_path, result)
        if not section(card, "model").get("version") and not config.get("version"):
            result.warn(f"{key}: model card/config should include model.version")
        if not section(card, "evaluation") and not section(card, "metrics"):
            result.warn(f"{key}: model card should include evaluation or metrics")
    elif strict_sidecars:
        result.error(f"{key}: model card not found: {card_path}")
    else:
        result.warn(f"{key}: model card not found: {card_path}")

    labels_name = artifact.get("labels")
    labels_path = safe_sidecar(model_path, labels_name.strip(), result) if isinstance(labels_name, str) and labels_name.strip() else model_path.with_suffix(".labels.txt")
    classes = output.get("classes")
    has_inline_labels = isinstance(classes, list) or (isinstance(classes, str) and classes.lower() == "coco")
    if labels_path and labels_path.is_file():
        labels = list_labels(labels_path)
        sidecar_info["labels"] = {"path": str(labels_path), "count": len(labels)}
        if not labels:
            result.error(f"{key}: labels file is empty: {labels_path}")
    elif not has_inline_labels and config.get("task") in {"detection", "classification"}:
        if strict_sidecars:
            result.error(f"{key}: labels file not found and output.classes is not provided: {labels_path}")
        else:
            result.warn(f"{key}: labels file not found and output.classes is not provided: {labels_path}")

    return sidecar_info


def validate_model(
    key: str,
    config: dict[str, Any],
    models_root: Path,
    strict_hash: bool,
    strict_sidecars: bool,
    result: CheckResult,
) -> dict[str, Any]:
    model_info: dict[str, Any] = {"key": key, "exists": False}
    split = split_model_key(key, result)
    if split is None:
        return model_info

    project, model = split
    model_path = (models_root / project / model).resolve()
    try:
        model_path.relative_to(models_root.resolve())
    except ValueError:
        result.error(f"{key}: model path escapes models root")
        return model_info

    model_info["path"] = str(model_path)
    if not model_path.is_file():
        result.error(f"{key}: model file not found: {model_path}")
        return model_info

    model_info["exists"] = True
    model_info["bytes"] = model_path.stat().st_size

    task = str(config.get("task") or config.get("type") or "").lower()
    if task not in {"detection", "classification", "reid", "yolo"}:
        result.warn(f"{key}: task/type is not a known first-stage task: {task or '<missing>'}")

    input_config = section(config, "input")
    output_config = section(config, "output")
    if not input_config.get("size") and not config.get("input_size"):
        result.warn(f"{key}: input.size is missing; service will infer from ONNX shape")
    if not output_config and task in {"detection", "classification", "reid", "yolo"}:
        result.warn(f"{key}: output section is missing")

    expected_sha = str(section(config, "artifact").get("sha256") or config.get("sha256") or "").strip().lower()
    digest = sha256_file(model_path)
    model_info["sha256"] = digest
    if expected_sha:
        model_info["expected_sha256"] = expected_sha
        if expected_sha != digest:
            result.error(f"{key}: sha256 mismatch: expected {expected_sha}, actual {digest}")
    elif strict_hash:
        result.error(f"{key}: artifact.sha256 is required in strict mode")
    else:
        result.warn(f"{key}: artifact.sha256 is empty")

    model_info["sidecars"] = validate_sidecars(key, config, model_path, result, strict_sidecars)
    return model_info


def validate_config(args: argparse.Namespace) -> dict[str, Any]:
    result = CheckResult()
    config_path = Path(args.config).resolve()
    models_root = Path(args.models_root or os.getenv("MODELS_ROOT", "/models")).resolve()
    raw = load_yaml(config_path, result)
    models = raw.get("models", raw)
    aliases = raw.get("aliases", {})
    if not isinstance(models, dict):
        result.error("models must be a mapping")
        models = {}
    if not isinstance(aliases, dict):
        result.error("aliases must be a mapping")
        aliases = {}

    selected = set(args.model_id or [])
    model_items = {str(key): value for key, value in models.items() if isinstance(value, dict)}
    if selected:
        resolved: set[str] = set()
        for item in selected:
            if item in aliases:
                target = alias_target(item, aliases[item], result)
                if target:
                    resolved.add(target)
            else:
                resolved.add(item)
        model_items = {key: value for key, value in model_items.items() if key in resolved}
        missing = sorted(resolved - set(model_items))
        for key in missing:
            result.error(f"selected model is not configured: {key}")

    alias_info = {}
    for name, config in aliases.items():
        target = alias_target(str(name), config, result)
        alias_info[str(name)] = target
        if target and target not in models:
            result.warn(f"alias target is not in models mapping: {name} -> {target}")

    checked_models = [
        validate_model(
            key,
            config,
            models_root,
            strict_hash=args.strict_hash,
            strict_sidecars=args.strict_sidecars,
            result=result,
        )
        for key, config in sorted(model_items.items())
    ]

    return {
        "ok": result.ok,
        "config_path": str(config_path),
        "models_root": str(models_root),
        "model_count": len(checked_models),
        "aliases": alias_info,
        "models": checked_models,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model package files referenced by models.yml.")
    parser.add_argument("--config", default="models.yml", help="Path to models.yml.")
    parser.add_argument("--models-root", default=None, help="Shared model root. Defaults to MODELS_ROOT or /models.")
    parser.add_argument("--model-id", action="append", help="Only validate this model key or alias. Can be repeated.")
    parser.add_argument("--strict-hash", action="store_true", help="Require artifact.sha256 and verify it.")
    parser.add_argument("--strict-sidecars", action="store_true", help="Require model cards and labels where applicable.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report = validate_config(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "OK" if report["ok"] else "FAILED"
        print(f"model package validation: {status}")
        print(f"config: {report['config_path']}")
        print(f"models_root: {report['models_root']}")
        print(f"models_checked: {report['model_count']}")
        for warning in report["warnings"]:
            print(f"warning: {warning}")
        for error in report["errors"]:
            print(f"error: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
