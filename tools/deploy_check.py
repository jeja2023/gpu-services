"""Static deployment checks for gpu-services."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DeployReport:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "ok": ok, "detail": detail})

    @property
    def ok(self) -> bool:
        return all(item["ok"] for item in self.checks)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_files(root: Path, report: DeployReport) -> None:
    required = [
        "main.py",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "models.yml",
        "app/server.py",
        "app/routes.py",
        "app/runtime.py",
        "app/inference.py",
        "app/vision.py",
        "tools/validate_model_package.py",
        "tools/service_smoke_test.py",
        "tools/regression_check.py",
        "tools/worker_control.py",
    ]
    missing = [item for item in required if not (root / item).is_file()]
    report.add("required_files", not missing, {"missing": missing})


def check_python_syntax(root: Path, report: DeployReport) -> None:
    errors = []
    for path in [root / "main.py", *sorted((root / "app").glob("*.py")), *sorted((root / "tools").glob("*.py"))]:
        try:
            ast.parse(read_text(path), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path}: {exc}")
    report.add("python_syntax", not errors, {"errors": errors, "file_count": len(list((root / "app").glob("*.py"))) + len(list((root / "tools").glob("*.py"))) + 1})


def check_models_config(root: Path, report: DeployReport) -> None:
    path = root / "models.yml"
    try:
        raw = yaml.safe_load(read_text(path)) or {}
    except Exception as exc:
        report.add("models_yml_parse", False, str(exc))
        return
    if not isinstance(raw, dict):
        report.add("models_yml_parse", False, "root must be a mapping")
        return
    models = raw.get("models", raw)
    aliases = raw.get("aliases", {})
    model_ok = isinstance(models, dict) and bool(models)
    alias_ok = isinstance(aliases, dict)
    report.add("models_yml_models", model_ok, {"model_count": len(models) if isinstance(models, dict) else 0})
    report.add("models_yml_aliases", alias_ok, {"alias_count": len(aliases) if isinstance(aliases, dict) else 0})
    if isinstance(models, dict):
        missing_task = [str(key) for key, value in models.items() if isinstance(value, dict) and not (value.get("task") or value.get("type"))]
        report.add("models_yml_task_fields", not missing_task, {"missing_task": missing_task})


def check_docker_files(root: Path, report: DeployReport) -> None:
    dockerfile = read_text(root / "Dockerfile")
    compose = yaml.safe_load(read_text(root / "docker-compose.yml")) or {}
    services = compose.get("services", {}) if isinstance(compose, dict) else {}
    service_names = sorted(services) if isinstance(services, dict) else []
    report.add("dockerfile_copies_app", "COPY app /workspace/app" in dockerfile, None)
    report.add("dockerfile_copies_main", "COPY main.py /workspace/main.py" in dockerfile, None)
    report.add("compose_services", bool(service_names), {"services": service_names})
    gpu_like = [
        name
        for name, service in services.items()
        if isinstance(service, dict)
        and ("NVIDIA_VISIBLE_DEVICES" in str(service.get("environment", "")) or "gpus" in service)
    ] if isinstance(services, dict) else []
    report.add("compose_gpu_configuration", bool(gpu_like), {"gpu_services": gpu_like})
    volumes = [
        volume
        for service in services.values()
        if isinstance(service, dict)
        for volume in service.get("volumes", [])
    ] if isinstance(services, dict) else []
    volume_targets = [
        str(volume.get("target"))
        if isinstance(volume, dict)
        else str(volume).split(":")[1] if ":" in str(volume) else str(volume)
        for volume in volumes
    ]
    report.add(
        "compose_model_config_mount",
        "/workspace/models.yml" in volume_targets,
        {"volumes": volumes},
    )
    report.add(
        "compose_model_config_no_autocreate",
        any(
            isinstance(volume, dict)
            and volume.get("target") == "/workspace/models.yml"
            and isinstance(volume.get("bind"), dict)
            and volume["bind"].get("create_host_path") is False
            for volume in volumes
        ),
        {"volumes": volumes},
    )
    report.add(
        "compose_runtime_state_mount",
        "/workspace/runtime-state" in volume_targets,
        {"volumes": volumes},
    )
    ready_healthchecks = [
        name
        for name, service in services.items()
        if isinstance(service, dict) and "/ready" in str(service.get("healthcheck", ""))
    ] if isinstance(services, dict) else []
    report.add("compose_ready_healthcheck", bool(ready_healthchecks), {"services": ready_healthchecks})


def check_import_app(root: Path, report: DeployReport) -> None:
    try:
        sys.path.insert(0, str(root))
        import main

        paths = {route.path for route in main.app.routes}
        required = {
            "/health",
            "/ready",
            "/models",
            "/predict",
            "/vision/infer",
            "/vision/batch-infer",
            "/rollout/aliases",
            "/rollout/aliases/preview",
            "/rollout/aliases/switch",
            "/rollout/aliases/weighted",
            "/rollout/aliases/rollback",
        }
        missing = sorted(required - paths)
        report.add("app_import", True, {"route_count": len(paths)})
        report.add("app_required_routes", not missing, {"missing": missing})
    except Exception as exc:
        report.add("app_import", False, str(exc))


def run_checks(args: argparse.Namespace) -> DeployReport:
    root = Path(args.root).resolve()
    report = DeployReport()
    check_required_files(root, report)
    check_python_syntax(root, report)
    check_models_config(root, report)
    check_docker_files(root, report)
    if args.import_app:
        check_import_app(root, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run static deployment checks for gpu-services.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--import-app", action="store_true", help="Import main.app and verify key routes.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report = run_checks(args)
    output = {"ok": report.ok, "checks": report.checks}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"deploy check: {'OK' if report.ok else 'FAILED'}")
        for item in report.checks:
            marker = "ok" if item["ok"] else "fail"
            print(f"{marker}: {item['name']}")
            if not item["ok"]:
                print(f"  detail: {item['detail']}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
