"""Run fixed-sample regression checks against a running inference service."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CompareResult:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            raw = json.load(file)
        else:
            raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a mapping")
    return raw


def load_expected(case: dict[str, Any], base_dir: Path) -> Any:
    if "expected" in case:
        return case["expected"]
    expected_path = case.get("expected_path")
    if not isinstance(expected_path, str):
        return None
    path = (base_dir / expected_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            return json.load(file)
        return yaml.safe_load(file)


def compare_values(
    actual: Any,
    expected: Any,
    result: CompareResult,
    path: str = "$",
    tolerance: float = 1e-6,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            result.error(path, f"expected mapping, got {type(actual).__name__}")
            return
        for key, expected_value in expected.items():
            if key not in actual:
                result.error(f"{path}.{key}", "missing key")
                continue
            compare_values(actual[key], expected_value, result, f"{path}.{key}", tolerance)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            result.error(path, f"expected list, got {type(actual).__name__}")
            return
        if len(actual) < len(expected):
            result.error(path, f"expected at least {len(expected)} items, got {len(actual)}")
            return
        for index, expected_value in enumerate(expected):
            compare_values(actual[index], expected_value, result, f"{path}[{index}]", tolerance)
        return

    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            result.error(path, f"expected number, got {type(actual).__name__}")
            return
        if abs(float(actual) - expected) > tolerance:
            result.error(path, f"expected {expected} +/- {tolerance}, got {actual}")
        return

    if actual != expected:
        result.error(path, f"expected {expected!r}, got {actual!r}")


def open_case_files(case: dict[str, Any], base_dir: Path) -> tuple[list[tuple[str, tuple[str, Any, str]]], list[Any]]:
    handles = []
    file_items = []
    files = case.get("files") or {}
    if not isinstance(files, dict):
        raise ValueError("case.files must be a mapping of form field to path or paths")

    for field, raw_paths in files.items():
        paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
        for raw_path in paths:
            if not isinstance(raw_path, str):
                raise ValueError(f"file path for field {field} must be a string")
            path = (base_dir / raw_path).resolve()
            handle = path.open("rb")
            handles.append(handle)
            file_items.append((field, (path.name, handle, "application/octet-stream")))
    return file_items, handles


def run_case(client: Any, base_url: str, token: str | None, case: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    method = str(case.get("method", "GET")).upper()
    path = str(case.get("path") or case.get("endpoint") or "")
    if not path.startswith("/"):
        raise ValueError("case.path must start with /")

    headers = dict(case.get("headers") or {})
    if token:
        headers.setdefault("Authorization", f"Bearer {token}")
        headers.setdefault("X-API-Key", token)

    files, handles = open_case_files(case, base_dir)
    try:
        response = client.request(
            method,
            base_url.rstrip("/") + path,
            params=case.get("query"),
            headers=headers,
            json=case.get("json"),
            data=case.get("form"),
            files=files or None,
        )
    finally:
        for handle in handles:
            handle.close()

    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return {"status_code": response.status_code, "payload": payload}


def run_regression(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required. Install requirements-dev.txt before running regression checks.") from exc

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest must include cases list")

    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise ValueError(f"case #{index} must be a mapping")
            name = str(case.get("name") or f"case_{index}")
            expected_status = int(case.get("expected_status", 200))
            tolerance = float(case.get("tolerance", manifest.get("tolerance", args.tolerance)))
            actual = run_case(client, args.base_url, args.token, case, base_dir)
            comparison = CompareResult()
            if actual["status_code"] != expected_status:
                comparison.error("$status_code", f"expected {expected_status}, got {actual['status_code']}")
            expected = load_expected(case, base_dir)
            if expected is not None:
                compare_values(actual["payload"], expected, comparison, tolerance=tolerance)
            results.append(
                {
                    "name": name,
                    "ok": comparison.ok,
                    "status_code": actual["status_code"],
                    "errors": comparison.errors,
                }
            )

    return {"ok": all(item["ok"] for item in results), "case_count": len(results), "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed regression cases against a running gpu-services endpoint.")
    parser.add_argument("--manifest", required=True, help="Regression manifest YAML/JSON.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9001", help="Service base URL.")
    parser.add_argument("--token", default=None, help="API token for protected endpoints.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Default float comparison tolerance.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report = run_regression(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"regression check: {'OK' if report['ok'] else 'FAILED'}")
        for item in report["cases"]:
            marker = "ok" if item["ok"] else "fail"
            print(f"{marker}: {item['name']} status={item['status_code']}")
            for error in item["errors"]:
                print(f"  {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
