"""Run operational actions across multiple gpu-services workers."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}", "X-API-Key": token}


def split_model_id(model_id: str) -> dict[str, str]:
    if "/" not in model_id:
        raise ValueError(f"model must use project/model.onnx format: {model_id}")
    project_name, model_name = model_id.split("/", 1)
    if not project_name or not model_name:
        raise ValueError(f"model must use project/model.onnx format: {model_id}")
    return {"project_name": project_name, "model_name": model_name}


def request_worker(base_url: str, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required. Install requirements-dev.txt before running worker control.") from exc

    base_url = base_url.rstrip("/")
    headers = auth_headers(args.token)
    timeout = args.timeout
    action = args.action
    with httpx.Client(timeout=timeout) as client:
        if action == "health":
            response = client.get(f"{base_url}/health")
        elif action == "ready":
            response = client.get(f"{base_url}/ready")
        elif action == "reload-config":
            response = client.post(f"{base_url}/reload-config", headers=headers)
        elif action == "aliases":
            response = client.get(f"{base_url}/rollout/aliases", headers=headers)
        elif action == "warmup":
            models = [split_model_id(item) for item in args.model]
            response = client.post(f"{base_url}/warmup", headers=headers, json={"models": models})
        elif action in {"reload", "unload"}:
            if len(args.model) != 1:
                raise ValueError(f"{action} requires exactly one --model")
            response = client.post(f"{base_url}/{action}", headers=headers, json=split_model_id(args.model[0]))
        else:
            raise ValueError(f"unsupported action: {action}")

    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return {
        "base_url": base_url,
        "action": action,
        "ok": 200 <= response.status_code < 300,
        "status_code": response.status_code,
        "payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an operational action across multiple gpu-services workers.")
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="Worker base URL. Can be repeated. Defaults to 9001 and 9002.",
    )
    parser.add_argument("--token", default=None, help="API token for protected endpoints.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--action",
        choices=["health", "ready", "reload-config", "aliases", "warmup", "reload", "unload"],
        required=True,
        help="Action to run on each worker.",
    )
    parser.add_argument("--model", action="append", default=[], help="Model id project/model.onnx for warmup/reload/unload.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    base_urls = args.base_url or ["http://127.0.0.1:9001", "http://127.0.0.1:9002"]
    results = []
    with ThreadPoolExecutor(max_workers=len(base_urls)) as executor:
        futures = {executor.submit(request_worker, base_url, args): base_url for base_url in base_urls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"base_url": futures[future], "action": args.action, "ok": False, "error": str(exc)})

    report = {"ok": all(item["ok"] for item in results), "results": sorted(results, key=lambda item: item["base_url"])}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"worker control: {'OK' if report['ok'] else 'FAILED'}")
        for item in report["results"]:
            marker = "ok" if item["ok"] else "fail"
            print(f"{marker}: {item['base_url']} action={item['action']}")
            if not item["ok"]:
                print(f"  detail: {item.get('error') or item.get('payload')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
