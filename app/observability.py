import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import Request


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("gpu-worker")


def now() -> float:
    return time.perf_counter()


def wall_time() -> float:
    return time.time()


def request_id_from_headers(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def log_json(level: int, event: str, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, ensure_ascii=False))
