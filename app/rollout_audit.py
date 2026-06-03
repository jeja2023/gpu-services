import json
from typing import Any

from app.observability import wall_time
from app.settings import ROLLOUT_AUDIT_PATH


def write_rollout_audit(event: str, payload: dict[str, Any]) -> None:
    record = {
        "time": wall_time(),
        "event": event,
        **payload,
    }
    ROLLOUT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ROLLOUT_AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
