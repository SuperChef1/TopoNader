from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .client import TopoNaderLLMClient


def parse_json_object(text: str):
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        left_candidates = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
        if not left_candidates:
            raise
        left = min(left_candidates)
        right = max(cleaned.rfind("}"), cleaned.rfind("]"))
        return json.loads(cleaned[left : right + 1])


class TopoNaderBaseAgent:
    def __init__(self, name: str, client: TopoNaderLLMClient, log_dir: str | Path):
        self.name = name
        self.client = client
        self.log_dir = Path(log_dir) / name
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def call(self, system: str, user: str):
        started = time.time()
        response = self.client.complete([{"role": "system", "content": system}, {"role": "user", "content": user}])
        record = {
            **response,
            "agent": self.name,
            "elapsed_seconds": time.time() - started,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        (self.log_dir / f"{stamp}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return parse_json_object(response["content"]), record

