from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .client import TopoNaderLLMClient


class TopoNaderStructuredOutputError(ValueError):
    """Raised when an agent response cannot be recovered as JSON."""


def parse_json_object(text: str):
    """Parse a JSON object/array using conservative, non-semantic recovery.

    Recovery is intentionally limited to presentation mistakes: Markdown JSON
    fences, leading/trailing commentary, and redundant trailing delimiters. It
    does not guess missing commas, quotes, keys, or values.
    """
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    if cleaned not in candidates:
        candidates.append(cleaned)

    errors: list[str] = []
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if not isinstance(value, (dict, list)):
                raise TypeError(f"top-level JSON must be an object or array, got {type(value).__name__}")
            return value
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(str(exc))

        left_candidates = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
        if not left_candidates:
            continue
        left = min(left_candidates)
        try:
            value, _ = decoder.raw_decode(candidate, left)
            if not isinstance(value, (dict, list)):
                raise TypeError(f"top-level JSON must be an object or array, got {type(value).__name__}")
            return value
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(str(exc))

    detail = errors[-1] if errors else "no JSON object or array was found"
    raise TopoNaderStructuredOutputError(f"unable to parse structured JSON output: {detail}")


class TopoNaderBaseAgent:
    def __init__(
        self,
        name: str,
        client: TopoNaderLLMClient,
        log_dir: str | Path,
        structured_output_retries: int = 2,
    ):
        self.name = name
        self.client = client
        self.log_dir = Path(log_dir) / name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.structured_output_retries = max(0, int(structured_output_retries))

    def _write_record(self, record: dict) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        (self.log_dir / f"{stamp}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def call(self, system: str, user: str):
        started = time.time()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        attempt_records: list[dict] = []
        maximum_attempts = self.structured_output_retries + 1
        for attempt_index in range(maximum_attempts):
            response = self.client.complete(messages)
            record = {
                **response,
                "agent": self.name,
                "structured_output_attempt": attempt_index + 1,
                "elapsed_seconds": time.time() - started,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                payload = parse_json_object(response["content"])
            except (TopoNaderStructuredOutputError, json.JSONDecodeError) as exc:
                record.update({"parse_status": "failed", "parse_error": str(exc)})
                attempt_records.append(record)
                self._write_record(record)
                if attempt_index + 1 >= maximum_attempts:
                    raise TopoNaderStructuredOutputError(
                        f"{self.name} returned invalid JSON after {maximum_attempts} attempts: {exc}"
                    ) from exc
                messages.extend([
                    {"role": "assistant", "content": response["content"]},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as the required JSON. "
                            f"Parser error: {exc}. Correct the formatting error and return the complete "
                            "JSON object only. Do not use Markdown fences or add explanatory text."
                        ),
                    },
                ])
                continue

            attempt_records.append(record)
            record.update({
                "parse_status": "ok",
                "structured_output_attempts": len(attempt_records),
                "structured_output_retries_used": attempt_index,
                "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in attempt_records),
                "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in attempt_records),
                "elapsed_seconds": time.time() - started,
            })
            if attempt_index:
                record["retry_parse_errors"] = [
                    item["parse_error"] for item in attempt_records[:-1] if "parse_error" in item
                ]
            self._write_record(record)
            return payload, record

        raise AssertionError("structured output retry loop terminated unexpectedly")
