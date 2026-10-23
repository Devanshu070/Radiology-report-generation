"""Local content-addressed cache for successful structured model responses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from pydantic import ValidationError

from ..schemas import GeneratedSections


def request_fingerprint(payload: dict[str, Any]) -> str:
    """Hash a canonical request without writing its contents to the cache index."""
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ResponseCache:
    """Read and atomically write validated successful generations only."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, fingerprint: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("Response cache fingerprints must be 64-character lowercase SHA-256 digests.")
        return self.directory / f"{fingerprint}.json"

    def get(self, fingerprint: str) -> GeneratedSections | None:
        path = self._path(fingerprint)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            if data.get("fingerprint") != fingerprint:
                return None
            return GeneratedSections.model_validate(data.get("response"))
        except (OSError, json.JSONDecodeError, ValidationError, TypeError):
            return None

    def put(self, fingerprint: str, response: GeneratedSections) -> Path:
        path = self._path(fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{fingerprint}.", suffix=".tmp",
                dir=path.parent, delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(json.dumps(
                    {"fingerprint": fingerprint, "response": response.model_dump()},
                    ensure_ascii=False,
                    indent=2,
                ) + "\n")
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path
