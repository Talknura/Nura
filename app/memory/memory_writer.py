"""
Append-only JSONL memory writer.
No LLM involvement in storage decisions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.memory.memory_classifier import classify_text
from config.settings import settings


@dataclass
class MemoryRecord:
    id: str
    user_id: int
    role: str
    text: str
    memory_type: str
    importance: float
    session_id: str
    ts: str
    source: str
    metadata: Dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_record(
    user_id: int,
    role: str,
    text: str,
    session_id: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    ts: Optional[str] = None,
) -> Optional[MemoryRecord]:
    """
    Append a memory record to disk (JSONL).
    Returns MemoryRecord or None if dropped by classifier.
    """
    metadata = metadata or {}
    if not text or not text.strip():
        return None

    cls = classify_text(text)
    if cls.action == "drop":
        return None

    record = MemoryRecord(
        id=str(uuid.uuid4()),
        user_id=int(user_id),
        role=role,
        text=text.strip(),
        memory_type=cls.memory_type,
        importance=float(cls.importance),
        session_id=session_id,
        ts=ts or _now_iso(),
        source=source,
        metadata=metadata,
    )

    path = Path(settings.memory_jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")

    return record


def clear_user_records(user_id: int) -> int:
    """
    Remove all records for a user from the JSONL file.
    Returns number of removed records.
    """
    path = Path(settings.memory_jsonl_path)
    if not path.exists():
        return 0

    kept = []
    removed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(obj.get("user_id", -1)) == int(user_id):
                removed += 1
                continue
            kept.append(obj)

    with path.open("w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")

    return removed
