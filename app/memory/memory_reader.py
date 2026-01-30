"""
JSONL memory reader and summarizer.
No LLM involvement in memory retrieval or summarization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.memory.memory_summarizer import summarize_turns
from config.settings import settings


@dataclass
class MemoryHit:
    record: Dict
    score: float


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in text.split() if t.strip()}


def _score_record(record: Dict, query_tokens: set[str], now: datetime) -> float:
    text = str(record.get("text", "")).lower()
    tokens = _tokenize(text)
    overlap = len(tokens.intersection(query_tokens))
    ts = _parse_ts(record.get("ts", ""))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
    recency = 1.0 / (1.0 + age_days)
    importance = float(record.get("importance", 0.5))
    return (overlap * 2.0) + (recency * 1.5) + (importance * 1.0)


def _iter_records(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_user_records(user_id: int) -> List[Dict]:
    path = Path(settings.memory_jsonl_path)
    records = [r for r in _iter_records(path) if int(r.get("user_id", -1)) == int(user_id)]
    return records


def get_relevant_records(user_id: int, query: str, max_records: int = 8) -> List[Dict]:
    records = load_user_records(user_id)
    if not records:
        return []
    now = datetime.now(timezone.utc)
    query_tokens = _tokenize(query)
    scored: List[MemoryHit] = []
    for r in records:
        score = _score_record(r, query_tokens, now)
        scored.append(MemoryHit(record=r, score=score))
    scored.sort(key=lambda h: h.score, reverse=True)
    return [h.record for h in scored[:max_records]]


def build_memory_summary(user_id: int, query: str, max_records: int = 8) -> str:
    records = get_relevant_records(user_id, query, max_records=max_records)
    if not records:
        return ""
    texts = [r.get("text", "") for r in records if r.get("text")]
    return summarize_turns(texts)
