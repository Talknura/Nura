from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, TypedDict


class MemoryEngineIngestInput(TypedDict):
    user_id: int
    role: str
    text: str
    session_id: str
    ts: datetime
    temporal_tags: Dict[str, Any]
    source: str
    metadata: Dict[str, Any]


class MemoryEngineSearchInput(TypedDict):
    user_id: int
    query: str
    k: int


class MemoryEngineFactsInput(TypedDict):
    user_id: int


class MemoryEngineOutput(TypedDict):
    id: int
    memory_type: str
    importance: float


MEMORY_ENGINE_FORBIDDEN_KEYS = {"embedding", "vector", "prompt"}
MEMORY_ENGINE_MAX_KB = 16.0
