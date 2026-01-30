from pydantic import BaseModel
from datetime import datetime
from typing import Literal, Any

MemoryType = Literal["episodic", "semantic", "summary"]

class Event(BaseModel):
    user_id: int
    role: Literal["user", "assistant", "tool"]
    text: str
    session_id: str
    ts: datetime
    source: str = "chat"
    metadata: dict[str, Any] = {}

class MemoryItem(BaseModel):
    id: int | None = None
    user_id: int
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    created_at: datetime | None = None
    temporal_tags: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

class SemanticFact(BaseModel):
    user_id: int
    key: str
    value: str
    confidence: float = 0.6
    provenance_memory_id: int | None = None
