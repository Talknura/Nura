"""
Memory Engine for Nura.

Three-tier memory architecture:
1. FACTS (Semantic Memory) - Personal truths, ONE value per key, NO decay
2. MILESTONES (Life Events) - Timestamped events, NO decay
3. EPISODES (Episodic Memory) - Day-to-day conversations, DOES decay

Uses semantic classification to determine which tier each memory belongs to.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Protocol

from config.settings import settings
from app.memory.memory_classifier import classify_text
from app.memory.memory_summarizer import summarize_turns
from app.memory.memory_store import (
    insert_memory, search_memories, recent_memories,
    upsert_fact, get_facts,
    # New three-tier functions
    upsert_fact_v2, get_all_facts, search_facts,
    insert_milestone, get_milestones, search_milestones, check_duplicate_milestone
)
from app.vector.embedding_service import EmbeddingService
from app.db.session import get_db_context

# Try to import semantic memory classifier (three-tier architecture)
try:
    from app.semantic.memory_architecture import (
        get_semantic_memory_classifier,
        MemoryType,
        ClassifiedMemory
    )
    SEMANTIC_MEMORY_CLASSIFIER_AVAILABLE = True
except ImportError:
    SEMANTIC_MEMORY_CLASSIFIER_AVAILABLE = False
    print("[MemoryEngine] Semantic memory classifier not available, using legacy mode")

# Fallback: Try to import semantic fact extractor
try:
    from app.semantic.fact_concepts import get_semantic_fact_extractor
    SEMANTIC_FACTS_AVAILABLE = True
except ImportError:
    SEMANTIC_FACTS_AVAILABLE = False


class MemoryEngineProtocol(Protocol):
    def ingest_event(self, user_id: int, role: str, text: str, session_id: str, ts: datetime, temporal_tags: dict[str, Any], source: str = "chat", metadata: dict[str, Any] | None = None) -> dict | None: ...
    def search(self, user_id: int, query: str, k: int = 10) -> list[dict]: ...
    def recent(self, user_id: int, k: int) -> list[dict]: ...
    def facts(self, user_id: int) -> dict[str, str]: ...


class MemoryEngine:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def ingest_event(
        self,
        user_id: int,
        role: str,
        text: str,
        session_id: str,
        ts: datetime,
        temporal_tags: dict[str, Any],
        source: str = "chat",
        metadata: dict[str, Any] | None = None
    ) -> dict | None:
        """
        Ingest a conversation event using three-tier memory architecture.

        1. Classify text as FACT, MILESTONE, or EPISODE
        2. Store in appropriate table
        3. No decay for FACTS and MILESTONES
        """
        metadata = metadata or {}

        # Basic noise filtering
        cls = classify_text(text)
        if cls.action == "drop":
            return None

        # Generate embedding once
        embedding = self.embedding_service.embed(text)
        embedding_bytes = embedding.tobytes()

        # =================================================================
        # THREE-TIER SEMANTIC CLASSIFICATION (preferred)
        # =================================================================
        if SEMANTIC_MEMORY_CLASSIFIER_AVAILABLE:
            try:
                classifier = get_semantic_memory_classifier()
                classified = classifier.classify(text, threshold=0.40)

                return self._store_classified_memory(
                    user_id=user_id,
                    classified=classified,
                    embedding_bytes=embedding_bytes,
                    ts=ts,
                    temporal_tags=temporal_tags,
                    role=role,
                    session_id=session_id,
                    source=source,
                    metadata=metadata
                )
            except Exception as e:
                print(f"[MemoryEngine] Semantic classification failed, using fallback: {e}")

        # =================================================================
        # FALLBACK: Legacy single-table storage
        # =================================================================
        return self._legacy_ingest(
            user_id=user_id,
            role=role,
            text=text,
            session_id=session_id,
            ts=ts,
            temporal_tags=temporal_tags,
            source=source,
            metadata=metadata,
            cls=cls,
            embedding_bytes=embedding_bytes
        )

    def _store_classified_memory(
        self,
        user_id: int,
        classified: ClassifiedMemory,
        embedding_bytes: bytes,
        ts: datetime,
        temporal_tags: dict,
        role: str,
        session_id: str,
        source: str,
        metadata: dict
    ) -> dict:
        """Store memory in the appropriate tier based on classification."""

        if classified.memory_type == MemoryType.FACT:
            # =================================================================
            # FACTS TABLE - One value per key, no decay
            # =================================================================
            upsert_fact_v2(
                user_id=user_id,
                key=classified.fact_key,
                value=classified.fact_value,
                confidence=classified.confidence,
                provenance_memory_id=None,  # Will be set after episode insert
                embedding=embedding_bytes
            )

            # Also store in episodes for conversation continuity
            mem_id = insert_memory(
                user_id=user_id,
                content=classified.original_text,
                memory_type="semantic",  # Mark as semantic (fact-bearing)
                importance=0.9,  # Facts are high importance
                created_at=ts,
                temporal_tags=temporal_tags,
                metadata={
                    "role": role,
                    "session_id": session_id,
                    "source": source,
                    "memory_tier": "fact",
                    "fact_key": classified.fact_key,
                    **metadata
                },
                embedding=embedding_bytes
            )

            print(f"[MemoryEngine] Stored FACT: {classified.fact_key}")
            return {
                "id": mem_id,
                "memory_type": "fact",
                "memory_tier": "fact",
                "fact_key": classified.fact_key,
                "confidence": classified.confidence
            }

        elif classified.memory_type == MemoryType.MILESTONE:
            # =================================================================
            # MILESTONES TABLE - Life events, no decay
            # =================================================================

            # Check for duplicates (don't store "I got married" twice)
            is_duplicate = check_duplicate_milestone(
                user_id=user_id,
                event_type=classified.event_type,
                description=classified.event_description,
                threshold=0.85
            )

            if not is_duplicate:
                milestone_id = insert_milestone(
                    user_id=user_id,
                    event_type=classified.event_type,
                    description=classified.event_description,
                    event_date=classified.event_date,
                    confidence=classified.confidence,
                    provenance_memory_id=None,
                    embedding=embedding_bytes,
                    metadata={"source": source, "session_id": session_id}
                )
                print(f"[MemoryEngine] Stored MILESTONE: {classified.event_type}")
            else:
                print(f"[MemoryEngine] Skipped duplicate milestone: {classified.event_type}")
                milestone_id = None

            # Also store in episodes for conversation continuity
            mem_id = insert_memory(
                user_id=user_id,
                content=classified.original_text,
                memory_type="episodic",
                importance=0.85,  # Milestones are important
                created_at=ts,
                temporal_tags=temporal_tags,
                metadata={
                    "role": role,
                    "session_id": session_id,
                    "source": source,
                    "memory_tier": "milestone",
                    "event_type": classified.event_type,
                    **metadata
                },
                embedding=embedding_bytes
            )

            return {
                "id": mem_id,
                "memory_type": "milestone",
                "memory_tier": "milestone",
                "event_type": classified.event_type,
                "milestone_id": milestone_id,
                "confidence": classified.confidence
            }

        else:
            # =================================================================
            # EPISODES TABLE - Day-to-day, does decay
            # =================================================================
            mem_id = insert_memory(
                user_id=user_id,
                content=classified.original_text,
                memory_type="episodic",
                importance=0.5,  # Default importance for episodes
                created_at=ts,
                temporal_tags=temporal_tags,
                metadata={
                    "role": role,
                    "session_id": session_id,
                    "source": source,
                    "memory_tier": "episode",
                    **metadata
                },
                embedding=embedding_bytes
            )

            # Run summarization check
            self._check_summarization(user_id, ts, temporal_tags, session_id)

            return {
                "id": mem_id,
                "memory_type": "episodic",
                "memory_tier": "episode",
                "confidence": classified.confidence
            }

    def _check_summarization(self, user_id: int, ts: datetime, temporal_tags: dict, session_id: str):
        """Check if episodic memories need summarization."""
        with get_db_context() as conn:
            episodic_count_row = conn.execute(
                """SELECT COUNT(*) FROM memories
                   WHERE user_id=? AND memory_type='episodic'
                   AND id > IFNULL((SELECT MAX(id) FROM memories WHERE user_id=? AND memory_type='summary'), 0)""",
                (user_id, user_id),
            ).fetchone()
            episodic_count = int(episodic_count_row[0]) if episodic_count_row else 0

            if episodic_count >= settings.summary_every_n_turns:
                rows = conn.execute(
                    """SELECT id, content FROM memories
                       WHERE user_id=? AND memory_type='episodic'
                       AND id > IFNULL((SELECT MAX(id) FROM memories WHERE user_id=? AND memory_type='summary'), 0)
                       ORDER BY id ASC""",
                    (user_id, user_id),
                ).fetchall()
                texts = [r["content"] for r in rows]
                source_ids = [r["id"] for r in rows]
                summary_text = summarize_turns(texts)
                summary_embedding = self.embedding_service.embed(summary_text).tobytes()
                insert_memory(
                    user_id=user_id,
                    content=summary_text,
                    memory_type="summary",
                    importance=0.5,
                    created_at=ts,
                    temporal_tags=temporal_tags,
                    metadata={"source_memory_ids": source_ids, "role": "system", "session_id": session_id, "source": "summarization"},
                    embedding=summary_embedding,
                )

    def _legacy_ingest(
        self,
        user_id: int,
        role: str,
        text: str,
        session_id: str,
        ts: datetime,
        temporal_tags: dict,
        source: str,
        metadata: dict,
        cls,
        embedding_bytes: bytes
    ) -> dict:
        """Legacy single-table ingestion (fallback)."""
        mem_id = insert_memory(
            user_id=user_id,
            content=text,
            memory_type=cls.memory_type,
            importance=cls.importance,
            created_at=ts,
            temporal_tags=temporal_tags,
            metadata={"role": role, "session_id": session_id, "source": source, **metadata},
            embedding=embedding_bytes,
        )

        self._check_summarization(user_id, ts, temporal_tags, session_id)

        # Legacy fact extraction
        if SEMANTIC_FACTS_AVAILABLE:
            try:
                extractor = get_semantic_fact_extractor()
                result = extractor.extract(text, threshold=0.50)
                for fact in result.facts:
                    upsert_fact(user_id, fact.key, fact.value, fact.confidence, mem_id)
            except Exception:
                pass

        return {"id": mem_id, "memory_type": cls.memory_type, "importance": cls.importance}

    def search(self, user_id: int, query: str, k: int = 10, max_candidates: int = 1000) -> list[dict]:
        """
        Search across all three memory tiers.

        Facts and milestones are ALWAYS included (no decay).
        Episodes are searched with recency consideration.
        """
        query_emb = self.embedding_service.embed(query)

        results = []

        # Search FACTS (always relevant, no decay)
        try:
            fact_results = search_facts(user_id, query_emb, k=5)
            for f in fact_results:
                results.append({
                    "id": f"fact_{f['key']}",
                    "content": f["value"],
                    "memory_type": "fact",
                    "memory_tier": "fact",
                    "importance": 1.0,  # Facts are always important
                    "similarity": f["similarity"],
                    "fact_key": f["key"],
                    "created_at": None,  # Facts are timeless
                })
        except Exception:
            pass

        # Search MILESTONES (always relevant, no decay)
        try:
            milestone_results = search_milestones(user_id, query_emb, k=5)
            for m in milestone_results:
                results.append({
                    "id": m["id"],
                    "content": m["description"],
                    "memory_type": "milestone",
                    "memory_tier": "milestone",
                    "importance": 0.95,
                    "similarity": m["similarity"],
                    "event_type": m["event_type"],
                    "event_date": m["event_date"],
                    "created_at": m.get("event_date"),
                })
        except Exception:
            pass

        # Search EPISODES (with decay consideration)
        episode_results = search_memories(
            user_id=user_id,
            query=query,
            k=k,
            query_embedding=query_emb,
            max_candidates=max_candidates
        )
        for e in episode_results:
            e["memory_tier"] = "episode"
        results.extend(episode_results)

        # Sort by similarity (facts and milestones naturally rank high)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return results[:k]

    def recent(self, user_id: int, k: int) -> list[dict]:
        """Get recent memories from episodes table."""
        return recent_memories(user_id=user_id, k=k)

    def facts(self, user_id: int) -> dict[str, str]:
        """Get all facts for a user (key-value pairs)."""
        return get_facts(user_id=user_id)

    def all_facts(self, user_id: int) -> list[dict]:
        """Get all facts with full details including history."""
        return get_all_facts(user_id=user_id)

    def milestones(self, user_id: int, event_type: str | None = None) -> list[dict]:
        """Get milestones (life events) for a user."""
        return get_milestones(user_id=user_id, event_type=event_type)
