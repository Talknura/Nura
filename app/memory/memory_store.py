from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.db.session import get_db_context
from app.vector.vector_index import top_k_by_cosine

# Fast FAISS-based indexes (O(log n) instead of O(n))
try:
    from app.vector.memory_indexes import get_user_indexes, FAISS_AVAILABLE
    FAST_INDEX_AVAILABLE = FAISS_AVAILABLE
except ImportError:
    FAST_INDEX_AVAILABLE = False

def dt_to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def insert_memory(user_id: int, content: str, memory_type: str, importance: float,
                  created_at: datetime, temporal_tags: dict[str, Any], metadata: dict[str, Any],
                  embedding: bytes) -> int:
    with get_db_context() as conn:
        emb = embedding
        cur = conn.execute(
            """INSERT INTO memories(user_id, content, memory_type, importance, embedding, created_at, temporal_tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, content, memory_type, importance, emb, dt_to_iso(created_at), json.dumps(temporal_tags), json.dumps(metadata)),
        )
        conn.commit()
        memory_id = int(cur.lastrowid)

    # Add to FAISS index for fast retrieval
    if FAST_INDEX_AVAILABLE and embedding:
        try:
            indexes = get_user_indexes(user_id)
            indexes["episodes"].add(memory_id, embedding)
        except Exception:
            pass  # Index update failure shouldn't break insert

    return memory_id

def upsert_fact(user_id: int, key: str, value: str, confidence: float, provenance_memory_id: int | None) -> None:
    with get_db_context() as conn:
        conn.execute(
            """INSERT INTO facts(user_id, key, value, confidence, last_confirmed_at, provenance_memory_id)
               VALUES (?, ?, ?, ?, datetime('now'), ?)
               ON CONFLICT(user_id, key) DO UPDATE SET
                 value=excluded.value,
                 confidence=excluded.confidence,
                 last_confirmed_at=datetime('now'),
                 provenance_memory_id=excluded.provenance_memory_id
            """,
            (user_id, key, value, confidence, provenance_memory_id),
        )
        conn.commit()

def get_facts(user_id: int) -> dict[str, str]:
    with get_db_context() as conn:
        rows = conn.execute("SELECT key, value FROM facts WHERE user_id=?", (user_id,)).fetchall()
        return {r["key"]: r["value"] for r in rows}

def recent_memories(user_id: int, k: int) -> list[dict]:
    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT id, content, memory_type, importance, created_at, temporal_tags, metadata, last_accessed_at
               FROM memories WHERE user_id=? ORDER BY created_at DESC LIMIT ?""",
            (user_id, k),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "importance": r["importance"],
                "created_at": r["created_at"],
                "temporal_tags": json.loads(r["temporal_tags"] or "{}"),
                "metadata": json.loads(r["metadata"] or "{}"),
                "last_accessed_at": r["last_accessed_at"],
            })
        return out

def search_memories(user_id: int, query: str, k: int, query_embedding: np.ndarray, max_candidates: int = 1000, memory_type_filter: str | None = None) -> list[dict]:
    """
    Search memories with FAISS optimization for O(log n) retrieval.

    Args:
        user_id: User ID
        query: Search query text
        k: Number of results to return
        query_embedding: Query embedding vector
        max_candidates: Maximum number of memories to load (fallback only)
        memory_type_filter: Optional filter by memory_type
    """

    # =================================================================
    # FAST PATH: FAISS index (O(log n))
    # =================================================================
    if FAST_INDEX_AVAILABLE and not memory_type_filter:
        indexes = get_user_indexes(user_id)
        results = indexes["episodes"].search(query_embedding, k=k * 2)  # Over-fetch for filtering

        if results:
            now = datetime.now(timezone.utc)
            with get_db_context() as conn:
                hits = []
                for memory_id, similarity in results:
                    row = conn.execute(
                        """SELECT id, content, memory_type, importance, created_at, temporal_tags
                           FROM memories WHERE user_id=? AND id=?""",
                        (user_id, memory_id)
                    ).fetchone()
                    if row:
                        hits.append({
                            "id": row["id"],
                            "content": row["content"],
                            "memory_type": row["memory_type"],
                            "importance": row["importance"],
                            "created_at": row["created_at"],
                            "similarity": similarity,
                            "temporal_tags": json.loads(row["temporal_tags"] or "{}"),
                        })
                        # Update access time
                        conn.execute("UPDATE memories SET last_accessed_at=? WHERE id=?",
                                     (dt_to_iso(now), row["id"]))
                conn.commit()
                return hits[:k]

    # =================================================================
    # FALLBACK: Brute force O(n) - used when FAISS unavailable or filtering
    # =================================================================
    with get_db_context() as conn:
        if memory_type_filter:
            rows = conn.execute(
                """SELECT id, content, memory_type, importance, embedding, created_at, temporal_tags
                   FROM memories WHERE user_id=? AND memory_type=?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, memory_type_filter, max_candidates),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, content, memory_type, importance, embedding, created_at, temporal_tags
                   FROM memories WHERE user_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, max_candidates),
            ).fetchall()

        if not rows:
            return []

        vectors = []
        items = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            vectors.append(vec)
            items.append(r)

        qv = query_embedding
        top = top_k_by_cosine(qv, vectors, k=min(k, len(vectors)))

        hits = []
        now = datetime.now(timezone.utc)
        for idx, sim in top:
            r = items[idx]
            hits.append({
                "id": r["id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "importance": r["importance"],
                "created_at": r["created_at"],
                "similarity": float(sim),
                "temporal_tags": json.loads(r["temporal_tags"] or "{}"),
            })
            conn.execute("UPDATE memories SET last_accessed_at=? WHERE id=?", (dt_to_iso(now), r["id"]))
        conn.commit()
        return hits

def update_memory(memory_id: int, user_id: int, new_content: str, embedding_service) -> bool:
    """Update a specific memory's content. Returns True if updated, False if not found or unauthorized."""
    with get_db_context() as conn:
        row = conn.execute("SELECT id FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)).fetchone()
        if not row:
            return False
        new_embedding = embedding_service.embed(new_content).tobytes()
        now = datetime.now(timezone.utc)
        conn.execute(
            "UPDATE memories SET content=?, embedding=?, last_accessed_at=? WHERE id=? AND user_id=?",
            (new_content, new_embedding, dt_to_iso(now), memory_id, user_id),
        )
        conn.commit()
        _log_memory_correction(user_id, memory_id, "update", new_content)
        return True

def delete_memory(memory_id: int, user_id: int) -> bool:
    """Delete a specific memory. Returns True if deleted, False if not found or unauthorized."""
    with get_db_context() as conn:
        row = conn.execute("SELECT id, content FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)).fetchone()
        if not row:
            return False
        old_content = row["content"]
        conn.execute("DELETE FROM memories WHERE id=? AND user_id=?", (memory_id, user_id))
        conn.commit()
        _log_memory_correction(user_id, memory_id, "delete", old_content)
        return True

def update_fact(user_id: int, key: str, new_value: str, confidence: float) -> bool:
    """Update an existing fact. Returns True if updated, False if not found."""
    with get_db_context() as conn:
        row = conn.execute("SELECT key FROM facts WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE facts SET value=?, confidence=?, last_confirmed_at=datetime('now') WHERE user_id=? AND key=?",
            (new_value, confidence, user_id, key),
        )
        conn.commit()
        _log_fact_correction(user_id, key, "update", new_value)
        return True

def delete_fact(user_id: int, key: str) -> bool:
    """Delete a specific fact. Returns True if deleted, False if not found."""
    with get_db_context() as conn:
        row = conn.execute("SELECT key, value FROM facts WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        if not row:
            return False
        old_value = row["value"]
        conn.execute("DELETE FROM facts WHERE user_id=? AND key=?", (user_id, key))
        conn.commit()
        _log_fact_correction(user_id, key, "delete", old_value)
        return True

def _log_memory_correction(user_id: int, memory_id: int, action: str, content: str) -> None:
    """Log memory correction for audit trail."""
    import os
    log_dir = r"D:\Nura\Docs\memory_corrections"
    os.makedirs(log_dir, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "memory_id": memory_id,
        "action": action,
        "content": content,
    }
    log_file = os.path.join(log_dir, f"memory_corrections_{user_id}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=True) + "\n")

def _log_fact_correction(user_id: int, key: str, action: str, value: str) -> None:
    """Log fact correction for audit trail."""
    import os
    log_dir = r"D:\Nura\Docs\memory_corrections"
    os.makedirs(log_dir, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "fact_key": key,
        "action": action,
        "value": value,
    }
    log_file = os.path.join(log_dir, f"fact_corrections_{user_id}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=True) + "\n")


# =============================================================================
# FACTS TABLE - Enhanced with history tracking
# =============================================================================

def upsert_fact_v2(
    user_id: int,
    key: str,
    value: str,
    confidence: float,
    provenance_memory_id: int | None,
    embedding: bytes | None = None
) -> None:
    """
    Insert or update a fact with history tracking.

    If the fact already exists with a different value, the old value
    is preserved in the history field for contradiction tracking.
    """
    with get_db_context() as conn:
        # Check if fact exists
        existing = conn.execute(
            "SELECT value, history FROM facts WHERE user_id=? AND key=?",
            (user_id, key)
        ).fetchone()

        if existing:
            old_value = existing["value"]
            old_history = json.loads(existing["history"] or "[]")

            # If value changed, add old value to history
            if old_value != value:
                old_history.append({
                    "value": old_value,
                    "replaced_at": datetime.now(timezone.utc).isoformat()
                })

            conn.execute(
                """UPDATE facts SET
                    value=?, confidence=?, last_confirmed_at=datetime('now'),
                    provenance_memory_id=?, embedding=?, history=?
                   WHERE user_id=? AND key=?""",
                (value, confidence, provenance_memory_id, embedding,
                 json.dumps(old_history), user_id, key)
            )
        else:
            conn.execute(
                """INSERT INTO facts(user_id, key, value, confidence, last_confirmed_at,
                                     first_learned_at, provenance_memory_id, embedding, history)
                   VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), ?, ?, '[]')""",
                (user_id, key, value, confidence, provenance_memory_id, embedding)
            )
        conn.commit()

    # Add to FAISS index for fast retrieval
    if FAST_INDEX_AVAILABLE and embedding:
        try:
            indexes = get_user_indexes(user_id)
            indexes["facts"].add(key, embedding)
        except Exception:
            pass  # Index update failure shouldn't break upsert


def get_all_facts(user_id: int) -> list[dict]:
    """Get all facts for a user with full details."""
    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT key, value, confidence, last_confirmed_at, first_learned_at, history
               FROM facts WHERE user_id=?""",
            (user_id,)
        ).fetchall()
        return [
            {
                "key": r["key"],
                "value": r["value"],
                "confidence": r["confidence"],
                "last_confirmed_at": r["last_confirmed_at"],
                "first_learned_at": r["first_learned_at"],
                "history": json.loads(r["history"] or "[]")
            }
            for r in rows
        ]


def search_facts(user_id: int, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
    """Search facts by semantic similarity. Uses FAISS O(log n) when available."""

    # =================================================================
    # FAST PATH: FAISS index (O(log n))
    # =================================================================
    if FAST_INDEX_AVAILABLE:
        indexes = get_user_indexes(user_id)
        results = indexes["facts"].search(query_embedding, k=k)

        if results:
            # Fetch full data for matched keys
            with get_db_context() as conn:
                output = []
                for fact_key, similarity in results:
                    row = conn.execute(
                        "SELECT key, value, confidence FROM facts WHERE user_id=? AND key=?",
                        (user_id, fact_key)
                    ).fetchone()
                    if row:
                        output.append({
                            "key": row["key"],
                            "value": row["value"],
                            "confidence": row["confidence"],
                            "similarity": similarity
                        })
                return output

    # =================================================================
    # FALLBACK: Brute force O(n)
    # =================================================================
    with get_db_context() as conn:
        rows = conn.execute(
            "SELECT key, value, confidence, embedding FROM facts WHERE user_id=? AND embedding IS NOT NULL",
            (user_id,)
        ).fetchall()

        if not rows:
            return []

        vectors = []
        items = []
        for r in rows:
            if r["embedding"]:
                vec = np.frombuffer(r["embedding"], dtype=np.float32)
                vectors.append(vec)
                items.append(r)

        if not vectors:
            return []

        top = top_k_by_cosine(query_embedding, vectors, k=min(k, len(vectors)))

        return [
            {
                "key": items[idx]["key"],
                "value": items[idx]["value"],
                "confidence": items[idx]["confidence"],
                "similarity": float(sim)
            }
            for idx, sim in top
        ]


# =============================================================================
# MILESTONES TABLE - Life events, permanent, no decay
# =============================================================================

def insert_milestone(
    user_id: int,
    event_type: str,
    description: str,
    event_date: str | None,
    confidence: float,
    provenance_memory_id: int | None,
    embedding: bytes | None = None,
    metadata: dict | None = None
) -> int:
    """Insert a new milestone (life event)."""
    with get_db_context() as conn:
        cur = conn.execute(
            """INSERT INTO milestones(user_id, event_type, event_date, description,
                                      confidence, provenance_memory_id, embedding, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, event_type, event_date, description, confidence,
             provenance_memory_id, embedding, json.dumps(metadata or {}))
        )
        conn.commit()
        milestone_id = int(cur.lastrowid)

    # Add to FAISS index for fast retrieval
    if FAST_INDEX_AVAILABLE and embedding:
        try:
            indexes = get_user_indexes(user_id)
            indexes["milestones"].add(milestone_id, embedding)
        except Exception:
            pass  # Index update failure shouldn't break insert

    return milestone_id


def get_milestones(user_id: int, event_type: str | None = None) -> list[dict]:
    """Get milestones, optionally filtered by event type."""
    with get_db_context() as conn:
        if event_type:
            rows = conn.execute(
                """SELECT id, event_type, event_date, description, confidence, created_at, metadata
                   FROM milestones WHERE user_id=? AND event_type=?
                   ORDER BY event_date DESC, created_at DESC""",
                (user_id, event_type)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, event_type, event_date, description, confidence, created_at, metadata
                   FROM milestones WHERE user_id=?
                   ORDER BY event_date DESC, created_at DESC""",
                (user_id,)
            ).fetchall()

        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "event_date": r["event_date"],
                "description": r["description"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata"] or "{}")
            }
            for r in rows
        ]


def search_milestones(user_id: int, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
    """Search milestones by semantic similarity. Uses FAISS O(log n) when available."""

    # =================================================================
    # FAST PATH: FAISS index (O(log n))
    # =================================================================
    if FAST_INDEX_AVAILABLE:
        indexes = get_user_indexes(user_id)
        results = indexes["milestones"].search(query_embedding, k=k)

        if results:
            with get_db_context() as conn:
                output = []
                for milestone_id, similarity in results:
                    row = conn.execute(
                        """SELECT id, event_type, event_date, description, confidence
                           FROM milestones WHERE user_id=? AND id=?""",
                        (user_id, milestone_id)
                    ).fetchone()
                    if row:
                        output.append({
                            "id": row["id"],
                            "event_type": row["event_type"],
                            "event_date": row["event_date"],
                            "description": row["description"],
                            "confidence": row["confidence"],
                            "similarity": similarity
                        })
                return output

    # =================================================================
    # FALLBACK: Brute force O(n)
    # =================================================================
    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT id, event_type, event_date, description, confidence, embedding
               FROM milestones WHERE user_id=? AND embedding IS NOT NULL""",
            (user_id,)
        ).fetchall()

        if not rows:
            return []

        vectors = []
        items = []
        for r in rows:
            if r["embedding"]:
                vec = np.frombuffer(r["embedding"], dtype=np.float32)
                vectors.append(vec)
                items.append(r)

        if not vectors:
            return []

        top = top_k_by_cosine(query_embedding, vectors, k=min(k, len(vectors)))

        return [
            {
                "id": items[idx]["id"],
                "event_type": items[idx]["event_type"],
                "event_date": items[idx]["event_date"],
                "description": items[idx]["description"],
                "confidence": items[idx]["confidence"],
                "similarity": float(sim)
            }
            for idx, sim in top
        ]


def rebuild_indexes(user_id: int) -> dict[str, int]:
    """
    Rebuild FAISS indexes from database.
    Call on startup or when indexes are out of sync.

    Returns:
        Dict with count of vectors in each index
    """
    if not FAST_INDEX_AVAILABLE:
        return {"error": "FAISS not available"}

    from app.vector.memory_indexes import rebuild_user_indexes, get_user_indexes

    rebuild_user_indexes(user_id)
    indexes = get_user_indexes(user_id)

    return {
        "facts": indexes["facts"].size,
        "milestones": indexes["milestones"].size,
        "episodes": indexes["episodes"].size
    }


def check_duplicate_milestone(user_id: int, event_type: str, description: str, threshold: float = 0.85) -> bool:
    """
    Check if a similar milestone already exists.
    Prevents duplicate entries for the same life event.
    """
    from app.vector.embedder import embed_text

    # Get embedding for new description
    new_embedding = embed_text(description)

    # Search existing milestones of same type
    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT embedding FROM milestones
               WHERE user_id=? AND event_type=? AND embedding IS NOT NULL""",
            (user_id, event_type)
        ).fetchall()

        for r in rows:
            if r["embedding"]:
                existing_vec = np.frombuffer(r["embedding"], dtype=np.float32)
                # Compute cosine similarity
                sim = np.dot(new_embedding, existing_vec) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(existing_vec) + 1e-9
                )
                if sim >= threshold:
                    return True  # Duplicate found

    return False
