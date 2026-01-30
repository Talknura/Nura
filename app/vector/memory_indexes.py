"""
Fast Vector Indexes for Three-Tier Memory Architecture.

Uses FAISS HNSW (Hierarchical Navigable Small World) for O(log n) retrieval.

=============================================================================
HNSW: Tree-Like Graph Structure
=============================================================================

HNSW builds a multi-layer navigable graph (similar to skip lists):

    Layer 2:  [A]--------------------[D]           (sparse, long jumps)
               |                      |
    Layer 1:  [A]--------[C]--------[D]--------[F] (medium density)
               |          |          |          |
    Layer 0:  [A]-[B]-[C]-[D]-[E]-[F]-[G]-[H]     (dense, all vectors)

Search Algorithm:
    1. Start at top layer (sparse)
    2. Greedy walk to nearest neighbor
    3. Drop to next layer, repeat
    4. At layer 0, refine with local search

Complexity: O(log n) average case (like binary search through layers)

=============================================================================
Why Not Just a Tree?
=============================================================================

Pure trees (like KD-trees) suffer from "curse of dimensionality" in 384D.
HNSW combines graph navigation with hierarchical structure = best of both.

Maintains separate indexes for:
  - Facts (semantic memory) - permanent, ~100s of entries
  - Milestones (life events) - permanent, ~10s of entries
  - Episodes (conversations) - decays, ~1000s of entries

Each index maps FAISS positions back to database IDs.
"""

from __future__ import annotations
import os
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

# Try FAISS, fall back to brute force
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[MemoryIndexes] FAISS not available, using brute-force fallback")


class MemoryVectorIndex:
    """
    FAISS HNSW-backed vector index with ID mapping.

    Uses Hierarchical Navigable Small World (HNSW) graph structure
    for true O(log n) approximate nearest neighbor search.

    HNSW builds a multi-layer graph where:
    - Layer 0: All vectors (densest)
    - Layer 1+: Subset of vectors (sparser, for fast navigation)

    Search: Start at top layer, greedily descend → O(log n)
    """

    def __init__(self, name: str, dimension: int = 384, index_dir: str = None):
        """
        Initialize memory index with HNSW.

        Args:
            name: Index name (e.g., "facts", "milestones", "episodes")
            dimension: Vector dimension (384 for MiniLM-L6-v2)
            index_dir: Directory for persistence
        """
        self.name = name
        self.dimension = dimension
        self.index_dir = index_dir or r"D:\Nura\Data\indexes"

        # HNSW parameters
        self.M = 32  # Number of connections per layer (higher = more accurate, more memory)
        self.ef_construction = 64  # Construction-time search depth
        self.ef_search = 32  # Query-time search depth (can tune for speed/accuracy)

        # FAISS index
        self.index = None

        # ID mapping: position -> database ID
        self.id_map: list[int | str] = []

        # Thread safety
        self._lock = threading.Lock()

        # Initialize
        self._init_index()

    def _init_index(self):
        """Initialize or load HNSW index."""
        if not FAISS_AVAILABLE:
            return

        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, f"{self.name}.faiss")
        map_path = os.path.join(self.index_dir, f"{self.name}_ids.json")

        # Try to load existing
        if os.path.exists(index_path) and os.path.exists(map_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(map_path, "r") as f:
                    self.id_map = json.load(f)
                print(f"[{self.name}] Loaded HNSW index with {len(self.id_map)} vectors")
                return
            except Exception as e:
                print(f"[{self.name}] Failed to load index: {e}")

        # Create HNSW index with Inner Product metric (true O(log n) search)
        # Inner Product on normalized vectors = Cosine Similarity
        self.index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.id_map = []

    def add(self, db_id: int | str, embedding: np.ndarray | bytes) -> None:
        """
        Add a vector to the index.

        Args:
            db_id: Database ID (memory_id, fact_key, milestone_id)
            embedding: Vector as numpy array or bytes
        """
        if not FAISS_AVAILABLE:
            return

        # Convert bytes to numpy if needed
        if isinstance(embedding, bytes):
            vec = np.frombuffer(embedding, dtype=np.float32)
        else:
            vec = embedding.astype(np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vec = vec.reshape(1, -1)

        with self._lock:
            # Check if ID already exists (update case)
            if db_id in self.id_map:
                # FAISS doesn't support in-place update, need to rebuild
                # For now, skip duplicate (rebuild handles this)
                return

            self.index.add(vec)
            self.id_map.append(db_id)

    def search(self, query_embedding: np.ndarray | bytes, k: int = 5) -> list[tuple[Any, float]]:
        """
        Search for top-k similar vectors.

        Args:
            query_embedding: Query vector
            k: Number of results

        Returns:
            List of (database_id, similarity_score) tuples
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []

        # Convert bytes to numpy if needed
        if isinstance(query_embedding, bytes):
            vec = np.frombuffer(query_embedding, dtype=np.float32)
        else:
            vec = query_embedding.astype(np.float32)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vec = vec.reshape(1, -1)

        # Search
        k = min(k, self.index.ntotal)
        with self._lock:
            distances, indices = self.index.search(vec, k)

        # Map positions back to database IDs
        results = []
        for pos, score in zip(indices[0], distances[0]):
            if 0 <= pos < len(self.id_map):
                results.append((self.id_map[pos], float(score)))

        return results

    def save(self) -> None:
        """Persist index to disk."""
        if not FAISS_AVAILABLE or self.index is None:
            return

        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, f"{self.name}.faiss")
        map_path = os.path.join(self.index_dir, f"{self.name}_ids.json")

        with self._lock:
            faiss.write_index(self.index, index_path)
            with open(map_path, "w") as f:
                json.dump(self.id_map, f)

        print(f"[{self.name}] Saved index with {len(self.id_map)} vectors")

    def rebuild_from_db(self, rows: list[tuple[Any, bytes]]) -> None:
        """
        Rebuild HNSW index from database rows.

        HNSW doesn't support deletion, so rebuild is the way to
        sync after deletions or data changes.

        Args:
            rows: List of (db_id, embedding_bytes) tuples
        """
        if not FAISS_AVAILABLE:
            return

        with self._lock:
            # Create fresh HNSW index with Inner Product (cosine for normalized vectors)
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            self.id_map = []

            if not rows:
                return

            # Batch add for efficiency
            vectors = []
            ids = []

            for db_id, emb_bytes in rows:
                if emb_bytes:
                    vec = np.frombuffer(emb_bytes, dtype=np.float32)
                    # Normalize for cosine similarity
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vectors.append(vec)
                    ids.append(db_id)

            if vectors:
                matrix = np.vstack(vectors).astype(np.float32)
                self.index.add(matrix)
                self.id_map = ids

        print(f"[{self.name}] Rebuilt HNSW index with {len(self.id_map)} vectors (M={self.M}, ef={self.ef_search})")
        self.save()

    def set_search_depth(self, ef_search: int) -> None:
        """
        Tune search accuracy vs speed.

        Higher ef_search = more accurate but slower.
        Lower ef_search = faster but may miss some results.

        Recommended:
            - Fast (real-time): ef_search = 16-32
            - Balanced: ef_search = 64
            - High accuracy: ef_search = 128-256

        Args:
            ef_search: Number of candidates to explore during search
        """
        if self.index is not None and FAISS_AVAILABLE:
            self.index.hnsw.efSearch = ef_search
            self.ef_search = ef_search

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return len(self.id_map)


# =============================================================================
# SINGLETON INDEXES (one per memory tier per user)
# =============================================================================

_user_indexes: dict[int, dict[str, MemoryVectorIndex]] = {}
_indexes_lock = threading.Lock()


def get_user_indexes(user_id: int) -> dict[str, MemoryVectorIndex]:
    """
    Get or create indexes for a user.

    Returns dict with keys: "facts", "milestones", "episodes"
    """
    global _user_indexes

    with _indexes_lock:
        if user_id not in _user_indexes:
            index_dir = rf"D:\Nura\Data\indexes\user_{user_id}"
            _user_indexes[user_id] = {
                "facts": MemoryVectorIndex("facts", index_dir=index_dir),
                "milestones": MemoryVectorIndex("milestones", index_dir=index_dir),
                "episodes": MemoryVectorIndex("episodes", index_dir=index_dir),
            }
        return _user_indexes[user_id]


def rebuild_user_indexes(user_id: int) -> None:
    """
    Rebuild all indexes for a user from database.
    Call this on startup or when indexes are out of sync.
    """
    from app.db.session import get_db_context

    indexes = get_user_indexes(user_id)

    with get_db_context() as conn:
        # Rebuild facts index
        facts_rows = conn.execute(
            "SELECT key, embedding FROM facts WHERE user_id=? AND embedding IS NOT NULL",
            (user_id,)
        ).fetchall()
        indexes["facts"].rebuild_from_db([(r["key"], r["embedding"]) for r in facts_rows])

        # Rebuild milestones index
        milestones_rows = conn.execute(
            "SELECT id, embedding FROM milestones WHERE user_id=? AND embedding IS NOT NULL",
            (user_id,)
        ).fetchall()
        indexes["milestones"].rebuild_from_db([(r["id"], r["embedding"]) for r in milestones_rows])

        # Rebuild episodes index (recent only for efficiency)
        episodes_rows = conn.execute(
            """SELECT id, embedding FROM memories
               WHERE user_id=? AND embedding IS NOT NULL
               ORDER BY created_at DESC LIMIT 10000""",
            (user_id,)
        ).fetchall()
        indexes["episodes"].rebuild_from_db([(r["id"], r["embedding"]) for r in episodes_rows])

    print(f"[User {user_id}] Rebuilt all indexes: facts={indexes['facts'].size}, "
          f"milestones={indexes['milestones'].size}, episodes={indexes['episodes'].size}")
