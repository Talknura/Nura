# Memory Engine Architecture

## Three-Tier Memory System

Nura's Memory Engine uses a biologically-inspired three-tier architecture that separates memories by their nature and persistence requirements.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NURA MEMORY ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │     FACTS       │    │   MILESTONES    │    │    EPISODES     │        │
│   │  (Semantic)     │    │  (Life Events)  │    │   (Episodic)    │        │
│   ├─────────────────┤    ├─────────────────┤    ├─────────────────┤        │
│   │ Key-Value Pairs │    │ Timestamped     │    │ Conversations   │        │
│   │ ONE value/key   │    │ Major events    │    │ Day-to-day      │        │
│   │ UPDATES on      │    │ NO decay        │    │ DOES decay      │        │
│   │ contradiction   │    │ Permanent       │    │ 60-90 days      │        │
│   │ NO decay        │    │                 │    │ Summarizes      │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                      │                      │                  │
│           └──────────────────────┼──────────────────────┘                  │
│                                  │                                          │
│                    ┌─────────────▼─────────────┐                           │
│                    │    HNSW Vector Index      │                           │
│                    │    O(log n) Retrieval     │                           │
│                    └───────────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Tiers

### 1. FACTS (Semantic Memory)
Personal truths that define the user's identity.

| Property | Value |
|----------|-------|
| Storage | `facts` table |
| Key Constraint | ONE value per key (UPSERT) |
| Decay | None (permanent) |
| Examples | Name, occupation, location, preferences |

```python
# When user says "I'm a software engineer"
upsert_fact_v2(
    user_id=1,
    key="occupation",
    value="software engineer",
    confidence=0.95,
    embedding=embedding_bytes
)

# When user later says "I switched to product management"
# → Old value moved to history, new value stored
```

**History Tracking**: When a fact changes, the old value is preserved:
```json
{
    "key": "occupation",
    "value": "product manager",
    "history": [
        {"value": "software engineer", "replaced_at": "2024-01-15T10:00:00Z"}
    ]
}
```

### 2. MILESTONES (Life Events)
Significant life events that should never be forgotten.

| Property | Value |
|----------|-------|
| Storage | `milestones` table |
| Key Constraint | Multiple events allowed |
| Decay | None (permanent) |
| Duplicate Check | Semantic similarity (0.85 threshold) |
| Examples | Deaths, marriages, graduations, births |

```python
insert_milestone(
    user_id=1,
    event_type="death",
    description="User's father passed away",
    event_date="2024-01-10",
    confidence=0.95
)
```

### 3. EPISODES (Episodic Memory)
Day-to-day conversations that may fade over time.

| Property | Value |
|----------|-------|
| Storage | `memories` table |
| Decay | 60-90 days half-life |
| Summarization | Every N turns → summary |
| Examples | Casual conversations, temporary tasks |

## Semantic Classification

The Memory Engine uses ML embeddings (not regex) to classify incoming text:

```python
from app.semantic.memory_architecture import get_semantic_memory_classifier, MemoryType

classifier = get_semantic_memory_classifier()
result = classifier.classify("My name is Sam")

# result.memory_type = MemoryType.FACT
# result.fact_key = "name"
# result.fact_value = "Sam"
# result.confidence = 0.92
```

### Classification Flow

```
Input Text
    │
    ▼
┌─────────────────────────┐
│  Generate Embedding     │
│  (all-MiniLM-L6-v2)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Compare to Concept     │
│  Embeddings             │
│  - FACT patterns        │
│  - MILESTONE patterns   │
│  - EPISODE (default)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Classify & Extract     │
│  - Determine tier       │
│  - Extract key/value    │
│  - Set confidence       │
└───────────┬─────────────┘
            │
            ▼
    Store in appropriate
    tier (Facts/Milestones/Episodes)
```

## HNSW Vector Index

For fast O(log n) retrieval, we use FAISS HNSW (Hierarchical Navigable Small World):

```
Layer 2:  [A]────────────────────[D]           ← Sparse (long jumps)
           │                      │
Layer 1:  [A]────────[C]────────[D]────────[F] ← Medium density
           │          │          │          │
Layer 0:  [A]─[B]─[C]─[D]─[E]─[F]─[G]─[H]     ← Dense (all vectors)

Search: Start at top layer, greedily descend → O(log n)
```

### Index Parameters

```python
M = 32              # Connections per node (accuracy vs memory)
ef_construction = 64 # Build-time search depth
ef_search = 32      # Query-time search depth (tunable)
```

### Performance

| Memory Size | Brute Force | HNSW |
|-------------|-------------|------|
| 1,000 | ~10ms | ~2ms |
| 10,000 | ~100ms | ~3ms |
| 100,000 | ~1000ms | ~5ms |

## File Structure

```
app/memory/
├── memory_engine.py      # Main engine (ingest, search, retrieve)
├── memory_store.py       # Database operations + FAISS integration
├── memory_classifier.py  # Basic noise filtering
├── memory_summarizer.py  # Episode summarization
└── ARCHITECTURE.md       # This file

app/semantic/
├── memory_architecture.py # Three-tier semantic classifier
├── fact_concepts.py       # Fact extraction patterns
└── retrieval_concepts.py  # Adaptive decay rates

app/vector/
├── memory_indexes.py     # HNSW index management
├── vector_index.py       # Base FAISS utilities
└── embedding_service.py  # Embedding generation

app/db/
├── models.py            # SQLite schema (facts, milestones, memories)
└── session.py           # Database connection
```

## Database Schema

### facts
```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,           -- Unique per user
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.9,
    last_confirmed_at TEXT,
    first_learned_at TEXT,
    provenance_memory_id INTEGER,
    embedding BLOB,
    history TEXT,                -- JSON array of previous values
    UNIQUE(user_id, key)
);
```

### milestones
```sql
CREATE TABLE milestones (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,    -- death, marriage, graduation, etc.
    event_date TEXT,
    description TEXT NOT NULL,
    confidence REAL DEFAULT 0.9,
    created_at TEXT,
    provenance_memory_id INTEGER,
    embedding BLOB,
    metadata TEXT
);
```

### memories (episodes)
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT CHECK(memory_type IN ('episodic','semantic','summary')),
    importance REAL DEFAULT 0.5,
    embedding BLOB,
    created_at TEXT,
    last_accessed_at TEXT,
    temporal_tags TEXT,
    metadata TEXT
);
```

## Usage

### Initialization
```python
from app.memory.memory_engine import MemoryEngine
from app.vector.embedding_service import EmbeddingService
from app.memory.memory_store import rebuild_indexes

# Create engine
embedding_service = EmbeddingService()
memory_engine = MemoryEngine(embedding_service)

# Build HNSW indexes on startup
rebuild_indexes(user_id=1)
```

### Ingesting Memory
```python
from datetime import datetime, timezone

result = memory_engine.ingest_event(
    user_id=1,
    role="user",
    text="My name is Sam and I work at Google",
    session_id="abc123",
    ts=datetime.now(timezone.utc),
    temporal_tags={"hour": 14, "weekday": "Monday"},
    source="chat"
)

# result = {
#     "id": 42,
#     "memory_type": "fact",
#     "memory_tier": "fact",
#     "fact_key": "name",  # Also extracts "occupation"
#     "confidence": 0.92
# }
```

### Searching
```python
# Searches all three tiers, facts/milestones always surface
results = memory_engine.search(user_id=1, query="What's my job?", k=10)

# Results include:
# - Relevant facts (permanent, high rank)
# - Relevant milestones (permanent)
# - Relevant episodes (with recency decay)
```

### Retrieving Facts
```python
# Key-value dict
facts = memory_engine.facts(user_id=1)
# {"name": "Sam", "occupation": "software engineer", ...}

# Full details with history
all_facts = memory_engine.all_facts(user_id=1)
```

### Retrieving Milestones
```python
# All milestones
milestones = memory_engine.milestones(user_id=1)

# Filtered by type
deaths = memory_engine.milestones(user_id=1, event_type="death")
```

## Design Principles

1. **Semantic over Regex**: Classification uses ML embeddings, not hardcoded patterns
2. **Facts Update, Don't Accumulate**: "I moved to NYC" replaces "I live in SF"
3. **Permanent where it Matters**: Identity, family, health, grief never decay
4. **Fast Retrieval**: HNSW indexes for O(log n) search at any scale
5. **Graceful Fallback**: Works without FAISS (slower) or semantic classifier (legacy mode)

## Frozen: January 2025

This architecture is stable and ready for production use.
