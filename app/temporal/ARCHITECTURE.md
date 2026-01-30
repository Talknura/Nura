# Temporal Engine Architecture

## Integrated Temporal Understanding for Three-Tier Memory

The Temporal Engine provides unified temporal parsing that feeds directly into Memory Engine (for storage) and Retrieval Engine (for queries).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL ENGINE v2                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            User Input                                       │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │     Fast Path Cache   │ ◄── "today", "last week"       │
│                    │     (No Embedding)    │     (~0.1ms)                   │
│                    └───────────┬───────────┘                                │
│                                │ miss                                       │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │   Semantic Parser     │ ◄── "a couple weeks ago"       │
│                    │   (Embedding Match)   │     (~5ms, cached)             │
│                    └───────────┬───────────┘                                │
│                                │ low confidence                             │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │   Regex Fallback      │ ◄── "5 days ago"               │
│                    │   (Pattern Match)     │     (~0.5ms)                   │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │   TemporalResult      │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│           ┌────────────────────┼────────────────────┐                       │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                │
│   │    MEMORY     │   │   RETRIEVAL   │   │   PATTERNS    │                │
│   │  INTEGRATION  │   │  INTEGRATION  │   │   CONTEXT     │                │
│   │               │   │               │   │               │                │
│   │ • temporal_   │   │ • window_days │   │ • active_     │                │
│   │   tags        │   │ • strategy_   │   │   hours       │                │
│   │ • milestone_  │   │   hint        │   │ • active_     │                │
│   │   date        │   │ • disable_    │   │   days        │                │
│   │ • tier_hint   │   │   recency     │   │ • timezone    │                │
│   └───────────────┘   └───────────────┘   └───────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Latency Optimization

| Path | Latency | When Used |
|------|---------|-----------|
| Fast Path Cache | ~0.1ms | Common phrases (today, yesterday, this week) |
| Semantic Parser | ~5ms | Complex expressions, cached embeddings |
| Regex Fallback | ~0.5ms | Numeric patterns (5 days ago) |

### Fast Path Cache

Pre-computed results for common phrases - no embedding needed:

```python
FAST_PATH_CACHE = {
    "today": {"granularity": "day", "direction": 0, "days": 1},
    "yesterday": {"granularity": "day", "direction": -1, "days": 2},
    "this week": {"granularity": "week", "direction": 0, "days": 7},
    "last month": {"granularity": "month", "direction": -1, "days": 60},
    ...
}
```

## Integration with Memory Engine

### Temporal Tags

Generated for every memory stored:

```python
tags = engine.generate_temporal_tags(now)
# {
#     "day_of_week": 1,      # Monday=0, Sunday=6
#     "hour_of_day": 14,
#     "season": "winter",
#     "date": "2025-01-30",
#     "month": 1,
#     "year": 2025,
#     "is_weekend": False,
#     "time_of_day": "afternoon"
# }
```

### Milestone Date Extraction

For life events, extracts specific dates:

```python
date = engine.extract_milestone_date("My dad passed away on January 15, 2020", now)
# "2020-01-15"
```

### Memory Tier Hints

Suggests which tier based on temporal context:

| Pattern | Tier Hint |
|---------|-----------|
| "passed away", "married", "graduated" | milestone |
| "always", "usually", "every day" | fact |
| Specific time reference | episode |

## Integration with Retrieval Engine

### Time Window

```python
days = engine.get_retrieval_window("What did we discuss last week?", now)
# 14 (two weeks lookback)
```

### Disable Recency

For explicit past references:

```python
disable = engine.should_disable_recency("What did I tell you years ago?")
# True → Don't penalize old memories
```

### Strategy Hints

```python
result = engine.parse("What happened this month?", now)
# result.retrieval_strategy_hint = "timeline"
```

| Query Pattern | Strategy Hint |
|---------------|---------------|
| "this week/month" | timeline |
| "yesterday", "few days ago" | episodic |
| No time reference | None (hybrid) |

## TemporalResult Structure

```python
@dataclass
class TemporalResult:
    # Core parsing
    start_ts: Optional[str]           # ISO timestamp
    end_ts: Optional[str]             # ISO timestamp
    granularity: TemporalGranularity  # second → year
    confidence: float                 # 0.0 - 1.0
    direction: int                    # -1=past, 0=present, 1=future

    # Memory integration
    temporal_tags: Dict[str, Any]     # For memory storage
    milestone_date: Optional[str]     # For milestones tier
    memory_tier_hint: Optional[str]   # "fact", "milestone", "episode"

    # Retrieval integration
    retrieval_window_days: Optional[int]
    retrieval_strategy_hint: Optional[str]
    disable_recency: bool

    # Metadata
    concept_matched: Optional[str]
    requires_clarification: bool
```

## Temporal Concepts (Semantic Parser)

Full coverage from seconds to decades:

| Category | Examples |
|----------|----------|
| **Relative** | "5 minutes ago", "in 3 hours", "2 weeks later" |
| **Named** | "today", "tomorrow", "this week", "next month" |
| **Colloquial** | "a while ago", "recently", "ages ago", "soon" |
| **Time of Day** | "this morning", "tonight", "at noon" |
| **Seasons** | "this summer", "last winter" |

## Usage Examples

### Memory Storage

```python
from app.temporal.temporal_engine_v2 import get_temporal_engine

engine = get_temporal_engine()
now = datetime.now(timezone.utc)

# Generate tags for new memory
tags = engine.generate_temporal_tags(now)
memory_engine.ingest_event(
    ...,
    temporal_tags=tags
)

# Extract milestone date
text = "My mom passed away on March 5, 2019"
result = engine.parse(text, now)
if result.memory_tier_hint == "milestone":
    milestone_date = result.milestone_date  # "2019-03-05"
```

### Retrieval Query

```python
# Parse query for retrieval
result = engine.parse("What did we talk about last week?", now)

# Use in retrieval
retrieval_state = RetrievalState(
    query="What did we talk about last week?",
    temporal_rewrite={
        "start_ts": result.start_ts,
        "end_ts": result.end_ts,
        "window_days": result.retrieval_window_days
    }
)
```

## File Structure

```
app/temporal/
├── temporal_engine_v2.py     # Main integrated engine
├── temporal_engine.py        # Legacy wrapper
├── temporal_patterns.py      # User behavior patterns
├── temporal_query_parser.py  # Query parsing utilities
├── time_humanizer.py         # Human-readable time formatting
└── ARCHITECTURE.md           # This file

app/semantic/
└── temporal_concepts.py      # Semantic temporal parser
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Fast path lookup | <1ms | Common phrases |
| Semantic parse | ~5ms | First call (embedding) |
| Semantic parse | <1ms | Cached embeddings |
| Regex fallback | <1ms | Pattern matching |
| Full result | ~1-6ms | Depends on path taken |

## Design Principles

1. **Latency First**: Fast path cache for common phrases
2. **Semantic Understanding**: Embeddings for complex expressions
3. **Memory Integration**: Tags + tier hints + milestone dates
4. **Retrieval Integration**: Time windows + strategy hints + recency control
5. **Graceful Fallback**: Regex when semantic unavailable

## Frozen: January 2025

This architecture is stable and integrated with Memory/Retrieval engines.
