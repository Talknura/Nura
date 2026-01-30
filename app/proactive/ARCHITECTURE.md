# Proactive Engine Architecture

## Integrated Decision Engine for Follow-Up Questions

The Proactive Engine decides when and how to proactively ask follow-up questions. It integrates with Memory, Retrieval, and Temporal engines for intelligent, context-aware decisions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROACTIVE ENGINE v2                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            User Context                                     │
│                                │                                            │
│             ┌──────────────────┼──────────────────┐                         │
│             ▼                  ▼                  ▼                         │
│    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
│    │    TEMPORAL    │  │     MEMORY     │  │   RETRIEVAL    │              │
│    │    ENGINE      │  │     ENGINE     │  │    ENGINE      │              │
│    │                │  │                │  │                │              │
│    │ • Active hours │  │ • Facts tier   │  │ • Episodic     │              │
│    │ • Anniversaries│  │ • Milestones   │  │ • Factual      │              │
│    │ • Time context │  │ • Episodes     │  │ • Milestone    │              │
│    └───────┬────────┘  └───────┬────────┘  └───────┬────────┘              │
│            │                   │                   │                        │
│            └───────────────────┼───────────────────┘                        │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │    SEMANTIC ANALYSIS    │ ◄── Cached embeddings         │
│                   │    (all-MiniLM-L6-v2)   │     (~1ms cached)             │
│                   │                         │                               │
│                   │  • Importance level     │                               │
│                   │  • Memory type          │                               │
│                   │  • Follow-up triggers   │                               │
│                   │  • Suppression signals  │                               │
│                   └────────────┬────────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │    DECISION ENGINE      │                               │
│                   │                         │                               │
│                   │  • Salience scoring     │                               │
│                   │  • Obligation priority  │                               │
│                   │  • Cooldown/rate limits │                               │
│                   └────────────┬────────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                   ┌─────────────────────────┐                               │
│                   │     ProactiveResult     │                               │
│                   └─────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Latency Optimization

| Path | Latency | When Used |
|------|---------|-----------|
| Fast Path Cache | ~0.1ms | Suppression phrases ("drop it", "forget it") |
| Rate Limit Check | ~0.1ms | Cooldown and daily limit checks |
| Semantic Analysis | ~5ms | First analysis (embedding computation) |
| Semantic Cached | ~1ms | Subsequent analyses (same session) |
| Full Evaluation | ~5-10ms | Complete decision with all integrations |

### Fast Path Cache

Pre-computed decisions for common suppression phrases:

```python
FAST_PATH_DECISIONS = {
    "don't remind me": {"action": "suppress", "reason": "user_requested"},
    "drop it": {"action": "suppress", "reason": "user_requested"},
    "forget about it": {"action": "suppress", "reason": "user_requested"},
    "done": {"action": "mark_complete", "reason": "task_finished"},
    "finished": {"action": "mark_complete", "reason": "task_finished"},
}
```

## Engine Integrations

### Memory Engine Integration

Queries three-tier memory for context:

```python
# Access milestones for anniversary detection
milestones = memory_store.search_milestones(user_id, query, top_k=10)

# Check facts for relevant context
facts = memory_store.search_facts(user_id, query, top_k=5)

# Get recent episodes for context
episodes = memory_store.search_memories(user_id, query, top_k=20)
```

### Temporal Engine Integration

Uses temporal context for time-aware decisions:

```python
# Check if within user's active hours
is_active = temporal_engine.is_active_time(user_context)

# Detect milestone anniversaries
anniversaries = proactive_engine._check_milestone_anniversaries(state)
```

### Semantic Analysis Integration

ML-based understanding of memory importance and type:

```python
from app.semantic.proactive_concepts import get_semantic_proactive_analyzer

analyzer = get_semantic_proactive_analyzer()
analysis = analyzer.analyze(text, threshold=0.45)

# analysis.importance_level: "urgent", "high", "medium", "low", "trivial"
# analysis.memory_type: "task_memory", "event_memory", "personal_memory"
# analysis.followup_triggers: ["needs_closure", "time_sensitive"]
# analysis.should_suppress: True/False
# analysis.total_salience_boost: int
```

## Obligation Classes

Proactive asks are classified by obligation level:

| Class | Priority | Description |
|-------|----------|-------------|
| TASK_CLOSURE | 0 | Past-due task with unknown outcome |
| NARRATIVE_EXPECTATION | 1 | User promised to update |
| TASK_BLOCKING | 2 | Prerequisite blocking upcoming task |
| EMOTIONAL_CHECK_IN | 3 | User expressed concern/anxiety |
| MILESTONE_ANNIVERSARY | 4 | Anniversary of significant event |
| HABIT | 5 | Recurring habit check |
| OPTIONAL | 6 | General follow-up opportunity |

### Mandatory vs Optional

**Mandatory** (bypasses rate limits):
- TASK_CLOSURE
- NARRATIVE_EXPECTATION
- TASK_BLOCKING

**Optional** (respects rate limits):
- EMOTIONAL_CHECK_IN
- MILESTONE_ANNIVERSARY
- HABIT
- OPTIONAL

## Salience Scoring

```python
def compute_salience(memory, state) -> int:
    score = 0

    # Base scoring
    if memory_type == "task" and has_due:
        score += 3
    if importance == "high":
        score += 3
    if due_within_6_hours:
        score += 2
    if has_emotional_link:
        score += 2

    # Penalties
    if memory_type == "habit":
        score -= 2
    if is_generic_emotional:
        score -= 3

    # Semantic boost (from ML analysis)
    score += analysis.total_salience_boost

    if analysis.should_suppress:
        score -= 10

    return score
```

## ProactiveResult Structure

```python
@dataclass
class ProactiveResult:
    # Core decision
    should_ask: bool = False
    memory_id: Optional[str] = None
    question_type: Optional[str] = None  # follow_up, check_in, anniversary
    urgency: str = "low"  # low, medium, high
    cooldown_until: Optional[str] = None
    reason: Optional[str] = None

    # Integration context
    temporal_context: Optional[Dict[str, Any]] = None
    memory_tier: Optional[str] = None  # fact, milestone, episode

    # Obligation info
    obligation_state: str = "NONE"  # NONE, OPTIONAL, MANDATORY
    obligation_class: Optional[str] = None
    blocks_task_id: Optional[str] = None

    # Scoring
    salience_score: int = 0
    semantic_analysis: Optional[Dict[str, Any]] = None
```

## Rate Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Cooldown | 4 hours | Between optional asks |
| Daily Max | 4 asks | Per day for optional |
| Mandatory | No limit | Always allowed |

## Usage Examples

### Basic Evaluation

```python
from app.proactive import get_proactive_engine

engine = get_proactive_engine()
result = engine.evaluate({
    "user_id": "user123",
    "now_timestamp": "2025-01-30T14:00:00Z",
    "recent_memories": memories,
    "cooldown_state": {
        "last_asked_at": "2025-01-30T10:00:00Z",
        "asks_today": 1
    }
})

if result.should_ask:
    question = generate_followup(result.memory_id, result.question_type)
```

### With Temporal Context

```python
from app.temporal import UserTemporalContext

result = engine.evaluate({
    "user_id": "user123",
    "now_timestamp": "2025-01-30T14:00:00Z",
    "recent_memories": memories,
    "cooldown_state": cooldown,
    "temporal_context": UserTemporalContext(
        active_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        active_days=[0, 1, 2, 3, 4],  # Weekdays
        timezone="America/New_York"
    )
})
```

### Legacy Interface

```python
from app.proactive import evaluate, decide_followup

# These return dicts for backward compatibility
result = evaluate(payload)
# or
result = decide_followup(payload)
```

## File Structure

```
app/proactive/
├── proactive_engine.py      # Main integrated engine
├── __init__.py              # Module exports
└── ARCHITECTURE.md          # This file

app/semantic/
└── proactive_concepts.py    # Semantic concept definitions
```

## Semantic Concepts

The semantic analyzer recognizes these concept categories:

| Category | Concepts |
|----------|----------|
| Importance | urgent, high_importance, medium, low, trivial |
| Memory Type | task_memory, event_memory, personal_memory, learning_memory, general_knowledge |
| Status | pending_status, in_progress_status, completed_status, blocked_status, cancelled_status |
| Triggers | needs_closure, needs_checkin, time_sensitive, recurring_topic, commitment_made |
| Emotional | emotionally_significant, routine_matter, sensitive_topic, celebratory |
| Suppress | dont_ask_again, topic_closed |

## Design Principles

1. **Latency First**: Fast path cache and cached semantic embeddings
2. **Integration**: Works with Memory/Retrieval/Temporal engines
3. **Semantic Understanding**: ML-based importance detection, not keywords
4. **Rate Limiting**: Prevents over-prompting with cooldowns and daily limits
5. **User Respect**: Detects suppression signals ("drop it", "leave it alone")
6. **Priority System**: Mandatory obligations bypass rate limits

## Frozen: January 2025

This architecture is stable and integrated with Memory/Retrieval/Temporal engines.
