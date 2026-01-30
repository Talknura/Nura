# Adaptation Engine Architecture

## Integrated Communication Style Adaptation

The Adaptation Engine adjusts Nura's communication style based on user signals, historical patterns, and temporal context. It integrates with Memory, Temporal, and Semantic engines.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADAPTATION ENGINE v2                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            User Input                                       │
│                                │                                            │
│             ┌──────────────────┼──────────────────┐                         │
│             ▼                  ▼                  ▼                         │
│    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
│    │     MEMORY     │  │    TEMPORAL    │  │    SEMANTIC    │              │
│    │     ENGINE     │  │    ENGINE      │  │    ANALYZER    │              │
│    │                │  │                │  │                │              │
│    │ • Past prefs   │  │ • Time of day  │  │ • Vulnerability│              │
│    │ • Milestones   │  │ • Day of week  │  │ • Gratitude    │              │
│    │ • Emotional    │  │ • Patterns     │  │ • Engagement   │              │
│    │   history      │  │                │  │ • Style        │              │
│    └───────┬────────┘  └───────┬────────┘  └───────┬────────┘              │
│            │                   │                   │                        │
│            └───────────────────┼───────────────────┘                        │
│                                ▼                                            │
│                 ┌───────────────────────────┐                               │
│                 │    FAST PATH CACHE        │ ◄── "thank you", "i'm scared" │
│                 │    (~0.1ms)               │     Instant adaptation        │
│                 └────────────┬──────────────┘                               │
│                              │ miss                                         │
│                              ▼                                              │
│                 ┌───────────────────────────┐                               │
│                 │    SEMANTIC ANALYSIS      │ ◄── ML embeddings             │
│                 │    (~5ms first, ~1ms)     │     (all-MiniLM-L6-v2)        │
│                 └────────────┬──────────────┘                               │
│                              │                                              │
│                              ▼                                              │
│                 ┌───────────────────────────┐                               │
│                 │    ADAPTATION PROFILE     │                               │
│                 │                           │                               │
│                 │  • warmth (0.0-1.0)       │                               │
│                 │  • formality (0.0-1.0)    │                               │
│                 │  • initiative (0.0-1.0)   │                               │
│                 │  • check_in_frequency     │                               │
│                 └───────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Latency Optimization

| Path | Latency | When Used |
|------|---------|-----------|
| Fast Path Cache | ~0.1ms | Common phrases ("thank you", "i'm scared") |
| Semantic Analysis | ~5ms | First analysis (embedding computation) |
| Semantic Cached | ~1ms | Subsequent analyses (same session) |
| Full Update | ~5-10ms | Complete profile update with all integrations |

### Fast Path Cache

Pre-computed adaptation deltas for common signals:

```python
FAST_PATH_SIGNALS = {
    # Vulnerability
    "i'm struggling": {"warmth": 0.12, "check_in_frequency": 0.08},
    "i'm scared": {"warmth": 0.10, "check_in_frequency": 0.05},
    "i need help": {"warmth": 0.10, "initiative": 0.08},

    # Gratitude
    "thank you": {"initiative": 0.05},
    "you're amazing": {"initiative": 0.08, "warmth": 0.05},

    # Style preferences
    "get to the point": {"formality": -0.08},
    "no need to be formal": {"formality": -0.10},
    "be professional": {"formality": 0.10},
}
```

## Profile Dimensions

| Dimension | Range | Meaning |
|-----------|-------|---------|
| warmth | 0.0-1.0 | Emotional supportiveness (0=neutral, 1=highly empathetic) |
| formality | 0.0-1.0 | Language style (0=casual, 1=formal) |
| initiative | 0.0-1.0 | Proactive engagement (0=reactive only, 1=highly proactive) |
| check_in_frequency | 0.0-1.0 | Follow-up frequency (0=never, 1=frequent) |

## Semantic Signal Categories

| Category | Concepts | Profile Effect |
|----------|----------|----------------|
| **Vulnerability** | struggling, scared, sad, lonely, hopeless | ↑ warmth, ↑ check_in |
| **Gratitude** | direct_thanks, deep_gratitude, appreciation | ↑ initiative |
| **Prayer/Spiritual** | prayer_request, religious_expression, faith | ↑ warmth |
| **Communication Style** | direct, detailed, casual, formal, brief | ↕ formality |
| **Engagement** | high, low, distracted, enthusiastic | ↕ initiative |
| **Emotion** | positive, negative, neutral, excited | ↕ warmth |

## Engine Integrations

### Temporal Engine Integration

Adjusts profile based on time context:

```python
# Time of day modifiers
TIME_OF_DAY_MODIFIERS = {
    "morning": {"warmth": 0.02, "initiative": 0.01},
    "evening": {"warmth": 0.03, "initiative": -0.02},
    "night": {"warmth": 0.05, "initiative": -0.03},
}

# Weekend vs weekday
DAY_MODIFIERS = {
    "weekend": {"formality": -0.02, "initiative": -0.01},
    "weekday": {"formality": 0.01},
}
```

### Memory Engine Integration

Uses historical context:

```python
# Query past emotional patterns
past_vulnerabilities = memory_store.search_milestones(user_id, "emotional")

# Check if user recently shared something sensitive
# -> Sustain warmth increase for follow-up conversations
```

### Semantic Analysis Integration

```python
from app.semantic.adaptation_concepts import get_semantic_adaptation_analyzer

analyzer = get_semantic_adaptation_analyzer()
signals = analyzer.analyze(user_text, threshold=0.45)

# signals.vulnerability: True/False
# signals.gratitude: True/False
# signals.communication_style: "prefers_direct", "prefers_casual", etc.
# signals.adaptation_deltas: {"warmth": 0.15, "initiative": 0.08}
```

## Text Heuristics

Additional signals from text analysis:

| Signal | Detection | Effect |
|--------|-----------|--------|
| Long messages (>200 chars) | Length check | ↑ warmth +0.03, ↑ initiative +0.02 |
| Short messages (<20 chars) | Length check | ↑ formality +0.02, ↓ initiative -0.02 |
| Multiple questions | Count "?" | ↑ initiative +0.03 |
| Personal pronouns (≥3) | Word count | ↑ warmth +0.02 |

## AdaptationProfile Structure

```python
@dataclass
class AdaptationProfile:
    user_id: int
    warmth: float = 0.5
    formality: float = 0.5
    initiative: float = 0.5
    check_in_frequency: float = 0.5
    updated_at: Optional[datetime] = None

    # Extended profile
    preferred_style: Optional[str] = None  # direct, detailed, casual, formal
    engagement_trend: Optional[str] = None  # increasing, stable, decreasing
    emotional_baseline: Optional[str] = None  # positive, neutral, negative
```

## AdaptationResult Structure

```python
@dataclass
class AdaptationResult:
    profile: AdaptationProfile
    delta_applied: AdaptationDelta
    signals_detected: Dict[str, Any]  # vulnerability, gratitude, etc.
    temporal_modifiers: Dict[str, float]  # time-based adjustments
    memory_influence: Dict[str, float]  # history-based adjustments
```

## Usage Examples

### Basic Update

```python
from app.adaptation import get_adaptation_engine

engine = get_adaptation_engine()

# Simple update (backward compatible)
profile = engine.update(user_id=123, metrics=conversation_metrics)
# Returns: {"warmth": 0.62, "formality": 0.48, "initiative": 0.55, ...}
```

### Extended Update

```python
# Full update with all context
result = engine.update_full(user_id=123, metrics=conversation_metrics)

print(result.profile.warmth)  # 0.62
print(result.signals_detected)  # {"vulnerability": True, "gratitude": False, ...}
print(result.temporal_modifiers)  # {"warmth": 0.03, "initiative": -0.02}
```

### Get Profile

```python
# Simple profile (dict)
profile = engine.get_profile(user_id=123)

# Extended profile (AdaptationProfile object)
profile = engine.get_profile_extended(user_id=123)
print(profile.preferred_style)  # "prefers_direct"
```

## File Structure

```
app/adaptation/
├── adaptation_engine.py        # Main integrated engine
├── adaptation_rules.py         # Delta application logic
├── breakthrough_detector.py    # Legacy breakthrough detection
├── __init__.py                 # Module exports
└── ARCHITECTURE.md             # This file

app/semantic/
└── adaptation_concepts.py      # Semantic concept definitions
```

## Delta Application

```python
@dataclass
class AdaptationDelta:
    warmth: float = 0.0
    formality: float = 0.0
    initiative: float = 0.0
    check_in_frequency: float = 0.0

def apply_delta(profile: dict, delta: AdaptationDelta) -> dict:
    # Clamps all values to [0.0, 1.0]
    out["warmth"] = clamp01(profile["warmth"] + delta.warmth)
    out["formality"] = clamp01(profile["formality"] + delta.formality)
    out["initiative"] = clamp01(profile["initiative"] + delta.initiative)
    out["check_in_frequency"] = clamp01(profile["check_in_frequency"] + delta.check_in_frequency)
    return out
```

## Database Schema

```sql
CREATE TABLE adaptation_profiles (
    user_id INTEGER PRIMARY KEY,
    warmth REAL DEFAULT 0.5,
    formality REAL DEFAULT 0.5,
    initiative REAL DEFAULT 0.5,
    check_in_frequency REAL DEFAULT 0.5,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Design Principles

1. **Latency First**: Fast path cache for common phrases (~0.1ms)
2. **Semantic Understanding**: ML embeddings for complex emotional signals
3. **Memory Integration**: Historical context influences current adaptation
4. **Temporal Awareness**: Time-of-day and day-of-week adjustments
5. **Graceful Fallback**: Keyword detection when semantic unavailable
6. **Bounded Profiles**: All values clamped to [0.0, 1.0]

## Frozen: January 2025

This architecture is stable and integrated with Memory/Temporal engines.
