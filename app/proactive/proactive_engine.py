"""
Proactive Engine (Follow-Up Engine) v2.

Integrated decision engine for intelligent follow-up questions.
Uses Memory Engine for context, Retrieval Engine for finding relevant memories,
and Temporal Engine for time-based decisions.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      PROACTIVE ENGINE v2                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │                         User Context                                │
    │                              │                                      │
    │              ┌───────────────┼───────────────┐                      │
    │              ▼               ▼               ▼                      │
    │     ┌────────────┐   ┌────────────┐   ┌────────────┐               │
    │     │  TEMPORAL  │   │  MEMORY    │   │ RETRIEVAL  │               │
    │     │  ENGINE    │   │  ENGINE    │   │  ENGINE    │               │
    │     │            │   │            │   │            │               │
    │     │ • Time     │   │ • Facts    │   │ • Episodic │               │
    │     │   context  │   │ • Miles-   │   │ • Factual  │               │
    │     │ • Active   │   │   tones    │   │ • Milestone│               │
    │     │   hours    │   │ • Episodes │   │            │               │
    │     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘               │
    │           │                │                │                       │
    │           └────────────────┼────────────────┘                       │
    │                            ▼                                        │
    │                 ┌───────────────────┐                               │
    │                 │  SEMANTIC ANALYSIS │ ◄── Cached embeddings        │
    │                 │  (Importance,      │     (~1ms cached)            │
    │                 │   Memory Type,     │                              │
    │                 │   Follow-up        │                              │
    │                 │   Triggers)        │                              │
    │                 └─────────┬─────────┘                               │
    │                           │                                         │
    │                           ▼                                         │
    │                 ┌───────────────────┐                               │
    │                 │  DECISION ENGINE  │                               │
    │                 │  • Salience       │                               │
    │                 │  • Cooldown       │                               │
    │                 │  • Rate Limits    │                               │
    │                 └─────────┬─────────┘                               │
    │                           │                                         │
    │                           ▼                                         │
    │                 ┌───────────────────┐                               │
    │                 │  ProactiveResult  │                               │
    │                 └───────────────────┘                               │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Latency Optimization:
    - Fast path for cooldown/rate limit checks (~0.1ms)
    - Cached semantic analysis (~1ms after first call)
    - Batch memory queries via engines (~5ms)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache

# =============================================================================
# ENGINE INTEGRATIONS
# =============================================================================

# Semantic Proactive Analyzer (required)
try:
    from app.semantic.proactive_concepts import (
        get_semantic_proactive_analyzer,
        ProactiveAnalysis,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[ProactiveEngine] Semantic analyzer not available, using fallback")

# Temporal Engine Integration (optional but recommended)
try:
    from app.temporal import TemporalEngine, get_temporal_engine, UserTemporalContext
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    print("[ProactiveEngine] Temporal engine not available")

# Memory Engine Integration (optional but recommended)
try:
    from app.memory import MemoryStore, get_memory_store
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("[ProactiveEngine] Memory engine not available")

# Retrieval Engine Integration (optional but recommended)
try:
    from app.retrieval import RetrievalEngine, RetrievalState
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    print("[ProactiveEngine] Retrieval engine not available")


# =============================================================================
# CONSTANTS
# =============================================================================

_COOLDOWN_HOURS = 4
_MAX_DAILY_ASKS = 4
_SALIENCE_THRESHOLD_HOURS = 6

# Obligation states
OBLIGATION_NONE = "NONE"
OBLIGATION_OPTIONAL = "OPTIONAL"
OBLIGATION_MANDATORY = "MANDATORY"

# Obligation classes with priority
OBLIGATION_PRIORITY = {
    "TASK_CLOSURE": 0,
    "NARRATIVE_EXPECTATION": 1,
    "TASK_BLOCKING": 2,
    "EMOTIONAL_CHECK_IN": 3,
    "MILESTONE_ANNIVERSARY": 4,  # New: anniversary reminders
    "HABIT": 5,
    "OPTIONAL": 6,
}

# Fallback keywords (only used when semantic unavailable)
_IMPORTANCE_MARKERS = {"important", "matters", "long-term", "big", "serious"}


# =============================================================================
# FAST PATH CACHE
# =============================================================================

# Pre-computed patterns for instant decisions (no semantic needed)
FAST_PATH_DECISIONS = {
    # Suppression patterns
    "don't remind me": {"action": "suppress", "reason": "user_requested"},
    "no need to follow up": {"action": "suppress", "reason": "user_requested"},
    "drop it": {"action": "suppress", "reason": "user_requested"},
    "forget about it": {"action": "suppress", "reason": "user_requested"},
    "i've got this": {"action": "suppress", "reason": "user_requested"},
    "leave it alone": {"action": "suppress", "reason": "user_requested"},

    # Completion patterns
    "done": {"action": "mark_complete", "reason": "task_finished"},
    "finished": {"action": "mark_complete", "reason": "task_finished"},
    "completed": {"action": "mark_complete", "reason": "task_finished"},
    "all done": {"action": "mark_complete", "reason": "task_finished"},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProactiveResult:
    """Result of proactive evaluation."""
    should_ask: bool = False
    memory_id: Optional[str] = None
    question_type: Optional[str] = None  # follow_up, check_in, anniversary
    urgency: str = "low"  # low, medium, high
    cooldown_until: Optional[str] = None
    reason: Optional[str] = None

    # Integration context
    temporal_context: Optional[Dict[str, Any]] = None
    memory_tier: Optional[str] = None  # fact, milestone, episode
    retrieval_strategy: Optional[str] = None

    # Obligation info
    obligation_state: str = OBLIGATION_NONE
    obligation_class: Optional[str] = None
    blocks_task_id: Optional[str] = None

    # Scoring
    salience_score: int = 0
    semantic_analysis: Optional[Dict[str, Any]] = None


@dataclass
class ProactiveState:
    """State for proactive evaluation."""
    user_id: str
    now: datetime
    memories: List[Dict[str, Any]] = field(default_factory=list)
    cooldown_state: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Optional[UserTemporalContext] = None

    # Cached analysis results (avoid recomputation)
    _semantic_cache: Dict[str, ProactiveAnalysis] = field(default_factory=dict)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _iso(dt: datetime) -> str:
    """Convert datetime to ISO string."""
    return dt.isoformat()


def _bool(value: Any) -> bool:
    """Safe boolean conversion."""
    return bool(value is True)


# =============================================================================
# PROACTIVE ENGINE v2
# =============================================================================

class ProactiveEngineV2:
    """
    Integrated Proactive Engine with Memory/Retrieval/Temporal integration.

    Features:
        - Semantic classification for importance/triggers
        - Memory Engine integration for three-tier context
        - Temporal Engine integration for time-aware decisions
        - Fast path cache for common patterns
        - Latency-optimized with cached embeddings
    """

    def __init__(self):
        """Initialize the proactive engine."""
        self._semantic_analyzer = None
        self._temporal_engine = None
        self._memory_store = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of dependencies."""
        if self._initialized:
            return

        if SEMANTIC_AVAILABLE:
            self._semantic_analyzer = get_semantic_proactive_analyzer()

        if TEMPORAL_AVAILABLE:
            self._temporal_engine = get_temporal_engine()

        if MEMORY_AVAILABLE:
            try:
                self._memory_store = get_memory_store()
            except Exception:
                pass

        self._initialized = True

    # -------------------------------------------------------------------------
    # FAST PATH CHECKS
    # -------------------------------------------------------------------------

    def _check_fast_path(self, text: str) -> Optional[Dict[str, str]]:
        """
        Check fast path cache for instant decisions.

        Latency: ~0.1ms
        """
        if not text:
            return None

        lowered = text.lower().strip()
        return FAST_PATH_DECISIONS.get(lowered)

    def _check_rate_limits(
        self,
        state: ProactiveState,
        is_mandatory: bool = False
    ) -> Optional[ProactiveResult]:
        """
        Check cooldown and daily limits.

        Latency: ~0.1ms

        Mandatory obligations bypass rate limits.
        """
        if is_mandatory:
            return None  # Mandatory bypasses limits

        asks_today = int(state.cooldown_state.get("asks_today", 0) or 0)
        last_asked_at = _parse_iso(state.cooldown_state.get("last_asked_at"))

        # Check daily limit
        if asks_today >= _MAX_DAILY_ASKS:
            return ProactiveResult(
                should_ask=False,
                reason="daily_limit_reached"
            )

        # Check cooldown
        if last_asked_at:
            cooldown_until = last_asked_at + timedelta(hours=_COOLDOWN_HOURS)
            if state.now < cooldown_until:
                return ProactiveResult(
                    should_ask=False,
                    reason="cooldown_active",
                    cooldown_until=_iso(cooldown_until)
                )

        return None

    # -------------------------------------------------------------------------
    # SEMANTIC ANALYSIS (CACHED)
    # -------------------------------------------------------------------------

    def _analyze_semantic(
        self,
        text: str,
        state: ProactiveState,
        threshold: float = 0.45
    ) -> Optional[ProactiveAnalysis]:
        """
        Get semantic analysis with caching.

        Latency: ~5ms first call, ~1ms cached
        """
        if not SEMANTIC_AVAILABLE or not self._semantic_analyzer:
            return None

        # Check cache first
        cache_key = text[:200]  # Truncate for cache key
        if cache_key in state._semantic_cache:
            return state._semantic_cache[cache_key]

        # Compute and cache
        analysis = self._semantic_analyzer.analyze(text, threshold)
        state._semantic_cache[cache_key] = analysis
        return analysis

    # -------------------------------------------------------------------------
    # MEMORY TYPE DETECTION
    # -------------------------------------------------------------------------

    def _is_general_knowledge(self, memory: Dict[str, Any], state: ProactiveState) -> bool:
        """Check if memory is general knowledge (not eligible for follow-up)."""
        # Explicit metadata
        if _bool(memory.get("is_general_knowledge")):
            return True

        category = str(memory.get("category", "")).lower()
        memory_type = str(memory.get("memory_type", "")).lower()

        if category in {"general_knowledge", "gk", "learning"}:
            return True
        if memory_type in {"general_knowledge", "gk", "learning"}:
            return True

        # Semantic detection
        text = self._get_memory_text(memory)
        if text:
            analysis = self._analyze_semantic(text, state, threshold=0.50)
            if analysis and analysis.memory_type in {"general_knowledge", "learning_memory"}:
                return True

        return False

    def _is_personal_or_task(self, memory: Dict[str, Any], state: ProactiveState) -> bool:
        """Check if memory is personal or task-related (eligible for follow-up)."""
        memory_type = str(memory.get("memory_type", "")).lower()

        if _bool(memory.get("is_personal")):
            return True
        if memory_type in {"task", "event", "note", "personal"}:
            return True

        # Semantic detection
        text = self._get_memory_text(memory)
        if text:
            analysis = self._analyze_semantic(text, state, threshold=0.45)
            if analysis and analysis.memory_type in {"task_memory", "event_memory", "personal_memory"}:
                return True

        return False

    def _get_memory_text(self, memory: Dict[str, Any]) -> str:
        """Extract text content from memory."""
        parts = [
            memory.get("text"),
            memory.get("content"),
            memory.get("summary"),
            memory.get("focus"),
        ]
        return " ".join([str(p) for p in parts if p])

    # -------------------------------------------------------------------------
    # SUPPRESSION DETECTION
    # -------------------------------------------------------------------------

    def _should_suppress(self, memory: Dict[str, Any], state: ProactiveState) -> bool:
        """Check if memory should be suppressed from follow-ups."""
        # Explicit flags
        if _bool(memory.get("do_not_remind")):
            return True
        if _bool(memory.get("suppress_proactive")):
            return True
        if _bool(memory.get("no_remind")):
            return True

        # Fast path check
        text = self._get_memory_text(memory)
        fast_result = self._check_fast_path(text)
        if fast_result and fast_result.get("action") == "suppress":
            return True

        # Semantic suppression
        if text:
            analysis = self._analyze_semantic(text, state, threshold=0.50)
            if analysis and analysis.should_suppress:
                return True

        return False

    # -------------------------------------------------------------------------
    # TEMPORAL INTEGRATION
    # -------------------------------------------------------------------------

    def _get_temporal_context(self, state: ProactiveState) -> Dict[str, Any]:
        """
        Get temporal context from Temporal Engine.

        Used for:
            - Active hours detection
            - Milestone anniversary detection
            - Time-based urgency
        """
        if not TEMPORAL_AVAILABLE or not self._temporal_engine:
            return {}

        try:
            # Get user's temporal context
            if state.temporal_context:
                return {
                    "active_hours": state.temporal_context.active_hours,
                    "active_days": state.temporal_context.active_days,
                    "timezone": state.temporal_context.timezone,
                    "is_active_time": self._is_active_time(state),
                }
        except Exception:
            pass

        return {}

    def _is_active_time(self, state: ProactiveState) -> bool:
        """Check if current time is within user's active hours."""
        if not state.temporal_context:
            return True  # Default to active if no context

        try:
            current_hour = state.now.hour
            active_hours = state.temporal_context.active_hours or []
            if active_hours:
                return current_hour in active_hours
        except Exception:
            pass

        return True

    def _check_milestone_anniversaries(
        self,
        state: ProactiveState
    ) -> List[Dict[str, Any]]:
        """
        Check for milestone anniversaries using Memory Engine.

        Returns list of milestones with upcoming anniversaries.
        """
        if not MEMORY_AVAILABLE or not self._memory_store:
            return []

        try:
            # Query milestones tier for anniversary candidates
            today = state.now.date()
            anniversaries = []

            for mem in state.memories:
                milestone_date = mem.get("milestone_date")
                if not milestone_date:
                    continue

                # Parse milestone date
                try:
                    if isinstance(milestone_date, str):
                        md = datetime.fromisoformat(milestone_date).date()
                    else:
                        md = milestone_date

                    # Check if anniversary is within next 3 days
                    anniversary_this_year = md.replace(year=today.year)
                    days_until = (anniversary_this_year - today).days

                    if 0 <= days_until <= 3:
                        anniversaries.append({
                            "memory": mem,
                            "days_until": days_until,
                            "anniversary_date": anniversary_this_year.isoformat(),
                        })
                except Exception:
                    continue

            return anniversaries
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # SALIENCE SCORING
    # -------------------------------------------------------------------------

    def compute_salience(
        self,
        memory: Dict[str, Any],
        state: ProactiveState
    ) -> int:
        """
        Compute salience score for a memory.

        Uses:
            - Base rule scoring
            - Semantic analysis boost
            - Temporal urgency boost
        """
        score = 0
        memory_type = str(memory.get("memory_type", "")).lower()
        due_at = _parse_iso(memory.get("due_at"))

        # Base scoring
        if memory_type == "task" and due_at:
            score += 3
        if str(memory.get("importance", "")).lower() == "high":
            score += 3

        # Time urgency
        if due_at:
            time_to_due = due_at - state.now
            if time_to_due.total_seconds() >= 0:
                if time_to_due <= timedelta(hours=_SALIENCE_THRESHOLD_HOURS):
                    score += 2
                elif time_to_due <= timedelta(days=1):
                    score += 1

        # Emotional context
        if _bool(memory.get("has_linked_emotion")) or memory.get("emotional_state"):
            score += 2

        # Penalties
        if memory_type == "habit":
            score -= 2
        if self._is_generic_emotional(memory):
            score -= 3

        # Semantic boost
        text = self._get_memory_text(memory)
        if text:
            analysis = self._analyze_semantic(text, state)
            if analysis:
                score += analysis.total_salience_boost
                if analysis.should_suppress:
                    score -= 10

        return score

    def _is_generic_emotional(self, memory: Dict[str, Any]) -> bool:
        """Check if memory is generic emotional state (lower priority)."""
        memory_type = str(memory.get("memory_type", "")).lower()
        category = str(memory.get("category", "")).lower()

        if _bool(memory.get("is_emotional_state")) and not memory.get("linked_task_id"):
            return True
        if category in {"emotion", "emotional_state"} and not memory.get("linked_task_id"):
            return True

        return False

    # -------------------------------------------------------------------------
    # CANDIDATE SELECTION
    # -------------------------------------------------------------------------

    def _get_mandatory_candidates(
        self,
        state: ProactiveState
    ) -> List[Dict[str, Any]]:
        """Get mandatory obligation candidates."""
        candidates = []

        # Task closure (past due)
        candidates.extend(self._get_closure_candidates(state))

        # Task blocking (risk window)
        candidates.extend(self._get_blocking_candidates(state))

        # Narrative expectation
        candidates.extend(self._get_narrative_candidates(state))

        # Milestone anniversaries
        candidates.extend(self._get_anniversary_candidates(state))

        return candidates

    def _get_closure_candidates(self, state: ProactiveState) -> List[Dict[str, Any]]:
        """Get task closure candidates (past due, outcome unknown)."""
        candidates = []

        for mem in state.memories:
            if self._is_general_knowledge(mem, state):
                continue
            if str(mem.get("memory_type", "")).lower() != "task":
                continue

            due_at = _parse_iso(mem.get("due_at"))
            if not due_at or state.now <= due_at:
                continue
            if _bool(mem.get("outcome_known")):
                continue

            created_at = _parse_iso(mem.get("created_at"))
            if not created_at:
                continue

            score = self.compute_salience(mem, state)
            candidates.append({
                "memory": mem,
                "obligation_state": OBLIGATION_MANDATORY,
                "obligation_class": "TASK_CLOSURE",
                "violation_at": due_at,
                "score": score,
                "due_at": due_at,
                "created_at": created_at,
            })

        return candidates

    def _get_blocking_candidates(self, state: ProactiveState) -> List[Dict[str, Any]]:
        """Get task blocking candidates (risk window, unresolved prerequisites)."""
        candidates = []
        memory_index = {str(m.get("memory_id")): m for m in state.memories if m.get("memory_id")}

        for mem in state.memories:
            if self._is_general_knowledge(mem, state):
                continue
            if str(mem.get("memory_type", "")).lower() != "task":
                continue

            due_at = _parse_iso(mem.get("due_at"))
            if not due_at or state.now >= due_at:
                continue

            risk_window = mem.get("risk_window_hours")
            if not isinstance(risk_window, (int, float)):
                continue

            risk_start = due_at - timedelta(hours=float(risk_window))
            if state.now < risk_start:
                continue

            prerequisites = mem.get("prerequisites") or []
            if not isinstance(prerequisites, list):
                continue

            # Find unresolved prerequisites
            unresolved = []
            for prereq in prerequisites:
                if not isinstance(prereq, dict):
                    continue
                prereq_id = prereq.get("id") or prereq.get("memory_id")
                if prereq_id is None:
                    continue

                status = str(prereq.get("status", "")).lower()
                if status in {"done", "complete", "completed"}:
                    continue

                prereq_mem = memory_index.get(str(prereq_id))
                if prereq_mem and not _bool(prereq_mem.get("outcome_known")):
                    unresolved.append(prereq_mem)

            if not unresolved:
                continue

            # Use first unresolved prerequisite
            prereq_mem = unresolved[0]
            created_at = _parse_iso(prereq_mem.get("created_at"))
            if not created_at:
                continue

            score = self.compute_salience(prereq_mem, state)
            candidates.append({
                "memory": prereq_mem,
                "obligation_state": OBLIGATION_MANDATORY,
                "obligation_class": "TASK_BLOCKING",
                "violation_at": risk_start,
                "blocks_task_id": str(mem.get("memory_id")),
                "score": score,
                "due_at": _parse_iso(prereq_mem.get("due_at")),
                "created_at": created_at,
            })

        return candidates

    def _get_narrative_candidates(self, state: ProactiveState) -> List[Dict[str, Any]]:
        """Get narrative expectation candidates (promised updates)."""
        candidates = []

        for mem in state.memories:
            if self._is_general_knowledge(mem, state):
                continue
            if _bool(mem.get("asked_before")) or _bool(mem.get("proactive_asked")):
                continue

            boundary = _parse_iso(mem.get("narrative_boundary_at"))
            if not boundary or state.now <= boundary:
                continue
            if _bool(mem.get("acknowledged")):
                continue

            created_at = _parse_iso(mem.get("created_at"))
            if not created_at:
                continue

            score = self.compute_salience(mem, state)
            candidates.append({
                "memory": mem,
                "obligation_state": OBLIGATION_MANDATORY,
                "obligation_class": "NARRATIVE_EXPECTATION",
                "violation_at": boundary,
                "score": score,
                "due_at": _parse_iso(mem.get("due_at")),
                "created_at": created_at,
            })

        return candidates

    def _get_anniversary_candidates(self, state: ProactiveState) -> List[Dict[str, Any]]:
        """Get milestone anniversary candidates."""
        candidates = []
        anniversaries = self._check_milestone_anniversaries(state)

        for anniv in anniversaries:
            mem = anniv["memory"]
            created_at = _parse_iso(mem.get("created_at"))
            if not created_at:
                continue

            score = self.compute_salience(mem, state)
            score += 3  # Anniversary boost

            candidates.append({
                "memory": mem,
                "obligation_state": OBLIGATION_OPTIONAL,  # Anniversaries are optional
                "obligation_class": "MILESTONE_ANNIVERSARY",
                "violation_at": None,
                "score": score,
                "due_at": None,
                "created_at": created_at,
                "days_until_anniversary": anniv["days_until"],
            })

        return candidates

    def _select_best_mandatory(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the best mandatory candidate by priority."""
        if not candidates:
            return None

        def sort_key(item: Dict[str, Any]) -> Tuple[int, datetime, datetime]:
            priority = OBLIGATION_PRIORITY.get(item.get("obligation_class"), 99)
            violation_at = item.get("violation_at") or datetime.max.replace(tzinfo=timezone.utc)
            created_at = item.get("created_at") or datetime.max.replace(tzinfo=timezone.utc)
            return (priority, violation_at, created_at)

        candidates.sort(key=sort_key)
        return candidates[0]

    def _get_optional_candidates(
        self,
        state: ProactiveState
    ) -> List[Dict[str, Any]]:
        """Get optional follow-up candidates."""
        candidates = []

        for mem in state.memories:
            # Eligibility checks
            if not self._is_personal_or_task(mem, state):
                continue
            if self._is_general_knowledge(mem, state):
                continue
            if self._should_suppress(mem, state):
                continue
            if _bool(mem.get("asked_before")) or _bool(mem.get("proactive_asked")):
                continue
            if self._recently_discussed(mem, state.now):
                continue

            # Must be unresolved or time-relevant
            if not (self._is_unresolved(mem) or self._is_time_relevant(mem, state.now)):
                continue

            # Skip tasks before due date
            memory_type = str(mem.get("memory_type", "")).lower()
            due_at = _parse_iso(mem.get("due_at"))
            if memory_type == "task" and due_at and state.now <= due_at:
                continue

            created_at = _parse_iso(mem.get("created_at"))
            if not created_at:
                continue

            score = self.compute_salience(mem, state)
            candidates.append({
                "memory": mem,
                "obligation_state": OBLIGATION_OPTIONAL,
                "obligation_class": "OPTIONAL",
                "score": score,
                "due_at": due_at,
                "created_at": created_at,
            })

        # Sort by score (descending), due date, created date
        candidates.sort(key=lambda x: (
            -x["score"],
            x["due_at"] or datetime.max.replace(tzinfo=timezone.utc),
            -x["created_at"].timestamp()
        ))

        return candidates

    def _is_unresolved(self, memory: Dict[str, Any]) -> bool:
        """Check if memory is unresolved."""
        if _bool(memory.get("outcome_known")):
            return False
        if _bool(memory.get("is_resolved")):
            return False
        if memory.get("resolved") is False:
            return True

        status = str(memory.get("status", "")).lower()
        return status in {"open", "pending", "unresolved"}

    def _is_time_relevant(self, memory: Dict[str, Any], now: datetime) -> bool:
        """Check if memory is time-relevant."""
        if _bool(memory.get("time_relevant")):
            return True

        due_at = _parse_iso(memory.get("due_at"))
        return due_at is not None and now <= due_at

    def _recently_discussed(self, memory: Dict[str, Any], now: datetime) -> bool:
        """Check if memory was recently discussed."""
        for key in ("last_discussed_at", "last_accessed_at", "last_mentioned_at"):
            ts = _parse_iso(memory.get(key))
            if ts and now - ts < timedelta(hours=_COOLDOWN_HOURS):
                return True
        return False

    # -------------------------------------------------------------------------
    # MAIN EVALUATION
    # -------------------------------------------------------------------------

    def evaluate(self, payload: Dict[str, Any]) -> ProactiveResult:
        """
        Evaluate whether a proactive ask is permitted.

        Expected input:
            {
                user_id: str,
                now_timestamp: ISO-8601,
                recent_memories: List[Memory],
                cooldown_state: { last_asked_at: ISO-8601, asks_today: int },
                temporal_context: Optional[UserTemporalContext]
            }

        Returns:
            ProactiveResult with decision and context
        """
        self._ensure_initialized()

        # Parse input
        now = _parse_iso(payload.get("now_timestamp"))
        if not now:
            return ProactiveResult(should_ask=False, reason="no_timestamp")

        memories = payload.get("recent_memories", [])
        if not isinstance(memories, list):
            memories = []

        # Build state
        state = ProactiveState(
            user_id=payload.get("user_id", ""),
            now=now,
            memories=memories,
            cooldown_state=payload.get("cooldown_state") or {},
            temporal_context=payload.get("temporal_context"),
        )

        # Check if in active window
        if self._in_active_window(state):
            return ProactiveResult(should_ask=False, reason="active_window")

        # Get temporal context for the result
        temporal_ctx = self._get_temporal_context(state)

        # Check mandatory obligations first (bypass rate limits)
        mandatory = self._get_mandatory_candidates(state)
        if mandatory:
            best = self._select_best_mandatory(mandatory)
            if best:
                return self._build_result(best, state, temporal_ctx, is_mandatory=True)

        # Check rate limits for optional
        rate_limit_result = self._check_rate_limits(state, is_mandatory=False)
        if rate_limit_result:
            return rate_limit_result

        # Get optional candidates
        optional = self._get_optional_candidates(state)
        if optional:
            best = optional[0]
            return self._build_result(best, state, temporal_ctx, is_mandatory=False)

        return ProactiveResult(should_ask=False, reason="no_trigger")

    def _in_active_window(self, state: ProactiveState) -> bool:
        """Check if user is in an active conversation window."""
        for mem in state.memories:
            start = _parse_iso(mem.get("active_window_start"))
            end = _parse_iso(mem.get("active_window_end"))
            if start and end and start <= state.now <= end:
                return True
        return False

    def _build_result(
        self,
        candidate: Dict[str, Any],
        state: ProactiveState,
        temporal_ctx: Dict[str, Any],
        is_mandatory: bool
    ) -> ProactiveResult:
        """Build ProactiveResult from selected candidate."""
        memory = candidate["memory"]
        memory_id = str(memory.get("memory_id"))
        obligation_class = candidate.get("obligation_class")

        # Determine question type
        if obligation_class == "MILESTONE_ANNIVERSARY":
            question_type = "anniversary"
        elif obligation_class in {"TASK_BLOCKING", "TASK_CLOSURE"}:
            question_type = "follow_up"
        else:
            memory_type = str(memory.get("memory_type", "")).lower()
            question_type = "follow_up" if memory_type == "task" else "check_in"

        # Compute cooldown
        cooldown_until = _iso(state.now + timedelta(hours=_COOLDOWN_HOURS))

        # Get semantic analysis for result
        text = self._get_memory_text(memory)
        analysis = self._analyze_semantic(text, state) if text else None

        # Determine memory tier
        tier = memory.get("tier") or "episode"
        if memory.get("milestone_date"):
            tier = "milestone"
        elif str(memory.get("memory_type", "")).lower() == "fact":
            tier = "fact"

        # Log decision
        self._log_decision(
            decision="ask",
            memory_id=memory_id,
            reason="eligible",
            state=state,
            obligation_class=obligation_class,
            is_mandatory=is_mandatory,
        )

        return ProactiveResult(
            should_ask=True,
            memory_id=memory_id,
            question_type=question_type,
            urgency="high" if is_mandatory else "low",
            cooldown_until=cooldown_until,
            reason="eligible",
            temporal_context=temporal_ctx,
            memory_tier=tier,
            obligation_state=candidate.get("obligation_state", OBLIGATION_OPTIONAL),
            obligation_class=obligation_class,
            blocks_task_id=candidate.get("blocks_task_id"),
            salience_score=candidate.get("score", 0),
            semantic_analysis=analysis.__dict__ if analysis else None,
        )

    def _log_decision(
        self,
        decision: str,
        memory_id: Optional[str],
        reason: str,
        state: ProactiveState,
        obligation_class: Optional[str] = None,
        is_mandatory: bool = False,
    ) -> None:
        """Log proactive decision for debugging."""
        out_dir = r"D:\Nura\Docs\runtime_logs\proactive_engine"
        try:
            os.makedirs(out_dir, exist_ok=True)

            asks_today = int(state.cooldown_state.get("asks_today", 0) or 0)
            payload = {
                "decision": decision,
                "memory_id": memory_id,
                "reason": reason,
                "asks_today": asks_today,
                "obligation_class": obligation_class,
                "is_mandatory": is_mandatory,
                "timestamp": state.now.isoformat(),
            }

            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f.json")
            path = os.path.join(out_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True)
        except Exception:
            pass  # Logging failure shouldn't break the engine


# =============================================================================
# SINGLETON & LEGACY INTERFACE
# =============================================================================

_engine_instance: Optional[ProactiveEngineV2] = None


def get_proactive_engine() -> ProactiveEngineV2:
    """Get or create the singleton proactive engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ProactiveEngineV2()
    return _engine_instance


def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy interface for backward compatibility.

    Returns dict instead of ProactiveResult.
    """
    engine = get_proactive_engine()
    result = engine.evaluate(payload)

    return {
        "should_ask": result.should_ask,
        "memory_id": result.memory_id,
        "question_type": result.question_type,
        "urgency": result.urgency,
        "cooldown_until": result.cooldown_until,
        "reason": result.reason,
    }


def decide_followup(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for evaluate to match existing orchestration hooks."""
    return evaluate(payload)


# =============================================================================
# EXPORTS
# =============================================================================

# Re-export for compatibility
ProactiveEngine = ProactiveEngineV2

__all__ = [
    "ProactiveEngineV2",
    "ProactiveEngine",
    "ProactiveResult",
    "ProactiveState",
    "get_proactive_engine",
    "evaluate",
    "decide_followup",
    "OBLIGATION_MANDATORY",
    "OBLIGATION_OPTIONAL",
    "OBLIGATION_NONE",
]
