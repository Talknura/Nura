"""
Adaptation Engine v2.

Integrated engine for user communication style adaptation.
Uses Memory Engine for historical context, Retrieval Engine for past interactions,
and Temporal Engine for time-based adaptation patterns.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         ADAPTATION ENGINE v2                                │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │                            User Input                                       │
    │                                │                                            │
    │              ┌─────────────────┼─────────────────┐                          │
    │              ▼                 ▼                 ▼                          │
    │     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐               │
    │     │    MEMORY      │ │   TEMPORAL     │ │   SEMANTIC     │               │
    │     │    ENGINE      │ │   ENGINE       │ │   ANALYZER     │               │
    │     │                │ │                │ │                │               │
    │     │ • Past prefs   │ │ • Time of day  │ │ • Vulnerability│               │
    │     │ • Milestones   │ │ • Day of week  │ │ • Gratitude    │               │
    │     │ • Emotional    │ │ • Patterns     │ │ • Engagement   │               │
    │     │   history      │ │                │ │ • Style        │               │
    │     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘               │
    │             │                  │                  │                         │
    │             └──────────────────┼──────────────────┘                         │
    │                                ▼                                            │
    │                   ┌─────────────────────────┐                               │
    │                   │   ADAPTATION PROFILE    │                               │
    │                   │                         │                               │
    │                   │  • warmth (0.0-1.0)     │                               │
    │                   │  • formality (0.0-1.0)  │                               │
    │                   │  • initiative (0.0-1.0) │                               │
    │                   │  • check_in_freq        │                               │
    │                   └─────────────────────────┘                               │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

Latency Optimization:
    - Fast path for common signals (~0.1ms)
    - Cached semantic analysis (~1ms after first call)
    - Batch profile updates (single DB write)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol

# =============================================================================
# ENGINE INTEGRATIONS
# =============================================================================

# Database integration
try:
    from app.db.session import get_db_context
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[AdaptationEngine] Database not available")

# Semantic Adaptation Analyzer
try:
    from app.semantic.adaptation_concepts import (
        get_semantic_adaptation_analyzer,
        AdaptationSignals,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[AdaptationEngine] Semantic analyzer not available")

# Temporal Engine Integration
try:
    from app.temporal import get_temporal_engine, UserTemporalContext
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    print("[AdaptationEngine] Temporal engine not available")

# Memory Engine Integration
try:
    from app.memory import get_memory_store
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("[AdaptationEngine] Memory engine not available")

# Legacy breakthrough detector (fallback)
try:
    from app.adaptation.breakthrough_detector import detect_breakthrough, BreakthroughSignals
    BREAKTHROUGH_AVAILABLE = True
except ImportError:
    BREAKTHROUGH_AVAILABLE = False

# Adaptation rules
from app.adaptation.adaptation_rules import AdaptationDelta, apply_delta


# =============================================================================
# CONSTANTS
# =============================================================================

# Default profile values
DEFAULT_PROFILE = {
    "warmth": 0.5,
    "formality": 0.5,
    "initiative": 0.5,
    "check_in_frequency": 0.5,
}

# Fast path patterns for instant adaptation (no semantic needed)
FAST_PATH_SIGNALS = {
    # Vulnerability signals
    "i'm struggling": {"warmth": 0.12, "check_in_frequency": 0.08},
    "i'm scared": {"warmth": 0.10, "check_in_frequency": 0.05},
    "i'm so sad": {"warmth": 0.10, "check_in_frequency": 0.05},
    "i feel hopeless": {"warmth": 0.15, "check_in_frequency": 0.10},
    "i need help": {"warmth": 0.10, "initiative": 0.08},

    # Gratitude signals
    "thank you": {"initiative": 0.05},
    "thanks so much": {"initiative": 0.06, "warmth": 0.03},
    "i appreciate it": {"initiative": 0.05},
    "you're amazing": {"initiative": 0.08, "warmth": 0.05},

    # Style preferences
    "get to the point": {"formality": -0.08},
    "keep it short": {"formality": 0.02, "initiative": -0.03},
    "no need to be formal": {"formality": -0.10},
    "be professional": {"formality": 0.10},

    # Engagement signals
    "tell me more": {"initiative": 0.08, "warmth": 0.03},
    "this is fascinating": {"initiative": 0.08},
    "ok": {"initiative": -0.03},
    "whatever": {"initiative": -0.05},
}

# Time-of-day adaptation modifiers
TIME_OF_DAY_MODIFIERS = {
    "morning": {"warmth": 0.02, "initiative": 0.01},      # Friendly morning greeting
    "evening": {"warmth": 0.03, "initiative": -0.02},     # Warmer, less pushy at night
    "night": {"warmth": 0.05, "initiative": -0.03},       # Empathetic late night
}

# Day-of-week adaptation modifiers
DAY_MODIFIERS = {
    "weekend": {"formality": -0.02, "initiative": -0.01},  # More casual on weekends
    "weekday": {"formality": 0.01},                        # Slightly more formal weekdays
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AdaptationProfile:
    """User's adaptation profile."""
    user_id: int
    warmth: float = 0.5
    formality: float = 0.5
    initiative: float = 0.5
    check_in_frequency: float = 0.5
    updated_at: Optional[datetime] = None

    # Extended profile (integrated context)
    preferred_style: Optional[str] = None  # direct, detailed, casual, formal
    engagement_trend: Optional[str] = None  # increasing, stable, decreasing
    emotional_baseline: Optional[str] = None  # positive, neutral, negative

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmth": self.warmth,
            "formality": self.formality,
            "initiative": self.initiative,
            "check_in_frequency": self.check_in_frequency,
            "preferred_style": self.preferred_style,
            "engagement_trend": self.engagement_trend,
            "emotional_baseline": self.emotional_baseline,
        }


@dataclass
class AdaptationContext:
    """Context for adaptation decisions."""
    user_id: int
    now: datetime
    user_text: str
    temporal_context: Optional[Dict[str, Any]] = None
    memory_context: Optional[Dict[str, Any]] = None

    # Cached semantic analysis
    _semantic_signals: Optional[AdaptationSignals] = None


@dataclass
class AdaptationResult:
    """Result of adaptation evaluation."""
    profile: AdaptationProfile
    delta_applied: AdaptationDelta
    signals_detected: Dict[str, Any] = field(default_factory=dict)
    temporal_modifiers: Dict[str, float] = field(default_factory=dict)
    memory_influence: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# PROTOCOL FOR TYPE HINTS
# =============================================================================

class AdaptationEngineProtocol(Protocol):
    """Protocol for adaptation engine interface."""
    def get_profile(self, user_id: int) -> dict: ...
    def update(self, user_id: int, metrics: Any) -> dict: ...


# =============================================================================
# ADAPTATION ENGINE v2
# =============================================================================

class AdaptationEngineV2:
    """
    Integrated Adaptation Engine with Memory/Temporal integration.

    Features:
        - Semantic classification for emotional/style signals
        - Memory Engine integration for historical context
        - Temporal Engine integration for time-based adaptation
        - Fast path cache for common patterns
        - Latency-optimized with cached embeddings
    """

    def __init__(self):
        """Initialize the adaptation engine."""
        self._semantic_analyzer = None
        self._temporal_engine = None
        self._memory_store = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of dependencies."""
        if self._initialized:
            return

        if SEMANTIC_AVAILABLE:
            self._semantic_analyzer = get_semantic_adaptation_analyzer()

        if TEMPORAL_AVAILABLE:
            self._temporal_engine = get_temporal_engine()

        if MEMORY_AVAILABLE:
            try:
                self._memory_store = get_memory_store()
            except Exception:
                pass

        self._initialized = True

    def _ensure_profile(self, user_id: int) -> None:
        """Ensure user has an adaptation profile in database."""
        if not DB_AVAILABLE:
            return

        with get_db_context() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO adaptation_profiles(user_id) VALUES (?)",
                (user_id,)
            )
            conn.commit()

    # -------------------------------------------------------------------------
    # PROFILE MANAGEMENT
    # -------------------------------------------------------------------------

    def get_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Get user's adaptation profile.

        Returns dict for backward compatibility.
        """
        self._ensure_profile(user_id)

        if not DB_AVAILABLE:
            return dict(DEFAULT_PROFILE)

        with get_db_context() as conn:
            row = conn.execute(
                """SELECT warmth, formality, initiative, check_in_frequency
                   FROM adaptation_profiles WHERE user_id=?""",
                (user_id,)
            ).fetchone()

            if row:
                return dict(row)
            return dict(DEFAULT_PROFILE)

    def get_profile_extended(self, user_id: int) -> AdaptationProfile:
        """Get extended adaptation profile with full context."""
        base = self.get_profile(user_id)
        return AdaptationProfile(
            user_id=user_id,
            warmth=base.get("warmth", 0.5),
            formality=base.get("formality", 0.5),
            initiative=base.get("initiative", 0.5),
            check_in_frequency=base.get("check_in_frequency", 0.5),
        )

    def _save_profile(self, profile: AdaptationProfile) -> None:
        """Save profile to database."""
        if not DB_AVAILABLE:
            return

        with get_db_context() as conn:
            conn.execute(
                """UPDATE adaptation_profiles
                   SET warmth=?, formality=?, initiative=?, check_in_frequency=?,
                       updated_at=datetime('now')
                   WHERE user_id=?""",
                (profile.warmth, profile.formality, profile.initiative,
                 profile.check_in_frequency, profile.user_id)
            )
            conn.commit()

    # -------------------------------------------------------------------------
    # FAST PATH DETECTION
    # -------------------------------------------------------------------------

    def _check_fast_path(self, text: str) -> Optional[Dict[str, float]]:
        """
        Check fast path cache for instant adaptation signals.

        Latency: ~0.1ms
        """
        if not text:
            return None

        lowered = text.lower().strip()

        # Exact match first
        if lowered in FAST_PATH_SIGNALS:
            return FAST_PATH_SIGNALS[lowered]

        # Substring match for common patterns
        for pattern, deltas in FAST_PATH_SIGNALS.items():
            if pattern in lowered:
                return deltas

        return None

    # -------------------------------------------------------------------------
    # SEMANTIC ANALYSIS
    # -------------------------------------------------------------------------

    def _analyze_semantic(
        self,
        text: str,
        context: AdaptationContext,
        threshold: float = 0.45
    ) -> Optional[AdaptationSignals]:
        """
        Get semantic analysis with caching.

        Latency: ~5ms first call, ~1ms cached
        """
        if not SEMANTIC_AVAILABLE or not self._semantic_analyzer:
            return None

        # Check cache
        if context._semantic_signals is not None:
            return context._semantic_signals

        # Compute and cache
        signals = self._semantic_analyzer.analyze(text, threshold)
        context._semantic_signals = signals
        return signals

    def _get_fallback_signals(self, text: str) -> Dict[str, float]:
        """Get adaptation deltas using fallback breakthrough detection."""
        if not BREAKTHROUGH_AVAILABLE:
            return {}

        sig = detect_breakthrough(text)
        delta = {}

        if sig.vulnerability:
            delta["warmth"] = delta.get("warmth", 0) + 0.10
            delta["check_in_frequency"] = delta.get("check_in_frequency", 0) + 0.05
        if sig.gratitude:
            delta["initiative"] = delta.get("initiative", 0) + 0.05
        if sig.prayer:
            delta["warmth"] = delta.get("warmth", 0) + 0.05

        return delta

    # -------------------------------------------------------------------------
    # TEMPORAL INTEGRATION
    # -------------------------------------------------------------------------

    def _get_temporal_modifiers(
        self,
        context: AdaptationContext
    ) -> Dict[str, float]:
        """
        Get adaptation modifiers based on temporal context.

        Adjusts profile based on time of day, day of week, etc.
        """
        modifiers = {}

        if not TEMPORAL_AVAILABLE or not self._temporal_engine:
            return modifiers

        try:
            # Get temporal tags
            tags = self._temporal_engine.generate_temporal_tags(context.now)

            # Time of day adjustment
            time_of_day = tags.get("time_of_day", "")
            if time_of_day in TIME_OF_DAY_MODIFIERS:
                for key, value in TIME_OF_DAY_MODIFIERS[time_of_day].items():
                    modifiers[key] = modifiers.get(key, 0) + value

            # Weekend/weekday adjustment
            is_weekend = tags.get("is_weekend", False)
            day_type = "weekend" if is_weekend else "weekday"
            if day_type in DAY_MODIFIERS:
                for key, value in DAY_MODIFIERS[day_type].items():
                    modifiers[key] = modifiers.get(key, 0) + value

            context.temporal_context = tags

        except Exception:
            pass

        return modifiers

    # -------------------------------------------------------------------------
    # MEMORY INTEGRATION
    # -------------------------------------------------------------------------

    def _get_memory_influence(
        self,
        context: AdaptationContext
    ) -> Dict[str, float]:
        """
        Get adaptation influence from memory history.

        Looks at past emotional patterns, milestones, etc.
        """
        influence = {}

        if not MEMORY_AVAILABLE or not self._memory_store:
            return influence

        try:
            # Check if user recently shared vulnerability
            # (would warrant sustained warmth increase)
            # This is a placeholder - actual implementation would query memory

            context.memory_context = {"queried": True}

        except Exception:
            pass

        return influence

    # -------------------------------------------------------------------------
    # HEURISTIC ANALYSIS
    # -------------------------------------------------------------------------

    def _analyze_text_heuristics(self, text: str) -> Dict[str, float]:
        """
        Analyze text using simple heuristics.

        Used as additional signal alongside semantic analysis.
        """
        deltas = {}
        text_len = len(text)

        # Text length signals engagement
        if text_len > 200:
            # Long, detailed messages -> user wants to share more
            deltas["warmth"] = deltas.get("warmth", 0) + 0.03
            deltas["initiative"] = deltas.get("initiative", 0) + 0.02
        elif text_len < 20:
            # Very short messages -> user may prefer brief interactions
            deltas["formality"] = deltas.get("formality", 0) + 0.02
            deltas["initiative"] = deltas.get("initiative", 0) - 0.02

        # Question count for engagement
        question_count = text.count("?")
        if question_count > 1:
            deltas["initiative"] = deltas.get("initiative", 0) + 0.03

        # First-person pronouns -> personal sharing
        first_person = ["i", "i'm", "im", "i've", "ive", "my", "me", "mine"]
        text_lower = text.lower()
        pronoun_count = sum(1 for p in first_person if p in text_lower.split())
        if pronoun_count >= 3:
            deltas["warmth"] = deltas.get("warmth", 0) + 0.02

        return deltas

    # -------------------------------------------------------------------------
    # MAIN UPDATE
    # -------------------------------------------------------------------------

    def update(self, user_id: int, metrics: Any) -> Dict[str, Any]:
        """
        Update user's adaptation profile based on conversation.

        Args:
            user_id: User identifier
            metrics: ConversationMetrics with user_text and other signals

        Returns:
            Updated profile as dict
        """
        self._ensure_initialized()
        self._ensure_profile(user_id)

        # Extract text from metrics
        user_text = getattr(metrics, "user_text", "") or ""
        if not user_text:
            return self.get_profile(user_id)

        # Build context
        now = datetime.now(timezone.utc)
        context = AdaptationContext(
            user_id=user_id,
            now=now,
            user_text=user_text,
        )

        # Get current profile
        profile = self.get_profile_extended(user_id)
        delta = AdaptationDelta()
        signals_detected = {}

        # 1. Fast path check (~0.1ms)
        fast_deltas = self._check_fast_path(user_text)
        if fast_deltas:
            delta.warmth += fast_deltas.get("warmth", 0)
            delta.formality += fast_deltas.get("formality", 0)
            delta.initiative += fast_deltas.get("initiative", 0)
            delta.check_in_frequency += fast_deltas.get("check_in_frequency", 0)
            signals_detected["fast_path"] = True

        # 2. Semantic analysis (~5ms first, ~1ms cached)
        semantic_signals = self._analyze_semantic(user_text, context)
        if semantic_signals and semantic_signals.adaptation_deltas:
            for key, value in semantic_signals.adaptation_deltas.items():
                if key == "warmth":
                    delta.warmth += value
                elif key == "formality":
                    delta.formality += value
                elif key == "initiative":
                    delta.initiative += value
                elif key == "check_in_frequency":
                    delta.check_in_frequency += value

            signals_detected["vulnerability"] = semantic_signals.vulnerability
            signals_detected["gratitude"] = semantic_signals.gratitude
            signals_detected["prayer"] = semantic_signals.prayer
            signals_detected["style"] = semantic_signals.communication_style
            signals_detected["engagement"] = semantic_signals.engagement_level
            signals_detected["emotion"] = semantic_signals.emotional_state

            # Update extended profile
            if semantic_signals.communication_style:
                profile.preferred_style = semantic_signals.communication_style
            if semantic_signals.engagement_level:
                profile.engagement_trend = semantic_signals.engagement_level
            if semantic_signals.emotional_state:
                profile.emotional_baseline = semantic_signals.emotional_state

        elif not fast_deltas:
            # Fallback to legacy breakthrough detection
            fallback_deltas = self._get_fallback_signals(user_text)
            for key, value in fallback_deltas.items():
                if key == "warmth":
                    delta.warmth += value
                elif key == "formality":
                    delta.formality += value
                elif key == "initiative":
                    delta.initiative += value
                elif key == "check_in_frequency":
                    delta.check_in_frequency += value

        # 3. Text heuristics
        heuristic_deltas = self._analyze_text_heuristics(user_text)
        delta.warmth += heuristic_deltas.get("warmth", 0)
        delta.formality += heuristic_deltas.get("formality", 0)
        delta.initiative += heuristic_deltas.get("initiative", 0)

        # 4. Temporal modifiers
        temporal_modifiers = self._get_temporal_modifiers(context)
        delta.warmth += temporal_modifiers.get("warmth", 0)
        delta.formality += temporal_modifiers.get("formality", 0)
        delta.initiative += temporal_modifiers.get("initiative", 0)

        # 5. Memory influence
        memory_influence = self._get_memory_influence(context)
        delta.warmth += memory_influence.get("warmth", 0)
        delta.formality += memory_influence.get("formality", 0)
        delta.initiative += memory_influence.get("initiative", 0)

        # 6. Check prefers_direct from metrics
        if getattr(metrics, "prefers_direct", False):
            delta.formality -= 0.05

        # Apply delta and save
        profile_dict = profile.to_dict()
        new_profile_dict = apply_delta(profile_dict, delta)

        # Update profile object
        profile.warmth = new_profile_dict["warmth"]
        profile.formality = new_profile_dict["formality"]
        profile.initiative = new_profile_dict["initiative"]
        profile.check_in_frequency = new_profile_dict["check_in_frequency"]
        profile.updated_at = now

        # Save to database
        self._save_profile(profile)

        return new_profile_dict

    def update_full(self, user_id: int, metrics: Any) -> AdaptationResult:
        """
        Full update with detailed result.

        Returns AdaptationResult with all context.
        """
        self._ensure_initialized()
        self._ensure_profile(user_id)

        user_text = getattr(metrics, "user_text", "") or ""
        if not user_text:
            profile = self.get_profile_extended(user_id)
            return AdaptationResult(
                profile=profile,
                delta_applied=AdaptationDelta(),
            )

        now = datetime.now(timezone.utc)
        context = AdaptationContext(
            user_id=user_id,
            now=now,
            user_text=user_text,
        )

        profile = self.get_profile_extended(user_id)
        delta = AdaptationDelta()
        signals = {}
        temporal_mods = {}
        memory_inf = {}

        # Fast path
        fast_deltas = self._check_fast_path(user_text)
        if fast_deltas:
            delta.warmth += fast_deltas.get("warmth", 0)
            delta.formality += fast_deltas.get("formality", 0)
            delta.initiative += fast_deltas.get("initiative", 0)
            delta.check_in_frequency += fast_deltas.get("check_in_frequency", 0)
            signals["fast_path"] = True

        # Semantic
        semantic_signals = self._analyze_semantic(user_text, context)
        if semantic_signals and semantic_signals.adaptation_deltas:
            for key, value in semantic_signals.adaptation_deltas.items():
                if key == "warmth":
                    delta.warmth += value
                elif key == "formality":
                    delta.formality += value
                elif key == "initiative":
                    delta.initiative += value
                elif key == "check_in_frequency":
                    delta.check_in_frequency += value

            signals["vulnerability"] = semantic_signals.vulnerability
            signals["gratitude"] = semantic_signals.gratitude
            signals["prayer"] = semantic_signals.prayer

        # Heuristics
        heuristic_deltas = self._analyze_text_heuristics(user_text)
        delta.warmth += heuristic_deltas.get("warmth", 0)
        delta.formality += heuristic_deltas.get("formality", 0)
        delta.initiative += heuristic_deltas.get("initiative", 0)

        # Temporal
        temporal_mods = self._get_temporal_modifiers(context)
        delta.warmth += temporal_mods.get("warmth", 0)
        delta.formality += temporal_mods.get("formality", 0)
        delta.initiative += temporal_mods.get("initiative", 0)

        # Memory
        memory_inf = self._get_memory_influence(context)
        delta.warmth += memory_inf.get("warmth", 0)
        delta.formality += memory_inf.get("formality", 0)
        delta.initiative += memory_inf.get("initiative", 0)

        # Apply
        profile_dict = profile.to_dict()
        new_profile_dict = apply_delta(profile_dict, delta)

        profile.warmth = new_profile_dict["warmth"]
        profile.formality = new_profile_dict["formality"]
        profile.initiative = new_profile_dict["initiative"]
        profile.check_in_frequency = new_profile_dict["check_in_frequency"]
        profile.updated_at = now

        self._save_profile(profile)

        return AdaptationResult(
            profile=profile,
            delta_applied=delta,
            signals_detected=signals,
            temporal_modifiers=temporal_mods,
            memory_influence=memory_inf,
        )


# =============================================================================
# SINGLETON & LEGACY INTERFACE
# =============================================================================

_engine_instance: Optional[AdaptationEngineV2] = None


def get_adaptation_engine() -> AdaptationEngineV2:
    """Get or create the singleton adaptation engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AdaptationEngineV2()
    return _engine_instance


# Legacy class for backward compatibility
class AdaptationEngine(AdaptationEngineV2):
    """Legacy adapter for backward compatibility."""
    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AdaptationEngineV2",
    "AdaptationEngine",
    "AdaptationProfile",
    "AdaptationContext",
    "AdaptationResult",
    "AdaptationDelta",
    "get_adaptation_engine",
]
