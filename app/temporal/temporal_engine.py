"""
Temporal Engine v2 - Integrated with Three-Tier Memory Architecture.

Provides unified temporal understanding that feeds into:
  1. Memory Engine - Temporal tags for storage, milestone date extraction
  2. Retrieval Engine - Time windows, timeline queries, pattern context

Optimizations:
  - Cached embeddings (singleton parser)
  - Fast path for common phrases
  - Batch processing support

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                     TEMPORAL ENGINE v2                          │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │   Input Text ──► Semantic Parser ──► Temporal Result            │
  │                       │                    │                    │
  │                       │          ┌─────────┴─────────┐          │
  │                       │          │                   │          │
  │                       ▼          ▼                   ▼          │
  │               ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
  │               │   MEMORY    │ │  RETRIEVAL  │ │  PATTERNS   │  │
  │               │  INTEGRATION│ │ INTEGRATION │ │  CONTEXT    │  │
  │               │             │ │             │ │             │  │
  │               │ - Tags      │ │ - Time      │ │ - User      │  │
  │               │ - Milestone │ │   window    │ │   habits    │  │
  │               │   dates     │ │ - Timeline  │ │ - Active    │  │
  │               │ - Tier hint │ │   strategy  │ │   hours     │  │
  │               └─────────────┘ └─────────────┘ └─────────────┘  │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re

from app.time.calendar_model import CalendarModel
from app.time.time_authority import TimeAuthority

# Import semantic parser (already optimized with singleton)
try:
    from app.semantic.temporal_concepts import get_semantic_temporal_parser
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


class TemporalGranularity(Enum):
    """Time granularity levels."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


@dataclass
class TemporalResult:
    """
    Unified temporal parsing result.

    Used by both Memory Engine (for storage) and Retrieval Engine (for queries).
    """
    # Core parsing result
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    granularity: TemporalGranularity = TemporalGranularity.NONE
    confidence: float = 0.0
    direction: int = 0  # -1=past, 0=present, 1=future

    # For Memory Engine integration
    temporal_tags: Dict[str, Any] = field(default_factory=dict)
    milestone_date: Optional[str] = None  # Extracted date for milestones
    memory_tier_hint: Optional[str] = None  # "fact", "milestone", "episode"

    # For Retrieval Engine integration
    retrieval_window_days: Optional[int] = None
    retrieval_strategy_hint: Optional[str] = None  # "timeline", "episodic", etc.
    disable_recency: bool = False  # For explicit past references

    # Metadata
    concept_matched: Optional[str] = None
    requires_clarification: bool = False


@dataclass
class UserTemporalContext:
    """User's temporal patterns and context."""
    active_hours: List[int] = field(default_factory=list)  # Hours user is usually active
    active_days: List[int] = field(default_factory=list)   # Days user is usually active
    timezone: str = "UTC"
    patterns: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# FAST PATH CACHE (common phrases, no embedding needed)
# =============================================================================

FAST_PATH_CACHE: Dict[str, Dict[str, Any]] = {
    # Immediate
    "now": {"granularity": "second", "direction": 0, "days": 0},
    "right now": {"granularity": "second", "direction": 0, "days": 0},

    # Today/Yesterday/Tomorrow
    "today": {"granularity": "day", "direction": 0, "days": 1},
    "yesterday": {"granularity": "day", "direction": -1, "days": 2},
    "tomorrow": {"granularity": "day", "direction": 1, "days": 1},

    # This week/month/year
    "this week": {"granularity": "week", "direction": 0, "days": 7},
    "this month": {"granularity": "month", "direction": 0, "days": 30},
    "this year": {"granularity": "year", "direction": 0, "days": 365},

    # Last week/month/year
    "last week": {"granularity": "week", "direction": -1, "days": 14},
    "last month": {"granularity": "month", "direction": -1, "days": 60},
    "last year": {"granularity": "year", "direction": -1, "days": 730},

    # Next week/month/year
    "next week": {"granularity": "week", "direction": 1, "days": 7},
    "next month": {"granularity": "month", "direction": 1, "days": 30},
    "next year": {"granularity": "year", "direction": 1, "days": 365},

    # Vague past references (disable recency penalty)
    "a while ago": {"granularity": "day", "direction": -1, "days": 30, "disable_recency": True},
    "long time ago": {"granularity": "year", "direction": -1, "days": 365, "disable_recency": True},
    "back then": {"granularity": "year", "direction": -1, "days": 365, "disable_recency": True},
    "years ago": {"granularity": "year", "direction": -1, "days": 730, "disable_recency": True},
}


class TemporalEngineV2:
    """
    Integrated Temporal Engine for three-tier memory architecture.

    Provides:
      1. Temporal parsing with semantic understanding
      2. Memory integration (tags, milestone dates, tier hints)
      3. Retrieval integration (time windows, strategy hints)
      4. User pattern context
    """

    def __init__(self):
        self._semantic_parser = None
        if SEMANTIC_AVAILABLE:
            self._semantic_parser = get_semantic_temporal_parser()

    # =========================================================================
    # MAIN API
    # =========================================================================

    def parse(
        self,
        text: str,
        now: Optional[datetime] = None,
        user_context: Optional[UserTemporalContext] = None
    ) -> TemporalResult:
        """
        Parse temporal references with full integration.

        Args:
            text: Input text containing temporal references
            now: Current datetime (defaults to UTC now)
            user_context: Optional user temporal patterns

        Returns:
            TemporalResult with parsing + memory + retrieval integration
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # 1. Try fast path first (no embedding needed)
        fast_result = self._try_fast_path(text, now)
        if fast_result is not None:
            return fast_result

        # 2. Use semantic parser for complex expressions
        if self._semantic_parser is not None:
            semantic_result = self._parse_semantic(text, now)
            if semantic_result.confidence > 0.4:
                return self._enrich_result(semantic_result, text, now, user_context)

        # 3. Fallback to regex patterns
        regex_result = self._parse_regex(text, now)
        return self._enrich_result(regex_result, text, now, user_context)

    def generate_temporal_tags(self, dt: datetime) -> Dict[str, Any]:
        """
        Generate temporal tags for memory storage.

        Used by Memory Engine when storing new memories.

        Args:
            dt: Datetime to generate tags for

        Returns:
            Dict with day_of_week, hour_of_day, season, date, etc.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        month = dt.month
        season = (
            "winter" if month in (12, 1, 2) else
            "spring" if month in (3, 4, 5) else
            "summer" if month in (6, 7, 8) else
            "fall"
        )

        return {
            "day_of_week": dt.weekday(),
            "hour_of_day": dt.hour,
            "season": season,
            "date": dt.date().isoformat(),
            "month": dt.month,
            "year": dt.year,
            "is_weekend": dt.weekday() >= 5,
            "time_of_day": self._get_time_of_day(dt.hour),
        }

    def extract_milestone_date(self, text: str, now: datetime) -> Optional[str]:
        """
        Extract a specific date from text for milestone storage.

        Examples:
            "My dad passed away on January 15, 2020" → "2020-01-15"
            "I got married last June" → "2023-06" (approximate)

        Args:
            text: Text containing potential date
            now: Current datetime for relative dates

        Returns:
            ISO date string or None
        """
        # Try to find explicit dates
        date_patterns = [
            # YYYY-MM-DD
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
            # Month DD, YYYY
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
             lambda m: self._month_name_to_date(m.group(1), m.group(2), m.group(3))),
            # DD Month YYYY
            (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
             lambda m: self._month_name_to_date(m.group(2), m.group(1), m.group(3))),
        ]

        for pattern, extractor in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return extractor(match)
                except Exception:
                    continue

        # Try relative date extraction
        result = self.parse(text, now)
        if result.start_ts and result.direction <= 0:
            # Extract just the date part
            try:
                dt = datetime.fromisoformat(result.start_ts.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except Exception:
                pass

        return None

    def get_retrieval_window(self, text: str, now: datetime) -> Optional[int]:
        """
        Get retrieval time window in days for a query.

        Used by Retrieval Engine to filter memories by time.

        Args:
            text: Query text
            now: Current datetime

        Returns:
            Number of days to look back, or None for no time filter
        """
        result = self.parse(text, now)
        return result.retrieval_window_days

    def should_disable_recency(self, text: str) -> bool:
        """
        Check if query explicitly references past (disable recency penalty).

        Used by Retrieval Engine ranker.

        Args:
            text: Query text

        Returns:
            True if recency penalty should be disabled
        """
        result = self.parse(text)
        return result.disable_recency

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _try_fast_path(self, text: str, now: datetime) -> Optional[TemporalResult]:
        """Try fast path cache for common phrases."""
        text_lower = text.lower().strip()

        # Check exact matches
        if text_lower in FAST_PATH_CACHE:
            cached = FAST_PATH_CACHE[text_lower]
            return self._build_result_from_cache(cached, now)

        # Check if text contains a cached phrase
        for phrase, cached in FAST_PATH_CACHE.items():
            if phrase in text_lower:
                return self._build_result_from_cache(cached, now)

        return None

    def _build_result_from_cache(self, cached: Dict, now: datetime) -> TemporalResult:
        """Build TemporalResult from cached fast path data."""
        granularity = TemporalGranularity(cached["granularity"])
        direction = cached["direction"]
        days = cached["days"]
        disable_recency = cached.get("disable_recency", False)

        # Compute timestamps
        if direction == 0:
            start_ts = now.isoformat()
            end_ts = now.isoformat()
        else:
            delta = timedelta(days=days * (1 if direction > 0 else -1))
            target = now + delta
            start_ts = target.isoformat()
            end_ts = target.isoformat()

        return TemporalResult(
            start_ts=start_ts,
            end_ts=end_ts,
            granularity=granularity,
            confidence=1.0,
            direction=direction,
            temporal_tags=self.generate_temporal_tags(now),
            retrieval_window_days=days if direction <= 0 else None,
            disable_recency=disable_recency or direction < 0,
            retrieval_strategy_hint="timeline" if days > 0 else None,
        )

    def _parse_semantic(self, text: str, now: datetime) -> TemporalResult:
        """Parse using semantic parser."""
        try:
            result = self._semantic_parser.parse(text, now)

            granularity = TemporalGranularity.NONE
            if result.get("granularity"):
                try:
                    granularity = TemporalGranularity(result["granularity"])
                except ValueError:
                    granularity = TemporalGranularity.DAY

            direction = result.get("direction", 0)
            value = result.get("value", 1) or 1

            # Compute retrieval window
            retrieval_days = self._granularity_to_days(granularity, value)

            return TemporalResult(
                start_ts=result.get("start_ts"),
                end_ts=result.get("end_ts"),
                granularity=granularity,
                confidence=result.get("confidence", 0.0),
                direction=direction,
                concept_matched=result.get("concept"),
                requires_clarification=result.get("requires_clarification", False),
                retrieval_window_days=retrieval_days if direction <= 0 else None,
                disable_recency=direction < 0,
            )
        except Exception:
            return TemporalResult()

    def _parse_regex(self, text: str, now: datetime) -> TemporalResult:
        """Fallback regex parsing."""
        text_lower = text.lower()

        # Check for relative time patterns
        patterns = [
            (r'(\d+)\s*(?:seconds?|secs?)\s*ago', "second", -1),
            (r'(\d+)\s*(?:minutes?|mins?)\s*ago', "minute", -1),
            (r'(\d+)\s*(?:hours?|hrs?)\s*ago', "hour", -1),
            (r'(\d+)\s*(?:days?)\s*ago', "day", -1),
            (r'(\d+)\s*(?:weeks?)\s*ago', "week", -1),
            (r'(\d+)\s*(?:months?)\s*ago', "month", -1),
            (r'(\d+)\s*(?:years?)\s*ago', "year", -1),
        ]

        for pattern, unit, direction in patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = int(match.group(1))
                granularity = TemporalGranularity(unit)
                days = self._granularity_to_days(granularity, value)

                delta = timedelta(days=days * direction)
                target = now + delta

                return TemporalResult(
                    start_ts=target.isoformat(),
                    end_ts=target.isoformat(),
                    granularity=granularity,
                    confidence=0.8,
                    direction=direction,
                    retrieval_window_days=days,
                    disable_recency=True,
                )

        return TemporalResult()

    def _enrich_result(
        self,
        result: TemporalResult,
        text: str,
        now: datetime,
        user_context: Optional[UserTemporalContext]
    ) -> TemporalResult:
        """Enrich result with memory and retrieval integration."""

        # Generate temporal tags
        result.temporal_tags = self.generate_temporal_tags(now)

        # Determine memory tier hint
        result.memory_tier_hint = self._determine_tier_hint(text, result)

        # Extract milestone date if applicable
        if result.memory_tier_hint == "milestone":
            result.milestone_date = self.extract_milestone_date(text, now)

        # Determine retrieval strategy hint
        result.retrieval_strategy_hint = self._determine_retrieval_strategy(result)

        # Apply user context if available
        if user_context:
            result = self._apply_user_context(result, user_context)

        return result

    def _determine_tier_hint(self, text: str, result: TemporalResult) -> Optional[str]:
        """Determine which memory tier this temporal reference suggests."""
        text_lower = text.lower()

        # Milestone indicators (specific dates, life events)
        milestone_keywords = [
            "passed away", "died", "born", "married", "divorced",
            "graduated", "started", "quit", "moved", "retired"
        ]
        if any(kw in text_lower for kw in milestone_keywords):
            return "milestone"

        # Fact indicators (timeless truths)
        fact_keywords = ["always", "never", "every day", "usually"]
        if any(kw in text_lower for kw in fact_keywords):
            return "fact"

        # Default to episode for temporal references
        if result.start_ts:
            return "episode"

        return None

    def _determine_retrieval_strategy(self, result: TemporalResult) -> Optional[str]:
        """Determine retrieval strategy based on temporal result."""
        if result.granularity == TemporalGranularity.NONE:
            return None

        # Timeline strategy for explicit time ranges
        if result.retrieval_window_days and result.retrieval_window_days > 1:
            return "timeline"

        # Episodic for recent past
        if result.direction < 0 and result.retrieval_window_days and result.retrieval_window_days <= 7:
            return "episodic"

        return None

    def _apply_user_context(
        self,
        result: TemporalResult,
        user_context: UserTemporalContext
    ) -> TemporalResult:
        """Apply user patterns to result."""
        # Could boost confidence if temporal reference aligns with user's active times
        # For now, just pass through
        return result

    def _granularity_to_days(self, granularity: TemporalGranularity, value: int = 1) -> int:
        """Convert granularity and value to days."""
        mapping = {
            TemporalGranularity.SECOND: max(1, value // 86400),
            TemporalGranularity.MINUTE: max(1, value // 1440),
            TemporalGranularity.HOUR: max(1, value // 24),
            TemporalGranularity.DAY: value,
            TemporalGranularity.WEEK: value * 7,
            TemporalGranularity.MONTH: value * 30,
            TemporalGranularity.YEAR: value * 365,
        }
        return mapping.get(granularity, 1) or 1

    def _get_time_of_day(self, hour: int) -> str:
        """Get time of day category."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _month_name_to_date(self, month_name: str, day: str, year: str) -> str:
        """Convert month name to ISO date."""
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        month_num = months.get(month_name.lower(), 1)
        return f"{year}-{month_num:02d}-{int(day):02d}"

    # =========================================================================
    # LEGACY API (backward compatibility)
    # =========================================================================

    def best_time_phrase(self, created_at_iso: str, now_ts: str, timezone: str = "UTC") -> str:
        """Legacy API - returns ISO timestamp."""
        authority = TimeAuthority(now_ts, timezone)
        try:
            dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        except Exception:
            dt = authority.now
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=authority.tzinfo)
        return dt.astimezone(authority.tzinfo).isoformat()

    def parse_time_window(self, text: str) -> Optional[int]:
        """Legacy API - returns retrieval window days."""
        result = self.parse(text)
        return result.retrieval_window_days

    def rewrite_time_phrases(self, text: str, now_ts: str, timezone: str = "UTC") -> Dict[str, Any]:
        """Legacy API - returns parsing result dict."""
        authority = TimeAuthority(now_ts, timezone)
        result = self.parse(text, authority.now)

        return {
            "start_ts": result.start_ts,
            "end_ts": result.end_ts,
            "granularity": result.granularity.value,
            "confidence": result.confidence,
            "requires_clarification": result.requires_clarification,
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_engine_instance: Optional[TemporalEngineV2] = None


def get_temporal_engine() -> TemporalEngineV2:
    """Get singleton temporal engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TemporalEngineV2()
    return _engine_instance
