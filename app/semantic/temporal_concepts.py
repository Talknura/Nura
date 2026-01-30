"""
Semantic Temporal Concepts for Nura.

Replaces regex-based temporal parsing with embedding-based semantic understanding.
Covers ALL time measures used in human conversation.

Architecture:
    1. Pre-compute embeddings for temporal concepts at init
    2. Embed user input
    3. Find closest matching concept(s) via cosine similarity
    4. Extract numeric values and compute timestamps

Time Units Covered:
    - Seconds, minutes, hours (short-term)
    - Days, weeks, fortnights (medium-term)
    - Months, quarters, seasons (calendar)
    - Years, decades, centuries (long-term)
    - Named days (today, tomorrow, yesterday)
    - Named periods (this week, next month, last year)
    - Colloquial (moment, bit, while, ages, forever)
    - Relative directions (ago, later, from now, in)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.vector.embedder import embed_text


# =============================================================================
# TEMPORAL CONCEPT DEFINITIONS
# =============================================================================

@dataclass
class TemporalConcept:
    """A semantic concept for temporal understanding."""
    name: str
    exemplars: List[str]  # Example phrases that represent this concept
    unit: str  # second, minute, hour, day, week, month, year, etc.
    direction: int  # -1 = past, 0 = present, 1 = future
    multiplier: float  # How many base units (e.g., fortnight = 14 days)
    is_relative: bool  # True if needs a number (e.g., "5 minutes ago")
    is_named: bool  # True if it's a named period (e.g., "tomorrow")
    default_value: Optional[int] = None  # Default numeric value if not specified


# -----------------------------------------------------------------------------
# RELATIVE TIME CONCEPTS (need a number: "in 5 minutes", "3 hours ago")
# -----------------------------------------------------------------------------

RELATIVE_CONCEPTS = [
    # SECONDS
    TemporalConcept(
        name="seconds_ago",
        exemplars=[
            "seconds ago", "second ago", "secs ago", "sec ago",
            "a few seconds ago", "some seconds ago", "couple seconds ago"
        ],
        unit="second", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="seconds_later",
        exemplars=[
            "in seconds", "in a few seconds", "seconds from now", "seconds later",
            "after some seconds", "wait seconds"
        ],
        unit="second", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # MINUTES
    TemporalConcept(
        name="minutes_ago",
        exemplars=[
            "minutes ago", "minute ago", "mins ago", "min ago",
            "a few minutes ago", "some minutes ago", "couple minutes ago"
        ],
        unit="minute", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="minutes_later",
        exemplars=[
            "in minutes", "in a few minutes", "minutes from now", "minutes later",
            "after some minutes", "wait minutes", "give me minutes"
        ],
        unit="minute", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # HOURS
    TemporalConcept(
        name="hours_ago",
        exemplars=[
            "hours ago", "hour ago", "hrs ago", "hr ago",
            "a few hours ago", "some hours ago", "couple hours ago"
        ],
        unit="hour", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="hours_later",
        exemplars=[
            "in hours", "in a few hours", "hours from now", "hours later",
            "after some hours", "in a couple hours"
        ],
        unit="hour", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # DAYS
    TemporalConcept(
        name="days_ago",
        exemplars=[
            "days ago", "day ago", "a few days ago", "some days ago",
            "couple days ago", "several days ago"
        ],
        unit="day", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="days_later",
        exemplars=[
            "in days", "in a few days", "days from now", "days later",
            "after some days", "in a couple days"
        ],
        unit="day", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # WEEKS
    TemporalConcept(
        name="weeks_ago",
        exemplars=[
            "weeks ago", "week ago", "a few weeks ago", "some weeks ago",
            "couple weeks ago", "several weeks ago"
        ],
        unit="week", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="weeks_later",
        exemplars=[
            "in weeks", "in a few weeks", "weeks from now", "weeks later",
            "after some weeks", "in a couple weeks"
        ],
        unit="week", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # FORTNIGHTS (2 weeks)
    TemporalConcept(
        name="fortnights_ago",
        exemplars=["fortnight ago", "fortnights ago", "a fortnight ago"],
        unit="day", direction=-1, multiplier=14.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="fortnights_later",
        exemplars=["in a fortnight", "fortnight from now", "fortnights later"],
        unit="day", direction=1, multiplier=14.0, is_relative=True, is_named=False
    ),

    # MONTHS
    TemporalConcept(
        name="months_ago",
        exemplars=[
            "months ago", "month ago", "a few months ago", "some months ago",
            "couple months ago", "several months ago"
        ],
        unit="month", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="months_later",
        exemplars=[
            "in months", "in a few months", "months from now", "months later",
            "after some months", "in a couple months"
        ],
        unit="month", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # QUARTERS (3 months)
    TemporalConcept(
        name="quarters_ago",
        exemplars=["quarter ago", "quarters ago", "a quarter ago", "last quarter"],
        unit="month", direction=-1, multiplier=3.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="quarters_later",
        exemplars=["in a quarter", "next quarter", "quarters from now"],
        unit="month", direction=1, multiplier=3.0, is_relative=True, is_named=False
    ),

    # YEARS
    TemporalConcept(
        name="years_ago",
        exemplars=[
            "years ago", "year ago", "a few years ago", "some years ago",
            "couple years ago", "several years ago"
        ],
        unit="year", direction=-1, multiplier=1.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="years_later",
        exemplars=[
            "in years", "in a few years", "years from now", "years later",
            "after some years", "in a couple years"
        ],
        unit="year", direction=1, multiplier=1.0, is_relative=True, is_named=False
    ),

    # DECADES
    TemporalConcept(
        name="decades_ago",
        exemplars=["decades ago", "decade ago", "a decade ago"],
        unit="year", direction=-1, multiplier=10.0, is_relative=True, is_named=False
    ),
    TemporalConcept(
        name="decades_later",
        exemplars=["in decades", "in a decade", "decades from now"],
        unit="year", direction=1, multiplier=10.0, is_relative=True, is_named=False
    ),
]


# -----------------------------------------------------------------------------
# NAMED TIME CONCEPTS (don't need a number: "today", "next week")
# -----------------------------------------------------------------------------

NAMED_CONCEPTS = [
    # IMMEDIATE
    TemporalConcept(
        name="now",
        exemplars=[
            "right now", "now", "at this moment", "this instant", "immediately",
            "as we speak", "currently", "presently", "at present"
        ],
        unit="second", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),

    # TODAY/YESTERDAY/TOMORROW
    TemporalConcept(
        name="today",
        exemplars=["today", "this day", "today's", "on this day"],
        unit="day", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="yesterday",
        exemplars=["yesterday", "the day before", "yesterday's", "previous day"],
        unit="day", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),
    TemporalConcept(
        name="tomorrow",
        exemplars=["tomorrow", "the day after", "tomorrow's", "next day"],
        unit="day", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),
    TemporalConcept(
        name="day_after_tomorrow",
        exemplars=["day after tomorrow", "in two days", "overmorrow"],
        unit="day", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=2
    ),
    TemporalConcept(
        name="day_before_yesterday",
        exemplars=["day before yesterday", "two days ago", "ereyesterday"],
        unit="day", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=2
    ),

    # THIS/NEXT/LAST WEEK
    TemporalConcept(
        name="this_week",
        exemplars=["this week", "current week", "the week"],
        unit="week", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="next_week",
        exemplars=["next week", "the coming week", "following week"],
        unit="week", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),
    TemporalConcept(
        name="last_week",
        exemplars=["last week", "previous week", "past week", "the week before"],
        unit="week", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),

    # WEEKEND
    TemporalConcept(
        name="this_weekend",
        exemplars=["this weekend", "the weekend", "coming weekend"],
        unit="weekend", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="next_weekend",
        exemplars=["next weekend", "following weekend"],
        unit="weekend", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),
    TemporalConcept(
        name="last_weekend",
        exemplars=["last weekend", "previous weekend", "past weekend"],
        unit="weekend", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),

    # THIS/NEXT/LAST MONTH
    TemporalConcept(
        name="this_month",
        exemplars=["this month", "current month", "the month"],
        unit="month", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="next_month",
        exemplars=["next month", "the coming month", "following month"],
        unit="month", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),
    TemporalConcept(
        name="last_month",
        exemplars=["last month", "previous month", "past month", "the month before"],
        unit="month", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),

    # THIS/NEXT/LAST YEAR
    TemporalConcept(
        name="this_year",
        exemplars=["this year", "current year", "the year"],
        unit="year", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="next_year",
        exemplars=["next year", "the coming year", "following year"],
        unit="year", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),
    TemporalConcept(
        name="last_year",
        exemplars=["last year", "previous year", "past year", "the year before"],
        unit="year", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=1
    ),

    # SEASONS
    TemporalConcept(
        name="this_spring",
        exemplars=["this spring", "current spring", "the spring"],
        unit="season", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="this_summer",
        exemplars=["this summer", "current summer", "the summer"],
        unit="season", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="this_fall",
        exemplars=["this fall", "this autumn", "current fall", "the fall"],
        unit="season", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
    TemporalConcept(
        name="this_winter",
        exemplars=["this winter", "current winter", "the winter"],
        unit="season", direction=0, multiplier=0.0, is_relative=False, is_named=True
    ),
]


# -----------------------------------------------------------------------------
# COLLOQUIAL/VAGUE TIME CONCEPTS
# -----------------------------------------------------------------------------

COLLOQUIAL_CONCEPTS = [
    # VERY SHORT TERM
    TemporalConcept(
        name="moment",
        exemplars=[
            "a moment", "just a moment", "in a moment", "moment ago",
            "a sec", "just a sec", "one sec", "hang on"
        ],
        unit="second", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=30
    ),
    TemporalConcept(
        name="shortly",
        exemplars=[
            "shortly", "in a bit", "in a little bit", "pretty soon",
            "any minute now", "any second now", "momentarily"
        ],
        unit="minute", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=5
    ),

    # SHORT TERM VAGUE
    TemporalConcept(
        name="soon",
        exemplars=[
            "soon", "before long", "in the near future", "not long from now",
            "in no time", "quickly", "shortly"
        ],
        unit="hour", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=2
    ),
    TemporalConcept(
        name="recently",
        exemplars=[
            "recently", "lately", "not long ago", "just recently",
            "a short while ago", "the other day"
        ],
        unit="day", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=3
    ),

    # MEDIUM TERM VAGUE
    TemporalConcept(
        name="a_while",
        exemplars=[
            "a while", "for a while", "in a while", "a while ago",
            "a while back", "some time ago", "for some time"
        ],
        unit="day", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=7
    ),
    TemporalConcept(
        name="eventually",
        exemplars=[
            "eventually", "at some point", "someday", "one day",
            "sooner or later", "in due time", "when the time comes"
        ],
        unit="day", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=30
    ),

    # LONG TERM VAGUE
    TemporalConcept(
        name="ages_ago",
        exemplars=[
            "ages ago", "a long time ago", "way back", "back in the day",
            "once upon a time", "in the distant past", "long ago"
        ],
        unit="year", direction=-1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=2
    ),
    TemporalConcept(
        name="forever",
        exemplars=[
            "forever", "for ages", "for eternity", "indefinitely",
            "for the foreseeable future", "for a very long time"
        ],
        unit="year", direction=1, multiplier=1.0, is_relative=False, is_named=True,
        default_value=10
    ),
]


# -----------------------------------------------------------------------------
# TIME OF DAY CONCEPTS
# -----------------------------------------------------------------------------

TIME_OF_DAY_CONCEPTS = [
    TemporalConcept(
        name="morning",
        exemplars=[
            "this morning", "in the morning", "tomorrow morning",
            "yesterday morning", "early morning"
        ],
        unit="hour", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=9
    ),
    TemporalConcept(
        name="afternoon",
        exemplars=[
            "this afternoon", "in the afternoon", "tomorrow afternoon",
            "yesterday afternoon"
        ],
        unit="hour", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=14
    ),
    TemporalConcept(
        name="evening",
        exemplars=[
            "this evening", "in the evening", "tomorrow evening",
            "yesterday evening", "tonight"
        ],
        unit="hour", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=19
    ),
    TemporalConcept(
        name="night",
        exemplars=[
            "tonight", "at night", "tomorrow night", "last night",
            "late at night", "during the night"
        ],
        unit="hour", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=22
    ),
    TemporalConcept(
        name="midnight",
        exemplars=["midnight", "at midnight", "around midnight"],
        unit="hour", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=0
    ),
    TemporalConcept(
        name="noon",
        exemplars=["noon", "at noon", "around noon", "midday", "at midday"],
        unit="hour", direction=0, multiplier=1.0, is_relative=False, is_named=True,
        default_value=12
    ),
]


# Combine all concepts
ALL_CONCEPTS = RELATIVE_CONCEPTS + NAMED_CONCEPTS + COLLOQUIAL_CONCEPTS + TIME_OF_DAY_CONCEPTS


# =============================================================================
# SEMANTIC TEMPORAL PARSER
# =============================================================================

class SemanticTemporalParser:
    """
    Semantic-based temporal parser using embeddings.
    Replaces regex-based temporal parsing with ML understanding.
    """

    def __init__(self):
        """Initialize with pre-computed concept embeddings."""
        self._concepts = ALL_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings (expensive operation)."""
        if self._initialized:
            return

        print("[SemanticTemporal] Initializing temporal concept embeddings...")

        for concept in self._concepts:
            # Embed all exemplars and average them for robust matching
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            # Store average embedding for this concept
            avg_embedding = np.mean(embeddings, axis=0)
            # Normalize for cosine similarity
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticTemporal] Initialized {len(self._concept_embeddings)} temporal concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_matching_concepts(
        self,
        text: str,
        threshold: float = 0.5
    ) -> List[Tuple[TemporalConcept, float]]:
        """
        Find temporal concepts that match the input text.

        Args:
            text: Input text to analyze
            threshold: Minimum similarity score to consider a match

        Returns:
            List of (concept, score) tuples, sorted by score descending
        """
        self._ensure_initialized()

        # Embed input text
        text_embedding = embed_text(text.lower())
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

        matches = []
        for concept in self._concepts:
            concept_emb = self._concept_embeddings[concept.name]
            score = self._cosine_similarity(text_embedding, concept_emb)

            if score >= threshold:
                matches.append((concept, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract a numeric value from text.
        Handles: "5", "five", "a few", "couple", etc.
        """
        text_lower = text.lower()

        # Try to find explicit numbers
        number_match = re.search(r'\b(\d+)\b', text)
        if number_match:
            return int(number_match.group(1))

        # Word numbers
        word_numbers = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
            "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
            "a": 1, "an": 1,
        }
        for word, value in word_numbers.items():
            if re.search(rf'\b{word}\b', text_lower):
                return value

        # Vague quantities
        vague_quantities = {
            "few": 3, "couple": 2, "several": 5, "some": 3,
            "many": 10, "lots": 10, "bunch": 5,
        }
        for word, value in vague_quantities.items():
            if re.search(rf'\b{word}\b', text_lower):
                return value

        return None

    def _compute_timedelta(
        self,
        concept: TemporalConcept,
        value: Optional[int]
    ) -> timedelta:
        """
        Compute timedelta from concept and numeric value.

        Args:
            concept: The matched temporal concept
            value: Numeric value (or None to use default)

        Returns:
            timedelta representing the time offset
        """
        # Use provided value, concept default, or 1
        num = value or concept.default_value or 1

        # Apply multiplier and direction
        total_units = num * concept.multiplier * concept.direction

        # Convert to timedelta based on unit
        if concept.unit == "second":
            return timedelta(seconds=total_units)
        elif concept.unit == "minute":
            return timedelta(minutes=total_units)
        elif concept.unit == "hour":
            return timedelta(hours=total_units)
        elif concept.unit == "day":
            return timedelta(days=total_units)
        elif concept.unit == "week":
            return timedelta(weeks=total_units)
        elif concept.unit == "weekend":
            # Weekend handling - find next/last weekend
            return timedelta(weeks=total_units)
        elif concept.unit == "month":
            # Approximate: 30 days per month
            return timedelta(days=total_units * 30)
        elif concept.unit == "year":
            # Approximate: 365 days per year
            return timedelta(days=total_units * 365)
        elif concept.unit == "season":
            # Approximate: 90 days per season
            return timedelta(days=total_units * 90)
        else:
            return timedelta()

    def parse(
        self,
        text: str,
        now: datetime,
        threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Parse temporal references from text using semantic matching.

        Args:
            text: Input text containing temporal references
            now: Current datetime for relative calculations
            threshold: Minimum similarity score for concept matching

        Returns:
            Dict with:
                - start_ts: ISO timestamp of the referenced time
                - end_ts: ISO timestamp of the end of the period
                - granularity: Time unit (second, minute, hour, day, etc.)
                - confidence: How confident we are in the match
                - concept: Name of the matched concept
                - requires_clarification: True if ambiguous
        """
        if not text or not text.strip():
            return {
                "start_ts": None,
                "end_ts": None,
                "granularity": "none",
                "confidence": 0.0,
                "concept": None,
                "requires_clarification": False,
            }

        # Find matching concepts
        matches = self._find_matching_concepts(text, threshold)

        if not matches:
            return {
                "start_ts": None,
                "end_ts": None,
                "granularity": "none",
                "confidence": 0.0,
                "concept": None,
                "requires_clarification": False,
            }

        # Check for ambiguity (multiple strong matches)
        if len(matches) > 1 and matches[0][1] - matches[1][1] < 0.1:
            return {
                "start_ts": None,
                "end_ts": None,
                "granularity": "ambiguous",
                "confidence": matches[0][1],
                "concept": matches[0][0].name,
                "requires_clarification": True,
            }

        # Use the best match
        concept, score = matches[0]

        # Extract numeric value if this is a relative concept
        value = None
        if concept.is_relative:
            value = self._extract_number(text)

        # Compute the timestamp
        delta = self._compute_timedelta(concept, value)
        target_time = now + delta

        # Ensure timezone awareness
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        return {
            "start_ts": target_time.isoformat(),
            "end_ts": target_time.isoformat(),
            "granularity": concept.unit,
            "confidence": score,
            "concept": concept.name,
            "direction": concept.direction,
            "value": value or concept.default_value,
            "requires_clarification": False,
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_parser_instance: Optional[SemanticTemporalParser] = None


def get_semantic_temporal_parser() -> SemanticTemporalParser:
    """Get or create the singleton semantic temporal parser."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = SemanticTemporalParser()
    return _parser_instance
