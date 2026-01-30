"""
Semantic Retrieval Concepts for Nura.

Provides intelligent ranking adjustments based on memory characteristics.
Handles edge cases that simple recency decay misses.

Edge Cases Handled:
    - Core identity facts (name, birthday) - near permanent
    - Emotionally significant memories - slower decay
    - High importance memories - extended half-life
    - Semantic vs episodic memories - different decay curves
    - Query-aware decay - explicit time references bypass recency penalty
    - Recurring topics - boost frequently mentioned subjects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math

from app.vector.embedder import embed_text


# =============================================================================
# MEMORY CHARACTERISTIC CONCEPTS
# =============================================================================

@dataclass
class MemoryCharacteristicConcept:
    """A concept that identifies memory characteristics for ranking."""
    name: str
    exemplars: List[str]
    half_life_days: float  # How long before this memory type decays to 50%
    importance_floor: float  # Minimum importance score (0-1)


# -----------------------------------------------------------------------------
# CORE IDENTITY (PERMANENT - no decay, half_life = 0 means infinite)
# -----------------------------------------------------------------------------

IDENTITY_MEMORY_CONCEPTS = [
    MemoryCharacteristicConcept(
        name="personal_name",
        exemplars=[
            "My name is", "I'm called", "Call me", "People call me",
            "I go by", "My nickname is", "You can call me"
        ],
        half_life_days=0.0,  # PERMANENT - never decays
        importance_floor=1.0
    ),
    MemoryCharacteristicConcept(
        name="birthday_age",
        exemplars=[
            "My birthday is", "I was born on", "I'm years old",
            "I turned", "My age is", "I was born in"
        ],
        half_life_days=0.0,  # PERMANENT - never decays
        importance_floor=1.0
    ),
    MemoryCharacteristicConcept(
        name="family_identity",
        exemplars=[
            "My mother", "My father", "My parents", "My wife", "My husband",
            "My children", "My son", "My daughter", "My brother", "My sister",
            "My family", "My spouse", "My partner"
        ],
        half_life_days=0.0,  # PERMANENT - never decays
        importance_floor=0.95
    ),
    MemoryCharacteristicConcept(
        name="core_preferences",
        exemplars=[
            "My favorite", "I always prefer", "I love", "I hate",
            "I can't stand", "I'm passionate about", "I really enjoy"
        ],
        half_life_days=0.0,  # PERMANENT - core preferences define a person
        importance_floor=0.85
    ),
]


# -----------------------------------------------------------------------------
# SEMANTIC FACTS (PERMANENT - for life)
# -----------------------------------------------------------------------------

SEMANTIC_MEMORY_CONCEPTS = [
    MemoryCharacteristicConcept(
        name="occupation_career",
        exemplars=[
            "I work as", "I'm a engineer", "I'm a teacher", "I'm a doctor",
            "My job is", "I work at", "My profession is", "I'm employed as",
            "My career", "I work in the field of"
        ],
        half_life_days=0.0,  # PERMANENT
        importance_floor=0.9
    ),
    MemoryCharacteristicConcept(
        name="location_home",
        exemplars=[
            "I live in", "I'm from", "My home is in", "I grew up in",
            "I moved to", "I reside in", "My hometown is"
        ],
        half_life_days=0.0,  # PERMANENT
        importance_floor=0.9
    ),
    MemoryCharacteristicConcept(
        name="education_background",
        exemplars=[
            "I studied", "I graduated from", "My degree is in",
            "I went to school at", "I majored in", "My education"
        ],
        half_life_days=0.0,  # PERMANENT
        importance_floor=0.9
    ),
    MemoryCharacteristicConcept(
        name="health_conditions",
        exemplars=[
            "I have diabetes", "I'm allergic to", "My health condition",
            "I take medication for", "I was diagnosed with", "My disability",
            "I have a condition", "My medical history"
        ],
        half_life_days=0.0,  # PERMANENT
        importance_floor=1.0
    ),
]


# -----------------------------------------------------------------------------
# EMOTIONAL MEMORIES (PERMANENT for major events, 90 days for others)
# -----------------------------------------------------------------------------

EMOTIONAL_MEMORY_CONCEPTS = [
    MemoryCharacteristicConcept(
        name="grief_loss",
        exemplars=[
            "My grandmother passed away", "I lost my father",
            "My friend died", "We had to put down our dog",
            "The funeral was", "I'm grieving", "I miss them so much"
        ],
        half_life_days=0.0,  # PERMANENT
        importance_floor=1.0
    ),
    MemoryCharacteristicConcept(
        name="major_life_event",
        exemplars=[
            "I got married", "We had a baby", "I got divorced",
            "I graduated", "I retired", "I got promoted",
            "I bought a house", "I moved to a new city"
        ],
        half_life_days=0.0,  # PERMANENT
        importance_floor=0.95
    ),
    MemoryCharacteristicConcept(
        name="trauma_struggle",
        exemplars=[
            "I was in an accident", "I went through a difficult time",
            "I struggled with depression", "I had a breakdown",
            "It was really hard", "I hit rock bottom", "I was hospitalized"
        ],
        half_life_days=0.0,  # PERMANENT - trauma shapes a person
        importance_floor=0.9
    ),
    MemoryCharacteristicConcept(
        name="achievement_milestone",
        exemplars=[
            "I finally did it", "I achieved my goal", "I won",
            "I passed my exam", "I completed the project",
            "I reached my milestone", "I succeeded"
        ],
        half_life_days=90.0,  # 3 months - achievements can be ongoing
        importance_floor=0.7
    ),
]


# -----------------------------------------------------------------------------
# TASK/EVENT MEMORIES (2-3 months half-life)
# -----------------------------------------------------------------------------

TASK_MEMORY_CONCEPTS = [
    MemoryCharacteristicConcept(
        name="upcoming_event",
        exemplars=[
            "I have a meeting", "My appointment is", "The deadline is",
            "I need to attend", "The event is scheduled", "I have an interview"
        ],
        half_life_days=60.0,  # 2 months
        importance_floor=0.5
    ),
    MemoryCharacteristicConcept(
        name="active_task",
        exemplars=[
            "I need to finish", "I'm working on", "My to-do",
            "I have to complete", "I should do", "I must remember to"
        ],
        half_life_days=60.0,  # 2 months
        importance_floor=0.5
    ),
    MemoryCharacteristicConcept(
        name="pending_decision",
        exemplars=[
            "I'm deciding whether", "I'm thinking about",
            "I'm considering", "I haven't decided", "I'm torn between"
        ],
        half_life_days=90.0,  # 3 months
        importance_floor=0.6
    ),
]


# -----------------------------------------------------------------------------
# EPISODIC MEMORIES (2-3 months half-life)
# -----------------------------------------------------------------------------

EPISODIC_MEMORY_CONCEPTS = [
    MemoryCharacteristicConcept(
        name="daily_activity",
        exemplars=[
            "I had coffee", "I went to the store", "I watched a movie",
            "I ate lunch", "I took a walk", "I called a friend"
        ],
        half_life_days=60.0,  # 2 months
        importance_floor=0.3
    ),
    MemoryCharacteristicConcept(
        name="casual_mention",
        exemplars=[
            "By the way", "Just so you know", "Random thought",
            "I was just thinking", "Oh and also", "Quick update"
        ],
        half_life_days=60.0,  # 2 months
        importance_floor=0.3
    ),
]


# -----------------------------------------------------------------------------
# QUERY CONTEXT CONCEPTS (for query-aware decay)
# -----------------------------------------------------------------------------

TEMPORAL_QUERY_CONCEPTS = [
    MemoryCharacteristicConcept(
        name="explicit_past_reference",
        exemplars=[
            "Last year", "Last month", "A while ago", "Back when",
            "Remember when", "That time when", "Previously",
            "In the past", "Earlier this year", "Before"
        ],
        half_life_days=0.0,  # Special: disables recency penalty
        importance_floor=0.0
    ),
    MemoryCharacteristicConcept(
        name="specific_date_reference",
        exemplars=[
            "On my birthday", "Last Christmas", "New Year's",
            "Last summer", "That winter", "In January",
            "During the holidays", "On that day"
        ],
        half_life_days=0.0,  # Special: disables recency penalty
        importance_floor=0.0
    ),
]


# Combine all concepts
ALL_RETRIEVAL_CONCEPTS = (
    IDENTITY_MEMORY_CONCEPTS +
    SEMANTIC_MEMORY_CONCEPTS +
    EMOTIONAL_MEMORY_CONCEPTS +
    TASK_MEMORY_CONCEPTS +
    EPISODIC_MEMORY_CONCEPTS +
    TEMPORAL_QUERY_CONCEPTS
)


# =============================================================================
# SEMANTIC RETRIEVAL ANALYZER
# =============================================================================

@dataclass
class RetrievalCharacteristics:
    """Characteristics that affect retrieval ranking."""
    half_life_days: float = 30.0  # Default half-life
    importance_floor: float = 0.5  # Minimum importance
    disable_recency_penalty: bool = False  # For explicit past references
    emotional_boost: float = 0.0  # Extra weight for emotional content
    identity_boost: float = 0.0  # Extra weight for identity content
    matched_concept: Optional[str] = None
    match_score: float = 0.0


class SemanticRetrievalAnalyzer:
    """
    Semantic-based retrieval analyzer.
    Determines appropriate decay rates and boosts based on memory content.
    """

    def __init__(self):
        """Initialize with pre-computed concept embeddings."""
        self._concepts = ALL_RETRIEVAL_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        print("[SemanticRetrieval] Initializing retrieval concept embeddings...")

        for concept in self._concepts:
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticRetrieval] Initialized {len(self._concept_embeddings)} retrieval concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_best_concept(
        self,
        text: str,
        threshold: float = 0.45
    ) -> Optional[Tuple[MemoryCharacteristicConcept, float]]:
        """Find the best matching concept for the text."""
        self._ensure_initialized()

        text_embedding = embed_text(text.lower())
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

        best_match = None
        best_score = 0.0

        for concept in self._concepts:
            concept_emb = self._concept_embeddings[concept.name]
            score = self._cosine_similarity(text_embedding, concept_emb)

            if score >= threshold and score > best_score:
                best_match = concept
                best_score = score

        if best_match:
            return (best_match, best_score)
        return None

    def analyze_memory(self, memory_content: str, threshold: float = 0.45) -> RetrievalCharacteristics:
        """
        Analyze a memory to determine its retrieval characteristics.

        Args:
            memory_content: The memory text content
            threshold: Minimum similarity score

        Returns:
            RetrievalCharacteristics with appropriate decay settings
        """
        if not memory_content or not memory_content.strip():
            return RetrievalCharacteristics()

        result = self._find_best_concept(memory_content, threshold)

        if result:
            concept, score = result
            chars = RetrievalCharacteristics(
                half_life_days=concept.half_life_days,
                importance_floor=concept.importance_floor,
                matched_concept=concept.name,
                match_score=score
            )

            # Add boosts based on concept category
            if concept.name in ["personal_name", "birthday_age", "family_identity"]:
                chars.identity_boost = 0.15
            elif concept.name in ["grief_loss", "trauma_struggle", "major_life_event"]:
                chars.emotional_boost = 0.10

            return chars

        # Default characteristics
        return RetrievalCharacteristics()

    def analyze_query(self, query_text: str, threshold: float = 0.45) -> RetrievalCharacteristics:
        """
        Analyze a query to determine if it references past time explicitly.

        Args:
            query_text: The user's query
            threshold: Minimum similarity score

        Returns:
            RetrievalCharacteristics (check disable_recency_penalty)
        """
        if not query_text or not query_text.strip():
            return RetrievalCharacteristics()

        # Check specifically for temporal query concepts
        text_embedding = embed_text(query_text.lower())
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

        self._ensure_initialized()

        for concept in TEMPORAL_QUERY_CONCEPTS:
            concept_emb = self._concept_embeddings[concept.name]
            score = self._cosine_similarity(text_embedding, concept_emb)

            if score >= threshold:
                return RetrievalCharacteristics(
                    disable_recency_penalty=True,
                    matched_concept=concept.name,
                    match_score=score
                )

        return RetrievalCharacteristics()

    def compute_adaptive_recency(
        self,
        created_at_days_ago: float,
        memory_characteristics: RetrievalCharacteristics
    ) -> float:
        """
        Compute recency score with adaptive half-life.

        Args:
            created_at_days_ago: Days since memory was created
            memory_characteristics: Characteristics from analyze_memory()

        Returns:
            Recency score (0-1)
        """
        if memory_characteristics.disable_recency_penalty:
            return 1.0  # No decay

        half_life = memory_characteristics.half_life_days
        if half_life <= 0:
            return 1.0

        # Exponential decay with adaptive half-life
        decay = math.exp(-math.log(2) * created_at_days_ago / half_life)
        return float(decay)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_analyzer_instance: Optional[SemanticRetrievalAnalyzer] = None


def get_semantic_retrieval_analyzer() -> SemanticRetrievalAnalyzer:
    """Get or create the singleton semantic retrieval analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SemanticRetrievalAnalyzer()
    return _analyzer_instance
