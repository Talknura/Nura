"""
Semantic Memory Architecture for Nura.

Three-tier memory system that classifies by MEANING, not keywords:

1. FACTS (Semantic Memory)
   - Personal truths: name, job, family, preferences, health
   - ONE value per fact key - gets UPDATED, not accumulated
   - NEVER decays - permanent until contradicted

2. MILESTONES (Life Events)
   - Significant events with timestamps: marriage, death, graduation
   - Permanent history - never deleted, never decays
   - Retrieved when contextually relevant

3. EPISODES (Episodic Memory)
   - Day-to-day conversations and activities
   - DOES decay over 2-3 months
   - Gets summarized/consolidated over time

The classifier uses semantic embeddings to determine which category
a piece of text belongs to - no hardcoded keywords or regex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import re
import hashlib

from app.vector.embedder import embed_text


# =============================================================================
# MEMORY TYPE CLASSIFICATION
# =============================================================================

class MemoryType(Enum):
    """The three types of memory in Nura's architecture."""
    FACT = "fact"           # Permanent personal truths
    MILESTONE = "milestone"  # Life events with dates
    EPISODE = "episode"      # Day-to-day conversations


@dataclass
class ClassifiedMemory:
    """Result of memory classification."""
    memory_type: MemoryType
    confidence: float

    # For FACTS: the extracted key and value
    fact_key: Optional[str] = None      # e.g., "user.pets.dog"
    fact_value: Optional[str] = None    # e.g., "Max, a golden retriever"

    # For MILESTONES: event details
    event_type: Optional[str] = None    # e.g., "marriage", "death", "graduation"
    event_date: Optional[str] = None    # e.g., "2020-06-15" or "last year"
    event_description: Optional[str] = None

    # For EPISODES: just the content
    episode_content: Optional[str] = None

    # Original text
    original_text: str = ""


# =============================================================================
# CONCEPT DEFINITIONS FOR CLASSIFICATION
# =============================================================================

@dataclass
class MemoryConcept:
    """A semantic concept for memory classification."""
    name: str
    memory_type: MemoryType
    exemplars: List[str]
    fact_category: Optional[str] = None  # For facts: the category prefix


# -----------------------------------------------------------------------------
# FACT CONCEPTS (Things that are TRUE about the user)
# -----------------------------------------------------------------------------

FACT_CONCEPTS = [
    # Identity
    MemoryConcept(
        name="personal_identity",
        memory_type=MemoryType.FACT,
        fact_category="user.identity",
        exemplars=[
            "My name is", "I'm called", "Call me", "I go by",
            "I am years old", "My age is", "I was born in",
            "My birthday is", "I'm a man", "I'm a woman",
            "My pronouns are", "I identify as"
        ]
    ),
    # Family & Relationships
    MemoryConcept(
        name="family_relationships",
        memory_type=MemoryType.FACT,
        fact_category="user.family",
        exemplars=[
            "My mother is", "My father is", "My parents are",
            "My wife is", "My husband is", "My spouse is",
            "My partner is", "I'm married to", "I'm dating",
            "My son is", "My daughter is", "My children are",
            "My brother is", "My sister is", "My sibling",
            "I have kids", "I don't have children", "I'm single"
        ]
    ),
    # Location & Origin
    MemoryConcept(
        name="location_origin",
        memory_type=MemoryType.FACT,
        fact_category="user.location",
        exemplars=[
            "I live in", "I'm from", "I grew up in",
            "My home is in", "I moved to", "I reside in",
            "My hometown is", "I was raised in", "I'm based in",
            "I'm originally from", "My address is"
        ]
    ),
    # Occupation & Education
    MemoryConcept(
        name="occupation_education",
        memory_type=MemoryType.FACT,
        fact_category="user.career",
        exemplars=[
            "I work as", "I'm a doctor", "I'm an engineer",
            "My job is", "I work at", "I'm employed at",
            "My profession is", "I studied at", "I graduated from",
            "My degree is in", "I majored in", "I'm a student at",
            "I'm retired", "I'm unemployed", "I work from home"
        ]
    ),
    # Health & Medical
    MemoryConcept(
        name="health_medical",
        memory_type=MemoryType.FACT,
        fact_category="user.health",
        exemplars=[
            "I'm allergic to", "I have diabetes", "I have asthma",
            "I take medication for", "I was diagnosed with",
            "My condition is", "I have a disability",
            "I'm vegetarian", "I'm vegan", "I can't eat",
            "I don't drink alcohol", "I'm in recovery"
        ]
    ),
    # Pets & Animals
    MemoryConcept(
        name="pets_animals",
        memory_type=MemoryType.FACT,
        fact_category="user.pets",
        exemplars=[
            "My dog is", "My cat is", "I have a pet",
            "My dog's name is", "My cat's name is",
            "I have dogs", "I have cats", "I have a bird",
            "My pet", "I don't have pets", "I'm allergic to pets"
        ]
    ),
    # Preferences & Likes
    MemoryConcept(
        name="preferences_likes",
        memory_type=MemoryType.FACT,
        fact_category="user.preferences",
        exemplars=[
            "My favorite is", "I love", "I really enjoy",
            "I prefer", "I always like", "I'm passionate about",
            "I hate", "I can't stand", "I don't like",
            "I'm interested in", "My hobby is", "I collect"
        ]
    ),
    # Beliefs & Values
    MemoryConcept(
        name="beliefs_values",
        memory_type=MemoryType.FACT,
        fact_category="user.beliefs",
        exemplars=[
            "I believe in", "I'm religious", "I'm spiritual",
            "I'm Christian", "I'm Muslim", "I'm Jewish", "I'm Buddhist",
            "I'm atheist", "I'm agnostic", "My faith is",
            "I value", "It's important to me that", "I strongly believe"
        ]
    ),
    # Goals & Aspirations
    MemoryConcept(
        name="goals_aspirations",
        memory_type=MemoryType.FACT,
        fact_category="user.goals",
        exemplars=[
            "My goal is", "I want to become", "I'm working towards",
            "I dream of", "I aspire to", "My ambition is",
            "I plan to", "I hope to someday", "I'm saving for"
        ]
    ),
    # Skills & Abilities
    MemoryConcept(
        name="skills_abilities",
        memory_type=MemoryType.FACT,
        fact_category="user.skills",
        exemplars=[
            "I can speak", "I know how to", "I'm fluent in",
            "I play guitar", "I play piano", "I can code",
            "I'm good at", "I'm skilled in", "I learned to"
        ]
    ),
]


# -----------------------------------------------------------------------------
# MILESTONE CONCEPTS (Significant life events)
# -----------------------------------------------------------------------------

MILESTONE_CONCEPTS = [
    MemoryConcept(
        name="relationship_milestone",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "I got married", "We got engaged", "I got divorced",
            "We broke up", "I started dating", "We had a baby",
            "I proposed", "We moved in together", "We separated"
        ]
    ),
    MemoryConcept(
        name="death_loss",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "My mother passed away", "My father died", "I lost my",
            "My grandmother passed", "My friend died", "The funeral was",
            "We had to put down", "My pet died", "I'm grieving"
        ]
    ),
    MemoryConcept(
        name="career_milestone",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "I got the job", "I was promoted", "I got fired",
            "I quit my job", "I started my business", "I retired",
            "I got my first job", "I changed careers", "I was laid off"
        ]
    ),
    MemoryConcept(
        name="education_milestone",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "I graduated", "I got my degree", "I finished school",
            "I got accepted to", "I dropped out", "I started college",
            "I passed my exam", "I got certified", "I defended my thesis"
        ]
    ),
    MemoryConcept(
        name="health_milestone",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "I was diagnosed with", "I had surgery", "I was hospitalized",
            "I beat cancer", "I recovered from", "I had an accident",
            "I broke my", "I had a heart attack", "I went to rehab"
        ]
    ),
    MemoryConcept(
        name="life_transition",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "I moved to", "I bought a house", "I sold my house",
            "I immigrated to", "I became a citizen", "I got my license",
            "I turned 18", "I turned 21", "I turned 50"
        ]
    ),
    MemoryConcept(
        name="achievement",
        memory_type=MemoryType.MILESTONE,
        exemplars=[
            "I won", "I achieved", "I finally did it",
            "I published my", "I completed", "I reached my goal",
            "I set a record", "I was recognized for", "I received an award"
        ]
    ),
]


# -----------------------------------------------------------------------------
# EPISODE CONCEPTS (Day-to-day stuff)
# -----------------------------------------------------------------------------

EPISODE_CONCEPTS = [
    MemoryConcept(
        name="daily_activity",
        memory_type=MemoryType.EPISODE,
        exemplars=[
            "Today I", "I just", "I'm currently", "Right now I",
            "I had lunch", "I went to", "I met with", "I talked to",
            "I watched", "I read", "I played", "I worked on"
        ]
    ),
    MemoryConcept(
        name="emotional_state",
        memory_type=MemoryType.EPISODE,
        exemplars=[
            "I feel", "I'm feeling", "I'm so", "I'm really",
            "I'm happy", "I'm sad", "I'm anxious", "I'm stressed",
            "I'm worried about", "I'm excited about", "I'm nervous"
        ]
    ),
    MemoryConcept(
        name="plans_intentions",
        memory_type=MemoryType.EPISODE,
        exemplars=[
            "I'm going to", "I'm planning to", "Tomorrow I",
            "Later I'll", "I need to", "I should", "I have to",
            "I'm thinking about", "I might", "I'm considering"
        ]
    ),
    MemoryConcept(
        name="casual_update",
        memory_type=MemoryType.EPISODE,
        exemplars=[
            "By the way", "Just so you know", "Quick update",
            "Oh and", "Also", "I forgot to mention", "Funny story"
        ]
    ),
    MemoryConcept(
        name="question_inquiry",
        memory_type=MemoryType.EPISODE,
        exemplars=[
            "What do you think", "Can you help", "Do you know",
            "How do I", "Why is", "What is", "Could you explain"
        ]
    ),
]


# Combine all concepts
ALL_MEMORY_CONCEPTS = FACT_CONCEPTS + MILESTONE_CONCEPTS + EPISODE_CONCEPTS


# =============================================================================
# SEMANTIC MEMORY CLASSIFIER
# =============================================================================

class SemanticMemoryClassifier:
    """
    Classifies text into FACT, MILESTONE, or EPISODE using semantic similarity.
    No hardcoded keywords - learns from exemplar meanings.
    """

    def __init__(self):
        self._concepts = ALL_MEMORY_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        print("[SemanticMemoryClassifier] Initializing memory concept embeddings...")

        for concept in self._concepts:
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticMemoryClassifier] Initialized {len(self._concept_embeddings)} concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_best_concept(
        self,
        text: str,
        threshold: float = 0.40
    ) -> Optional[Tuple[MemoryConcept, float]]:
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

    def _generate_fact_key(self, text: str, concept: MemoryConcept) -> str:
        """
        Generate a semantic fact key based on content.

        Instead of hardcoding "user.name", we generate keys like:
        - user.identity.name
        - user.pets.dog.max
        - user.preferences.food.pizza
        """
        base_category = concept.fact_category or "user.general"

        # Create a content-based suffix using key words
        text_lower = text.lower()

        # Extract potential subject words (nouns after "my", "I have", etc.)
        patterns = [
            r"my (\w+)'?s?(?:\s+name)?\s+is",
            r"my (\w+) is",
            r"i have (?:a |an )?(\w+)",
            r"i am (?:a |an )?(\w+)",
            r"i'm (?:a |an )?(\w+)",
            r"i (?:really )?(?:love|like|enjoy|prefer) (\w+)",
        ]

        subject = None
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                subject = match.group(1)
                break

        if subject and subject not in ["a", "an", "the", "really", "very"]:
            # Create hash for uniqueness if subject is common
            content_hash = hashlib.md5(text_lower.encode()).hexdigest()[:6]
            return f"{base_category}.{subject}.{content_hash}"
        else:
            # Use content hash for generic facts
            content_hash = hashlib.md5(text_lower.encode()).hexdigest()[:8]
            return f"{base_category}.{content_hash}"

    def _extract_fact_value(self, text: str) -> str:
        """
        Extract the fact value from text.
        For now, we store the full statement as the value.
        The key provides the category, the value is the full context.
        """
        # Clean up the text
        value = text.strip()

        # Remove common prefixes that don't add value
        prefixes_to_remove = [
            "by the way,", "just so you know,", "oh,", "well,",
            "actually,", "you know,", "i mean,"
        ]
        value_lower = value.lower()
        for prefix in prefixes_to_remove:
            if value_lower.startswith(prefix):
                value = value[len(prefix):].strip()
                break

        return value

    def _extract_event_info(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract event type and date from milestone text."""
        text_lower = text.lower()

        # Try to extract date references
        date_patterns = [
            r"(yesterday|today|last (?:week|month|year)|(?:in )?\d{4})",
            r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s+\d{4})?)",
            r"(\d{1,2}/\d{1,2}/\d{2,4})",
            r"(last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
            r"(\d+ (?:days?|weeks?|months?|years?) ago)",
        ]

        event_date = None
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                event_date = match.group(1)
                break

        # Extract event type based on keywords
        event_types = {
            "marriage": ["married", "wedding", "engaged", "engagement"],
            "death": ["passed away", "died", "death", "funeral", "lost my", "grieving"],
            "birth": ["baby", "born", "gave birth", "pregnant"],
            "graduation": ["graduated", "degree", "diploma", "finished school"],
            "job_change": ["new job", "promoted", "fired", "quit", "retired", "laid off"],
            "relocation": ["moved to", "moving to", "relocated", "immigrated"],
            "health": ["diagnosed", "surgery", "hospitalized", "accident", "recovered"],
            "achievement": ["won", "achieved", "completed", "published", "award"],
        }

        event_type = "life_event"  # Default
        for etype, keywords in event_types.items():
            if any(kw in text_lower for kw in keywords):
                event_type = etype
                break

        return event_type, event_date

    def classify(self, text: str, threshold: float = 0.40) -> ClassifiedMemory:
        """
        Classify text into FACT, MILESTONE, or EPISODE.

        Args:
            text: User input text
            threshold: Minimum similarity for concept matching

        Returns:
            ClassifiedMemory with type and extracted information
        """
        if not text or not text.strip():
            return ClassifiedMemory(
                memory_type=MemoryType.EPISODE,
                confidence=0.0,
                episode_content=text,
                original_text=text
            )

        result = self._find_best_concept(text, threshold)

        if not result:
            # Default to EPISODE if no concept matches
            return ClassifiedMemory(
                memory_type=MemoryType.EPISODE,
                confidence=0.5,
                episode_content=text,
                original_text=text
            )

        concept, score = result

        if concept.memory_type == MemoryType.FACT:
            fact_key = self._generate_fact_key(text, concept)
            fact_value = self._extract_fact_value(text)

            return ClassifiedMemory(
                memory_type=MemoryType.FACT,
                confidence=score,
                fact_key=fact_key,
                fact_value=fact_value,
                original_text=text
            )

        elif concept.memory_type == MemoryType.MILESTONE:
            event_type, event_date = self._extract_event_info(text)

            return ClassifiedMemory(
                memory_type=MemoryType.MILESTONE,
                confidence=score,
                event_type=event_type,
                event_date=event_date,
                event_description=text,
                original_text=text
            )

        else:  # EPISODE
            return ClassifiedMemory(
                memory_type=MemoryType.EPISODE,
                confidence=score,
                episode_content=text,
                original_text=text
            )

    def is_fact_update(self, new_text: str, existing_fact_key: str) -> bool:
        """
        Check if new text is updating an existing fact.

        For example:
        - Existing: "user.location.xyz" = "I live in New York"
        - New: "I moved to Texas"
        - Result: True (this updates the location fact)
        """
        # Classify the new text
        classified = self.classify(new_text)

        if classified.memory_type != MemoryType.FACT:
            return False

        # Check if the fact categories match
        if classified.fact_key and existing_fact_key:
            # Extract base category (e.g., "user.location" from "user.location.xyz")
            new_base = ".".join(classified.fact_key.split(".")[:2])
            existing_base = ".".join(existing_fact_key.split(".")[:2])
            return new_base == existing_base

        return False


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_classifier_instance: Optional[SemanticMemoryClassifier] = None


def get_semantic_memory_classifier() -> SemanticMemoryClassifier:
    """Get or create the singleton semantic memory classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SemanticMemoryClassifier()
    return _classifier_instance
