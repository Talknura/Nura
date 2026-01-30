"""
Semantic Fact Extraction Concepts for Nura.

Replaces keyword-based fact extraction with semantic understanding.
Enables intelligent extraction of personal facts from natural conversation.

Fact Categories:
    - Identity: name, age, birthday
    - Location: where they live, where they're from
    - Occupation: job, profession, work
    - Relationships: family, friends, pets
    - Preferences: favorites, likes, dislikes
    - Goals: aspirations, plans, intentions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import re

from app.vector.embedder import embed_text


# =============================================================================
# FACT CONCEPT DEFINITIONS
# =============================================================================

@dataclass
class FactConcept:
    """A semantic concept for fact extraction."""
    name: str
    fact_key: str  # The key to store this fact under (e.g., "user.name")
    exemplars: List[str]  # Example phrases that reveal this fact
    extraction_hints: List[str]  # Words that typically precede the fact value
    confidence: float = 0.8  # Default confidence for this fact type


# -----------------------------------------------------------------------------
# IDENTITY CONCEPTS
# -----------------------------------------------------------------------------

IDENTITY_CONCEPTS = [
    FactConcept(
        name="preferred_name",
        fact_key="user.preferred_name",
        exemplars=[
            "My name is Sam", "I'm called John", "Call me Alex",
            "People call me Mike", "I go by Sarah", "My friends call me Chris",
            "You can call me David", "Everyone calls me Tom",
            "I'm known as James", "My nickname is Jake"
        ],
        extraction_hints=["is", "called", "me", "by", "as"],
        confidence=0.9
    ),
    FactConcept(
        name="age",
        fact_key="user.age",
        exemplars=[
            "I am 25 years old", "I'm 30", "I just turned 28",
            "I'm in my twenties", "I'm 35 years of age",
            "I turned 40 last month", "I'll be 22 next week",
            "I'm almost 50", "I'm a 33 year old"
        ],
        extraction_hints=["am", "turned", "be", "old"],
        confidence=0.9
    ),
    FactConcept(
        name="birthday",
        fact_key="user.birthday",
        exemplars=[
            "My birthday is March 15", "I was born on July 4th",
            "My birthday is in December", "I was born in 1990",
            "My birthday is next week", "I'm a Pisces, born in February",
            "My birthday falls on Christmas", "I celebrate my birthday in August"
        ],
        extraction_hints=["is", "on", "in", "born"],
        confidence=0.85
    ),
]


# -----------------------------------------------------------------------------
# LOCATION CONCEPTS
# -----------------------------------------------------------------------------

LOCATION_CONCEPTS = [
    FactConcept(
        name="current_location",
        fact_key="user.location",
        exemplars=[
            "I live in New York", "I'm based in London",
            "I live in California", "My home is in Seattle",
            "I reside in Chicago", "I'm currently living in Austin",
            "I moved to Denver", "I stay in Miami"
        ],
        extraction_hints=["in", "to", "at"],
        confidence=0.8
    ),
    FactConcept(
        name="origin",
        fact_key="user.origin",
        exemplars=[
            "I'm from Texas", "I grew up in Ohio",
            "I was raised in Florida", "I come from Canada",
            "I'm originally from Boston", "I was born in India",
            "My hometown is Chicago", "I hail from Australia"
        ],
        extraction_hints=["from", "in"],
        confidence=0.8
    ),
]


# -----------------------------------------------------------------------------
# OCCUPATION CONCEPTS
# -----------------------------------------------------------------------------

OCCUPATION_CONCEPTS = [
    FactConcept(
        name="occupation",
        fact_key="user.occupation",
        exemplars=[
            "I work as a software engineer", "I'm a teacher",
            "I work in marketing", "I'm a doctor",
            "I'm a student", "I work at a bank",
            "I'm an accountant", "I work as a nurse",
            "I'm in sales", "I'm a freelance writer",
            "I'm a manager", "I work in construction"
        ],
        extraction_hints=["as", "a", "an", "in", "at"],
        confidence=0.8
    ),
    FactConcept(
        name="company",
        fact_key="user.company",
        exemplars=[
            "I work at Google", "I work for Microsoft",
            "I'm employed at Amazon", "I work for a startup",
            "My company is called", "I work at a hospital",
            "I'm with IBM", "I work for the government"
        ],
        extraction_hints=["at", "for", "with"],
        confidence=0.75
    ),
]


# -----------------------------------------------------------------------------
# RELATIONSHIP CONCEPTS
# -----------------------------------------------------------------------------

RELATIONSHIP_CONCEPTS = [
    FactConcept(
        name="spouse_name",
        fact_key="user.spouse_name",
        exemplars=[
            "My wife's name is Sarah", "My husband is called John",
            "My partner's name is Alex", "My spouse is named Chris",
            "I'm married to Emma", "My wife Emma",
            "My husband David", "My partner Michael"
        ],
        extraction_hints=["is", "called", "named", "to"],
        confidence=0.85
    ),
    FactConcept(
        name="children",
        fact_key="user.has_children",
        exemplars=[
            "I have two kids", "I have a son", "I have a daughter",
            "My children are", "I'm a parent of three",
            "I have a 5 year old", "My kids are in school",
            "I don't have children", "I'm childless", "No kids yet"
        ],
        extraction_hints=["have", "am", "my"],
        confidence=0.8
    ),
    FactConcept(
        name="pet_dog",
        fact_key="user.dog_name",
        exemplars=[
            "My dog's name is Max", "My dog is called Buddy",
            "I have a dog named Rex", "My puppy's name is Luna",
            "My dog Max", "I got a dog called Charlie",
            "My golden retriever is named Cooper"
        ],
        extraction_hints=["is", "called", "named"],
        confidence=0.9
    ),
    FactConcept(
        name="pet_cat",
        fact_key="user.cat_name",
        exemplars=[
            "My cat's name is Whiskers", "My cat is called Luna",
            "I have a cat named Mittens", "My kitten's name is Shadow",
            "My cat Oliver", "I got a cat called Bella",
            "My tabby is named Tiger"
        ],
        extraction_hints=["is", "called", "named"],
        confidence=0.9
    ),
]


# -----------------------------------------------------------------------------
# PREFERENCE CONCEPTS
# -----------------------------------------------------------------------------

PREFERENCE_CONCEPTS = [
    FactConcept(
        name="favorite_food",
        fact_key="user.favorite_food",
        exemplars=[
            "My favorite food is pizza", "I love sushi",
            "I really like Italian food", "My favorite meal is steak",
            "I can't get enough of tacos", "I'm obsessed with chocolate",
            "Nothing beats a good burger", "I prefer vegetarian food"
        ],
        extraction_hints=["is", "love", "like", "prefer"],
        confidence=0.75
    ),
    FactConcept(
        name="favorite_color",
        fact_key="user.favorite_color",
        exemplars=[
            "My favorite color is blue", "I love the color green",
            "I prefer red", "Blue is my favorite",
            "I really like purple", "I'm drawn to yellow",
            "Black is my go-to color", "I love wearing white"
        ],
        extraction_hints=["is", "love", "prefer", "like"],
        confidence=0.75
    ),
    FactConcept(
        name="hobby",
        fact_key="user.hobbies",
        exemplars=[
            "I love playing guitar", "My hobby is painting",
            "I enjoy reading books", "I like hiking on weekends",
            "I'm into photography", "I spend my free time cooking",
            "I play video games", "I love watching movies",
            "I enjoy gardening", "I'm passionate about music"
        ],
        extraction_hints=["love", "enjoy", "like", "into", "hobby"],
        confidence=0.7
    ),
]


# -----------------------------------------------------------------------------
# GOAL CONCEPTS
# -----------------------------------------------------------------------------

GOAL_CONCEPTS = [
    FactConcept(
        name="career_goal",
        fact_key="user.career_goal",
        exemplars=[
            "I want to become a manager", "My goal is to start a business",
            "I'm working towards a promotion", "I aspire to be a CEO",
            "I want to change careers", "I'm aiming for a leadership role",
            "My dream job is", "I hope to work in"
        ],
        extraction_hints=["to", "is", "be", "become"],
        confidence=0.7
    ),
    FactConcept(
        name="personal_goal",
        fact_key="user.personal_goal",
        exemplars=[
            "I want to lose weight", "My goal is to learn Spanish",
            "I'm trying to exercise more", "I want to travel more",
            "I'm working on my mental health", "I want to read more books",
            "My resolution is to save money", "I'm trying to be more mindful"
        ],
        extraction_hints=["to", "is", "more"],
        confidence=0.7
    ),
]


# Combine all fact concepts
ALL_FACT_CONCEPTS = (
    IDENTITY_CONCEPTS +
    LOCATION_CONCEPTS +
    OCCUPATION_CONCEPTS +
    RELATIONSHIP_CONCEPTS +
    PREFERENCE_CONCEPTS +
    GOAL_CONCEPTS
)


# =============================================================================
# SEMANTIC FACT EXTRACTOR
# =============================================================================

@dataclass
class ExtractedFact:
    """A fact extracted from user text."""
    key: str  # e.g., "user.preferred_name"
    value: str  # e.g., "Sam"
    confidence: float  # 0.0 to 1.0
    source_text: str  # The original text it was extracted from


@dataclass
class FactExtractionResult:
    """Result of fact extraction analysis."""
    facts: List[ExtractedFact] = field(default_factory=list)
    matched_concepts: List[Tuple[str, float]] = field(default_factory=list)


class SemanticFactExtractor:
    """
    Semantic-based fact extractor using embeddings.
    Replaces keyword-based fact extraction with ML understanding.
    """

    def __init__(self):
        """Initialize with pre-computed concept embeddings."""
        self._concepts = ALL_FACT_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        print("[SemanticFact] Initializing fact concept embeddings...")

        for concept in self._concepts:
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticFact] Initialized {len(self._concept_embeddings)} fact concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_matching_concepts(
        self,
        text: str,
        threshold: float = 0.50
    ) -> List[Tuple[FactConcept, float]]:
        """Find fact concepts matching the input text."""
        self._ensure_initialized()

        text_embedding = embed_text(text.lower())
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

        matches = []
        for concept in self._concepts:
            concept_emb = self._concept_embeddings[concept.name]
            score = self._cosine_similarity(text_embedding, concept_emb)

            if score >= threshold:
                matches.append((concept, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _extract_value(self, text: str, concept: FactConcept) -> Optional[str]:
        """
        Extract the actual fact value from text.

        This uses simple heuristics to find the value after trigger phrases.
        """
        text_lower = text.lower()

        # Special handling for different fact types
        if concept.name == "preferred_name":
            # Try patterns: "my name is X", "call me X", "I'm called X"
            patterns = [
                r"my name is\s+(\w+)",
                r"i'?m called\s+(\w+)",
                r"call me\s+(\w+)",
                r"people call me\s+(\w+)",
                r"i go by\s+(\w+)",
                r"you can call me\s+(\w+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Return original case from text
                    start_pos = match.start(1)
                    end_pos = match.end(1)
                    return text[start_pos:end_pos].strip()

        elif concept.name == "age":
            # Try patterns: "I am X years old", "I'm X"
            patterns = [
                r"i(?:'m| am)\s+(\d+)\s*(?:years?\s*old)?",
                r"i(?:'m| am)\s+a\s+(\d+)\s*year",
                r"(?:turned|turn)\s+(\d+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(1)

        elif concept.name in ["current_location", "origin"]:
            # Try patterns: "I live in X", "I'm from X"
            patterns = [
                r"i live in\s+(.+?)(?:\.|,|$)",
                r"i'?m (?:based|located) in\s+(.+?)(?:\.|,|$)",
                r"i'?m from\s+(.+?)(?:\.|,|$)",
                r"i grew up in\s+(.+?)(?:\.|,|$)",
                r"i was (?:raised|born) in\s+(.+?)(?:\.|,|$)",
                r"my (?:home|hometown) is (?:in\s+)?(.+?)(?:\.|,|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1).strip()
                    if len(value) < 50:  # Sanity check
                        return value

        elif concept.name == "occupation":
            # Try patterns: "I work as a X", "I'm a X"
            patterns = [
                r"i work as (?:a |an )?(.+?)(?:\.|,|$)",
                r"i'?m (?:a |an )?(\w+(?:\s+\w+)?)\s*(?:by profession|for work)?",
                r"i work in\s+(.+?)(?:\.|,|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1).strip()
                    if len(value) < 50:
                        return value

        elif concept.name in ["pet_dog", "pet_cat"]:
            # Try patterns: "My dog's name is X", "My dog X"
            animal = "dog" if concept.name == "pet_dog" else "cat"
            patterns = [
                rf"my {animal}(?:'s name)? is (?:called |named )?(\w+)",
                rf"my {animal} (\w+)",
                rf"i have a {animal} (?:called |named )(\w+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(1)

        # Generic extraction: look for value after hint words
        for hint in concept.extraction_hints:
            pattern = rf"\b{hint}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if len(value) < 50:
                    return value

        return None

    def extract(self, text: str, threshold: float = 0.50) -> FactExtractionResult:
        """
        Extract facts from user text.

        Args:
            text: User input text
            threshold: Minimum similarity score for concept matching

        Returns:
            FactExtractionResult with extracted facts
        """
        if not text or not text.strip():
            return FactExtractionResult()

        result = FactExtractionResult()

        # Find matching fact concepts
        matches = self._find_matching_concepts(text, threshold)
        result.matched_concepts = [(c.name, s) for c, s in matches[:5]]

        # Try to extract values for matched concepts
        for concept, score in matches:
            value = self._extract_value(text, concept)
            if value:
                # Adjust confidence based on match score
                confidence = concept.confidence * min(1.0, score / 0.6)
                result.facts.append(ExtractedFact(
                    key=concept.fact_key,
                    value=value,
                    confidence=round(confidence, 2),
                    source_text=text
                ))

        return result


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_extractor_instance: Optional[SemanticFactExtractor] = None


def get_semantic_fact_extractor() -> SemanticFactExtractor:
    """Get or create the singleton semantic fact extractor."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = SemanticFactExtractor()
    return _extractor_instance
