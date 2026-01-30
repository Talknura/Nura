"""
Semantic Adaptation Concepts for Nura.

Replaces keyword-based adaptation signals with embedding-based semantic understanding.
Detects emotional states, communication preferences, and engagement levels.

Categories:
    - Emotional Vulnerability (struggling, overwhelmed, need support)
    - Gratitude & Appreciation (thankful, grateful, appreciative)
    - Spiritual/Prayer Context (faith, worship, religious)
    - Communication Style (direct, formal, casual, detailed)
    - Engagement Level (high engagement, low engagement, distracted)
    - Emotional State (positive, negative, neutral, mixed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.vector.embedder import embed_text


# =============================================================================
# ADAPTATION CONCEPT DEFINITIONS
# =============================================================================

@dataclass
class AdaptationConcept:
    """A semantic concept for adaptation understanding."""
    name: str
    exemplars: List[str]  # Example phrases that represent this concept
    category: str  # vulnerability, gratitude, prayer, style, engagement, emotion
    adaptation_effect: Dict[str, float]  # Effect on profile: warmth, formality, initiative


# -----------------------------------------------------------------------------
# VULNERABILITY CONCEPTS (user is struggling, needs support)
# -----------------------------------------------------------------------------

VULNERABILITY_CONCEPTS = [
    AdaptationConcept(
        name="struggling_emotionally",
        exemplars=[
            "I'm struggling", "I'm having a hard time", "I can't cope",
            "I'm falling apart", "I'm breaking down", "I feel overwhelmed",
            "It's too much", "I can't handle this", "I'm at my limit",
            "Everything is falling apart", "I don't know what to do"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.15, "check_in_frequency": 0.10}
    ),
    AdaptationConcept(
        name="feeling_scared",
        exemplars=[
            "I'm scared", "I'm afraid", "I'm terrified", "I feel scared",
            "I'm frightened", "I'm worried", "I'm anxious", "I'm nervous",
            "I'm panicking", "I feel fear", "I'm dreading this"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.12, "check_in_frequency": 0.08}
    ),
    AdaptationConcept(
        name="feeling_sad",
        exemplars=[
            "I'm sad", "I'm depressed", "I feel down", "I'm feeling blue",
            "I'm heartbroken", "I'm devastated", "I'm miserable",
            "I'm crying", "I've been crying", "I feel empty",
            "I'm grieving", "I miss them so much"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.12, "check_in_frequency": 0.08}
    ),
    AdaptationConcept(
        name="feeling_lonely",
        exemplars=[
            "I'm lonely", "I feel alone", "I'm isolated", "No one understands me",
            "I have no one to talk to", "I feel disconnected", "I'm by myself",
            "Nobody cares", "I feel invisible", "I'm all alone in this"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.15, "initiative": 0.10}
    ),
    AdaptationConcept(
        name="feeling_hopeless",
        exemplars=[
            "I feel hopeless", "There's no point", "I've given up",
            "Nothing will change", "I see no way out", "I'm losing hope",
            "What's the point", "It's useless", "I can't go on",
            "I don't see a future", "Everything is pointless"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.18, "check_in_frequency": 0.15}
    ),
    AdaptationConcept(
        name="seeking_help",
        exemplars=[
            "I need help", "Can you help me", "Please help", "I need support",
            "I don't know who to turn to", "I need someone to talk to",
            "I need advice", "I'm reaching out", "I need guidance"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.10, "initiative": 0.08}
    ),
    AdaptationConcept(
        name="stress_overload",
        exemplars=[
            "I'm so stressed", "I'm burnt out", "I'm exhausted",
            "I can't take it anymore", "Work is killing me", "Too much pressure",
            "I'm running on empty", "I need a break", "I'm drained"
        ],
        category="vulnerability",
        adaptation_effect={"warmth": 0.10, "check_in_frequency": 0.05}
    ),
]


# -----------------------------------------------------------------------------
# GRATITUDE CONCEPTS (user expresses appreciation)
# -----------------------------------------------------------------------------

GRATITUDE_CONCEPTS = [
    AdaptationConcept(
        name="direct_thanks",
        exemplars=[
            "Thank you", "Thanks", "Thanks so much", "Thank you so much",
            "I appreciate it", "I appreciate you", "Thanks a lot",
            "Many thanks", "Cheers", "Ta"
        ],
        category="gratitude",
        adaptation_effect={"initiative": 0.05}
    ),
    AdaptationConcept(
        name="deep_gratitude",
        exemplars=[
            "I'm so grateful", "I'm deeply thankful", "I can't thank you enough",
            "You've been amazing", "This means the world to me",
            "I don't know how to thank you", "You've helped me so much",
            "I'm forever grateful", "You're a lifesaver"
        ],
        category="gratitude",
        adaptation_effect={"initiative": 0.08, "warmth": 0.05}
    ),
    AdaptationConcept(
        name="appreciation_for_support",
        exemplars=[
            "Thanks for listening", "Thanks for being there", "Thanks for understanding",
            "I appreciate you taking the time", "Thanks for your support",
            "Thanks for helping me through this", "Thanks for caring"
        ],
        category="gratitude",
        adaptation_effect={"initiative": 0.06, "warmth": 0.03}
    ),
    AdaptationConcept(
        name="positive_feedback",
        exemplars=[
            "That was helpful", "That really helped", "You made my day",
            "That's exactly what I needed", "Perfect advice", "Great suggestion",
            "You're the best", "You always know what to say"
        ],
        category="gratitude",
        adaptation_effect={"initiative": 0.05}
    ),
]


# -----------------------------------------------------------------------------
# SPIRITUAL/PRAYER CONCEPTS
# -----------------------------------------------------------------------------

PRAYER_CONCEPTS = [
    AdaptationConcept(
        name="prayer_request",
        exemplars=[
            "Please pray for me", "Keep me in your prayers", "Pray for us",
            "I need prayers", "Send prayers", "Pray that everything works out",
            "I'm praying about it", "Lift me up in prayer"
        ],
        category="prayer",
        adaptation_effect={"warmth": 0.08}
    ),
    AdaptationConcept(
        name="religious_expression",
        exemplars=[
            "God bless", "Thank God", "Praise the Lord", "Amen",
            "By God's grace", "God willing", "In Jesus name",
            "Hallelujah", "Glory to God", "The Lord is good"
        ],
        category="prayer",
        adaptation_effect={"warmth": 0.05}
    ),
    AdaptationConcept(
        name="faith_discussion",
        exemplars=[
            "I've been reading the Bible", "I went to church", "My faith",
            "I'm a believer", "Scripture says", "In my devotions",
            "I've been praying", "God has shown me", "My spiritual journey"
        ],
        category="prayer",
        adaptation_effect={"warmth": 0.05}
    ),
    AdaptationConcept(
        name="spiritual_struggle",
        exemplars=[
            "I'm questioning my faith", "I feel distant from God",
            "Why would God allow this", "I'm losing my faith",
            "Spiritually lost", "I don't feel His presence"
        ],
        category="prayer",
        adaptation_effect={"warmth": 0.12, "check_in_frequency": 0.05}
    ),
]


# -----------------------------------------------------------------------------
# COMMUNICATION STYLE CONCEPTS
# -----------------------------------------------------------------------------

COMMUNICATION_STYLE_CONCEPTS = [
    AdaptationConcept(
        name="prefers_direct",
        exemplars=[
            "Just tell me straight", "Get to the point", "No need to sugarcoat",
            "Be honest with me", "Don't beat around the bush", "Cut to the chase",
            "I want the truth", "Just say it", "No BS please"
        ],
        category="style",
        adaptation_effect={"formality": -0.08}
    ),
    AdaptationConcept(
        name="prefers_detailed",
        exemplars=[
            "Can you explain more", "I need all the details", "Walk me through it",
            "Give me the full picture", "I want to understand completely",
            "Don't leave anything out", "Tell me everything"
        ],
        category="style",
        adaptation_effect={"formality": 0.03}
    ),
    AdaptationConcept(
        name="prefers_casual",
        exemplars=[
            "No need to be formal", "Just chat with me", "Let's keep it casual",
            "Talk to me like a friend", "Don't be so stiff", "Relax a bit",
            "We're cool", "Just chill", "Keep it real"
        ],
        category="style",
        adaptation_effect={"formality": -0.10}
    ),
    AdaptationConcept(
        name="prefers_formal",
        exemplars=[
            "Please be professional", "Keep it formal", "I prefer proper language",
            "Let's keep this official", "Business-like please",
            "I expect professionalism", "Formal tone please"
        ],
        category="style",
        adaptation_effect={"formality": 0.10}
    ),
    AdaptationConcept(
        name="wants_brevity",
        exemplars=[
            "Keep it short", "Just the key points", "TLDR", "Summarize please",
            "I don't have much time", "Quick answer please", "In brief",
            "Short version", "Just the essentials"
        ],
        category="style",
        adaptation_effect={"formality": 0.02, "initiative": -0.03}
    ),
]


# -----------------------------------------------------------------------------
# ENGAGEMENT LEVEL CONCEPTS
# -----------------------------------------------------------------------------

ENGAGEMENT_CONCEPTS = [
    AdaptationConcept(
        name="high_engagement",
        exemplars=[
            "I love talking to you", "This is so interesting", "Tell me more",
            "I'm really curious about this", "Let's dive deeper",
            "I could talk about this all day", "This is fascinating"
        ],
        category="engagement",
        adaptation_effect={"initiative": 0.08, "warmth": 0.03}
    ),
    AdaptationConcept(
        name="low_engagement",
        exemplars=[
            "Ok", "Sure", "Fine", "Whatever", "I guess", "If you say so",
            "Mhm", "Yeah", "Alright", "K"
        ],
        category="engagement",
        adaptation_effect={"initiative": -0.05}
    ),
    AdaptationConcept(
        name="distracted",
        exemplars=[
            "Sorry what was that", "I wasn't paying attention",
            "Can you repeat that", "I got distracted", "Lost my train of thought",
            "Where were we", "What were you saying"
        ],
        category="engagement",
        adaptation_effect={"initiative": -0.03}
    ),
    AdaptationConcept(
        name="enthusiastic",
        exemplars=[
            "That's amazing", "Wow", "Incredible", "I'm so excited",
            "This is great", "I can't wait", "Awesome", "Fantastic",
            "Love it", "Perfect"
        ],
        category="engagement",
        adaptation_effect={"initiative": 0.05, "warmth": 0.03}
    ),
]


# -----------------------------------------------------------------------------
# EMOTIONAL STATE CONCEPTS
# -----------------------------------------------------------------------------

EMOTIONAL_STATE_CONCEPTS = [
    AdaptationConcept(
        name="positive_mood",
        exemplars=[
            "I'm happy", "I'm feeling good", "Great day", "I'm in a good mood",
            "Things are going well", "I'm content", "Life is good",
            "I'm cheerful", "Feeling blessed", "Everything is great"
        ],
        category="emotion",
        adaptation_effect={"warmth": 0.03}
    ),
    AdaptationConcept(
        name="negative_mood",
        exemplars=[
            "I'm upset", "I'm angry", "I'm frustrated", "Having a bad day",
            "I'm annoyed", "I'm irritated", "I'm pissed", "This sucks",
            "I'm so mad", "I'm furious"
        ],
        category="emotion",
        adaptation_effect={"warmth": 0.08, "formality": 0.03}
    ),
    AdaptationConcept(
        name="neutral_state",
        exemplars=[
            "I'm okay", "I'm fine", "Nothing special", "Same as usual",
            "Just normal", "Average day", "Can't complain", "Doing alright"
        ],
        category="emotion",
        adaptation_effect={}
    ),
    AdaptationConcept(
        name="excited",
        exemplars=[
            "I'm so excited", "Can't wait", "I'm thrilled", "I'm pumped",
            "This is amazing news", "I'm over the moon", "Best day ever",
            "I'm ecstatic", "So happy right now"
        ],
        category="emotion",
        adaptation_effect={"warmth": 0.05, "initiative": 0.03}
    ),
]


# Combine all adaptation concepts
ALL_ADAPTATION_CONCEPTS = (
    VULNERABILITY_CONCEPTS +
    GRATITUDE_CONCEPTS +
    PRAYER_CONCEPTS +
    COMMUNICATION_STYLE_CONCEPTS +
    ENGAGEMENT_CONCEPTS +
    EMOTIONAL_STATE_CONCEPTS
)


# =============================================================================
# SEMANTIC ADAPTATION ANALYZER
# =============================================================================

@dataclass
class AdaptationSignals:
    """Detected adaptation signals from user input."""
    vulnerability: bool = False
    vulnerability_score: float = 0.0
    gratitude: bool = False
    gratitude_score: float = 0.0
    prayer: bool = False
    prayer_score: float = 0.0
    communication_style: Optional[str] = None  # direct, detailed, casual, formal, brief
    engagement_level: Optional[str] = None  # high, low, distracted, enthusiastic
    emotional_state: Optional[str] = None  # positive, negative, neutral, excited
    adaptation_deltas: Dict[str, float] = None

    def __post_init__(self):
        if self.adaptation_deltas is None:
            self.adaptation_deltas = {}


class SemanticAdaptationAnalyzer:
    """
    Semantic-based adaptation analyzer using embeddings.
    Replaces keyword-based breakthrough detection with ML understanding.
    """

    def __init__(self):
        """Initialize with pre-computed concept embeddings."""
        self._concepts = ALL_ADAPTATION_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        print("[SemanticAdaptation] Initializing adaptation concept embeddings...")

        for concept in self._concepts:
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticAdaptation] Initialized {len(self._concept_embeddings)} adaptation concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_matching_concepts(
        self,
        text: str,
        category: Optional[str] = None,
        threshold: float = 0.45
    ) -> List[Tuple[AdaptationConcept, float]]:
        """Find adaptation concepts matching the input text."""
        self._ensure_initialized()

        text_embedding = embed_text(text.lower())
        text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

        matches = []
        for concept in self._concepts:
            if category and concept.category != category:
                continue

            concept_emb = self._concept_embeddings[concept.name]
            score = self._cosine_similarity(text_embedding, concept_emb)

            if score >= threshold:
                matches.append((concept, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def analyze(self, text: str, threshold: float = 0.45) -> AdaptationSignals:
        """
        Analyze user text for adaptation signals.

        Args:
            text: User input text
            threshold: Minimum similarity score for concept matching

        Returns:
            AdaptationSignals with detected signals and adaptation deltas
        """
        if not text or not text.strip():
            return AdaptationSignals()

        signals = AdaptationSignals()
        total_deltas: Dict[str, float] = {}

        # Check vulnerability
        vuln_matches = self._find_matching_concepts(text, "vulnerability", threshold)
        if vuln_matches:
            signals.vulnerability = True
            signals.vulnerability_score = vuln_matches[0][1]
            for concept, score in vuln_matches[:2]:  # Top 2
                for key, delta in concept.adaptation_effect.items():
                    total_deltas[key] = total_deltas.get(key, 0) + delta * score

        # Check gratitude
        grat_matches = self._find_matching_concepts(text, "gratitude", threshold)
        if grat_matches:
            signals.gratitude = True
            signals.gratitude_score = grat_matches[0][1]
            for concept, score in grat_matches[:2]:
                for key, delta in concept.adaptation_effect.items():
                    total_deltas[key] = total_deltas.get(key, 0) + delta * score

        # Check prayer/spiritual
        prayer_matches = self._find_matching_concepts(text, "prayer", threshold)
        if prayer_matches:
            signals.prayer = True
            signals.prayer_score = prayer_matches[0][1]
            for concept, score in prayer_matches[:2]:
                for key, delta in concept.adaptation_effect.items():
                    total_deltas[key] = total_deltas.get(key, 0) + delta * score

        # Check communication style
        style_matches = self._find_matching_concepts(text, "style", threshold)
        if style_matches:
            signals.communication_style = style_matches[0][0].name
            for concept, score in style_matches[:1]:
                for key, delta in concept.adaptation_effect.items():
                    total_deltas[key] = total_deltas.get(key, 0) + delta * score

        # Check engagement level
        engagement_matches = self._find_matching_concepts(text, "engagement", threshold)
        if engagement_matches:
            signals.engagement_level = engagement_matches[0][0].name
            for concept, score in engagement_matches[:1]:
                for key, delta in concept.adaptation_effect.items():
                    total_deltas[key] = total_deltas.get(key, 0) + delta * score

        # Check emotional state
        emotion_matches = self._find_matching_concepts(text, "emotion", threshold)
        if emotion_matches:
            signals.emotional_state = emotion_matches[0][0].name
            for concept, score in emotion_matches[:1]:
                for key, delta in concept.adaptation_effect.items():
                    total_deltas[key] = total_deltas.get(key, 0) + delta * score

        signals.adaptation_deltas = total_deltas
        return signals


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_analyzer_instance: Optional[SemanticAdaptationAnalyzer] = None


def get_semantic_adaptation_analyzer() -> SemanticAdaptationAnalyzer:
    """Get or create the singleton semantic adaptation analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SemanticAdaptationAnalyzer()
    return _analyzer_instance
