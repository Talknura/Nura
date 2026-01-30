"""
Semantic Proactive Concepts for Nura.

Replaces keyword-based importance and classification with semantic understanding.
Enables intelligent follow-up decisions based on meaning, not patterns.

Categories:
    - Importance Levels (urgent, high, medium, low, trivial)
    - Memory Types (task, event, personal, learning, general)
    - Task Status (pending, in_progress, completed, blocked, cancelled)
    - Follow-up Triggers (needs closure, needs check-in, time-sensitive)
    - Emotional Context (emotionally significant, routine, sensitive)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.vector.embedder import embed_text


# =============================================================================
# PROACTIVE CONCEPT DEFINITIONS
# =============================================================================

@dataclass
class ProactiveConcept:
    """A semantic concept for proactive understanding."""
    name: str
    exemplars: List[str]  # Example phrases representing this concept
    category: str  # importance, memory_type, status, trigger, emotional
    salience_boost: int  # How much to boost salience score


# -----------------------------------------------------------------------------
# IMPORTANCE LEVEL CONCEPTS
# -----------------------------------------------------------------------------

IMPORTANCE_CONCEPTS = [
    ProactiveConcept(
        name="urgent",
        exemplars=[
            "This is urgent", "I need this ASAP", "Time-sensitive matter",
            "Critical deadline", "Emergency situation", "Can't wait",
            "Drop everything", "Top priority", "Must happen now",
            "Extremely important", "Life or death", "Crisis"
        ],
        category="importance",
        salience_boost=5
    ),
    ProactiveConcept(
        name="high_importance",
        exemplars=[
            "This is really important", "This matters a lot", "High priority",
            "Very significant", "Crucial matter", "Essential task",
            "Big deal", "Serious matter", "Can't mess this up",
            "Important deadline", "Major event", "Key milestone"
        ],
        category="importance",
        salience_boost=3
    ),
    ProactiveConcept(
        name="medium_importance",
        exemplars=[
            "Should get done", "Would be good to finish", "Moderately important",
            "On my list", "Need to handle", "Should take care of",
            "Part of my plans", "On my radar", "In my queue"
        ],
        category="importance",
        salience_boost=1
    ),
    ProactiveConcept(
        name="low_importance",
        exemplars=[
            "Not a big deal", "Whenever I get to it", "Low priority",
            "No rush", "Can wait", "Not urgent", "Minor thing",
            "If I have time", "Optional", "Nice to have"
        ],
        category="importance",
        salience_boost=-1
    ),
    ProactiveConcept(
        name="trivial",
        exemplars=[
            "Doesn't matter", "Who cares", "Whatever", "Not important at all",
            "Totally trivial", "Meaningless", "Irrelevant",
            "Just random", "Nothing significant", "Don't bother"
        ],
        category="importance",
        salience_boost=-3
    ),
]


# -----------------------------------------------------------------------------
# MEMORY TYPE CONCEPTS
# -----------------------------------------------------------------------------

MEMORY_TYPE_CONCEPTS = [
    ProactiveConcept(
        name="task_memory",
        exemplars=[
            "I need to do", "I have to complete", "Task for me",
            "On my to-do list", "I should finish", "Assignment due",
            "Work to be done", "Action item", "Deliverable",
            "Homework", "Project deadline", "Submit by"
        ],
        category="memory_type",
        salience_boost=2
    ),
    ProactiveConcept(
        name="event_memory",
        exemplars=[
            "I have a meeting", "There's an event", "Appointment scheduled",
            "Party happening", "Wedding coming up", "Birthday celebration",
            "Conference next week", "Doctor's appointment", "Interview scheduled",
            "Dinner reservation", "Flight booked", "Concert tickets"
        ],
        category="memory_type",
        salience_boost=2
    ),
    ProactiveConcept(
        name="personal_memory",
        exemplars=[
            "My personal life", "About myself", "My feelings",
            "Something about me", "My experience", "How I feel",
            "My relationship", "My family", "My friends",
            "Personal matter", "Private thing", "Just for me"
        ],
        category="memory_type",
        salience_boost=1
    ),
    ProactiveConcept(
        name="learning_memory",
        exemplars=[
            "I learned that", "Interesting fact", "Did you know",
            "I read that", "According to", "Scientific study",
            "Historical fact", "Trivia", "General knowledge",
            "Educational content", "I discovered", "Research shows"
        ],
        category="memory_type",
        salience_boost=-2
    ),
    ProactiveConcept(
        name="general_knowledge",
        exemplars=[
            "What is", "How does", "Why does", "Can you explain",
            "Definition of", "Tell me about", "I wonder about",
            "Random question", "Just curious", "Quick question"
        ],
        category="memory_type",
        salience_boost=-3
    ),
]


# -----------------------------------------------------------------------------
# TASK STATUS CONCEPTS
# -----------------------------------------------------------------------------

STATUS_CONCEPTS = [
    ProactiveConcept(
        name="pending_status",
        exemplars=[
            "Haven't started yet", "Still need to do", "Waiting to begin",
            "On hold", "Not started", "Queued up", "Planning to do",
            "Will do later", "Upcoming task", "In the backlog"
        ],
        category="status",
        salience_boost=1
    ),
    ProactiveConcept(
        name="in_progress_status",
        exemplars=[
            "Working on it", "In progress", "Currently doing",
            "Halfway through", "Making progress", "Getting there",
            "Still working", "Ongoing", "Active work", "Underway"
        ],
        category="status",
        salience_boost=2
    ),
    ProactiveConcept(
        name="completed_status",
        exemplars=[
            "Done", "Finished", "Completed", "All done",
            "Mission accomplished", "Task complete", "Wrapped up",
            "Checked off", "Submitted", "Delivered", "Finalized"
        ],
        category="status",
        salience_boost=-5
    ),
    ProactiveConcept(
        name="blocked_status",
        exemplars=[
            "I'm stuck", "Can't proceed", "Blocked by something",
            "Waiting for response", "Depends on others", "Need approval",
            "Hit a roadblock", "Can't move forward", "Waiting on"
        ],
        category="status",
        salience_boost=3
    ),
    ProactiveConcept(
        name="cancelled_status",
        exemplars=[
            "Cancelled", "Called off", "Not happening anymore",
            "Abandoned", "Gave up on", "No longer needed",
            "Plans changed", "Scrapped", "Dropped"
        ],
        category="status",
        salience_boost=-5
    ),
]


# -----------------------------------------------------------------------------
# FOLLOW-UP TRIGGER CONCEPTS
# -----------------------------------------------------------------------------

FOLLOWUP_TRIGGER_CONCEPTS = [
    ProactiveConcept(
        name="needs_closure",
        exemplars=[
            "I'll let you know how it goes", "Will update you",
            "I'll tell you what happened", "Update coming",
            "Results pending", "Outcome unknown", "Waiting to hear",
            "Should know soon", "Will find out", "Stay tuned"
        ],
        category="trigger",
        salience_boost=3
    ),
    ProactiveConcept(
        name="needs_checkin",
        exemplars=[
            "I'm worried about", "Anxious about the outcome",
            "Nervous for tomorrow", "Stressed about this",
            "Hope it goes well", "Fingers crossed",
            "Wish me luck", "Pray for me", "Thinking about it a lot"
        ],
        category="trigger",
        salience_boost=2
    ),
    ProactiveConcept(
        name="time_sensitive",
        exemplars=[
            "Deadline is coming", "Due soon", "Happening tomorrow",
            "Just around the corner", "Time is running out",
            "Clock is ticking", "Almost time", "Very soon",
            "In the next few hours", "Today is the day"
        ],
        category="trigger",
        salience_boost=3
    ),
    ProactiveConcept(
        name="recurring_topic",
        exemplars=[
            "Like I mentioned before", "As I said earlier",
            "We talked about this", "Remember when I said",
            "Going back to that topic", "Following up on",
            "Update on what we discussed", "Regarding our conversation"
        ],
        category="trigger",
        salience_boost=1
    ),
    ProactiveConcept(
        name="commitment_made",
        exemplars=[
            "I promised to", "I said I would", "I committed to",
            "My goal is to", "I plan to", "I'm going to",
            "I will definitely", "I intend to", "I pledged to"
        ],
        category="trigger",
        salience_boost=2
    ),
]


# -----------------------------------------------------------------------------
# EMOTIONAL CONTEXT CONCEPTS
# -----------------------------------------------------------------------------

EMOTIONAL_CONTEXT_CONCEPTS = [
    ProactiveConcept(
        name="emotionally_significant",
        exemplars=[
            "This really matters to me", "I care deeply about this",
            "It's personal", "Means a lot to me", "Close to my heart",
            "Emotionally invested", "I feel strongly about",
            "Very sentimental", "Important relationship", "Life changing"
        ],
        category="emotional",
        salience_boost=2
    ),
    ProactiveConcept(
        name="routine_matter",
        exemplars=[
            "Just the usual", "Regular stuff", "Everyday thing",
            "Same old same old", "Nothing special", "Routine task",
            "Standard procedure", "Normal day", "Like always"
        ],
        category="emotional",
        salience_boost=-1
    ),
    ProactiveConcept(
        name="sensitive_topic",
        exemplars=[
            "This is hard to talk about", "Sensitive matter",
            "Please be gentle", "Difficult subject", "Touchy topic",
            "Handle with care", "Delicate situation", "Private matter",
            "Not easy to discuss", "Vulnerable moment"
        ],
        category="emotional",
        salience_boost=2
    ),
    ProactiveConcept(
        name="celebratory",
        exemplars=[
            "Great news", "Exciting update", "Celebration time",
            "Something to celebrate", "Happy announcement",
            "Wonderful news", "Achievement unlocked", "Success story",
            "Victory", "Milestone reached", "Won"
        ],
        category="emotional",
        salience_boost=1
    ),
]


# -----------------------------------------------------------------------------
# DO NOT REMIND CONCEPTS (signals user doesn't want follow-up)
# -----------------------------------------------------------------------------

DO_NOT_REMIND_CONCEPTS = [
    ProactiveConcept(
        name="dont_ask_again",
        exemplars=[
            "Don't remind me", "No need to follow up", "I'll handle it",
            "Stop asking about this", "Drop it", "Forget about it",
            "Not your concern", "I've got this", "Leave it alone",
            "Don't bring it up again", "Let it go"
        ],
        category="suppress",
        salience_boost=-10
    ),
    ProactiveConcept(
        name="topic_closed",
        exemplars=[
            "That's over now", "We're past that", "Ancient history",
            "Water under the bridge", "Moving on", "Chapter closed",
            "Done talking about it", "Resolved", "Behind me now"
        ],
        category="suppress",
        salience_boost=-10
    ),
]


# Combine all proactive concepts
ALL_PROACTIVE_CONCEPTS = (
    IMPORTANCE_CONCEPTS +
    MEMORY_TYPE_CONCEPTS +
    STATUS_CONCEPTS +
    FOLLOWUP_TRIGGER_CONCEPTS +
    EMOTIONAL_CONTEXT_CONCEPTS +
    DO_NOT_REMIND_CONCEPTS
)


# =============================================================================
# SEMANTIC PROACTIVE ANALYZER
# =============================================================================

@dataclass
class ProactiveAnalysis:
    """Analysis result for proactive decision-making."""
    importance_level: Optional[str] = None  # urgent, high, medium, low, trivial
    importance_score: float = 0.0
    memory_type: Optional[str] = None  # task, event, personal, learning, general
    memory_type_score: float = 0.0
    status: Optional[str] = None  # pending, in_progress, completed, blocked, cancelled
    status_score: float = 0.0
    followup_triggers: List[str] = None  # needs_closure, needs_checkin, etc.
    emotional_context: Optional[str] = None
    should_suppress: bool = False  # True if user doesn't want reminders
    total_salience_boost: int = 0

    def __post_init__(self):
        if self.followup_triggers is None:
            self.followup_triggers = []


class SemanticProactiveAnalyzer:
    """
    Semantic-based proactive analyzer using embeddings.
    Replaces keyword-based importance detection with ML understanding.
    """

    def __init__(self):
        """Initialize with pre-computed concept embeddings."""
        self._concepts = ALL_PROACTIVE_CONCEPTS
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        print("[SemanticProactive] Initializing proactive concept embeddings...")

        for concept in self._concepts:
            embeddings = []
            for exemplar in concept.exemplars:
                emb = embed_text(exemplar)
                embeddings.append(emb)

            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            self._concept_embeddings[concept.name] = avg_embedding

        self._initialized = True
        print(f"[SemanticProactive] Initialized {len(self._concept_embeddings)} proactive concepts")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _find_matching_concepts(
        self,
        text: str,
        category: Optional[str] = None,
        threshold: float = 0.45
    ) -> List[Tuple[ProactiveConcept, float]]:
        """Find proactive concepts matching the input text."""
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

    def analyze(self, text: str, threshold: float = 0.45) -> ProactiveAnalysis:
        """
        Analyze text for proactive decision signals.

        Args:
            text: Memory content or user input
            threshold: Minimum similarity score

        Returns:
            ProactiveAnalysis with detected signals
        """
        if not text or not text.strip():
            return ProactiveAnalysis()

        analysis = ProactiveAnalysis()
        total_boost = 0

        # Check importance level
        importance_matches = self._find_matching_concepts(text, "importance", threshold)
        if importance_matches:
            best = importance_matches[0]
            analysis.importance_level = best[0].name
            analysis.importance_score = best[1]
            total_boost += best[0].salience_boost

        # Check memory type
        type_matches = self._find_matching_concepts(text, "memory_type", threshold)
        if type_matches:
            best = type_matches[0]
            analysis.memory_type = best[0].name
            analysis.memory_type_score = best[1]
            total_boost += best[0].salience_boost

        # Check status
        status_matches = self._find_matching_concepts(text, "status", threshold)
        if status_matches:
            best = status_matches[0]
            analysis.status = best[0].name
            analysis.status_score = best[1]
            total_boost += best[0].salience_boost

        # Check follow-up triggers (can have multiple)
        trigger_matches = self._find_matching_concepts(text, "trigger", threshold)
        for concept, score in trigger_matches[:3]:  # Top 3 triggers
            analysis.followup_triggers.append(concept.name)
            total_boost += concept.salience_boost

        # Check emotional context
        emotional_matches = self._find_matching_concepts(text, "emotional", threshold)
        if emotional_matches:
            best = emotional_matches[0]
            analysis.emotional_context = best[0].name
            total_boost += best[0].salience_boost

        # Check suppression signals
        suppress_matches = self._find_matching_concepts(text, "suppress", threshold)
        if suppress_matches:
            analysis.should_suppress = True
            total_boost += suppress_matches[0][0].salience_boost

        analysis.total_salience_boost = total_boost
        return analysis

    def compute_semantic_salience(self, memory_content: str, base_score: int = 0) -> int:
        """
        Compute salience score using semantic analysis.

        Args:
            memory_content: Content of the memory
            base_score: Starting salience score

        Returns:
            Adjusted salience score
        """
        analysis = self.analyze(memory_content)
        return base_score + analysis.total_salience_boost


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_analyzer_instance: Optional[SemanticProactiveAnalyzer] = None


def get_semantic_proactive_analyzer() -> SemanticProactiveAnalyzer:
    """Get or create the singleton semantic proactive analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SemanticProactiveAnalyzer()
    return _analyzer_instance
