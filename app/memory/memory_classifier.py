from __future__ import annotations
from dataclasses import dataclass
from config import thresholds
from app.memory.trigger_manager import get_trigger_manager

@dataclass
class Classification:
    action: str  # 'store' or 'drop'
    memory_type: str  # episodic/semantic/summary
    importance: float

def classify_text(text: str) -> Classification:
    """
    Classify text using data-driven TriggerManager.

    Phase 6.4 Refactor: Replaced hardcoded keyword logic with TriggerManager.
    Now uses CSV-based trigger word matching for more flexible classification.
    """
    # Get TriggerManager singleton
    tm = get_trigger_manager()

    # Classify using trigger words
    result = tm.classify(text)
    category = result["category"]

    # Map trigger classification to memory classification
    if category == "noise":
        # Noise: drop (not worth storing)
        return Classification(
            action="drop",
            memory_type="episodic",
            importance=0.0
        )
    elif category == "long_term":
        # Long-term: semantic memory, high importance
        # Includes: identity, relationships, preferences, goals
        return Classification(
            action="store",
            memory_type="semantic",
            importance=thresholds.IMPORTANCE_HIGH
        )
    elif category == "short_term":
        # Short-term: episodic memory, default importance
        # Includes: emotions, activities, temporal events
        return Classification(
            action="store",
            memory_type="episodic",
            importance=thresholds.IMPORTANCE_DEFAULT
        )
    else:
        # Fallback: default to episodic store
        return Classification(
            action="store",
            memory_type="episodic",
            importance=thresholds.IMPORTANCE_DEFAULT
        )
