"""
Adaptation Engine Module.

Integrated engine for user communication style adaptation.
Uses Memory Engine for historical context, Temporal Engine for time-based adaptation,
and Semantic analysis for emotional/style signal detection.
"""

from app.adaptation.adaptation_engine import (
    AdaptationEngineV2 as AdaptationEngine,
    AdaptationProfile,
    AdaptationContext,
    AdaptationResult,
    get_adaptation_engine,
)
from app.adaptation.adaptation_rules import AdaptationDelta, apply_delta
from app.adaptation.breakthrough_detector import (
    detect_breakthrough,
    BreakthroughSignals,
)

__all__ = [
    "AdaptationEngine",
    "AdaptationProfile",
    "AdaptationContext",
    "AdaptationResult",
    "AdaptationDelta",
    "get_adaptation_engine",
    "apply_delta",
    "detect_breakthrough",
    "BreakthroughSignals",
]
