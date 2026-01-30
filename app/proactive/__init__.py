"""
Proactive Engine Module.

Integrated decision engine for intelligent follow-up questions.
Uses Memory Engine for context, Retrieval Engine for finding relevant memories,
and Temporal Engine for time-based decisions.
"""

from app.proactive.proactive_engine import (
    ProactiveEngineV2 as ProactiveEngine,
    ProactiveResult,
    ProactiveState,
    get_proactive_engine,
    evaluate,
    decide_followup,
    OBLIGATION_MANDATORY,
    OBLIGATION_OPTIONAL,
    OBLIGATION_NONE,
)

__all__ = [
    "ProactiveEngine",
    "ProactiveResult",
    "ProactiveState",
    "get_proactive_engine",
    "evaluate",
    "decide_followup",
    "OBLIGATION_MANDATORY",
    "OBLIGATION_OPTIONAL",
    "OBLIGATION_NONE",
]
