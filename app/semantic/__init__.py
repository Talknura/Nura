"""
Semantic understanding module for Nura.
Uses ML embeddings instead of regex for natural language understanding.

Modules:
    - temporal_concepts: Time/date parsing (seconds to years)
    - adaptation_concepts: Emotional states, communication style
    - proactive_concepts: Importance, task status, follow-up triggers
    - intent_concepts: User intent classification
    - fact_concepts: Personal fact extraction
    - retrieval_concepts: Adaptive memory decay and ranking
    - memory_architecture: Three-tier memory classification (FACT/MILESTONE/EPISODE)
"""

from app.semantic.temporal_concepts import SemanticTemporalParser, get_semantic_temporal_parser
from app.semantic.adaptation_concepts import SemanticAdaptationAnalyzer, get_semantic_adaptation_analyzer
from app.semantic.proactive_concepts import SemanticProactiveAnalyzer, get_semantic_proactive_analyzer
from app.semantic.intent_concepts import SemanticIntentAnalyzer, get_semantic_intent_analyzer
from app.semantic.fact_concepts import SemanticFactExtractor, get_semantic_fact_extractor
from app.semantic.retrieval_concepts import SemanticRetrievalAnalyzer, get_semantic_retrieval_analyzer
from app.semantic.memory_architecture import SemanticMemoryClassifier, get_semantic_memory_classifier, MemoryType

__all__ = [
    "SemanticTemporalParser",
    "get_semantic_temporal_parser",
    "SemanticAdaptationAnalyzer",
    "get_semantic_adaptation_analyzer",
    "SemanticProactiveAnalyzer",
    "get_semantic_proactive_analyzer",
    "SemanticIntentAnalyzer",
    "get_semantic_intent_analyzer",
    "SemanticFactExtractor",
    "get_semantic_fact_extractor",
    "SemanticRetrievalAnalyzer",
    "get_semantic_retrieval_analyzer",
    "SemanticMemoryClassifier",
    "get_semantic_memory_classifier",
    "MemoryType",
]
