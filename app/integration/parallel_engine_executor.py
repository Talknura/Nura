"""
Parallel Engine Executor - Run engines concurrently to minimize latency.

Key optimizations:
1. Run independent engines in parallel (not sequential)
2. Async/non-blocking operations for non-critical writes
3. Early LLM start (don't wait for all engines)
4. Smart priority system

Target: <100ms total engine overhead
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from app.temporal.temporal_engine import TemporalEngine, temporal_tags_from_dt


@dataclass
class EngineContext:
    """Context passed to engines."""
    user_id: int
    text: str
    now: datetime
    session_id: str
    intent: str
    confidence: float
    ambiguity_flags: set[str]


@dataclass
class EngineResults:
    """Results from parallel engine execution."""
    temporal_context: Optional[Dict[str, Any]] = None
    temporal_rewrite: Optional[Dict[str, Any]] = None
    retrieval_results: Optional[Any] = None
    memory_task_id: Optional[str] = None  # Async task ID
    adaptation_task_id: Optional[str] = None  # Async task ID

    # Timing info
    total_time_ms: float = 0
    breakdown: Dict[str, float] = None


class ParallelEngineExecutor:
    """
    Execute engines in parallel to minimize latency.

    Execution strategy:
    - Group 1 (CRITICAL - run first, block on completion):
      * Temporal rewrite (if ambiguous, need to ask user)
      * Retrieval (if PAST_SELF_REFERENCE, need results for LLM)

    - Group 2 (ASYNC - fire and forget):
      * Memory ingestion (5652ms, don't wait!)
      * Adaptation update (4ms, don't need results for LLM)

    Target: <100ms total
    """

    def __init__(
        self,
        memory_engine,
        temporal_engine: TemporalEngine,
        retrieval_engine,
        adaptation_engine,
        async_memory_queue,
        max_workers: int = 4
    ):
        self.memory_engine = memory_engine
        self.temporal_engine = temporal_engine
        self.retrieval_engine = retrieval_engine
        self.adaptation_engine = adaptation_engine
        self.async_queue = async_memory_queue
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Engine")

    def execute(self, ctx: EngineContext) -> EngineResults:
        """
        Execute engines in optimal parallel configuration.

        Returns results in <100ms (target).
        """
        start = time.perf_counter()
        results = EngineResults(breakdown={})

        # === PHASE 1: Critical Sequential Operations (fast, must complete) ===

        # Temporal rewrite (1-2ms, needed for ambiguity check)
        temporal_start = time.perf_counter()
        temporal_rewrite = self.temporal_engine.rewrite_time_phrases(
            ctx.text,
            ctx.now.isoformat(),
            "UTC"
        )
        temporal_context = temporal_tags_from_dt(ctx.now)
        results.temporal_rewrite = temporal_rewrite
        results.temporal_context = temporal_context
        results.breakdown['temporal'] = (time.perf_counter() - temporal_start) * 1000

        # === PHASE 2: Parallel Critical Operations ===

        futures = {}

        # Retrieval (if needed) - CRITICAL, must complete before LLM
        if ctx.intent == "PAST_SELF_REFERENCE":
            def _retrieve():
                from app.retrieval.retrieval_engine import RetrievalState
                retrieval_state = RetrievalState(
                    user_id=ctx.user_id,
                    query=ctx.text,
                    now=ctx.now,
                    current_temporal_tags=temporal_context,
                    temporal_rewrite=temporal_rewrite
                )
                return self.retrieval_engine.retrieve(retrieval_state)

            futures['retrieval'] = self.executor.submit(_retrieve)

        # === PHASE 3: Async Non-Critical Operations ===

        # Memory write - ASYNC (don't wait!)
        # Also save PAST_SELF_REFERENCE queries so we remember what user asked about
        if ctx.intent in ["PERSONAL_STATE", "EXPLICIT_MEMORY_COMMAND", "PAST_SELF_REFERENCE"]:
            memory_task_id = self.async_queue.enqueue_memory_write(
                memory_engine=self.memory_engine,
                user_id=ctx.user_id,
                role="user",
                text=ctx.text,
                session_id=ctx.session_id,
                ts=ctx.now,
                temporal_tags=temporal_context,
                source="parallel_pipeline",
                metadata={"intent": ctx.intent},
                priority=1  # High priority
            )
            results.memory_task_id = memory_task_id
            results.breakdown['memory'] = 0  # Async, no wait time

        # Adaptation update - ASYNC (low priority)
        # Run for personal state AND memory queries (may contain emotional context)
        if ctx.intent in ["PERSONAL_STATE", "PAST_SELF_REFERENCE"]:
            from app.metrics.relationship_metrics import ConversationMetrics
            metrics = ConversationMetrics.from_turn(ctx.text, "")

            adaptation_task_id = self.async_queue.enqueue_adaptation_update(
                adaptation_engine=self.adaptation_engine,
                user_id=ctx.user_id,
                metrics=metrics,
                priority=2  # Normal priority
            )
            results.adaptation_task_id = adaptation_task_id
            results.breakdown['adaptation'] = 0  # Async, no wait time

        # === PHASE 4: Wait for Critical Operations Only ===

        # Wait for retrieval if it was submitted
        if 'retrieval' in futures:
            retrieval_start = time.perf_counter()
            results.retrieval_results = futures['retrieval'].result(timeout=5.0)
            results.breakdown['retrieval'] = (time.perf_counter() - retrieval_start) * 1000

        results.total_time_ms = (time.perf_counter() - start) * 1000
        return results

    def execute_with_policy(self, ctx: EngineContext) -> tuple[EngineResults, set[str]]:
        """
        Execute engines based on intent policy.

        Returns:
            (results, allowed_engines_set)
        """
        # Determine which engines to run based on intent
        required, optional, forbidden = self._policy_for_intent(
            ctx.intent,
            ctx.ambiguity_flags,
            ctx.confidence
        )

        allowed = set(required)
        if ctx.confidence >= 0.4:
            allowed.update(optional)
        allowed.difference_update(forbidden)

        results = self.execute(ctx)
        return results, allowed

    def _policy_for_intent(
        self,
        intent: str,
        ambiguity_flags: set[str],
        confidence: float
    ) -> tuple[set[str], set[str], set[str]]:
        """Determine which engines to run for each intent."""
        required = set()
        optional = set()
        forbidden = set()

        if intent == "GENERAL_KNOWLEDGE":
            forbidden.update({
                "MemoryEngine", "RetrievalEngine", "TemporalEngine",
                "AdaptationEngine", "ProactiveEngine"
            })
        elif intent == "EXPLICIT_MEMORY_COMMAND":
            required.add("MemoryEngine")
            forbidden.update({"RetrievalEngine", "AdaptationEngine", "ProactiveEngine"})
        elif intent == "PAST_SELF_REFERENCE":
            required.add("RetrievalEngine")
            forbidden.update({"MemoryEngine", "AdaptationEngine", "ProactiveEngine"})
        elif intent == "PERSONAL_STATE":
            required.update({"MemoryEngine", "AdaptationEngine"})
            optional.add("RetrievalEngine")
            if "future_event" in ambiguity_flags or "future_evaluative" in ambiguity_flags:
                required.discard("AdaptationEngine")
                forbidden.add("AdaptationEngine")

        return required, optional, forbidden

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=False)


def create_parallel_executor(
    memory_engine,
    embedding_service,
    async_memory_queue
):
    """Create a parallel engine executor with all engines."""
    from app.temporal.temporal_engine import TemporalEngine
    from app.retrieval.retrieval_engine import RetrievalEngine
    from app.adaptation.adaptation_engine import AdaptationEngine

    temporal_engine = TemporalEngine()
    retrieval_engine = RetrievalEngine(memory_engine, temporal_engine)
    adaptation_engine = AdaptationEngine()

    return ParallelEngineExecutor(
        memory_engine=memory_engine,
        temporal_engine=temporal_engine,
        retrieval_engine=retrieval_engine,
        adaptation_engine=adaptation_engine,
        async_memory_queue=async_memory_queue,
        max_workers=4
    )
