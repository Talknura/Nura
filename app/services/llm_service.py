"""
LLM Service - Qwen3-4B via llama.cpp (No LoRA needed)
Uses the optimized streaming LLM for fast inference.
"""

import json
from typing import Optional, Dict, List, Any

from app.guards.memory_hallucination import guard_memories, is_memory_query


class LLMService:
    """
    Service for generating responses using Qwen3-4B (llama.cpp).
    No LoRA adapter needed - uses base model directly.
    """

    def __init__(self):
        """Initialize LLM service with Qwen3-4B."""
        self._llm = None

    def _get_llm(self):
        """Lazy load LLM to avoid startup delay."""
        if self._llm is None:
            try:
                from app.services.optimized_llm import get_streaming_llm
                self._llm = get_streaming_llm()
            except Exception as e:
                print(f"[LLMService] Failed to load Qwen3-4B: {e}")
                self._llm = None
        return self._llm

    def generate_response(
        self,
        user_message: str,
        mode: str,
        profile: Dict[str, float],
        memories: List[Dict],
        temporal_tags: List[str],
        emotion: str,
        use_gpt4: bool = True,
        llm_input: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate AI response using Qwen3-4B.

        Args:
            user_message: User's input message
            mode: Conversation mode (spiritual, calm, logical, emotional)
            profile: Adaptation profile {warmth, formality, initiative}
            memories: List of relevant memories from retrieval engine
            temporal_tags: Temporal context (morning, night, etc.)
            emotion: Emotion label from upstream (anxious, grateful, etc.)
            use_gpt4: Ignored (kept for compatibility)

        Returns:
            str: Generated response text
        """
        # Check memory guard
        guard_result = guard_memories(memories)
        if guard_result.should_fallback and is_memory_query(user_message):
            return guard_result.fallback_text

        # Get LLM
        llm = self._get_llm()
        if llm is None:
            return self._fallback_response(user_message, profile, emotion)

        # Build prompt using nura_prompt
        try:
            from app.core.nura_prompt import build_nura_prompt

            # Extract memory content
            memory_strings = []
            for mem in memories[:5]:
                content = mem.get("content") or mem.get("text") or ""
                if content:
                    memory_strings.append(content[:150])

            # Get time of day
            time_of_day = None
            for t in ["morning", "afternoon", "evening", "night"]:
                if t in temporal_tags:
                    time_of_day = t
                    break

            prompt = build_nura_prompt(
                user_input=user_message,
                memories=memory_strings if memory_strings else None,
                time_of_day=time_of_day,
                adaptation_profile=profile
            )
        except ImportError:
            # Fallback prompt
            prompt = f"You are Nura, a caring AI companion.\n\nUser: {user_message}\nNura:"

        # Generate response
        try:
            response = ""
            for chunk in llm.stream_generate(prompt):
                if chunk.is_final:
                    response = chunk.text
                    break
            return response.strip()
        except Exception as e:
            print(f"[LLMService] Generation failed: {e}")
            return self._fallback_response(user_message, profile, emotion)

    def _fallback_response(
        self,
        user_message: str,
        profile: Dict[str, float],
        emotion: str
    ) -> str:
        """Rule-based fallback when LLM unavailable."""
        warmth = profile.get("warmth", 0.5)

        if warmth > 0.7:
            return "I'm here for you. What's on your mind?"
        elif warmth < 0.3:
            return "I understand. Tell me more."
        else:
            return "I hear you. What would you like to talk about?"

    def generate_simple_response(
        self,
        user_message: str,
        system_prompt: str,
        use_gpt4: bool = False
    ) -> str:
        """
        Generate response with simple prompt (for warmup/testing).
        """
        llm = self._get_llm()
        if llm is None:
            return "OK"

        prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"

        try:
            response = ""
            for chunk in llm.stream_generate(prompt):
                if chunk.is_final:
                    response = chunk.text
                    break
            return response.strip()
        except Exception:
            return "OK"

    def check_connection(self) -> bool:
        """Check if LLM is available."""
        llm = self._get_llm()
        return llm is not None


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
