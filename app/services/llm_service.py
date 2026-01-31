"""
LLM Service - Qwen3-4B via llama.cpp (No LoRA needed)
Uses the optimized streaming LLM for fast inference.

Supports two modes:
- Fast mode (default): No thinking, <800ms responses
- Thinking mode: Full reasoning captured for debugging/training
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

from app.guards.memory_hallucination import guard_memories, is_memory_query


# =============================================================================
# THINKING LOGGER - Captures Qwen3 reasoning for debugging/training
# =============================================================================

class ThinkingLogger:
    """
    Logs Qwen3 <think> content for debugging and training data collection.

    Log files are stored in logs/thinking/ with daily rotation.
    Each entry includes: timestamp, user_message, thinking_content, response, metadata
    """

    def __init__(self, log_dir: str = "logs/thinking"):
        self.log_dir = log_dir
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_log_path(self) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"thinking_{date_str}.jsonl")

    def log(
        self,
        user_message: str,
        thinking_content: str,
        response: str,
        metadata: Optional[Dict] = None
    ):
        """Log a thinking trace to the daily log file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "thinking": thinking_content,
            "response": response,
            "metadata": metadata or {}
        }
        try:
            with open(self._get_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ThinkingLogger] Failed to write log: {e}")


# Global thinking logger instance
_thinking_logger: Optional[ThinkingLogger] = None


def get_thinking_logger() -> ThinkingLogger:
    """Get or create thinking logger singleton."""
    global _thinking_logger
    if _thinking_logger is None:
        _thinking_logger = ThinkingLogger()
    return _thinking_logger


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

        # Build prompt using Qwen3 chat format
        try:
            from app.core.nura_prompt import NURA_IDENTITY

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

            # Build context
            context_parts = []
            if memory_strings:
                context_parts.append("What you know about them:\n- " + "\n- ".join(memory_strings))
            if time_of_day:
                context_parts.append(f"It's {time_of_day}.")

            system_content = NURA_IDENTITY
            if context_parts:
                system_content += "\n\n" + "\n".join(context_parts)

            # Qwen3 chat format (no thinking mode - prefill empty think block)
            prompt = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""
        except ImportError:
            # Fallback prompt (with no-think prefill)
            prompt = f"""<|im_start|>system
You are Nura, a caring AI companion. Be concise and friendly.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        # Generate response
        try:
            response = ""
            for chunk in llm.stream_generate(prompt):
                if chunk.is_final:
                    response = chunk.text
                    break
            # Strip Qwen3 thinking tags (if any leaked through)
            response = self._strip_thinking(response)
            return response.strip()
        except Exception as e:
            print(f"[LLMService] Generation failed: {e}")
            return self._fallback_response(user_message, profile, emotion)

    def _strip_thinking(self, text: str) -> str:
        """Strip Qwen3 thinking tags from response."""
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Also handle unclosed think tags (stop mid-thinking)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        return text.strip()

    def _extract_thinking(self, text: str) -> Tuple[str, str]:
        """
        Extract thinking content and response from Qwen3 output.

        Returns:
            Tuple of (thinking_content, response_text)
        """
        thinking = ""
        response = text

        # Extract <think>...</think> content
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            response = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        else:
            # Handle unclosed think tags
            match = re.search(r'<think>(.*)', text, flags=re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                response = ""

        return thinking, response

    def generate_with_thinking(
        self,
        user_message: str,
        mode: str,
        profile: Dict[str, float],
        memories: List[Dict],
        temporal_tags: List[str],
        emotion: str,
        log_thinking: bool = True
    ) -> Tuple[str, str]:
        """
        Generate AI response WITH thinking enabled.

        This is slower than generate_response() but captures the model's
        reasoning process for debugging and training data collection.

        Args:
            user_message: User's input message
            mode: Conversation mode
            profile: Adaptation profile
            memories: Retrieved memories
            temporal_tags: Temporal context
            emotion: Detected emotion
            log_thinking: Whether to log thinking to file

        Returns:
            Tuple of (response_text, thinking_content)
        """
        # Check memory guard
        guard_result = guard_memories(memories)
        if guard_result.should_fallback and is_memory_query(user_message):
            return guard_result.fallback_text, ""

        # Get LLM
        llm = self._get_llm()
        if llm is None:
            return self._fallback_response(user_message, profile, emotion), ""

        # Build prompt WITH thinking enabled (no prefill)
        try:
            from app.core.nura_prompt import NURA_IDENTITY

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

            # Build context
            context_parts = []
            if memory_strings:
                context_parts.append("What you know about them:\n- " + "\n- ".join(memory_strings))
            if time_of_day:
                context_parts.append(f"It's {time_of_day}.")

            system_content = NURA_IDENTITY
            if context_parts:
                system_content += "\n\n" + "\n".join(context_parts)

            # Qwen3 chat format WITH thinking enabled (NO prefill)
            prompt = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        except ImportError:
            prompt = f"""<|im_start|>system
You are Nura, a caring AI companion. Be concise and friendly.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

        # Generate response with thinking (use more tokens for thinking overhead)
        try:
            raw_response = ""
            for chunk in llm.stream_generate(prompt, max_tokens=400):
                if chunk.is_final:
                    raw_response = chunk.text
                    break

            # Extract thinking and response
            thinking, response = self._extract_thinking(raw_response)

            # Log if enabled
            if log_thinking and thinking:
                logger = get_thinking_logger()
                logger.log(
                    user_message=user_message,
                    thinking_content=thinking,
                    response=response,
                    metadata={
                        "mode": mode,
                        "emotion": emotion,
                        "memory_count": len(memories),
                        "temporal_tags": temporal_tags
                    }
                )

            return response.strip(), thinking

        except Exception as e:
            print(f"[LLMService] Generation with thinking failed: {e}")
            return self._fallback_response(user_message, profile, emotion), ""

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
