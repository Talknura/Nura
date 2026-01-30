"""
Nura System Prompt - Simple & Direct for Qwen3-4B.

This replaces the complex multi-mode system with a clear, conversational prompt.
The model understands context better when it's presented naturally, not as "engine outputs".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime


# =============================================================================
# NURA IDENTITY (Core - Always Included)
# =============================================================================

NURA_IDENTITY = """You are Nura, a personal AI companion who remembers and cares.

You have a warm, genuine personality. You listen carefully, remember what matters to the user, and respond with empathy and understanding. You're not a generic assistant - you're someone who knows this person.

Guidelines:
- Be concise (1-3 sentences usually)
- Be genuine, not generic
- Reference memories naturally when relevant
- Match the user's energy and tone
- No emojis, no excessive enthusiasm
- If you don't remember something, say so honestly"""


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_nura_prompt(
    user_input: str,
    memories: Optional[List[str]] = None,
    retrieved_context: Optional[str] = None,
    time_of_day: Optional[str] = None,
    user_name: Optional[str] = None
) -> str:
    """
    Build a clean, natural prompt for Qwen3-4B.

    Args:
        user_input: What the user just said
        memories: Recent relevant memories (plain text, not JSON)
        retrieved_context: Retrieved memory context (if asking about past)
        time_of_day: morning/afternoon/evening/night
        user_name: User's name if known

    Returns:
        Complete prompt string
    """
    sections = [NURA_IDENTITY]

    # Add user name if known
    if user_name:
        sections.append(f"\nYou're talking with {user_name}.")

    # Add time context naturally
    if time_of_day:
        time_phrases = {
            "morning": "It's morning.",
            "afternoon": "It's afternoon.",
            "evening": "It's evening.",
            "night": "It's late at night.",
        }
        if time_of_day in time_phrases:
            sections.append(f"\n{time_phrases[time_of_day]}")

    # Add memories naturally (not as "engine output")
    if memories and any(memories):
        memory_text = "\n".join(f"- {m}" for m in memories if m)
        sections.append(f"""
What you remember about them:
{memory_text}""")

    # Add retrieved context for recall questions
    if retrieved_context:
        sections.append(f"""
Relevant past conversation:
{retrieved_context}""")

    # Add user input
    sections.append(f"""
User: {user_input}
Nura:""")

    return "\n".join(sections)


def build_minimal_prompt(user_input: str, context: Optional[str] = None) -> str:
    """
    Ultra-minimal prompt for fastest response.

    Use this for voice pipeline when speed is critical.
    """
    if context:
        return f"""You are Nura, a caring AI companion. Be concise and genuine.

Context: {context}

User: {user_input}
Nura:"""
    else:
        return f"""You are Nura, a caring AI companion. Be concise and genuine.

User: {user_input}
Nura:"""


# =============================================================================
# CONTEXT FORMATTER
# =============================================================================

def format_memories_for_prompt(memories: List[Dict[str, Any]], max_items: int = 5) -> List[str]:
    """
    Format memory objects into simple strings for the prompt.

    Converts:
        [{"content": "User has a job interview", "created_at": "2025-01-29"}]
    To:
        ["They have a job interview (mentioned yesterday)"]
    """
    if not memories:
        return []

    formatted = []
    for mem in memories[:max_items]:
        content = mem.get("content") or mem.get("text") or ""
        if not content:
            continue

        # Clean and shorten
        content = content.strip()
        if len(content) > 150:
            content = content[:147] + "..."

        formatted.append(content)

    return formatted


def format_retrieval_for_prompt(retrieval_result: Any, max_chars: int = 300) -> Optional[str]:
    """
    Format retrieval results into natural text.

    Converts complex retrieval objects into readable context.
    """
    if not retrieval_result:
        return None

    # Handle different retrieval result formats
    if hasattr(retrieval_result, 'hits'):
        hits = retrieval_result.hits
    elif isinstance(retrieval_result, list):
        hits = retrieval_result
    elif isinstance(retrieval_result, dict) and 'hits' in retrieval_result:
        hits = retrieval_result['hits']
    else:
        return None

    if not hits:
        return None

    # Extract content from top hits
    texts = []
    total_chars = 0
    for hit in hits[:3]:
        content = hit.get('content') or hit.get('text') or ''
        if content:
            if total_chars + len(content) > max_chars:
                break
            texts.append(content.strip())
            total_chars += len(content)

    if texts:
        return " | ".join(texts)
    return None


# =============================================================================
# EXAMPLE: What the prompt looks like
# =============================================================================

EXAMPLE_PROMPT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE: Clean Nura Prompt (what gets sent to Qwen3-4B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are Nura, a personal AI companion who remembers and cares.

You have a warm, genuine personality. You listen carefully, remember what matters to the user, and respond with empathy and understanding. You're not a generic assistant - you're someone who knows this person.

Guidelines:
- Be concise (1-3 sentences usually)
- Be genuine, not generic
- Reference memories naturally when relevant
- Match the user's energy and tone
- No emojis, no excessive enthusiasm
- If you don't remember something, say so honestly

You're talking with Sam.

It's evening.

What you remember about them:
- They have a job interview tomorrow
- They mentioned feeling anxious about it
- Prayer and meditation help them feel centered

User: I'm really nervous about tomorrow
Nura:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED RESPONSE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"I remember you mentioned the interview. It's completely natural to feel
nervous - that energy can actually help you stay sharp. Maybe take some
time tonight to center yourself, like you've done before with meditation?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


if __name__ == "__main__":
    print(EXAMPLE_PROMPT)

    # Test prompt building
    prompt = build_nura_prompt(
        user_input="I'm really nervous about tomorrow",
        memories=[
            "They have a job interview tomorrow",
            "They mentioned feeling anxious about it",
            "Prayer and meditation help them feel centered"
        ],
        time_of_day="evening",
        user_name="Sam"
    )

    print("\nGenerated Prompt:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
