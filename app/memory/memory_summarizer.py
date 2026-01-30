def summarize_turns(texts: list[str]) -> str:
    # V1: naive summary. Replace with LLM later.
    joined = " ".join(t.strip() for t in texts if t.strip())
    if len(joined) <= 400:
        return joined
    return joined[:400] + "…"
