import pytest
from app.memory.memory_classifier import classify_text

def test_classify_text_store():
    result = classify_text("I prefer Python over JavaScript")
    assert result.action == "store"
    assert result.memory_type == "semantic"
    assert result.importance > 0.5

def test_classify_text_drop():
    result = classify_text("ok")
    assert result.action == "drop"
    assert result.importance == 0.0
