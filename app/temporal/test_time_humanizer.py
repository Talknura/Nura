import pytest
from datetime import datetime, timezone, timedelta
from app.temporal.time_humanizer import human_time_diff

def test_human_time_diff_recent():
    now = datetime.now(timezone.utc)
    past = now - timedelta(seconds=30)
    result = human_time_diff(past, now)
    assert result == "just now"

def test_human_time_diff_days_ago():
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=3)
    result = human_time_diff(past, now)
    assert result == "3 days ago"
