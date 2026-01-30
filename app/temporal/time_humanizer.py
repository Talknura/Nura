from __future__ import annotations
from datetime import datetime, timezone

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def human_time_diff(past: datetime, now: datetime) -> str:
    if past.tzinfo is None:
        past = past.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta = now - past
    seconds = int(delta.total_seconds())
    days = delta.days

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = seconds // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if days == 0:
        h = seconds // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        w = days // 7
        return f"{w} week{'s' if w != 1 else ''} ago"
    if days < 365:
        mo = days // 30
        return f"{mo} month{'s' if mo != 1 else ''} ago"
    y = days // 365
    return f"{y} year{'s' if y != 1 else ''} ago"

def contextual_time(past: datetime) -> str:
    if past.tzinfo is None:
        past = past.replace(tzinfo=timezone.utc)
    hour = past.hour
    if 5 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 17:
        tod = "afternoon"
    elif 17 <= hour < 21:
        tod = "evening"
    else:
        tod = "night"
    day = DAY_NAMES[past.weekday()]
    return f"{day} {tod}"
