from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from app.db.session import get_conn

@dataclass
class TemporalPattern:
    pattern_type: str
    confidence: float
    example_memory_ids: list[int]

def store_pattern(user_id: int, pattern: TemporalPattern) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO temporal_patterns(user_id, pattern_type, confidence, example_memory_ids)
           VALUES (?, ?, ?, ?)""",
        (user_id, pattern.pattern_type, pattern.confidence, json.dumps(pattern.example_memory_ids)),
    )
    conn.commit()

def detect_patterns(user_id: int) -> list[TemporalPattern]:
    conn = get_conn()
    rows = conn.execute("SELECT id, temporal_tags FROM memories WHERE user_id=?", (user_id,))
    day_counts: dict[int, int] = {}
    hour_counts: dict[int, int] = {}
    day_examples: dict[int, list[int]] = {}
    hour_examples: dict[int, list[int]] = {}
    parsed_count = 0

    for row in rows:
        memory_id, temporal_tags = row
        if temporal_tags is None or temporal_tags == "":
            continue
        try:
            parsed = json.loads(temporal_tags)
        except Exception:
            continue
        parsed_count += 1

        day_of_week = parsed.get("day_of_week")
        if day_of_week is not None:
            day_counts[day_of_week] = day_counts.get(day_of_week, 0) + 1
            day_examples.setdefault(day_of_week, []).append(memory_id)

        hour_of_day = parsed.get("hour_of_day")
        if hour_of_day is not None:
            hour_counts[hour_of_day] = hour_counts.get(hour_of_day, 0) + 1
            hour_examples.setdefault(hour_of_day, []).append(memory_id)

    if parsed_count == 0:
        return []

    patterns: list[TemporalPattern] = []
    for day_value, count in day_counts.items():
        if count >= 3:
            confidence = count / parsed_count
            confidence = max(0.0, min(1.0, confidence))
            example_ids = day_examples.get(day_value, [])[:5]
            patterns.append(TemporalPattern(
                pattern_type=f"day_of_week_{day_value}",
                confidence=confidence,
                example_memory_ids=example_ids,
            ))

    for hour_value, count in hour_counts.items():
        if count >= 3:
            confidence = count / parsed_count
            confidence = max(0.0, min(1.0, confidence))
            example_ids = hour_examples.get(hour_value, [])[:5]
            patterns.append(TemporalPattern(
                pattern_type=f"hour_of_day_{hour_value}",
                confidence=confidence,
                example_memory_ids=example_ids,
            ))

    return patterns


def get_patterns(user_id: int) -> list[TemporalPattern]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT pattern_type, confidence, example_memory_ids FROM temporal_patterns WHERE user_id=?",
        (user_id,),
    )
    patterns: list[TemporalPattern] = []
    for row in rows:
        pattern_type, confidence, example_memory_ids = row
        patterns.append(TemporalPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            example_memory_ids=json.loads(example_memory_ids),
        ))
    return patterns
