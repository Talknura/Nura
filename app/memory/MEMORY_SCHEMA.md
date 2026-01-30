Memory JSONL schema

Each line in the JSONL file is a single memory record.

Fields:
- id: UUID string
- user_id: integer
- role: "user" | "assistant"
- text: string (plain text only)
- memory_type: "episodic" | "semantic" | "summary"
- importance: float (0.0 - 1.0)
- session_id: string
- ts: ISO 8601 timestamp (UTC)
- source: string (e.g., "chat", "api", "debug")
- metadata: object (freeform, optional)

Example:
{
  "id": "2d3e9d88-0b4a-4d9f-bc3b-22a34c7b23a2",
  "user_id": 1,
  "role": "user",
  "text": "I have a job interview tomorrow.",
  "memory_type": "episodic",
  "importance": 0.5,
  "session_id": "session_1_1736039999",
  "ts": "2026-01-05T21:10:12.123456+00:00",
  "source": "chat",
  "metadata": {"mode": "calm"}
}
