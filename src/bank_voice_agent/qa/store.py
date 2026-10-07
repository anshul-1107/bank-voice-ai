"""QA store. SQLite for dev; the same schema runs on Postgres in prod (swap the connection)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY, started_at TEXT, direction TEXT, purpose TEXT, outcome TEXT,
    prompt_version TEXT, llm_model TEXT, duration_s REAL, record_json TEXT
);
CREATE TABLE IF NOT EXISTS qa_results (
    call_id TEXT PRIMARY KEY REFERENCES calls(call_id), scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
    verdict TEXT, score REAL, reasons_json TEXT, metrics_json TEXT, rubric_json TEXT, judge TEXT,
    needs_human_review INTEGER, review_reason TEXT,
    human_verdict TEXT, human_score REAL, human_notes TEXT, reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_qa_review ON qa_results(needs_human_review, human_verdict);
CREATE INDEX IF NOT EXISTS ix_calls_started ON calls(started_at);
"""


class QAStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def save_call(self, rec: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
            (rec["call_id"], rec["started_at"], rec["direction"], rec["purpose"], rec["outcome"],
             rec["prompt_version"], rec["llm_model"], rec["duration_s"], json.dumps(rec, default=str)))
        self.db.commit()

    def save_qa(self, call_id: str, qa: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO qa_results (call_id, verdict, score, reasons_json, metrics_json, rubric_json, "
            "judge, needs_human_review, review_reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (call_id, qa["verdict"], qa["score"], json.dumps(qa["reasons"]), json.dumps(qa["metrics"]),
             json.dumps(qa.get("rubric")), qa.get("judge", ""), int(qa["needs_human_review"]),
             qa.get("review_reason", "")))
        self.db.commit()

    def record_human_review(self, call_id: str, verdict: str, score: float | None, notes: str) -> None:
        self.db.execute("UPDATE qa_results SET human_verdict=?, human_score=?, human_notes=?, "
                        "reviewed_at=CURRENT_TIMESTAMP WHERE call_id=?", (verdict, score, notes, call_id))
        self.db.commit()

    def get_call(self, call_id: str) -> dict | None:
        row = self.db.execute("SELECT record_json FROM calls WHERE call_id=?", (call_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def review_queue(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT q.call_id, q.verdict, q.score, q.review_reason, c.purpose, c.started_at FROM qa_results q "
            "JOIN calls c USING(call_id) WHERE q.needs_human_review=1 AND q.human_verdict IS NULL "
            "ORDER BY CASE q.verdict WHEN 'FAIL' THEN 0 WHEN 'REVIEW' THEN 1 ELSE 2 END, c.started_at LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    def rows_between(self, start_iso: str, end_iso: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT c.record_json, q.verdict, q.score, q.reasons_json, q.metrics_json, q.rubric_json, "
            "q.needs_human_review, q.human_verdict FROM calls c LEFT JOIN qa_results q USING(call_id) "
            "WHERE c.started_at >= ? AND c.started_at < ?", (start_iso, end_iso)).fetchall()
        return [dict(r) for r in rows]

    def recent_calls(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT c.call_id, c.started_at, c.direction, c.purpose, c.outcome, c.duration_s, "
            "q.verdict, q.score, q.needs_human_review, q.human_verdict, q.judge "
            "FROM calls c LEFT JOIN qa_results q USING(call_id) "
            "ORDER BY c.started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
