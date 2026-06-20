"""
memory/store.py

MemoryStore — persists past startup idea runs to a local SQLite database.

Used by the orchestrator to optionally inject prior run summaries as
context, letting the system learn across sessions (e.g. "last time you
analysed a retinopathy detection idea, the market score was 14/20 —
this time consider addressing the FDA pathway more explicitly").

Schema (single table: runs):
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp   TEXT    ISO-8601 UTC
    idea        TEXT    original startup idea
    domain      TEXT    target domain
    score       INTEGER viability score (nullable)
    verdict     TEXT    strong | promising | needs-work | not-viable
    flags       TEXT    JSON array of guardrail flag strings
    synthesis   TEXT    full synthesis markdown
    state_json  TEXT    full final state as JSON (for replay)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path(__file__).parent.parent / "output" / "memory.db"


# ── store ─────────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Thin SQLite wrapper for persisting and querying past startup runs.

    Usage:
        store = MemoryStore()
        store.save(final_state)
        recent = store.get_recent(n=3)
        similar = store.search("retinopathy")
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Creates the runs table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    idea       TEXT    NOT NULL,
                    domain     TEXT    NOT NULL DEFAULT 'Healthcare / MedTech',
                    score      INTEGER,
                    verdict    TEXT,
                    flags      TEXT    DEFAULT '[]',
                    synthesis  TEXT,
                    state_json TEXT
                )
            """)
            conn.commit()

    # ── write ─────────────────────────────────────────────────────────────────

    def save(self, state: dict) -> int:
        """
        Persists a completed graph state to the database.
        Returns the row id of the inserted record.
        """
        # Extract verdict from synthesis scorecard if available
        verdict = None
        synthesis = state.get("synthesis") or ""
        for v in ["strong", "promising", "needs-work", "not-viable"]:
            if v.upper() in synthesis.upper():
                verdict = v
                break

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs
                    (timestamp, idea, domain, score, verdict, flags, synthesis, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    state.get("idea", "")[:500],          # cap at 500 chars
                    state.get("domain", "Healthcare / MedTech"),
                    state.get("viability_score"),
                    verdict,
                    json.dumps(state.get("guardrail_flags") or []),
                    synthesis[:10000],                     # cap synthesis at 10k
                    json.dumps(state, default=str)[:50000],
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid

        print(f"[memory] Run saved to DB (id={row_id}, score={state.get('viability_score')})")
        return row_id

    # ── read ──────────────────────────────────────────────────────────────────

    def get_recent(self, n: int = 5) -> list[dict]:
        """Returns the n most recent runs as dicts, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, keyword: str, limit: int = 5) -> list[dict]:
        """
        Full-text search across idea and synthesis fields.
        Returns matching runs as dicts, newest first.
        """
        pattern = f"%{keyword.lower()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE LOWER(idea) LIKE ? OR LOWER(synthesis) LIKE ?
                ORDER BY id DESC LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_id(self, run_id: int) -> Optional[dict]:
        """Retrieves a single run by primary key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        """Returns total number of stored runs."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    # ── context injection ─────────────────────────────────────────────────────

    def build_prior_context(self, idea: str, n: int = 3) -> str:
        """
        Builds a short context string summarising the n most relevant
        past runs for injection into the orchestrator framing.

        Called by orchestrator.py if memory is enabled.
        Returns empty string if no prior runs exist.
        """
        total = self.count()
        if total == 0:
            return ""

        # Try keyword search first, fall back to recent
        keyword = idea.split()[0] if idea else ""
        runs = self.search(keyword, limit=n) or self.get_recent(n=n)

        if not runs:
            return ""

        lines = [f"## Prior Startup Runs (last {len(runs)} relevant)\n"]
        for r in runs:
            flags = json.loads(r.get("flags") or "[]")
            flag_summary = f"{len(flags)} flag(s)" if flags else "no flags"
            lines.append(
                f"- [{r['timestamp'][:10]}] Score: {r.get('score', '?')}/100 "
                f"({r.get('verdict', '?')}) | {flag_summary} | "
                f"Idea: {r.get('idea', '')[:80]}..."
            )

        return "\n".join(lines)

    # ── housekeeping ──────────────────────────────────────────────────────────

    def delete_run(self, run_id: int) -> bool:
        """Deletes a run by id. Returns True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Drops and recreates the runs table. Use with caution."""
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS runs")
            conn.commit()
        self._init_db()
        print("[memory] All runs cleared.")