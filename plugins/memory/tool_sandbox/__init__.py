"""
ToolSandbox memory provider — context-mode equivalent for Hermes.

Stores large tool outputs externally in SQLite + FTS5, replacing them
with compact references in the conversation context. On context compaction,
retrieves only the relevant past outputs via BM25 instead of summarizing
everything.

Tools exposed:
  ctx_search      — BM25 full-text search over stored tool outputs
  ctx_fetch      — retrieve a stored tool output by ID
  ctx_stats      — token savings report

Hook: on_tool_result — called after every tool execution with the result.
      Providers can return a replacement string to substitute in context.

Config (env vars or $HERMES_HOME/tool_sandbox/config.json):
  TOOL_SANDBOX_MIN_CHARS  — minimum result size to store (default: 500)
  TOOL_SANDBOX_ENABLED    — 'true' or 'false' (default: true)
  TOOL_SANDBOX_MAX_STORED — max entries in store (default: 5000)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_MIN_CHARS = 500
DEFAULT_MAX_STORED = 5000


def _load_config() -> dict:
    from hermes_constants import get_hermes_home

    profile_path = Path(str(get_hermes_home())) / "tool_sandbox" / "config.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    legacy = Path.home() / ".tool_sandbox" / "config.json"
    if legacy.exists():
        try:
            return json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "enabled": os.environ.get("TOOL_SANDBOX_ENABLED", "true").lower() == "true",
        "min_chars": int(os.environ.get("TOOL_SANDBOX_MIN_CHARS", str(DEFAULT_MIN_CHARS))),
        "max_stored": int(os.environ.get("TOOL_SANDBOX_MAX_STORED", str(DEFAULT_MAX_STORED))),
    }


# ─── Schema ───────────────────────────────────────────────────────────────────

_TOOL_SCHEMAS = [
    {
        "name": "ctx_search",
        "description": (
            "Search stored tool outputs using BM25 full-text search. "
            "Use after context compaction to retrieve past tool results. "
            "Returns tool name, timestamp, result ID, and a relevance-ranked snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — describe what information you need from past tool outputs.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 5, max: 20).",
                    "default": 5,
                },
                "tool_name": {
                    "type": "string",
                    "description": "Filter by tool name (e.g. 'terminal', 'read_file'). Optional.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ctx_fetch",
        "description": (
            "Fetch a stored tool output by its result ID (from ctx_search). "
            "Returns the full stored output so you can see details of a past result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "The result ID returned by ctx_search.",
                },
            },
            "required": ["result_id"],
        },
    },
    {
        "name": "ctx_stats",
        "description": (
            "Show tool sandbox statistics: entries stored, total chars saved, "
            "average compression ratio, and per-tool breakdown."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "ctx_clear",
        "description": (
            "Clear all stored tool outputs for the current session. "
            "Use when starting a new task to keep the store lean."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to clear. Defaults to current session.",
                },
            },
            "required": [],
        },
    },
]


# ─── Storage ─────────────────────────────────────────────────────────────────

class _ToolStore:
    """Thread-safe SQLite + FTS5 store for tool outputs."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        # Main table: tool_results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_results (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                tool_name   TEXT NOT NULL,
                args_json   TEXT NOT NULL,
                result_text TEXT NOT NULL,
                result_len  INTEGER NOT NULL,
                stored_len  INTEGER NOT NULL,
                created_at  REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON tool_results(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool ON tool_results(tool_name)")

        # FTS5 virtual table for BM25 search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS tool_results_fts USING fts5(
                result_text,
                tool_name,
                content='tool_results',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS tool_results_ai AFTER INSERT ON tool_results BEGIN
                INSERT INTO tool_results_fts(rowid, result_text, tool_name)
                VALUES (NEW.rowid, NEW.result_text, NEW.tool_name);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS tool_results_ad AFTER DELETE ON tool_results BEGIN
                INSERT INTO tool_results_fts(tool_results_fts, rowid, result_text, tool_name)
                VALUES ('delete', OLD.rowid, OLD.result_text, OLD.tool_name);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS tool_results_au AFTER UPDATE ON tool_results BEGIN
                INSERT INTO tool_results_fts(tool_results_fts, rowid, result_text, tool_name)
                VALUES ('delete', OLD.rowid, OLD.result_text, OLD.tool_name);
                INSERT INTO tool_results_fts(rowid, result_text, tool_name)
                VALUES (NEW.rowid, NEW.result_text, NEW.tool_name);
            END
        """)

        conn.commit()

    def open(self) -> None:
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def store(
        self,
        tool_name: str,
        args_json: str,
        result_text: str,
        session_id: str,
    ) -> Optional[str]:
        """Store a tool result. Returns the result ID, or None if below threshold."""
        if not result_text or len(result_text) < 500:
            return None

        result_id = f"{tool_name}_{int(time.time() * 1000)}"
        stored_len = len(result_text)

        with self._lock:
            cur = self._conn.cursor()

            # Enforce max entries per session
            cur.execute(
                "SELECT COUNT(*) FROM tool_results WHERE session_id = ?",
                (session_id,),
            )
            count = cur.fetchone()[0]
            if count >= 5000:
                # Collect old rowids before deleting
                cur.execute("""
                    SELECT rowid FROM tool_results
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (session_id, max(1, int(count * 0.2))))
                old_rowids = [r[0] for r in cur.fetchall()]
                if old_rowids:
                    placeholders = ",".join("?" * len(old_rowids))
                    cur.execute(f"DELETE FROM tool_results_fts WHERE rowid IN ({placeholders})", old_rowids)
                    cur.execute(f"DELETE FROM tool_results WHERE rowid IN ({placeholders})", old_rowids)
                    self._conn.commit()

            cur.execute("""
                INSERT INTO tool_results
                    (id, session_id, tool_name, args_json, result_text, result_len, stored_len, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id, session_id, tool_name, args_json,
                result_text, stored_len, stored_len, time.time(),
            ))
            # Manually sync to FTS (triggers can misfire with WAL journal mode)
            cur.execute("""
                INSERT INTO tool_results_fts(rowid, result_text, tool_name)
                SELECT rowid, result_text, tool_name FROM tool_results WHERE id = ?
            """, (result_id,))
            self._conn.commit()
            return result_id

    def search(
        self,
        query: str,
        session_id: str,
        limit: int = 5,
        tool_name: Optional[str] = None,
    ) -> List[dict]:
        """BM25 search over stored tool results."""
        if not query:
            return []

        with self._lock:
            cur = self._conn.cursor()

            # Build params: query goes first (for MATCH), then session_id, then tool_name if present
            base_params: list = [session_id]
            where_parts = ["r.session_id = ?"]
            if tool_name:
                where_parts.append("r.tool_name = ?")
                base_params.append(tool_name)
            where_clause = " AND ".join(where_parts)

            # BM25 via fts5 match + join
            # Note: for wildcard queries like "*", MATCH can fail; fall back to no-match query
            try:
                sql = f"""
                    SELECT r.id, r.tool_name, r.args_json,
                           snippet(tool_results_fts, 0, '[', ']', '...', 64) AS snippet,
                           r.result_len,
                           r.created_at,
                           bm25(tool_results_fts) AS rank
                    FROM tool_results_fts fts
                    JOIN tool_results r ON fts.rowid = r.rowid
                    WHERE tool_results_fts MATCH ?
                      AND {where_clause}
                    ORDER BY rank
                    LIMIT ?
                """
                cur.execute(sql, [query] + base_params + [limit])
            except sqlite3.OperationalError:
                # Fallback for invalid FTS queries (e.g., wildcard "*" alone)
                sql = f"""
                    SELECT r.id, r.tool_name, r.args_json,
                           substr(r.result_text, 1, 200) AS snippet,
                           r.result_len,
                           r.created_at,
                           0.0 AS rank
                    FROM tool_results r
                    WHERE {where_clause}
                    ORDER BY r.created_at DESC
                    LIMIT ?
                """
                cur.execute(sql, base_params + [limit])
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "tool_name": row[1],
                "args": row[2],
                "snippet": row[3],
                "result_len": row[4],
                "created_at": datetime.fromtimestamp(row[5]).isoformat(),
                "rank": row[6],
            }
            for row in rows
        ]

    def fetch(self, result_id: str) -> Optional[dict]:
        """Fetch a single stored result by ID."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT id, session_id, tool_name, args_json, result_text, result_len, created_at
                FROM tool_results WHERE id = ?
            """, (result_id,))
            row = cur.fetchone()

        if not row:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "tool_name": row[2],
            "args": row[3],
            "result_text": row[4],
            "result_len": row[5],
            "created_at": datetime.fromtimestamp(row[6]).isoformat(),
        }

    def stats(self, session_id: str) -> dict:
        """Return storage statistics."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*), SUM(result_len), SUM(stored_len) FROM tool_results WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            count, total_raw, total_stored = row
            count = count or 0

            cur.execute("""
                SELECT tool_name, COUNT(*), SUM(result_len)
                FROM tool_results
                WHERE session_id = ?
                GROUP BY tool_name
                ORDER BY SUM(result_len) DESC
                LIMIT 10
            """, (session_id,))
            per_tool = [
                {"tool": r[0], "count": r[1], "chars": r[2]}
                for r in cur.fetchall()
            ]

        saved = (total_raw or 0) - (total_stored or 0)
        ratio = (total_stored or 0) / (total_raw or 1)

        return {
            "entries": count,
            "total_raw_chars": total_raw or 0,
            "total_stored_chars": total_stored or 0,
            "chars_saved": saved,
            "compression_ratio": round(ratio, 3),
            "per_tool": per_tool,
        }

    def clear_session(self, session_id: str) -> int:
        """Delete all entries for a session. Returns count deleted."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT rowid FROM tool_results WHERE session_id = ?", (session_id,))
            rowids = [r[0] for r in cur.fetchall()]
            count = len(rowids)
            if rowids:
                placeholders = ",".join("?" * len(rowids))
                cur.execute(f"DELETE FROM tool_results_fts WHERE rowid IN ({placeholders})", rowids)
                cur.execute(f"DELETE FROM tool_results WHERE rowid IN ({placeholders})", rowids)
                self._conn.commit()
        return count


# ─── Provider ─────────────────────────────────────────────────────────────────

class ToolSandboxProvider(MemoryProvider):
    """
    Tool output sandbox — stores large results in SQLite, replaces with
    compact references, retrieves on-demand during compaction.

    Architecture equivalent to context-mode's sandbox tools, but built
    as a Hermes MemoryProvider so it integrates with the compaction
    lifecycle without requiring Claude Code hooks.
    """

    def __init__(self):
        self._config: Optional[dict] = None
        self._store: Optional[_ToolStore] = None
        self._session_id: str = ""
        self._enabled: bool = True
        self._min_chars: int = DEFAULT_MIN_CHARS
        self._stats_lock = threading.Lock()
        self._total_saved: int = 0
        self._total_stored: int = 0
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "tool_sandbox"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            return cfg.get("enabled", True)
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "enabled",
                "description": "Enable tool sandbox (stores large tool outputs externally)",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "min_chars",
                "description": "Minimum tool result size to store externally (chars)",
                "default": str(DEFAULT_MIN_CHARS),
            },
            {
                "key": "max_stored",
                "description": "Maximum entries to keep per session",
                "default": str(DEFAULT_MAX_STORED),
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._session_id = session_id
        self._config = _load_config()
        self._enabled = self._config.get("enabled", True)
        self._min_chars = self._config.get("min_chars", DEFAULT_MIN_CHARS)

        db_dir = Path(str(get_hermes_home())) / "tool_sandbox"
        db_path = db_dir / "tool_outputs.db"
        self._store = _ToolStore(db_path)
        self._store.open()

        logger.info(
            "ToolSandbox initialized: session=%s min_chars=%d",
            session_id, self._min_chars,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return _TOOL_SCHEMAS

    # ─── Tool dispatch ────────────────────────────────────────────────────────

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "ctx_search":
            return self._handle_search(args)
        elif tool_name == "ctx_fetch":
            return self._handle_fetch(args)
        elif tool_name == "ctx_stats":
            return self._handle_stats(args)
        elif tool_name == "ctx_clear":
            return self._handle_clear(args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "")
        limit = min(int(args.get("limit", 5)), 20)
        tool_name = args.get("tool_name")

        results = self._store.search(
            query=query,
            session_id=self._session_id,
            limit=limit,
            tool_name=tool_name if tool_name else None,
        )
        if not results:
            return json.dumps({
                "results": [],
                "message": "No matching tool outputs found.",
            })
        return json.dumps({"results": results}, indent=2)

    def _handle_fetch(self, args: dict) -> str:
        result_id = args.get("result_id", "")
        if not result_id:
            return json.dumps({"error": "Missing required parameter: result_id"})

        result = self._store.fetch(result_id)
        if not result:
            return json.dumps({"error": f"Result not found: {result_id}"})

        return json.dumps({
            "result": {
                "id": result["id"],
                "tool_name": result["tool_name"],
                "args": result["args"],
                "created_at": result["created_at"],
                "result_text": result["result_text"],
            }
        }, indent=2)

    def _handle_stats(self, args: dict) -> str:
        stats = self._store.stats(self._session_id)
        return json.dumps(stats, indent=2)

    def _handle_clear(self, args: dict) -> str:
        session_id = args.get("session_id", self._session_id)
        count = self._store.clear_session(session_id)
        return json.dumps({"cleared": count})

    # ─── Tool result interception ─────────────────────────────────────────────
    # Called by run_agent.py after each tool execution, before the result
    # is appended to the conversation. Return value replaces the result.

    def on_tool_result(
        self,
        tool_name: str,
        args_json: str,
        result: str,
        stored_len: int = 0,
    ) -> str:
        """
        Hook called after every tool execution.

        If the result is large enough, store it externally and return
        a compact placeholder. Otherwise return the result unchanged.

        Returns the string to use as the tool result in the conversation.
        """
        if not self._enabled or not self._store:
            return result

        result_len = len(result)
        if result_len < self._min_chars:
            return result

        try:
            result_id = self._store.store(
                tool_name=tool_name,
                args_json=args_json,
                result_text=result,
                session_id=self._session_id,
            )
            if result_id:
                with self._stats_lock:
                    self._total_saved += result_len
                    self._total_stored += result_len

                placeholder = (
                    f"[Tool output stored externally (id: {result_id}). "
                    f"Use ctx_search to find it, ctx_fetch to retrieve it, "
                    f"or ctx_stats to see storage stats.]"
                )
                logger.debug(
                    "ToolSandbox: stored %s result %s (%d chars -> placeholder)",
                    tool_name, result_id, result_len,
                )
                return placeholder
        except Exception as e:
            logger.warning("ToolSandbox store failed: %s", e)

        return result

    # ─── Pre-compress hook ────────────────────────────────────────────────────
    # When context compaction fires, retrieve relevant past outputs instead
    # of losing them to summarization.

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """
        Before compaction discards old messages, retrieve relevant
        stored tool outputs via BM25. Return them as a formatted
        block to be injected into the compression summary prompt.
        """
        if not self._enabled or not self._store:
            return ""

        # Count how many tool_result messages will be lost
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) < 3:
            return ""

        # Search for all stored results for this session
        stats = self._store.stats(self._session_id)
        if stats["entries"] == 0:
            return ""

        # Retrieve up to 10 most recent stored results
        try:
            results = self._store.search(
                query="*",
                session_id=self._session_id,
                limit=10,
            )
        except Exception:
            return ""

        if not results:
            return ""

        lines = ["## Stored Tool Outputs (context-mode retrieval)"]
        lines.append(
            f"{stats['entries']} tool outputs stored ({stats['chars_saved']:,} chars saved). "
            "Recent outputs:"
        )
        for r in results[:10]:
            lines.append(
                f"\n### [{r['tool_name']}] {r['created_at']} (id: {r['id']})"
            )
            lines.append(f"```\n{r['snippet']}\n```")

        return "\n".join(lines)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
        logger.info(
            "ToolSandbox shutdown: total_saved=%d chars", self._total_saved,
        )


# ─── Registration ────────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Register ToolSandbox as a memory provider plugin."""
    ctx.register_memory_provider(ToolSandboxProvider())
