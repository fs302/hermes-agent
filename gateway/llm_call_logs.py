"""LLM call observability store.

Records one row per LLM call so operators (and the user) can answer:

  * Which model actually served the request?
  * Did the resolver change the model from what was requested?
  * Did the provider fall back to a different model?
  * How many tokens / how much cost did this session accumulate?

Schema lives in a dedicated SQLite file (``llm_call_logs.db``) so it
can be inspected, archived, or truncated independently of the main
session DB (``state.db``).  The file is opened with WAL mode for
concurrent reads and single-writer semantics.

The gateway writes here from two places:

1. ``run_agent.py`` — the *actual* model returned by the provider is
   captured at the moment we receive the response, so the value is
   authoritative (not inferred from the request).
2. ``gateway/run.py`` — fallback events are recorded with the
   original ``requested_model`` so users can see what was asked for
   vs. what was used.

This module never raises into the caller — observability must NEVER
break a live conversation.  All public functions log failures at
WARNING and return safe defaults.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from hermes_constants import get_hermes_home
except ImportError:  # pragma: no cover - allow direct import
    get_hermes_home = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# `model_resolution_source` is one of the constants below.  Storing it
# as TEXT (not ENUM) keeps SQLite migrations painless.
RESOLUTION_SOURCE_VALUES = frozenset({
    "message_override",
    "session_model",
    "thread_model",
    "chat_model",
    "user_default",
    "agent_default",
    "system_default",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_call_logs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    user_id TEXT,
    chat_id TEXT,
    thread_id TEXT,
    session_key TEXT,
    agent_name TEXT,
    requested_model TEXT,
    requested_provider TEXT,
    resolved_model TEXT,
    resolved_provider TEXT,
    actual_model TEXT,
    actual_provider TEXT,
    model_resolution_source TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    cache_hit INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    status TEXT,
    error_message TEXT,
    fallback_used INTEGER DEFAULT 0,
    fallback_reason TEXT,
    request_id TEXT,
    provider_request_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_created
    ON llm_call_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_session
    ON llm_call_logs(session_key, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_user
    ON llm_call_logs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_logs_actual_model
    ON llm_call_logs(actual_model, created_at);
"""

_INSERT_SQL = """
INSERT INTO llm_call_logs (
    id, created_at, user_id, chat_id, thread_id, session_key, agent_name,
    requested_model, requested_provider,
    resolved_model, resolved_provider,
    actual_model, actual_provider,
    model_resolution_source,
    input_tokens, output_tokens, total_tokens, cost_usd,
    cache_hit, cache_read_tokens, cache_write_tokens,
    latency_ms, status, error_message,
    fallback_used, fallback_reason,
    request_id, provider_request_id
) VALUES (
    :id, :created_at, :user_id, :chat_id, :thread_id, :session_key, :agent_name,
    :requested_model, :requested_provider,
    :resolved_model, :resolved_provider,
    :actual_model, :actual_provider,
    :model_resolution_source,
    :input_tokens, :output_tokens, :total_tokens, :cost_usd,
    :cache_hit, :cache_read_tokens, :cache_write_tokens,
    :latency_ms, :status, :error_message,
    :fallback_used, :fallback_reason,
    :request_id, :provider_request_id
)
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_DB_PATH_OVERRIDE: Optional[Path] = None
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_init_lock = threading.Lock()
_init_done = False


def get_db_path() -> Path:
    if _DB_PATH_OVERRIDE is not None:
        return _DB_PATH_OVERRIDE
    if get_hermes_home is not None:
        return get_hermes_home() / "llm_call_logs.db"
    return Path.home() / ".hermes" / "llm_call_logs.db"


def set_db_path(path: Path) -> None:
    """Override the DB location (used by tests)."""
    global _DB_PATH_OVERRIDE, _conn, _init_done
    with _init_lock:
        _DB_PATH_OVERRIDE = Path(path)
        _conn = None
        _init_done = False


def _get_conn() -> Optional[sqlite3.Connection]:
    """Lazy-init the SQLite connection.  Never raises."""
    global _conn, _init_done
    if _init_done and _conn is not None:
        return _conn
    with _init_lock:
        if _init_done and _conn is not None:
            return _conn
        try:
            db_path = get_db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=2.0,
                isolation_level=None,  # we manage transactions ourselves
            )
            _conn.row_factory = sqlite3.Row
            # Pragmas — WAL for concurrent readers, NORMAL sync for
            # acceptable write latency.  This is observability data,
            # not user-critical state, so a crash mid-write only loses
            # the row being inserted.
            try:
                _conn.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass  # WAL is best-effort
            _conn.execute("PRAGMA synchronous=NORMAL")
            _conn.executescript(_SCHEMA)
            _init_done = True
            logger.debug("[llm_call_logs] DB ready at %s", db_path)
        except Exception as e:
            logger.warning(
                "[llm_call_logs] DB init failed at %s: %s",
                get_db_path(), e, exc_info=True,
            )
            _conn = None
            _init_done = False
            return None
    return _conn


# ---------------------------------------------------------------------------
# Public data shapes
# ---------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    """One row of llm_call_logs.

    Every field defaults to a safe sentinel so the caller can pass
    only what it knows.  ``id`` and ``created_at`` are auto-filled
    when not provided.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_key: Optional[str] = None
    agent_name: Optional[str] = None

    requested_model: Optional[str] = None
    requested_provider: Optional[str] = None
    resolved_model: Optional[str] = None
    resolved_provider: Optional[str] = None
    actual_model: Optional[str] = None
    actual_provider: Optional[str] = None
    model_resolution_source: Optional[str] = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self):
        # BOSS spec: total_tokens is the sum of input + output.
        # Auto-fill when the caller forgot, so SUM(total_tokens)
        # in aggregate_by_period() doesn't return NULL.
        if not self.total_tokens:
            try:
                self.total_tokens = int(self.input_tokens or 0) + int(self.output_tokens or 0)
            except (TypeError, ValueError):
                self.total_tokens = 0

    cache_hit: bool = False
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    latency_ms: int = 0
    status: str = "ok"  # "ok" | "error" | "timeout" | "cancelled"
    error_message: Optional[str] = None

    fallback_used: bool = False
    fallback_reason: Optional[str] = None

    request_id: Optional[str] = None
    provider_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def record_call(record: LLMCallRecord) -> bool:
    """Insert one LLM call row.  Returns True on success, False on failure.

    This function MUST NEVER raise — observability is best-effort.
    A failure here logs a WARNING and returns False so the caller can
    continue processing the user's request.
    """
    conn = _get_conn()
    if conn is None:
        return False
    try:
        payload = record.to_dict()
        # SQLite booleans → 0/1
        payload["cache_hit"] = 1 if payload.get("cache_hit") else 0
        payload["fallback_used"] = 1 if payload.get("fallback_used") else 0
        with _lock:
            conn.execute(_INSERT_SQL, payload)
        return True
    except Exception as e:
        logger.warning(
            "[llm_call_logs] record_call failed: %s | record=%r",
            e, {k: record.to_dict().get(k) for k in (
                "session_key", "actual_model", "status", "id"
            )},
        )
        return False


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _rows_to_records(rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        d["cache_hit"] = bool(d.get("cache_hit"))
        d["fallback_used"] = bool(d.get("fallback_used"))
        out.append(d)
    return out


def query_recent(
    *,
    session_key: Optional[str] = None,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return the most recent N records, newest first.

    Filters are AND-combined.  Empty filter means "all sessions".
    """
    conn = _get_conn()
    if conn is None:
        return []
    clauses: List[str] = []
    params: List[Any] = []
    if session_key:
        clauses.append("session_key = ?")
        params.append(session_key)
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    if chat_id:
        clauses.append("chat_id = ?")
        params.append(chat_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM llm_call_logs {where} ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    try:
        cur = conn.execute(sql, params)
        return _rows_to_records(cur.fetchall())
    except Exception as e:
        logger.warning("[llm_call_logs] query_recent failed: %s", e)
        return []


def aggregate_by_period(
    *,
    period: str = "today",  # "today" | "7d" | "session" | "all"
    session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate token / cost / counts for a period.

    Returns a dict with ``totals`` (overall) and ``by_model`` /
    ``by_provider`` / ``by_agent`` breakdowns.  When the DB is
    unavailable, returns a structure with all zeros and empty
    breakdowns so callers can render a friendly "no data" UI.
    """
    conn = _get_conn()
    if conn is None:
        return _empty_aggregate()
    now = time.time()
    since = _period_to_since(period, now)
    clauses: List[str] = ["created_at >= ?"]
    params: List[Any] = [since]
    if session_key:
        clauses.append("session_key = ?")
        params.append(session_key)
    where = "WHERE " + " AND ".join(clauses)

    out: Dict[str, Any] = _empty_aggregate()
    try:
        # Totals
        cur = conn.execute(
            f"SELECT COUNT(*) AS req, "
            f"COALESCE(SUM(input_tokens),0) AS inp, "
            f"COALESCE(SUM(output_tokens),0) AS out, "
            f"COALESCE(SUM(total_tokens),0) AS tot, "
            f"COALESCE(SUM(cost_usd),0) AS cost "
            f"FROM llm_call_logs {where}",
            params,
        )
        row = cur.fetchone()
        if row:
            out["total_requests"] = int(row["req"] or 0)
            out["total_input_tokens"] = int(row["inp"] or 0)
            out["total_output_tokens"] = int(row["out"] or 0)
            out["total_tokens"] = int(row["tot"] or 0)
            out["estimated_cost_usd"] = round(float(row["cost"] or 0.0), 6)

        # By model
        cur = conn.execute(
            f"SELECT actual_model AS model, actual_provider AS provider, "
            f"COUNT(*) AS requests, "
            f"COALESCE(SUM(input_tokens),0) AS input_tokens, "
            f"COALESCE(SUM(output_tokens),0) AS output_tokens, "
            f"COALESCE(SUM(total_tokens),0) AS total_tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost "
            f"FROM llm_call_logs {where} "
            f"GROUP BY actual_model, actual_provider "
            f"ORDER BY requests DESC",
            params,
        )
        out["by_model"] = _rows_to_records(cur.fetchall())

        # By provider
        cur = conn.execute(
            f"SELECT actual_provider AS provider, "
            f"COUNT(*) AS requests, "
            f"COALESCE(SUM(input_tokens),0) AS input_tokens, "
            f"COALESCE(SUM(output_tokens),0) AS output_tokens, "
            f"COALESCE(SUM(total_tokens),0) AS total_tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost "
            f"FROM llm_call_logs {where} "
            f"GROUP BY actual_provider "
            f"ORDER BY requests DESC",
            params,
        )
        out["by_provider"] = _rows_to_records(cur.fetchall())

        # By agent
        cur = conn.execute(
            f"SELECT agent_name, "
            f"COUNT(*) AS requests, "
            f"COALESCE(SUM(input_tokens),0) AS input_tokens, "
            f"COALESCE(SUM(output_tokens),0) AS output_tokens, "
            f"COALESCE(SUM(total_tokens),0) AS total_tokens, "
            f"COALESCE(SUM(cost_usd),0) AS cost "
            f"FROM llm_call_logs {where} "
            f"GROUP BY agent_name "
            f"ORDER BY requests DESC",
            params,
        )
        out["by_agent"] = _rows_to_records(cur.fetchall())
        return out
    except Exception as e:
        logger.warning("[llm_call_logs] aggregate_by_period failed: %s", e)
        return _empty_aggregate()


def _empty_aggregate() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "by_model": [],
        "by_provider": [],
        "by_agent": [],
    }


def _period_to_since(period: str, now: float) -> float:
    p = (period or "").lower()
    if p == "today":
        # Local-day start
        dt = datetime.fromtimestamp(now)
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()
    if p == "7d":
        return now - 7 * 24 * 3600
    if p == "session":
        # 24h window — sessions don't have a stable "since" boundary
        # in this store, so we approximate with the last 24h.
        return now - 24 * 3600
    if p == "all":
        return 0.0
    # Unknown period — fail safe to last 24h
    return now - 24 * 3600


def clear(*, session_key: Optional[str] = None) -> int:
    """Delete records.  Returns row count deleted.  Used by /reset."""
    conn = _get_conn()
    if conn is None:
        return 0
    try:
        if session_key:
            cur = conn.execute(
                "DELETE FROM llm_call_logs WHERE session_key = ?",
                (session_key,),
            )
        else:
            cur = conn.execute("DELETE FROM llm_call_logs")
        return int(cur.rowcount or 0)
    except Exception as e:
        logger.warning("[llm_call_logs] clear failed: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Helpers for the agent runtime
# ---------------------------------------------------------------------------


def _extract_usage(response: Any) -> Tuple[int, int, int, bool, int, int]:
    """Pull (input_tokens, output_tokens, total_tokens, cache_hit,
    cache_read_tokens, cache_write_tokens) from a provider response.

    Handles OpenAI and Anthropic shapes — both attach ``usage`` to
    the response object, though the field names differ.
    """
    if response is None:
        return 0, 0, 0, False, 0, 0
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0, False, 0, 0
    # OpenAI ChatCompletion.usage
    inp = int(getattr(usage, "prompt_tokens", 0) or 0)
    out = int(getattr(usage, "completion_tokens", 0) or 0)
    # Anthropic Messages.usage — input_tokens / output_tokens
    if not inp:
        inp = int(getattr(usage, "input_tokens", 0) or 0)
    if not out:
        out = int(getattr(usage, "output_tokens", 0) or 0)
    # Total — provider-reported first, fallback to inp+out.
    tot = int(getattr(usage, "total_tokens", 0) or 0)
    if not tot:
        tot = inp + out
    # Cache info
    cache_hit = False
    cache_read = 0
    cache_write = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cache_read = int(getattr(details, "cached_tokens", 0) or 0)
        if cache_read:
            cache_hit = True
    # Anthropic cache_creation_input_tokens / cache_read_input_tokens
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if not cache_read:
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        if cache_read:
            cache_hit = True
    return inp, out, tot, cache_hit, cache_read, cache_write


def _extract_actual_model(response: Any) -> Optional[str]:
    """Pull ``model`` from a provider response — the value returned
    by the upstream, not what we asked for."""
    if response is None:
        return None
    model = getattr(response, "model", None)
    if model:
        return str(model)
    return None


__all__ = [
    "LLMCallRecord",
    "record_call",
    "query_recent",
    "aggregate_by_period",
    "clear",
    "get_db_path",
    "set_db_path",
    "_extract_usage",
    "_extract_actual_model",
    "RESOLUTION_SOURCE_VALUES",
    "SCHEMA_VERSION",
]
