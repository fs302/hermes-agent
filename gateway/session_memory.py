"""
Session memory for the Hermes gateway.

Persists a compressed, cross-day task state per session_key so the agent
can pick up a conversation the next morning without re-reading every
historical message.

The persisted schema is intentionally narrow — it is a *task state* not
a transcript.  The full transcript still lives in
:class:`hermes_state.SessionDB`; this module only owns the slice that
must survive session expiry, idle resets, and "fresh" continuation.

Layout on disk
--------------
One JSON file per session_key under
``~/.hermes/session_memory/<session_key>.json`` (path overridable via
:func:`set_memory_dir`).  The file is rewritten atomically (temp file +
``os.replace``) on every update, so a crash mid-write cannot corrupt
the existing memory.

Schema
------
::

    {
        "session_key": "agent:main:feishu:group:oc_xxx:om_xxx",
        "platform": "feishu",
        "chat_id": "oc_xxx",
        "thread_id": "om_xxx",
        "parent_message_id": "om_parent",
        "project": "scorecard",
        "topic": "评分卡修复方案",
        "session_summary": "用户希望对评分卡的三项缺陷进行修复 ...",
        "current_task_state": {
            "status": "waiting_for_user_confirmation",
            "last_user_intent": "想看 Top 3 修复项",
            "last_agent_proposal": [
                "Risk Reward 模型: _calc_3scenarios 改用 vol_20d 校准",
                "Main Theme 区分度: 增加 refined ranking",
                "评级门槛: 调整 S/A/B/C/D 阈值"
            ],
            "next_action": "等待用户确认是否开工"
        },
        "open_todos": [
            {"id": "t1", "content": "实现 _calc_3scenarios vol_20d 校准", "owner": "agent"},
            ...
        ],
        "important_decisions": [
            {"id": "d1", "content": "采用 vol_20d 作为波动率基准", "by": "agent", "ts": "..."}
        ],
        "related_files_or_modules": [
            "alphaseek/backend/app/services/model_scoring_service.py"
        ],
        "updated_at": "2026-06-03T07:40:00+08:00"
    }

All fields except ``session_key`` and ``updated_at`` are optional.  The
loader is tolerant of missing keys — partial memories are still useful.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

VALID_TASK_STATUSES = frozenset({
    "open",                  # freshly created, no proposal yet
    "in_progress",           # user said "开工 / 开始修 / 就这么做"
    "waiting_for_user_confirmation",  # agent proposed, waiting for user
    "blocked",               # external dependency / unclear
    "done",                  # user marked complete
})


@dataclass
class TaskState:
    """The current task state machine."""

    status: str = "open"
    last_user_intent: str = ""
    last_agent_proposal: List[str] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.status not in VALID_TASK_STATUSES:
            logger.warning(
                "[session_memory] Unknown task status %r; normalising to 'open'",
                self.status,
            )
            d["status"] = "open"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        return cls(
            status=data.get("status", "open") or "open",
            last_user_intent=data.get("last_user_intent", "") or "",
            last_agent_proposal=list(data.get("last_agent_proposal", []) or []),
            next_action=data.get("next_action", "") or "",
        )


@dataclass
class SessionMemory:
    """The full persisted state for one session_key."""

    session_key: str
    platform: str = ""
    chat_id: str = ""
    thread_id: str = ""
    parent_message_id: str = ""
    project: str = ""
    topic: str = ""
    session_summary: str = ""
    current_task_state: TaskState = field(default_factory=TaskState)
    open_todos: List[Dict[str, Any]] = field(default_factory=list)
    important_decisions: List[Dict[str, Any]] = field(default_factory=list)
    related_files_or_modules: List[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": self.session_key,
            "platform": self.platform,
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "parent_message_id": self.parent_message_id,
            "project": self.project,
            "topic": self.topic,
            "session_summary": self.session_summary,
            "current_task_state": self.current_task_state.to_dict(),
            "open_todos": list(self.open_todos),
            "important_decisions": list(self.important_decisions),
            "related_files_or_modules": list(self.related_files_or_modules),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMemory":
        task_state_raw = data.get("current_task_state") or {}
        return cls(
            session_key=str(data.get("session_key", "")),
            platform=str(data.get("platform", "") or ""),
            chat_id=str(data.get("chat_id", "") or ""),
            thread_id=str(data.get("thread_id", "") or ""),
            parent_message_id=str(data.get("parent_message_id", "") or ""),
            project=str(data.get("project", "") or ""),
            topic=str(data.get("topic", "") or ""),
            session_summary=str(data.get("session_summary", "") or ""),
            current_task_state=TaskState.from_dict(task_state_raw),
            open_todos=list(data.get("open_todos", []) or []),
            important_decisions=list(data.get("important_decisions", []) or []),
            related_files_or_modules=list(
                data.get("related_files_or_modules", []) or []
            ),
            updated_at=str(data.get("updated_at", "") or ""),
        )

    def is_meaningfully_empty(self) -> bool:
        """True if there is no recoverable context for the next turn.

        Used by :func:`needs_thread_history_fallback` to decide whether
        to fetch Feishu thread history before letting the model respond.
        """
        return (
            not self.session_summary.strip()
            and self.current_task_state.status == "open"
            and not self.current_task_state.last_agent_proposal
            and not self.open_todos
            and not self.important_decisions
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_DEFAULT_DIR: Optional[Path] = None
_LOCK = threading.Lock()
_DIR_OVERRIDE: Optional[Path] = None


def _resolve_default_dir() -> Path:
    """Return the canonical memory directory.

    Honours ``HERMES_SESSION_MEMORY_DIR`` if set, otherwise falls back
    to ``~/.hermes/session_memory``.  The directory is created on
    first access.
    """
    override = os.environ.get("HERMES_SESSION_MEMORY_DIR", "").strip()
    if override:
        return Path(override).expanduser()

    home_env = os.environ.get("HERMES_HOME", "").strip()
    if home_env:
        return Path(home_env).expanduser() / "session_memory"

    return Path.home() / ".hermes" / "session_memory"


def get_memory_dir() -> Path:
    """Return the current memory directory, initialising on first call."""
    global _DEFAULT_DIR
    if _DEFAULT_DIR is not None:
        return _DEFAULT_DIR
    with _LOCK:
        if _DEFAULT_DIR is None:
            _DEFAULT_DIR = _resolve_default_dir()
        _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        return _DEFAULT_DIR


def set_memory_dir(path: Path) -> None:
    """Override the memory directory (used by tests)."""
    global _DEFAULT_DIR
    with _LOCK:
        _DEFAULT_DIR = Path(path).expanduser()
        _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)


_SESSION_KEY_SAFE = re.compile(r"[^A-Za-z0-9_.:-]+")


def _safe_session_key(session_key: str) -> str:
    """Map a session key to a filesystem-safe filename component.

    Session keys use ``:`` as the separator; on macOS / Linux ``:`` is
    a safe filename character but Windows reserves it for drive
    letters.  We replace anything outside ``[A-Za-z0-9_.:-]`` with
    ``_`` so the same code works on every platform.
    """
    cleaned = _SESSION_KEY_SAFE.sub("_", session_key or "unknown")
    cleaned = cleaned.strip("._:-") or "unknown"
    return cleaned[:200]


def _memory_path(session_key: str) -> Path:
    return get_memory_dir() / f"{_safe_session_key(session_key)}.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically — never leave a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_session_memory(session_key: str) -> Optional[SessionMemory]:
    """Return the persisted memory for *session_key*, or ``None`` if absent.

    Tolerant of corrupt / partial files: any JSON error yields ``None``
    and is logged at WARNING so a single bad file does not block the
    whole pipeline.
    """
    if not session_key:
        return None
    path = _memory_path(session_key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(
            "[session_memory] Corrupt memory file %s: %s; treating as empty",
            path, e,
        )
        return None
    except OSError as e:
        logger.warning(
            "[session_memory] Failed to read %s: %s; treating as empty",
            path, e,
        )
        return None

    if not isinstance(data, dict):
        return None
    data.setdefault("session_key", session_key)
    return SessionMemory.from_dict(data)


def save_session_memory(memory: SessionMemory) -> None:
    """Persist *memory* atomically.  No-op if ``session_key`` is empty."""
    if not memory.session_key:
        logger.warning("[session_memory] Refusing to save memory with empty session_key")
        return
    memory.updated_at = _now_iso()
    _atomic_write_json(_memory_path(memory.session_key), memory.to_dict())


def update_session_memory(
    session_key: str,
    *,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    project: Optional[str] = None,
    topic: Optional[str] = None,
    session_summary: Optional[str] = None,
    task_state: Optional[TaskState] = None,
    open_todos: Optional[List[Dict[str, Any]]] = None,
    important_decisions: Optional[List[Dict[str, Any]]] = None,
    related_files_or_modules: Optional[List[str]] = None,
) -> SessionMemory:
    """Patch the memory for *session_key* in place.

    Only the fields explicitly passed are overwritten.  ``None`` means
    "leave the existing value untouched".  Empty strings or empty
    containers are still applied verbatim — they represent deliberate
    resets (e.g. clearing todos after a fix lands).
    """
    if not session_key:
        raise ValueError("session_key is required")

    memory = load_session_memory(session_key) or SessionMemory(session_key=session_key)
    memory.session_key = session_key

    if platform is not None:
        memory.platform = platform
    if chat_id is not None:
        memory.chat_id = chat_id
    if thread_id is not None:
        memory.thread_id = thread_id
    if parent_message_id is not None:
        memory.parent_message_id = parent_message_id
    if project is not None:
        memory.project = project
    if topic is not None:
        memory.topic = topic
    if session_summary is not None:
        memory.session_summary = session_summary
    if task_state is not None:
        memory.current_task_state = task_state
    if open_todos is not None:
        memory.open_todos = open_todos
    if important_decisions is not None:
        memory.important_decisions = important_decisions
    if related_files_or_modules is not None:
        memory.related_files_or_modules = related_files_or_modules

    save_session_memory(memory)
    return memory


def clear_session_memory(session_key: str) -> bool:
    """Delete the memory file for *session_key*.  Returns True if removed."""
    if not session_key:
        return False
    path = _memory_path(session_key)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning(
            "[session_memory] Failed to delete %s: %s", path, e,
        )
        return False


def list_known_session_keys() -> List[str]:
    """Return every session_key that has a memory file on disk."""
    try:
        return sorted(
            p.stem for p in get_memory_dir().glob("*.json")
        )
    except OSError as e:
        logger.warning("[session_memory] list_known_session_keys failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def needs_thread_history_fallback(memory: Optional[SessionMemory]) -> bool:
    """Decide whether the inbound handler should fetch Feishu history.

    Returns True when memory is missing entirely OR when it is so
    sparse that the next turn would have no task context.  This is
    the BOSS-mandated escape hatch: never reply "I don't have
    context" before trying to rebuild it from the thread.
    """
    if memory is None:
        return True
    return memory.is_meaningfully_empty()


def detect_task_status_from_text(
    *,
    user_text: str,
    agent_text: str,
) -> Optional[str]:
    """Heuristically classify the new task status from a turn.

    Used as a *fallback* when a full LLM-based summary update is
    unavailable (e.g. the summary model is down).  The function only
    flips status; it never overwrites the proposal / todos / decisions.

    Returns ``None`` if the turn gives no clear signal — in that case
    leave the existing status untouched.
    """
    text = f"{user_text or ''} {agent_text or ''}".strip()
    if not text:
        return None
    lowered = text.lower()

    confirm_phrases = (
        "开工", "开始修", "开始改", "开始做", "开始实现",
        "开始", "就这么做", "就按这个", "就这个方案", "可以",
        "let's go", "go ahead", "proceed", "do it", "approved",
    )
    if any(p in lowered for p in confirm_phrases):
        return "in_progress"

    propose_phrases = (
        "建议", "方案", "top 3", "top3", "三个修复", "三项修复",
        "我会", "可以这么改", "考虑", "proposal", "propose",
    )
    if any(p in lowered for p in propose_phrases):
        return "waiting_for_user_confirmation"

    return None


def _now_iso() -> str:
    """Return the current local time in ISO-8601 form."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
