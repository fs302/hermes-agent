"""
Feishu thread history fallback.

When :class:`gateway.session_memory.SessionMemory` is missing or
meaningfully empty, the inbound handler should not reply "I don't have
context" — it should first try to rebuild a summary from the Feishu
thread itself.

Feishu's REST APIs make this a three-tier fallback:

  1. **By thread_id**: list messages where ``container_id_type=thread``
     and ``container_id=<thread_id>``.  This catches both forum topics
     and reply chains under a root message.
  2. **By parent_message_id**: list replies to the parent message via
     ``im/v1/messages`` with ``container_id_type=message``.
  3. **By chat_id + recent window**: list messages in the parent chat
     filtered to a 24-72 hour window that contains the most recent
     user intent.  Used only when both thread_id and parent_message_id
     are unavailable (e.g. raw text was sent into the parent chat
     without a topic).

The fetched messages are reduced to a single text transcript and
returned to the caller, which feeds them into a small LLM that
reconstructs the same :class:`SessionMemory` shape the rest of the
pipeline expects.

This module deliberately has **no dependency on the gateway hot
path** — it can be called from anywhere (inbound pipeline, debug
CLI, offline backfill) as long as a :class:`lark_oapi.Client` is
available.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class _RequestBuilder:
    """Minimal fallback builder used when lark_oapi is unavailable."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def container_id_type(self, value: str) -> "_RequestBuilder":
        self._data["container_id_type"] = value
        return self

    def container_id(self, value: str) -> "_RequestBuilder":
        self._data["container_id"] = value
        return self

    def sort_type(self, value: str) -> "_RequestBuilder":
        self._data["sort_type"] = value
        return self

    def page_size(self, value: int) -> "_RequestBuilder":
        self._data["page_size"] = value
        return self

    def start_time(self, value: str) -> "_RequestBuilder":
        self._data["start_time"] = value
        return self

    def end_time(self, value: str) -> "_RequestBuilder":
        self._data["end_time"] = value
        return self

    def page_token(self, value: str) -> "_RequestBuilder":
        self._data["page_token"] = value
        return self

    def build(self) -> SimpleNamespace:
        return SimpleNamespace(**self._data)


def _list_message_request_builder() -> Any:
    try:
        from lark_oapi.api.im.v1 import ListMessageRequest
    except ImportError:
        return _RequestBuilder()
    return ListMessageRequest.builder()


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class ThreadMessage:
    """A flattened view of a single Feishu message."""

    message_id: str
    sender_id: str
    sender_name: str
    text: str
    create_time_ms: int
    is_from_bot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "text": self.text,
            "create_time_ms": self.create_time_ms,
            "is_from_bot": self.is_from_bot,
        }


@dataclass
class ThreadHistoryResult:
    """Outcome of a :func:`fetch_thread_history` call."""

    messages: List[ThreadMessage] = field(default_factory=list)
    tier_used: str = ""              # "thread" | "parent" | "chat" | ""
    truncated: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_transcript(self, *, max_chars_per_message: int = 600) -> str:
        """Render the messages as a human-readable transcript.

        Used directly as input to the summary-reconstruction LLM.  The
        output is intentionally compact — long pastes get truncated so
        a 200-message thread still fits in a single prompt.
        """
        if not self.messages:
            return ""
        lines: List[str] = []
        for m in self.messages:
            ts = _ms_to_iso(m.create_time_ms)
            text = (m.text or "").strip().replace("\n", " ")
            if len(text) > max_chars_per_message:
                text = text[: max_chars_per_message - 1] + "…"
            who = "BOT" if m.is_from_bot else (m.sender_name or m.sender_id or "user")
            lines.append(f"[{ts}] {who}: {text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 1: by thread_id
# ---------------------------------------------------------------------------


async def fetch_by_thread_id(
    client: Any,
    thread_id: str,
    *,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    page_size: int = 50,
    bot_open_id: Optional[str] = None,
) -> ThreadHistoryResult:
    """List messages inside a Feishu thread / forum topic.

    Uses ``GET /open-apis/im/v1/messages?container_id_type=thread``
    (Lark doc: im/v1/messages).  ``container_id`` for a thread is
    the thread_id; the API returns messages in chronological order.
    """
    if not client or not thread_id:
        return ThreadHistoryResult(error="missing client or thread_id")

    items: List[ThreadMessage] = []
    page_token: Optional[str] = None
    pages = 0
    max_pages = 6  # 6 * 50 = 300 messages hard cap

    while pages < max_pages:
        pages += 1
        builder = (
            _list_message_request_builder()
            .container_id_type("thread")
            .container_id(thread_id)
            .sort_type("ByCreateTimeAsc")
            .page_size(page_size)
        )
        if start_time_ms is not None:
            builder = builder.start_time(str(start_time_ms))
        if end_time_ms is not None:
            builder = builder.end_time(str(end_time_ms))
        if page_token:
            builder = builder.page_token(page_token)
        request = builder.build()

        try:
            response = await asyncio.to_thread(client.im.v1.message.list, request)
        except Exception as e:
            return ThreadHistoryResult(
                messages=items,
                tier_used="thread",
                error=f"thread list failed: {e}",
            )

        if not response or not getattr(response, "success", lambda: False)():
            code = getattr(response, "code", "unknown")
            msg = getattr(response, "msg", "list failed")
            # -1 / 230020 typically means "thread does not exist" — not fatal,
            # we want the caller to fall through to tier 2.
            logger.info(
                "[feishu_thread_history] thread list rejected thread=%s code=%s msg=%s",
                thread_id, code, msg,
            )
            return ThreadHistoryResult(
                messages=items,
                tier_used="thread",
                error=f"thread list rejected: [{code}] {msg}",
            )

        data = getattr(response, "data", None)
        if not data:
            break
        raw_items = getattr(data, "items", None) or []
        for raw in raw_items:
            tm = _coerce_message(raw, bot_open_id=bot_open_id)
            if tm is not None:
                items.append(tm)

        has_more = bool(getattr(data, "has_more", False))
        page_token = getattr(data, "page_token", None) or None
        if not has_more or not page_token:
            break

    return ThreadHistoryResult(
        messages=items, tier_used="thread", truncated=(pages >= max_pages)
    )


# ---------------------------------------------------------------------------
# Tier 2: by parent_message_id
# ---------------------------------------------------------------------------


async def fetch_by_parent_message(
    client: Any,
    parent_message_id: str,
    *,
    page_size: int = 50,
    bot_open_id: Optional[str] = None,
) -> ThreadHistoryResult:
    """List replies to a single root message.

    Used when ``thread_id`` is empty but the inbound message references
    a parent / upper message — typical of Feishu group chats where a
    thread is just "all replies under message X".
    """
    if not client or not parent_message_id:
        return ThreadHistoryResult(error="missing client or parent_message_id")

    items: List[ThreadMessage] = []
    page_token: Optional[str] = None
    pages = 0
    max_pages = 4

    while pages < max_pages:
        pages += 1
        builder = (
            _list_message_request_builder()
            .container_id_type("message")
            .container_id(parent_message_id)
            .sort_type("ByCreateTimeAsc")
            .page_size(page_size)
        )
        if page_token:
            builder = builder.page_token(page_token)
        request = builder.build()

        try:
            response = await asyncio.to_thread(client.im.v1.message.list, request)
        except Exception as e:
            return ThreadHistoryResult(
                messages=items,
                tier_used="parent",
                error=f"parent list failed: {e}",
            )

        if not response or not getattr(response, "success", lambda: False)():
            code = getattr(response, "code", "unknown")
            msg = getattr(response, "msg", "list failed")
            logger.info(
                "[feishu_thread_history] parent list rejected id=%s code=%s msg=%s",
                parent_message_id, code, msg,
            )
            return ThreadHistoryResult(
                messages=items,
                tier_used="parent",
                error=f"parent list rejected: [{code}] {msg}",
            )

        data = getattr(response, "data", None)
        if not data:
            break
        raw_items = getattr(data, "items", None) or []
        for raw in raw_items:
            tm = _coerce_message(raw, bot_open_id=bot_open_id)
            if tm is not None:
                items.append(tm)

        has_more = bool(getattr(data, "has_more", False))
        page_token = getattr(data, "page_token", None) or None
        if not has_more or not page_token:
            break

    return ThreadHistoryResult(
        messages=items, tier_used="parent", truncated=(pages >= max_pages)
    )


# ---------------------------------------------------------------------------
# Tier 3: by chat_id + recent window
# ---------------------------------------------------------------------------


async def fetch_by_chat_recent(
    client: Any,
    chat_id: str,
    *,
    window_hours: int = 72,
    page_size: int = 30,
    bot_open_id: Optional[str] = None,
) -> ThreadHistoryResult:
    """Fallback: scan the parent chat for the most recent window.

    Used only when thread_id and parent_message_id are both missing.
    Fetches a 24-72h window of chat messages and lets the caller decide
    which subset is relevant.
    """
    if not client or not chat_id:
        return ThreadHistoryResult(error="missing client or chat_id")

    import time as _time

    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - int(window_hours * 3600 * 1000)

    items: List[ThreadMessage] = []
    page_token: Optional[str] = None
    pages = 0
    max_pages = 2

    while pages < max_pages:
        pages += 1
        builder = (
            _list_message_request_builder()
            .container_id_type("chat")
            .container_id(chat_id)
            .sort_type("ByCreateTimeDesc")
            .start_time(str(start_ms))
            .end_time(str(now_ms))
            .page_size(page_size)
        )
        if page_token:
            builder = builder.page_token(page_token)
        request = builder.build()

        try:
            response = await asyncio.to_thread(client.im.v1.message.list, request)
        except Exception as e:
            return ThreadHistoryResult(
                messages=items, tier_used="chat",
                error=f"chat list failed: {e}",
            )

        if not response or not getattr(response, "success", lambda: False)():
            code = getattr(response, "code", "unknown")
            msg = getattr(response, "msg", "list failed")
            return ThreadHistoryResult(
                messages=items, tier_used="chat",
                error=f"chat list rejected: [{code}] {msg}",
            )

        data = getattr(response, "data", None)
        if not data:
            break
        raw_items = getattr(data, "items", None) or []
        for raw in raw_items:
            tm = _coerce_message(raw, bot_open_id=bot_open_id)
            if tm is not None:
                items.append(tm)

        has_more = bool(getattr(data, "has_more", False))
        page_token = getattr(data, "page_token", None) or None
        if not has_more or not page_token:
            break

    # Reverse to chronological order — ByCreateTimeDesc returns newest-first.
    items.reverse()
    return ThreadHistoryResult(
        messages=items, tier_used="chat", truncated=(pages >= max_pages)
    )


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


async def fetch_thread_history(
    client: Any,
    *,
    thread_id: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    window_hours: int = 72,
    bot_open_id: Optional[str] = None,
) -> ThreadHistoryResult:
    """Three-tier fallback dispatcher.

    Tries ``thread_id`` → ``parent_message_id`` → ``chat_id`` in order.
    Returns the first *successful* result, even if it is empty.  Only
    returns an error if all three tiers fail.
    """
    if thread_id:
        result = await fetch_by_thread_id(
            client, thread_id, bot_open_id=bot_open_id
        )
        if result.ok:
            return result
        logger.info(
            "[feishu_thread_history] tier thread failed: %s — falling back",
            result.error,
        )

    if parent_message_id and parent_message_id != thread_id:
        result = await fetch_by_parent_message(
            client, parent_message_id, bot_open_id=bot_open_id
        )
        if result.ok:
            return result
        logger.info(
            "[feishu_thread_history] tier parent failed: %s — falling back",
            result.error,
        )

    if chat_id:
        result = await fetch_by_chat_recent(
            client, chat_id, window_hours=window_hours, bot_open_id=bot_open_id,
        )
        if result.ok:
            return result
        logger.warning(
            "[feishu_thread_history] all three tiers failed for thread=%s parent=%s chat=%s: %s",
            thread_id, parent_message_id, chat_id, result.error,
        )
        return result

    return ThreadHistoryResult(
        error="no identifiers (need thread_id, parent_message_id or chat_id)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BOT_NAME_RE = re.compile(r"@_user_\d+\s*")


def _coerce_message(raw: Any, *, bot_open_id: Optional[str] = None) -> Optional[ThreadMessage]:
    """Map a raw lark_oapi Message object to a flat :class:`ThreadMessage`."""
    if raw is None:
        return None
    message_id = str(getattr(raw, "message_id", "") or "")
    if not message_id:
        return None

    sender = getattr(raw, "sender", None)
    sender_id = str(getattr(sender, "id", "") or "") if sender else ""

    text = _extract_text(raw)
    text = _BOT_NAME_RE.sub("", text).strip()

    # Feishu returns create_time as a string in milliseconds
    raw_ct = getattr(raw, "create_time", None)
    try:
        create_time_ms = int(raw_ct) if raw_ct is not None else 0
    except (TypeError, ValueError):
        create_time_ms = 0

    is_bot = bool(bot_open_id) and sender_id == bot_open_id

    return ThreadMessage(
        message_id=message_id,
        sender_id=sender_id,
        sender_name="",
        text=text,
        create_time_ms=create_time_ms,
        is_from_bot=is_bot,
    )


def _extract_text(raw: Any) -> str:
    """Pull a plain-text representation out of a Feishu message body."""
    body = getattr(raw, "body", None)
    if not body:
        return ""
    content = getattr(body, "content", "") or ""
    if not content:
        return ""
    msg_type = str(getattr(raw, "msg_type", "") or "")
    # Reuse the platform's normaliser when available so post / interactive
    # messages come out as readable text instead of raw JSON.
    try:
        from gateway.platforms.feishu import normalize_feishu_message
        normalised = normalize_feishu_message(
            message_type=msg_type,
            raw_content=content,
            mentions=getattr(raw, "mentions", None),
        )
        if normalised.text_content:
            return normalised.text_content
    except Exception:
        pass

    # Best-effort fallback: strip JSON for text messages.
    if msg_type in ("text", ""):
        import json as _json
        try:
            data = _json.loads(content)
            if isinstance(data, dict):
                return str(data.get("text", "") or "")
        except (TypeError, ValueError):
            return content
    return content


def _ms_to_iso(ms: int) -> str:
    if not ms:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "unknown"
