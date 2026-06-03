"""Model resolution with explicit priority chain.

The BOSS spec mandates a 7-level priority chain.  This module
implements it as a single ``resolve()`` call that returns both
the chosen model/provider AND the source level it came from.

Priority (highest first — earlier wins):

    1. message_override  — set per-message by the user, e.g. /model
                            in the same turn
    2. session_model     — per-session_key override (Feishu thread,
                            Discord channel, etc.)
    3. thread_model      — per-thread override
    4. chat_model        — per-chat override
    5. user_default      — user-level default from config
    6. agent_default     — agent-level default from config
    7. system_default    — last-resort fallback

The resolver is **deliberately stateless across calls** so that an
``invalidated`` cache (set by /model) immediately takes effect on
the next LLM call.  Each call constructs a fresh resolver from
the current state of the session, the message, and the config.

Boss rule: ``agent_default`` MUST NOT silently override
``session_model``.  If the agent layer (e.g. tool/capability
guard) insists on a different model, it must declare an
``override_reason`` and pass that to ``resolve()`` so the reason
is recorded in llm_call_logs.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


RESOLUTION_PRIORITY = (
    "message_override",
    "session_model",
    "thread_model",
    "chat_model",
    "user_default",
    "agent_default",
    "system_default",
)


@dataclass
class ModelChoice:
    """A model+provider pair with provenance."""
    model: str
    provider: str
    base_url: str = ""
    api_key: str = ""
    api_mode: str = ""
    source: str = ""  # which priority level produced this choice


@dataclass
class ResolveRequest:
    """Everything the resolver needs to make a decision.

    All fields are optional except ``agent_name`` and ``system_default``,
    which the resolver falls back to when nothing else is set.
    """
    # Per-message override (rarely set)
    message_override: Optional[ModelChoice] = None

    # Per-session_key override — set by /model command
    session_model: Optional[ModelChoice] = None

    # Per-thread override
    thread_model: Optional[ModelChoice] = None

    # Per-chat override
    chat_model: Optional[ModelChoice] = None

    # User default (from config)
    user_default: Optional[ModelChoice] = None

    # Agent default (from agent config)
    agent_default: Optional[ModelChoice] = None

    # System default (last resort)
    system_default: Optional[ModelChoice] = None

    # Optional: if a safety/tool/capability check forces a different
    # model, pass it here with a reason.  This is the only way an
    # agent can override session_model.
    agent_override: Optional[ModelChoice] = None
    agent_override_reason: str = ""

    def pick(self) -> tuple[ModelChoice, str, Optional[str]]:
        """Apply the 7-level priority chain.

        Returns ``(choice, source, override_reason)``.

        The override_reason is non-empty ONLY when ``agent_override``
        won, and is propagated so the caller can record it in
        llm_call_logs.
        """
        # 1. message_override
        if self.message_override and self.message_override.model:
            return self.message_override, "message_override", None
        # 2. session_model
        if self.session_model and self.session_model.model:
            choice = self.session_model
            # If the agent layer forced a different model, prefer that
            # but record the override.  This is the ONLY path that
            # lets agent_default / a system policy outrank session_model.
            if self.agent_override and self.agent_override.model and \
                    self.agent_override.model != choice.model:
                return (
                    self.agent_override, "agent_override",
                    self.agent_override_reason or "agent forced override",
                )
            return choice, "session_model", None
        # 3. thread_model
        if self.thread_model and self.thread_model.model:
            return self.thread_model, "thread_model", None
        # 4. chat_model
        if self.chat_model and self.chat_model.model:
            return self.chat_model, "chat_model", None
        # 5. user_default
        if self.user_default and self.user_default.model:
            return self.user_default, "user_default", None
        # 6. agent_default
        if self.agent_default and self.agent_default.model:
            return self.agent_default, "agent_default", None
        # 7. system_default
        if self.system_default and self.system_default.model:
            return self.system_default, "system_default", None
        # All empty — return empty choice and let the caller handle.
        return (
            ModelChoice(model="", provider=""),
            "system_default",  # sentinel
            None,
        )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------
#
# /model invalidates the cache for one session_key.  The cache is
# keyed by (session_key, agent_name) and stores the resolved choice
# for a short TTL (default 60s) so that the resolver can short-circuit
# inside a single turn (the resolver is called multiple times in a
# turn — once per LLM call).  /model always invalidates.
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    choice: ModelChoice
    source: str
    override_reason: Optional[str]
    expires_at: float


class ModelResolverCache:
    """Tiny TTL cache for resolver decisions.

    Public methods never raise.  The cache is best-effort — a hit
    saves repeated work, a miss is harmless.
    """

    DEFAULT_TTL_S = 60.0

    def __init__(self, ttl_s: float = DEFAULT_TTL_S):
        self._ttl = float(ttl_s)
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def _key(self, session_key: str, agent_name: str) -> str:
        return f"{agent_name}::{session_key or '<none>'}"

    def get(self, session_key: str, agent_name: str) -> Optional[tuple[ModelChoice, str, Optional[str]]]:
        key = self._key(session_key, agent_name)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._store.pop(key, None)
                return None
            return entry.choice, entry.source, entry.override_reason

    def put(
        self,
        session_key: str,
        agent_name: str,
        choice: ModelChoice,
        source: str,
        override_reason: Optional[str],
    ) -> None:
        key = self._key(session_key, agent_name)
        with self._lock:
            self._store[key] = _CacheEntry(
                choice=choice,
                source=source,
                override_reason=override_reason,
                expires_at=time.time() + self._ttl,
            )

    def invalidate(self, session_key: str, agent_name: Optional[str] = None) -> int:
        """Drop one or all entries for a session.  Returns count evicted."""
        with self._lock:
            if not session_key:
                return 0
            if agent_name:
                key = self._key(session_key, agent_name)
                existed = self._store.pop(key, None) is not None
                return 1 if existed else 0
            # Agent-agnostic: drop everything for this session_key
            keys = [k for k in self._store if k.endswith(f"::{session_key}")]
            for k in keys:
                self._store.pop(k, None)
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n


# A process-wide default instance.  Tests can build their own.
_default_cache: Optional[ModelResolverCache] = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> ModelResolverCache:
    global _default_cache
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                _default_cache = ModelResolverCache()
    return _default_cache


def invalidate_model_resolver_cache(
    session_key: str, agent_name: Optional[str] = None
) -> int:
    """Public helper used by /model after switching.

    Returns count evicted.  Never raises.
    """
    try:
        return get_default_cache().invalidate(session_key, agent_name)
    except Exception as e:  # pragma: no cover
        logger.warning("[model_resolver] invalidate failed: %s", e)
        return 0


def resolve(
    request: ResolveRequest,
    *,
    session_key: str = "",
    agent_name: str = "main",
    use_cache: bool = True,
) -> tuple[ModelChoice, str, Optional[str]]:
    """Resolve the active model for one LLM call.

    Returns ``(choice, source, override_reason)``.
    """
    if use_cache:
        cached = get_default_cache().get(session_key, agent_name)
        if cached is not None:
            return cached

    choice, source, override_reason = request.pick()

    if use_cache and choice.model:
        try:
            get_default_cache().put(session_key, agent_name, choice, source, override_reason)
        except Exception as e:  # pragma: no cover
            logger.warning("[model_resolver] cache put failed: %s", e)

    if not choice.model:
        logger.warning(
            "[model_resolver] no model resolved for session=%s agent=%s — "
            "all 7 priority levels were empty.  Check config.yaml.",
            session_key, agent_name,
        )
    return choice, source, override_reason


__all__ = [
    "ModelChoice",
    "ResolveRequest",
    "ModelResolverCache",
    "get_default_cache",
    "invalidate_model_resolver_cache",
    "resolve",
    "RESOLUTION_PRIORITY",
]
