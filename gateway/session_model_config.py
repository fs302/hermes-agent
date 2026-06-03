"""Per-session model override store.

Wraps :mod:`gateway.session_memory` to add a ``model_config`` field
on the same record.  The existing in-memory dict in
``GatewayRunner._session_model_overrides`` is fast for the hot
turn path, but is lost on restart.  This module gives the
``/model`` command a durable home so Day-2 sessions continue
using the model the user picked on Day-1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from gateway.session_memory import (
    SessionMemory,
    load_session_memory,
    save_session_memory,
    update_session_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionModelConfig:
    """One model override attached to a session.

    Mirrors the fields stored in the in-memory dict in
    ``GatewayRunner._session_model_overrides`` so the two can
    interoperate without conversion.
    """

    model: str
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    api_mode: str = ""
    requested_at: float = 0.0  # unix seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_mode": self.api_mode,
            "requested_at": self.requested_at,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["SessionModelConfig"]:
        if not d or not d.get("model"):
            return None
        return cls(
            model=str(d.get("model") or ""),
            provider=str(d.get("provider") or ""),
            base_url=str(d.get("base_url") or ""),
            api_key=str(d.get("api_key") or ""),
            api_mode=str(d.get("api_mode") or ""),
            requested_at=float(d.get("requested_at") or 0.0),
        )


def get_session_model_config(session_key: str) -> Optional[SessionModelConfig]:
    """Read the persisted model override for a session.

    Returns ``None`` when no override has been set.
    """
    if not session_key:
        return None
    try:
        memory = load_session_memory(session_key)
    except Exception as e:
        logger.warning(
            "[session_model_config] load failed for %s: %s",
            session_key, e, exc_info=True,
        )
        return None
    if memory is None:
        return None
    return SessionModelConfig.from_dict(memory.model_config or {})


def set_session_model_config(
    session_key: str,
    config: SessionModelConfig,
) -> bool:
    """Persist a session model override.  Returns True on success."""
    if not session_key or not config or not config.model:
        return False
    try:
        # Ensure a memory record exists so we have a stable home for
        # the override.  update_session_memory merges fields into the
        # existing record without disturbing topic / summary / etc.
        memory = load_session_memory(session_key)
        if memory is None:
            memory = SessionMemory(session_key=session_key)
        update_session_memory(
            session_key,
            model_config=config.to_dict(),
        )
        return True
    except Exception as e:
        logger.warning(
            "[session_model_config] set failed for %s: %s",
            session_key, e, exc_info=True,
        )
        return False


def clear_session_model_config(session_key: str) -> bool:
    """Remove the override.  Returns True if a record was actually cleared."""
    if not session_key:
        return False
    try:
        memory = load_session_memory(session_key)
        if memory is None or not memory.model_config:
            return False
        # Empty dict + save reverts the field.
        memory.model_config = {}
        save_session_memory(memory)
        return True
    except Exception as e:
        logger.warning(
            "[session_model_config] clear failed for %s: %s",
            session_key, e, exc_info=True,
        )
        return False


__all__ = [
    "SessionModelConfig",
    "get_session_model_config",
    "set_session_model_config",
    "clear_session_model_config",
]
