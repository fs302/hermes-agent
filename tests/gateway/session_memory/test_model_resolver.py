"""Unit tests for gateway.model_resolver."""

import time
import unittest

from gateway.model_resolver import (
    ModelChoice,
    ModelResolverCache,
    ResolveRequest,
    RESOLUTION_PRIORITY,
    get_default_cache,
    invalidate_model_resolver_cache,
    resolve,
)


class TestPriorityChain(unittest.TestCase):
    """The 7-level priority order, top to bottom."""

    def setUp(self):
        invalidate_model_resolver_cache("test")

    def test_message_override_wins(self):
        req = ResolveRequest(
            message_override=ModelChoice("m_msg", "p_msg"),
            session_model=ModelChoice("m_sess", "p_sess"),
            user_default=ModelChoice("m_user", "p_user"),
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, ovr = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_msg")
        self.assertEqual(src, "message_override")
        self.assertIsNone(ovr)

    def test_session_beats_user_default(self):
        req = ResolveRequest(
            session_model=ModelChoice("m_sess", "p_sess"),
            user_default=ModelChoice("m_user", "p_user"),
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, _ = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_sess")
        self.assertEqual(src, "session_model")

    def test_user_default_beats_system(self):
        req = ResolveRequest(
            user_default=ModelChoice("m_user", "p_user"),
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, _ = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_user")
        self.assertEqual(src, "user_default")

    def test_system_default_is_last_resort(self):
        req = ResolveRequest(
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, _ = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_sys")
        self.assertEqual(src, "system_default")

    def test_priority_order_matches_spec(self):
        self.assertEqual(RESOLUTION_PRIORITY, (
            "message_override",
            "session_model",
            "thread_model",
            "chat_model",
            "user_default",
            "agent_default",
            "system_default",
        ))


class TestAgentOverride(unittest.TestCase):
    """BOSS rule: agent MUST NOT silently override session_model.

    If it MUST override, the reason must be visible."""

    def setUp(self):
        invalidate_model_resolver_cache("test")

    def test_no_override_passes_session(self):
        req = ResolveRequest(
            session_model=ModelChoice("m_sess", "p_sess"),
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, ovr = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_sess")
        self.assertEqual(src, "session_model")
        self.assertIsNone(ovr)

    def test_agent_override_with_reason(self):
        req = ResolveRequest(
            session_model=ModelChoice("m_sess", "p_sess"),
            agent_override=ModelChoice("m_forced", "p_forced"),
            agent_override_reason="no tool support on m_sess",
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, ovr = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_forced")
        self.assertEqual(src, "agent_override")
        self.assertEqual(ovr, "no tool support on m_sess")

    def test_agent_override_does_not_apply_when_no_session(self):
        # Without a session model, agent override has no one to beat.
        # The resolver should pick the highest non-override priority.
        req = ResolveRequest(
            user_default=ModelChoice("m_user", "p_user"),
            agent_override=ModelChoice("m_forced", "p_forced"),
            system_default=ModelChoice("m_sys", "p_sys"),
        )
        ch, src, ovr = resolve(req, session_key="test", use_cache=False)
        # agent_override is only consulted when there is a session_model
        # to override.  Without one, user_default wins.
        self.assertEqual(ch.model, "m_user")
        self.assertEqual(src, "user_default")


class TestCacheAndInvalidate(unittest.TestCase):
    def setUp(self):
        invalidate_model_resolver_cache("c1")
        invalidate_model_resolver_cache("c2")

    def test_cache_hit(self):
        req = ResolveRequest(
            session_model=ModelChoice("m_sess", "p_sess"),
        )
        ch1, src1, _ = resolve(req, session_key="c1")
        ch2, src2, _ = resolve(req, session_key="c1")
        self.assertEqual((ch1.model, src1), (ch2.model, src2))

    def test_invalidate_drops_entry(self):
        req = ResolveRequest(
            session_model=ModelChoice("m_sess", "p_sess"),
        )
        resolve(req, session_key="c1")
        evicted = invalidate_model_resolver_cache("c1", agent_name="main")
        self.assertEqual(evicted, 1)

    def test_invalidate_unknown_session(self):
        evicted = invalidate_model_resolver_cache("never_used")
        self.assertEqual(evicted, 0)

    def test_ttl_expiry(self):
        cache = ModelResolverCache(ttl_s=0.05)
        cache.put("k1", "main", ModelChoice("m1", "p1"), "session_model", None)
        # immediate hit
        got = cache.get("k1", "main")
        self.assertIsNotNone(got)
        time.sleep(0.1)
        # expired
        got = cache.get("k1", "main")
        self.assertIsNone(got)


class TestEmptyLevels(unittest.TestCase):
    def setUp(self):
        invalidate_model_resolver_cache("test")

    def test_no_levels_filled_returns_empty(self):
        req = ResolveRequest()
        ch, src, _ = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "")
        # The sentinel is "system_default" — see ResolveRequest.pick
        # contract.
        self.assertEqual(src, "system_default")

    def test_empty_model_string_skipped(self):
        # A ModelChoice with empty model should be ignored.
        req = ResolveRequest(
            session_model=ModelChoice("", "p_empty"),
            user_default=ModelChoice("m_user", "p_user"),
        )
        ch, src, _ = resolve(req, session_key="test", use_cache=False)
        self.assertEqual(ch.model, "m_user")
        self.assertEqual(src, "user_default")


if __name__ == "__main__":
    unittest.main()
