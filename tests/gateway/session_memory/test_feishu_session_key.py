"""Unit tests for the Feishu thread-scoped session key builder."""

import unittest

from gateway.session import Platform, SessionSource, build_session_key
from gateway.session import feishu_thread_session_key, build_session_key_with_diagnostics


def _feishu_source(
    *,
    chat_id: str = "oc_xxx",
    thread_id: str | None = "om_xxx",
    parent_chat_id: str | None = None,
    user_id: str | None = "ou_user_1",
    chat_type: str = "group",
) -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
        user_id=user_id,
    )


class TestFeishuThreadSessionKey(unittest.TestCase):
    def test_thread_id_is_most_specific(self):
        key = feishu_thread_session_key(
            chat_id="oc_1", thread_id="om_t1", user_id="ou_u1",
        )
        self.assertIn("thread:om_t1", key)
        self.assertIn("oc_1", key)
        # Group context — user is not part of the key (one task per thread).
        self.assertNotIn("ou_u1", key)

    def test_dm_keys_isolate_by_chat_id(self):
        # In Feishu, a DM chat_id IS the other party's open_id, so two
        # different DMs naturally produce different keys without an
        # explicit user_id component.
        key_a = feishu_thread_session_key(
            chat_id="oc_dm_a", chat_type="dm",
        )
        key_b = feishu_thread_session_key(
            chat_id="oc_dm_b", chat_type="dm",
        )
        self.assertIn("oc_dm_a", key_a)
        self.assertIn("oc_dm_b", key_b)
        self.assertNotEqual(key_a, key_b)

    def test_parent_message_id_used_as_thread_proxy(self):
        key = feishu_thread_session_key(
            chat_id="oc_1", parent_message_id="om_parent",
        )
        self.assertIn("parent:om_parent", key)

    def test_no_thread_no_parent(self):
        key = feishu_thread_session_key(chat_id="oc_1")
        # No thread / parent suffix should appear.
        self.assertNotIn("thread:", key)
        self.assertNotIn("parent:", key)
        self.assertIn("oc_1", key)

    def test_empty_chat_id_returns_empty(self):
        self.assertEqual(feishu_thread_session_key(chat_id=""), "")

    def test_dm_key_matches_user(self):
        # Two DMs that share the same chat_id (impossible in real Feishu
        # but possible in synthetic test data) collapse to one key when
        # no thread_id is present, since chat_id already identifies the
        # conversation.  This documents the collapse behaviour.
        key_a = feishu_thread_session_key(
            chat_id="oc_dm", user_id="ou_a", chat_type="dm",
        )
        key_b = feishu_thread_session_key(
            chat_id="oc_dm", user_id="ou_b", chat_type="dm",
        )
        self.assertEqual(key_a, key_b)

    def test_same_thread_same_key_across_days(self):
        # Day 1 and Day 2 of the same topic must produce the same key.
        k1 = feishu_thread_session_key(chat_id="oc_1", thread_id="om_t1")
        k2 = feishu_thread_session_key(chat_id="oc_1", thread_id="om_t1")
        self.assertEqual(k1, k2)


class TestGenericBuilderUnchanged(unittest.TestCase):
    def test_generic_key_includes_thread_id(self):
        src = _feishu_source(chat_id="oc_1", thread_id="om_t1")
        key = build_session_key(src, thread_sessions_per_user=False)
        self.assertIn("oc_1", key)
        self.assertIn("om_t1", key)

    def test_diagnostics_does_not_change_key(self):
        src = _feishu_source(chat_id="oc_1", thread_id="om_t1")
        k1 = build_session_key(src)
        k2 = build_session_key_with_diagnostics(src)
        self.assertEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
