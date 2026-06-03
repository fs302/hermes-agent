"""Unit tests for gateway.session_memory.

These tests are pure — no LLM calls, no Feishu API, no asyncio loop.
They cover:
  * round-trip persistence (save → load)
  * partial / corrupt file tolerance
  * thread-history fallback decision
  * status detection heuristics
  * safe session-key → filename mapping
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from gateway.session_memory import (
    SessionMemory,
    TaskState,
    VALID_TASK_STATUSES,
    _safe_session_key,
    clear_session_memory,
    detect_task_status_from_text,
    get_memory_dir,
    list_known_session_keys,
    load_session_memory,
    needs_thread_history_fallback,
    save_session_memory,
    set_memory_dir,
    update_session_memory,
)


class TestSessionKeySanitisation(unittest.TestCase):
    def test_keeps_colon_and_dot(self):
        # Session keys use ":" as a separator; must survive.
        self.assertEqual(
            _safe_session_key("agent:main:feishu:group:oc_xxx:om_xxx"),
            "agent:main:feishu:group:oc_xxx:om_xxx",
        )

    def test_replaces_path_traversal_chars(self):
        # Anything that could escape the directory must be neutralised.
        self.assertNotIn("/", _safe_session_key("a/b"))
        self.assertNotIn("\\", _safe_session_key("a\\b"))
        self.assertNotIn("..", _safe_session_key("../etc/passwd"))

    def test_empty_key_falls_back(self):
        self.assertEqual(_safe_session_key(""), "unknown")
        self.assertEqual(_safe_session_key("..."), "unknown")


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        set_memory_dir(Path(self.tmp.name))

    def test_save_then_load(self):
        m = SessionMemory(
            session_key="agent:main:feishu:group:oc_1:om_1",
            platform="feishu",
            chat_id="oc_1",
            thread_id="om_1",
            topic="评分卡修复方案",
            session_summary="用户在评估三项评分卡修复项",
            current_task_state=TaskState(
                status="waiting_for_user_confirmation",
                last_user_intent="看 Top 3 修复",
                last_agent_proposal=[
                    "Risk Reward: _calc_3scenarios 改用 vol_20d",
                    "Main Theme: 增加 refined ranking",
                    "评级门槛: 调整 S/A/B/C/D 阈值",
                ],
                next_action="等待用户确认是否开工",
            ),
            open_todos=[{"id": "t1", "content": "改 _calc_3scenarios"}],
            important_decisions=[{"id": "d1", "content": "采用 vol_20d"}],
            related_files_or_modules=[
                "alphaseek/backend/app/services/model_scoring_service.py"
            ],
        )
        save_session_memory(m)

        loaded = load_session_memory(m.session_key)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_key, m.session_key)
        self.assertEqual(loaded.platform, "feishu")
        self.assertEqual(loaded.chat_id, "oc_1")
        self.assertEqual(loaded.thread_id, "om_1")
        self.assertEqual(loaded.topic, "评分卡修复方案")
        self.assertEqual(loaded.session_summary, m.session_summary)
        self.assertEqual(
            loaded.current_task_state.status,
            "waiting_for_user_confirmation",
        )
        self.assertEqual(
            loaded.current_task_state.last_agent_proposal,
            m.current_task_state.last_agent_proposal,
        )
        self.assertEqual(loaded.open_todos, m.open_todos)
        self.assertEqual(loaded.important_decisions, m.important_decisions)
        self.assertEqual(
            loaded.related_files_or_modules, m.related_files_or_modules,
        )
        self.assertTrue(loaded.updated_at)  # set by save_session_memory

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_session_memory("agent:main:feishu:group:none"))

    def test_corrupt_file_returns_none(self):
        key = "agent:main:feishu:group:oc_corrupt"
        path = get_memory_dir() / f"{_safe_session_key(key)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("{not json}")
        self.assertIsNone(load_session_memory(key))

    def test_update_partial_fields_preserves_rest(self):
        key = "agent:main:feishu:group:oc_upd"
        m = SessionMemory(
            session_key=key,
            topic="原 topic",
            session_summary="原 summary",
            current_task_state=TaskState(
                status="in_progress",
                last_user_intent="开工",
                last_agent_proposal=["修复 #1"],
                next_action="实现中",
            ),
        )
        save_session_memory(m)

        # Update only the summary; everything else should be preserved.
        update_session_memory(key, session_summary="新的 summary")

        loaded = load_session_memory(key)
        self.assertEqual(loaded.session_summary, "新的 summary")
        self.assertEqual(loaded.topic, "原 topic")
        self.assertEqual(loaded.current_task_state.status, "in_progress")
        self.assertEqual(loaded.current_task_state.last_user_intent, "开工")
        self.assertEqual(
            loaded.current_task_state.last_agent_proposal, ["修复 #1"],
        )

    def test_clear_removes_file(self):
        key = "agent:main:feishu:group:oc_clr"
        save_session_memory(SessionMemory(session_key=key))
        self.assertTrue(clear_session_memory(key))
        self.assertFalse(clear_session_memory(key))  # idempotent
        self.assertIsNone(load_session_memory(key))

    def test_list_known_session_keys(self):
        for k in (
            "agent:main:feishu:group:oc_a:om_a",
            "agent:main:feishu:group:oc_b:om_b",
        ):
            save_session_memory(SessionMemory(session_key=k))
        keys = list_known_session_keys()
        self.assertEqual(len(keys), 2)
        self.assertIn("agent:main:feishu:group:oc_a:om_a", keys)
        self.assertIn("agent:main:feishu:group:oc_b:om_b", keys)

    def test_atomic_write_no_partial_files(self):
        # Force a write; the temp file must not survive.
        save_session_memory(SessionMemory(session_key="k1"))
        leftovers = list(get_memory_dir().glob(".k1.json.*.tmp"))
        self.assertEqual(leftovers, [])


class TestFallbackDecision(unittest.TestCase):
    def test_missing_memory_needs_fallback(self):
        self.assertTrue(needs_thread_history_fallback(None))

    def test_empty_memory_needs_fallback(self):
        m = SessionMemory(session_key="k1")
        self.assertTrue(needs_thread_history_fallback(m))

    def test_meaningful_memory_skips_fallback(self):
        m = SessionMemory(
            session_key="k1",
            session_summary="一些上下文",
            current_task_state=TaskState(status="in_progress"),
        )
        self.assertFalse(needs_thread_history_fallback(m))

    def test_proposal_only_memory_skips_fallback(self):
        m = SessionMemory(
            session_key="k1",
            current_task_state=TaskState(
                status="waiting_for_user_confirmation",
                last_agent_proposal=["修复 #1"],
            ),
        )
        self.assertFalse(needs_thread_history_fallback(m))


class TestStatusDetection(unittest.TestCase):
    def test_confirm_phrases_trigger_in_progress(self):
        for phrase in ("开工", "开始修", "可以", "go ahead", "proceed"):
            status = detect_task_status_from_text(
                user_text=phrase, agent_text="",
            )
            self.assertEqual(
                status, "in_progress", msg=f"phrase={phrase!r}",
            )

    def test_proposal_phrases_trigger_waiting(self):
        for phrase in ("建议", "方案", "Top 3", "我会这么改"):
            status = detect_task_status_from_text(
                user_text="", agent_text=phrase,
            )
            self.assertEqual(
                status, "waiting_for_user_confirmation",
                msg=f"phrase={phrase!r}",
            )

    def test_no_signal_returns_none(self):
        self.assertIsNone(
            detect_task_status_from_text(
                user_text="今天天气不错", agent_text="是的",
            )
        )

    def test_empty_input_returns_none(self):
        self.assertIsNone(detect_task_status_from_text(user_text="", agent_text=""))


class TestSchemaValidation(unittest.TestCase):
    def test_valid_statuses(self):
        self.assertIn("open", VALID_TASK_STATUSES)
        self.assertIn("in_progress", VALID_TASK_STATUSES)
        self.assertIn("waiting_for_user_confirmation", VALID_TASK_STATUSES)
        self.assertIn("blocked", VALID_TASK_STATUSES)
        self.assertIn("done", VALID_TASK_STATUSES)

    def test_unknown_status_normalised(self):
        # to_dict() should clamp unknown statuses to "open".
        ts = TaskState(status="mystery")
        d = ts.to_dict()
        self.assertEqual(d["status"], "open")


if __name__ == "__main__":
    unittest.main()
