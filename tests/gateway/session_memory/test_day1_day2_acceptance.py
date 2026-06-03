"""Day-1 → Day-2 acceptance simulation.

Walks the exact scenario from the BOSS spec:

* Day 1: BOSS asks Hermes to design scorecard fixes.
* Day 1: Hermes proposes Top-3 fixes and asks for go-ahead.
* Day 2: BOSS replies "可以，开始修这三项" in the same topic.
* Day 2: Hermes must NOT ask "修什么".  It must remember the
  three fixes and start listing files / entering the work flow.

The simulation is purely deterministic — it exercises the
session_memory round-trip + the Feishu key builder, then asserts
that the Day-2 prompt (after injection) contains the three fix
keywords.  It does NOT call the LLM.  A live LLM is exercised
end-to-end by the gateway itself.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.session import (
    Platform,
    SessionSource,
    feishu_thread_session_key,
)
from gateway.session_memory import (
    SessionMemory,
    TaskState,
    get_memory_dir,
    load_session_memory,
    save_session_memory,
    set_memory_dir,
)


# The three fix items from BOSS spec (used as the original Day-1
# agent proposal).
DAY1_PROPOSAL = [
    "Risk Reward 模型：_calc_3scenarios 改用 vol_20d 校准，消除动量股系统性 0 分",
    "Main Theme 区分度：增加 refined_ranking，避免 rank1 中多只股票并列",
    "评级门槛：调整 S/A/B/C/D 阈值，或提高 Research 上限",
]
DAY1_FILES = [
    "alphaseek/backend/app/services/model_scoring_service.py",
    "alphaseek/backend/app/services/stock_analysis_service.py",
]


def _day1_source(chat_id: str = "oc_eval_1") -> SessionSource:
    """Day-1 source: a topic is born from message `om_topic_root`."""
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_type="group",
        thread_id="om_topic_root",  # the topic root
        parent_chat_id=None,
        user_id="ou_11d6fe7f5f52d4053ede9f47c655fc55",
    )


def _day2_source(chat_id: str = "oc_eval_1") -> SessionSource:
    """Day-2 source: a reply inside the same topic."""
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_type="group",
        thread_id="om_topic_root",  # SAME topic
        parent_chat_id="om_day2_user_msg",
        user_id="ou_11d6fe7f5f52d4053ede9f47c655fc55",
    )


def _format_prompt(memory: SessionMemory) -> str:
    """Mirror the prompt block the Feishu adapter would inject."""
    if memory is None:
        return ""
    parts = ["<session_memory>"]
    if memory.topic:
        parts.append(f"topic: {memory.topic}")
    if memory.project:
        parts.append(f"project: {memory.project}")
    if memory.session_summary:
        parts.append(f"summary: {memory.session_summary}")
    ts = memory.current_task_state
    if ts:
        parts.append("current_task_state:")
        parts.append(f"  - status: {ts.status}")
        if ts.last_user_intent:
            parts.append(f"  - last_user_intent: {ts.last_user_intent}")
        for p in (ts.last_agent_proposal or [])[:6]:
            parts.append(f"  - last_agent_proposal: {p}")
        if ts.next_action:
            parts.append(f"  - next_action: {ts.next_action}")
    if memory.open_todos:
        parts.append("open_todos:")
        for t in memory.open_todos[:8]:
            if isinstance(t, dict):
                parts.append(f"  - {t.get('id', '?')}: {t.get('content', '')}")
            else:
                parts.append(f"  - {t}")
    if memory.important_decisions:
        parts.append("important_decisions:")
        for d in memory.important_decisions[:8]:
            if isinstance(d, dict):
                parts.append(f"  - {d.get('id', '?')}: {d.get('content', '')}")
            else:
                parts.append(f"  - {d}")
    if memory.related_files_or_modules:
        parts.append("related_files_or_modules:")
        for f in memory.related_files_or_modules[:6]:
            parts.append(f"  - {f}")
    parts.append("</session_memory>")
    return "\n".join(parts)


class Day1Day2Acceptance(unittest.TestCase):
    """Walks the BOSS spec scenario end-to-end at the memory layer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        set_memory_dir(Path(self.tmp.name))

    # ---------------------------------------------------------- Day 1
    def test_day1_creates_topic_keyed_memory(self):
        src = _day1_source()
        key = feishu_thread_session_key(
            chat_id=src.chat_id,
            thread_id=src.thread_id,
            parent_message_id=None,
            user_id=src.user_id,
            chat_type=src.chat_type,
        )
        self.assertIn("thread:om_topic_root", key)
        # Day 1 turns create the memory.
        save_session_memory(SessionMemory(
            session_key=key,
            platform="feishu",
            chat_id=src.chat_id,
            thread_id=src.thread_id,
            project="alphaseek 评分卡",
            topic="评分卡 Top-3 修复",
            session_summary=(
                "BOSS 要求设计评分卡修复方案。Hermes 给出三项修复，"
                "等待 BOSS 确认是否开工。"
            ),
            current_task_state=TaskState(
                status="waiting_for_user_confirmation",
                last_user_intent="设计评分卡修复方案",
                last_agent_proposal=DAY1_PROPOSAL,
                next_action="等待用户回复是否开工",
            ),
            open_todos=[
                {"id": "f1", "content": DAY1_PROPOSAL[0]},
                {"id": "f2", "content": DAY1_PROPOSAL[1]},
                {"id": "f3", "content": DAY1_PROPOSAL[2]},
            ],
            important_decisions=[
                {"id": "d1", "content": "采用 vol_20d 校准 _calc_3scenarios"},
            ],
            related_files_or_modules=DAY1_FILES,
        ))
        # File exists — load it back to confirm round-trip.
        loaded = load_session_memory(key)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.topic, "评分卡 Top-3 修复")

    # ---------------------------------------------------------- Day 2
    def test_day2_recovers_memory_via_same_topic(self):
        # Re-run Day 1's save (idempotent for the test).
        self.test_day1_creates_topic_keyed_memory()

        # Day 2 — same chat, same topic, but a fresh process / restart.
        set_memory_dir(Path(self.tmp.name))  # simulate fresh start
        src = _day2_source()
        key = feishu_thread_session_key(
            chat_id=src.chat_id,
            thread_id=src.thread_id,
            parent_message_id=src.parent_chat_id,
            user_id=src.user_id,
            chat_type=src.chat_type,
        )
        # KEY MUST MATCH — this is the whole point of the spec.
        self.assertIn("thread:om_topic_root", key)

        memory = load_session_memory(key)
        self.assertIsNotNone(memory, "Day-2 memory must load from Day-1 file")
        self.assertEqual(memory.current_task_state.status,
                         "waiting_for_user_confirmation")
        self.assertEqual(memory.current_task_state.last_agent_proposal,
                         DAY1_PROPOSAL)

    # ---------------------------------------------------------- Day 2
    def test_day2_prompt_contains_three_fixes(self):
        self.test_day2_recovers_memory_via_same_topic()

        src = _day2_source()
        key = feishu_thread_session_key(
            chat_id=src.chat_id,
            thread_id=src.thread_id,
            parent_message_id=src.parent_chat_id,
            user_id=src.user_id,
            chat_type=src.chat_type,
        )
        memory = load_session_memory(key)
        prompt = _format_prompt(memory)
        # All three fix items must appear in the injected context.
        for kw in ("_calc_3scenarios", "vol_20d",
                   "refined_ranking", "rank1",
                   "S/A/B/C/D"):
            self.assertIn(kw, prompt, msg=f"missing keyword: {kw}")
        # The agent must NOT have to ask "修什么" — the prompt already
        # contains the proposal, the todos, and the next action.
        self.assertIn("last_agent_proposal:", prompt)
        self.assertIn("open_todos:", prompt)
        self.assertIn("related_files_or_modules:", prompt)

    # ---------------------------------------------------------- Day 2 confirm
    def test_day2_user_confirm_flips_status(self):
        from gateway.session_memory import detect_task_status_from_text

        new_status = detect_task_status_from_text(
            user_text="可以，开始修这三项",
            agent_text="",
        )
        self.assertEqual(new_status, "in_progress")

    # ---------------------------------------------------------- key stability
    def test_same_topic_same_key_across_message_ids(self):
        # 100 different message_ids in the same topic → 1 session_key.
        keys = set()
        for i in range(100):
            src = SessionSource(
                platform=Platform.FEISHU,
                chat_id="oc_eval_1",
                chat_type="group",
                thread_id="om_topic_root",  # same
                parent_chat_id=f"om_msg_{i}",  # different!
                user_id="ou_11d6fe7f5f52d4053ede9f47c655fc55",
            )
            keys.add(feishu_thread_session_key(
                chat_id=src.chat_id,
                thread_id=src.thread_id,
                parent_message_id=src.parent_chat_id,
                user_id=src.user_id,
                chat_type=src.chat_type,
            ))
        # All collapse to one — message_id does NOT fragment the session.
        self.assertEqual(len(keys), 1)


if __name__ == "__main__":
    unittest.main()
