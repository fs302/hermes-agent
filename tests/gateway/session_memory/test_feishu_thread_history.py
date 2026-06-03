"""Unit tests for the Feishu thread-history fallback dispatcher.

These tests run the dispatcher against a fake ``lark_oapi.Client`` so we
can exercise the three-tier fallback logic without hitting the real
Feishu API.
"""

import asyncio
import unittest
from typing import Any, List, Optional

from gateway.platforms.feishu_thread_history import (
    ThreadHistoryResult,
    ThreadMessage,
    _coerce_message,
    _extract_text,
    fetch_thread_history,
)


class FakeListResponse:
    def __init__(
        self, success: bool, items: Optional[List[Any]] = None,
        code: str = "0", msg: str = "ok", has_more: bool = False,
        page_token: Optional[str] = None,
    ):
        self.success = (lambda: success)
        self.code = code
        self.msg = msg
        self._data = _FakeData(items or [], has_more, page_token)

    @property
    def data(self) -> Any:
        return self._data


class _FakeData:
    def __init__(self, items: List[Any], has_more: bool, page_token: Optional[str]):
        self.items = items
        self.has_more = has_more
        self.page_token = page_token


class _FakeMessageBody:
    def __init__(self, content: str):
        self.content = content


class _FakeMessage:
    def __init__(
        self, message_id: str, content: str, msg_type: str = "text",
        sender_id: str = "ou_user_1", create_time: str = "1700000000000",
    ):
        self.message_id = message_id
        self.body = _FakeMessageBody(content)
        self.msg_type = msg_type
        self.create_time = create_time
        sender = type("S", (), {})()
        sender.id = sender_id
        self.sender = sender
        self.mentions = None


class FakeClient:
    """Mimics lark_oapi.Client.im.v1.message.list."""

    def __init__(self, *, thread_response=None, parent_response=None, chat_response=None):
        self._thread_response = thread_response
        self._parent_response = parent_response
        self._chat_response = chat_response
        self.calls: List[dict] = []

    class _IM:
        def __init__(self, outer: "FakeClient"):
            self._outer = outer
            self.v1 = type("V1", (), {"message": type("M", (), {"list": self._outer._list})()})()

        def list(self, *args, **kwargs):  # pragma: no cover - never reached
            return self._outer._list(*args, **kwargs)

    def _list(self, request):
        # Inspect the request to figure out which tier we are serving.
        # The request has builder attributes, but the easiest is to look
        # at the URL queries, which the SDK builds from container_id_type.
        cid_type = None
        cid = None
        for attr in ("container_id_type",):
            try:
                cid_type = getattr(request, attr, None)
            except Exception:
                pass
        # Fallback: introspect via __dict__
        for k, v in getattr(request, "__dict__", {}).items():
            if "container" in k:
                if "type" in k and not cid_type:
                    cid_type = v
                elif "id" in k and not cid:
                    cid = v
        # Record the call for assertions.
        self.calls.append({"cid_type": cid_type, "cid": cid})
        if cid_type == "thread":
            return self._thread_response
        if cid_type == "message":
            return self._parent_response
        if cid_type == "chat":
            return self._chat_response
        # Default: thread
        return self._thread_response

    @property
    def im(self):
        # Build a callable namespace that ultimately calls our _list.
        client = self

        class _V1:
            class _M:
                @staticmethod
                def list(req):
                    return client._list(req)

            message = _M

        class _IM:
            v1 = _V1

        return _IM()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestCoerceMessage(unittest.TestCase):
    def test_text_message(self):
        raw = _FakeMessage(
            message_id="om_1",
            content='{"text": "hello world"}',
            msg_type="text",
        )
        m = _coerce_message(raw, bot_open_id="ou_bot")
        self.assertIsNotNone(m)
        self.assertEqual(m.message_id, "om_1")
        self.assertEqual(m.text, "hello world")
        self.assertEqual(m.create_time_ms, 1700000000000)
        self.assertFalse(m.is_from_bot)

    def test_bot_sender_flag(self):
        raw = _FakeMessage(
            message_id="om_2",
            content='{"text": "ok"}',
            sender_id="ou_bot",
        )
        m = _coerce_message(raw, bot_open_id="ou_bot")
        self.assertTrue(m.is_from_bot)


class TestExtractText(unittest.TestCase):
    def test_text_body(self):
        raw = _FakeMessage(
            message_id="om_1",
            content='{"text": "hi"}',
            msg_type="text",
        )
        self.assertEqual(_extract_text(raw), "hi")

    def test_empty_body(self):
        raw = _FakeMessage(message_id="om_1", content="", msg_type="text")
        self.assertEqual(_extract_text(raw), "")


class TestThreeTierFallback(unittest.TestCase):
    def test_tier1_thread_succeeds(self):
        thread_items = [
            _FakeMessage("om_a", '{"text": "old message"}'),
            _FakeMessage("om_b", '{"text": "newer message"}'),
        ]
        thread_resp = FakeListResponse(
            success=True, items=thread_items, has_more=False,
        )
        client = FakeClient(thread_response=thread_resp)

        result = run(fetch_thread_history(
            client, thread_id="om_root", parent_message_id="om_root",
            chat_id="oc_1",
        ))
        self.assertTrue(result.ok)
        self.assertEqual(result.tier_used, "thread")
        self.assertEqual(len(result.messages), 2)
        # Only one call should have been made.
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["cid_type"], "thread")

    def test_tier1_fails_tier2_succeeds(self):
        thread_resp = FakeListResponse(
            success=False, code="-1", msg="thread not found",
        )
        parent_items = [_FakeMessage("om_p", '{"text": "reply"}')]
        parent_resp = FakeListResponse(success=True, items=parent_items)
        client = FakeClient(
            thread_response=thread_resp, parent_response=parent_resp,
        )

        result = run(fetch_thread_history(
            client, thread_id="om_t", parent_message_id="om_p",
            chat_id="oc_1",
        ))
        self.assertTrue(result.ok)
        self.assertEqual(result.tier_used, "parent")
        self.assertEqual(len(result.messages), 1)
        # Two calls made (tier1 fail, tier2 success).
        self.assertEqual(len(client.calls), 2)

    def test_all_tiers_fail(self):
        thread_resp = FakeListResponse(success=False, code="-1", msg="x")
        parent_resp = FakeListResponse(success=False, code="-1", msg="x")
        chat_resp = FakeListResponse(success=False, code="-1", msg="x")
        client = FakeClient(
            thread_response=thread_resp,
            parent_response=parent_resp,
            chat_response=chat_resp,
        )
        result = run(fetch_thread_history(
            client, thread_id="om_t", parent_message_id="om_p",
            chat_id="oc_1",
        ))
        self.assertFalse(result.ok)
        self.assertEqual(result.tier_used, "chat")
        self.assertIn("rejected", result.error)

    def test_no_identifiers(self):
        client = FakeClient()
        result = run(fetch_thread_history(client))
        self.assertFalse(result.ok)
        self.assertIn("no identifiers", result.error)


class TestTranscriptRendering(unittest.TestCase):
    def test_transcript_includes_timestamps_and_sender(self):
        items = [
            ThreadMessage(
                message_id="m1", sender_id="u1", sender_name="alice",
                text="hello", create_time_ms=1700000000000,
            ),
            ThreadMessage(
                message_id="m2", sender_id="u2", sender_name="bob",
                text="hi there", create_time_ms=1700000060000,
            ),
        ]
        result = ThreadHistoryResult(messages=items, tier_used="thread")
        transcript = result.to_transcript()
        self.assertIn("alice", transcript)
        self.assertIn("hello", transcript)
        self.assertIn("bob", transcript)
        self.assertIn("hi there", transcript)

    def test_transcript_truncates_long_text(self):
        items = [
            ThreadMessage(
                message_id="m1", sender_id="u1", sender_name="alice",
                text="x" * 1000, create_time_ms=1700000000000,
            ),
        ]
        result = ThreadHistoryResult(messages=items, tier_used="thread")
        transcript = result.to_transcript(max_chars_per_message=100)
        # Should be truncated to 99 chars + "…"
        for line in transcript.splitlines():
            content = line.split(": ", 1)[1] if ": " in line else line
            self.assertLessEqual(len(content), 100)


if __name__ == "__main__":
    unittest.main()
