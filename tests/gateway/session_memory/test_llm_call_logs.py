"""Unit tests for gateway.llm_call_logs."""

import tempfile
import time
import unittest
from pathlib import Path

from gateway.llm_call_logs import (
    LLMCallRecord,
    RESOLUTION_SOURCE_VALUES,
    SCHEMA_VERSION,
    _extract_actual_model,
    _extract_usage,
    aggregate_by_period,
    clear,
    get_db_path,
    query_recent,
    record_call,
    set_db_path,
)


class TestRecordAndQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        set_db_path(Path(self.tmp.name) / "logs.db")

    def test_round_trip(self):
        rec = LLMCallRecord(
            session_key="agent:main:feishu:group:oc_1:thread:om_t1",
            user_id="ou_boss", chat_id="oc_1", thread_id="om_t1",
            agent_name="main",
            requested_model="deepseek-v4-pro",
            requested_provider="openrouter",
            resolved_model="deepseek/deepseek-v4-pro",
            resolved_provider="openrouter",
            actual_model="deepseek/deepseek-v4-pro",
            actual_provider="openrouter",
            model_resolution_source="session_model",
            input_tokens=100, output_tokens=200, total_tokens=300,
            cost_usd=0.001, latency_ms=1500,
        )
        self.assertTrue(record_call(rec))
        rows = query_recent(session_key=rec.session_key, limit=10)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["session_key"], rec.session_key)
        self.assertEqual(r["actual_model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(r["model_resolution_source"], "session_model")
        self.assertEqual(r["input_tokens"], 100)
        self.assertEqual(r["cache_hit"], False)
        self.assertEqual(r["fallback_used"], False)

    def test_fallback_round_trip(self):
        rec = LLMCallRecord(
            session_key="k1",
            requested_model="deepseek-v4-pro",
            actual_model="MiniMax-M3",
            fallback_used=True,
            fallback_reason="openrouter timeout",
        )
        record_call(rec)
        rows = query_recent(session_key="k1", limit=5)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["fallback_used"])
        self.assertEqual(rows[0]["fallback_reason"], "openrouter timeout")
        self.assertEqual(rows[0]["actual_model"], "MiniMax-M3")

    def test_filter_by_user(self):
        record_call(LLMCallRecord(session_key="k1", user_id="u_a"))
        record_call(LLMCallRecord(session_key="k2", user_id="u_b"))
        rows_a = query_recent(user_id="u_a", limit=10)
        rows_b = query_recent(user_id="u_b", limit=10)
        self.assertEqual(len(rows_a), 1)
        self.assertEqual(len(rows_b), 1)
        self.assertEqual(rows_a[0]["session_key"], "k1")
        self.assertEqual(rows_b[0]["session_key"], "k2")

    def test_query_recent_newest_first(self):
        for i in range(3):
            time.sleep(0.001)
            record_call(LLMCallRecord(
                session_key="k1",
                actual_model=f"m{i}",
                created_at=time.time() + i * 0.01,
            ))
        rows = query_recent(session_key="k1", limit=5)
        self.assertEqual(len(rows), 3)
        # Newest first
        self.assertEqual(rows[0]["actual_model"], "m2")
        self.assertEqual(rows[2]["actual_model"], "m0")


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        set_db_path(Path(self.tmp.name) / "logs.db")

    def test_totals_match_inserts(self):
        for i in range(5):
            record_call(LLMCallRecord(
                session_key="k1",
                actual_model="m1", actual_provider="p1",
                input_tokens=100, output_tokens=50, total_tokens=150,
                cost_usd=0.001,
            ))
        agg = aggregate_by_period(period="all")
        self.assertEqual(agg["total_requests"], 5)
        self.assertEqual(agg["total_input_tokens"], 500)
        self.assertEqual(agg["total_output_tokens"], 250)
        self.assertEqual(agg["total_tokens"], 750)
        self.assertAlmostEqual(agg["estimated_cost_usd"], 0.005)

    def test_breakdown_by_model(self):
        record_call(LLMCallRecord(
            session_key="k1", actual_model="m1", actual_provider="p1",
            input_tokens=10, output_tokens=20, total_tokens=30,
            cost_usd=0.0001,
        ))
        record_call(LLMCallRecord(
            session_key="k1", actual_model="m2", actual_provider="p1",
            input_tokens=40, output_tokens=50, total_tokens=90,
            cost_usd=0.0002,
        ))
        agg = aggregate_by_period(period="all")
        by_model = agg["by_model"]
        self.assertEqual(len(by_model), 2)
        # Sorted by requests desc — both 1, so insertion order
        self.assertIn(by_model[0]["model"], ("m1", "m2"))
        # Provider breakdown — same provider, so 2 rows merged
        self.assertEqual(len(agg["by_provider"]), 1)
        self.assertEqual(agg["by_provider"][0]["requests"], 2)

    def test_session_filter(self):
        record_call(LLMCallRecord(session_key="k1", input_tokens=10))
        record_call(LLMCallRecord(session_key="k2", input_tokens=99))
        agg = aggregate_by_period(period="all", session_key="k1")
        self.assertEqual(agg["total_requests"], 1)
        self.assertEqual(agg["total_input_tokens"], 10)

    def test_empty_db_returns_zeros(self):
        agg = aggregate_by_period(period="today")
        self.assertEqual(agg["total_requests"], 0)
        self.assertEqual(agg["total_tokens"], 0)
        self.assertEqual(agg["estimated_cost_usd"], 0.0)
        self.assertEqual(agg["by_model"], [])

    def test_period_today_only_counts_today(self):
        # Insert a record 2 days ago — should not count for "today"
        from gateway.llm_call_logs import _get_conn
        conn = _get_conn()
        old_time = time.time() - 2 * 24 * 3600
        record_call(LLMCallRecord(
            session_key="k1", input_tokens=999, created_at=old_time,
        ))
        record_call(LLMCallRecord(
            session_key="k1", input_tokens=1, created_at=time.time(),
        ))
        agg = aggregate_by_period(period="today")
        # The "today" filter should exclude the old record.
        self.assertEqual(agg["total_input_tokens"], 1)
        agg_all = aggregate_by_period(period="all")
        self.assertEqual(agg_all["total_input_tokens"], 1000)


class TestExtractHelpers(unittest.TestCase):
    def test_extract_usage_openai(self):
        from types import SimpleNamespace
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                prompt_tokens_details=SimpleNamespace(cached_tokens=50),
            ),
            model="m1",
        )
        inp, out, tot, cache_hit, cache_read, cache_write = _extract_usage(resp)
        self.assertEqual(inp, 100)
        self.assertEqual(out, 200)
        self.assertEqual(tot, 300)
        self.assertTrue(cache_hit)
        self.assertEqual(cache_read, 50)

    def test_extract_usage_anthropic(self):
        from types import SimpleNamespace
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=80,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=40,
            ),
            model="claude-sonnet-4.6",
        )
        inp, out, tot, cache_hit, cache_read, cache_write = _extract_usage(resp)
        self.assertEqual(inp, 120)
        self.assertEqual(out, 80)
        self.assertEqual(tot, 200)
        self.assertTrue(cache_hit)
        self.assertEqual(cache_read, 40)
        self.assertEqual(cache_write, 10)

    def test_extract_actual_model(self):
        from types import SimpleNamespace
        self.assertEqual(
            _extract_actual_model(SimpleNamespace(model="m1")), "m1",
        )
        self.assertIsNone(_extract_actual_model(None))
        self.assertIsNone(_extract_actual_model(SimpleNamespace()))


class TestClear(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        set_db_path(Path(self.tmp.name) / "logs.db")

    def test_clear_one_session(self):
        record_call(LLMCallRecord(session_key="k1"))
        record_call(LLMCallRecord(session_key="k2"))
        deleted = clear(session_key="k1")
        self.assertEqual(deleted, 1)
        self.assertEqual(len(query_recent(session_key="k1")), 0)
        self.assertEqual(len(query_recent(session_key="k2")), 1)

    def test_clear_all(self):
        record_call(LLMCallRecord(session_key="k1"))
        record_call(LLMCallRecord(session_key="k2"))
        deleted = clear()
        self.assertEqual(deleted, 2)


class TestSchema(unittest.TestCase):
    def test_resolution_source_values(self):
        for v in (
            "message_override", "session_model", "thread_model",
            "chat_model", "user_default", "agent_default",
            "system_default",
        ):
            self.assertIn(v, RESOLUTION_SOURCE_VALUES)

    def test_schema_version_is_int(self):
        self.assertIsInstance(SCHEMA_VERSION, int)


class TestFailureTolerance(unittest.TestCase):
    def test_record_call_with_db_unavailable(self):
        # Point to a path that cannot be created.
        set_db_path(Path("/this/path/should/not/exist/db.db"))
        # record_call must NOT raise, just return False.
        ok = record_call(LLMCallRecord(session_key="k1"))
        self.assertFalse(ok)
        # Recover for other tests
        with tempfile.TemporaryDirectory() as tmp:
            set_db_path(Path(tmp) / "ok.db")

    def test_query_with_db_unavailable(self):
        set_db_path(Path("/nonexistent/db.db"))
        rows = query_recent(session_key="anything")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
