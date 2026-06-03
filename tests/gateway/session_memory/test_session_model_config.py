"""Unit tests for gateway.session_model_config."""

import tempfile
import unittest
from pathlib import Path

import gateway.session_memory as sm_mod
from gateway.session_model_config import (
    SessionModelConfig,
    clear_session_model_config,
    get_session_model_config,
    set_session_model_config,
)


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        sm_mod.set_memory_dir(Path(self.tmp.name))

    def test_set_then_get(self):
        cfg = SessionModelConfig(
            model="deepseek-v4-pro",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            api_mode="chat_completions",
        )
        ok = set_session_model_config(
            "agent:main:feishu:group:oc_1:thread:om_t1", cfg,
        )
        self.assertTrue(ok)
        loaded = get_session_model_config(
            "agent:main:feishu:group:oc_1:thread:om_t1",
        )
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.model, "deepseek-v4-pro")
        self.assertEqual(loaded.provider, "openrouter")
        self.assertEqual(loaded.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(loaded.api_key, "sk-test")
        self.assertEqual(loaded.api_mode, "chat_completions")

    def test_get_none_for_missing_session(self):
        self.assertIsNone(get_session_model_config("never:seen"))

    def test_set_empty_session_key_returns_false(self):
        cfg = SessionModelConfig(model="m1")
        self.assertFalse(set_session_model_config("", cfg))

    def test_set_empty_model_returns_false(self):
        cfg = SessionModelConfig(model="")
        self.assertFalse(set_session_model_config("k1", cfg))

    def test_clear_removes_override(self):
        cfg = SessionModelConfig(model="m1", provider="p1")
        set_session_model_config("k1", cfg)
        # Now clear
        self.assertTrue(clear_session_model_config("k1"))
        self.assertIsNone(get_session_model_config("k1"))

    def test_clear_when_nothing_set_returns_false(self):
        self.assertFalse(clear_session_model_config("never:existed"))

    def test_clear_empty_session_key_returns_false(self):
        self.assertFalse(clear_session_model_config(""))

    def test_does_not_disturb_other_fields(self):
        # Pre-populate other fields
        sm_mod.update_session_memory(
            "k1",
            topic="评分卡",
            session_summary="一些历史",
            open_todos=[{"id": "t1", "content": "改 _calc_3scenarios"}],
        )
        set_session_model_config("k1", SessionModelConfig(
            model="m1", provider="p1",
        ))
        # Re-read and confirm the other fields survived.
        memory = sm_mod.load_session_memory("k1")
        self.assertIsNotNone(memory)
        self.assertEqual(memory.topic, "评分卡")
        self.assertEqual(memory.session_summary, "一些历史")
        self.assertEqual(memory.open_todos, [
            {"id": "t1", "content": "改 _calc_3scenarios"}
        ])
        self.assertEqual(memory.model_config, {
            "model": "m1", "provider": "p1",
            "base_url": "", "api_key": "", "api_mode": "",
            "requested_at": memory.model_config.get("requested_at", 0.0),
        })


class TestFromDict(unittest.TestCase):
    def test_from_dict_empty(self):
        self.assertIsNone(SessionModelConfig.from_dict(None))
        self.assertIsNone(SessionModelConfig.from_dict({}))
        self.assertIsNone(SessionModelConfig.from_dict({"model": ""}))

    def test_from_dict_partial(self):
        cfg = SessionModelConfig.from_dict({
            "model": "m1", "provider": "p1",
        })
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.model, "m1")
        self.assertEqual(cfg.provider, "p1")
        self.assertEqual(cfg.base_url, "")

    def test_to_dict_round_trip(self):
        cfg = SessionModelConfig(
            model="m1", provider="p1", base_url="u", api_key="k",
            api_mode="a", requested_at=12345.0,
        )
        d = cfg.to_dict()
        self.assertEqual(d["model"], "m1")
        self.assertEqual(d["provider"], "p1")
        self.assertEqual(d["base_url"], "u")
        self.assertEqual(d["api_key"], "k")
        self.assertEqual(d["api_mode"], "a")
        self.assertEqual(d["requested_at"], 12345.0)


if __name__ == "__main__":
    unittest.main()
