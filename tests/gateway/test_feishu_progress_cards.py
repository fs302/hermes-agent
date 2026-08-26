import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _ok(message_id="om_progress"):
    return SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(message_id=message_id),
    )


def test_feishu_progress_and_final_cards_use_their_supported_update_paths():
    adapter = FeishuAdapter(
        PlatformConfig(extra={"collapse_progress": True})
    )
    adapter._client = object()
    adapter._feishu_send_with_retry = AsyncMock(return_value=_ok())

    result = asyncio.run(
        adapter.send(
            "oc_chat",
            "🔎 Inspecting repository",
            metadata={"_hermes_progress": True},
        )
    )
    assert result.success
    progress_call = adapter._feishu_send_with_retry.await_args.kwargs
    assert progress_call["msg_type"] == "interactive"
    progress_card = json.loads(progress_call["payload"])
    assert progress_card["schema"] == "2.0"
    assert progress_card["config"]["update_multi"] is True
    assert progress_card["body"]["elements"][0]["tag"] == "collapsible_panel"

    patch_call = object()
    update_call = object()
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(
            v1=SimpleNamespace(
                message=SimpleNamespace(patch=patch_call, update=update_call)
            )
        )
    )
    adapter._run_blocking = AsyncMock(return_value=_ok())
    edited = asyncio.run(
        adapter.edit_message("oc_chat", "om_progress", "🔎 Inspecting\n🛠 Running")
    )
    assert edited.success
    assert adapter._run_blocking.await_args.args[0] is patch_call

    adapter._feishu_send_with_retry.reset_mock()
    adapter._feishu_send_with_retry.return_value = _ok("om_notice")
    notice = asyncio.run(adapter.send("oc_chat", "MiniMax-M3"))
    assert notice.success
    assert adapter._feishu_send_with_retry.await_args.kwargs["msg_type"] == "text"

    adapter._feishu_send_with_retry.reset_mock()
    adapter._feishu_send_with_retry.return_value = _ok("om_final")
    final = asyncio.run(
        adapter.send("oc_chat", "Done", metadata={"notify": True})
    )
    assert final.success
    final_call = adapter._feishu_send_with_retry.await_args.kwargs
    assert final_call["msg_type"] == "interactive"
    final_card = json.loads(final_call["payload"])
    assert final_card["schema"] == "2.0"
    assert final_card["body"]["elements"] == [
        {"tag": "markdown", "content": "Done"}
    ]
    assert adapter.prefers_fresh_final_streaming("Done") is True
