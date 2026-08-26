import importlib
import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class OpenAIHeadersTest(unittest.TestCase):
    def setUp(self):
        sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())
        sys.modules.setdefault(
            "aiohttp",
            types.SimpleNamespace(ClientSession=object, ClientTimeout=object),
        )
        sys.modules.setdefault("requests", types.SimpleNamespace(post=None, get=None))
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        sys.modules.pop("core.openai", None)
        self.manager = importlib.import_module("core.openai").OpenAIManager
        self.manager._api_key = ""
        self.manager._session_key = "agent:default:open-xiaoai-bridge"

    def test_default_sends_hermes_session_header(self):
        """Default config targets Hermes: session_key goes out as the header."""
        self.assertEqual(
            "agent:default:open-xiaoai-bridge",
            self.manager._headers()["X-Hermes-Session-Key"],
        )

    def test_empty_session_header_disables_it(self):
        """Setting session_header empty keeps requests header-free (plain OpenAI)."""
        self.manager._session_header = ""
        self.assertEqual({"Content-Type": "application/json"}, self.manager._headers())

    def test_session_header_sent_when_configured(self):
        self.manager._session_header = "X-Hermes-Session-Key"
        headers = self.manager._headers()
        self.assertEqual(
            "agent:default:open-xiaoai-bridge",
            headers["X-Hermes-Session-Key"],
        )

    def test_session_header_omitted_when_session_key_empty(self):
        self.manager._session_header = "X-Hermes-Session-Key"
        self.manager._session_key = ""
        self.assertNotIn("X-Hermes-Session-Key", self.manager._headers())

    def test_bearer_auth_and_session_header_coexist(self):
        self.manager._api_key = "secret"
        self.manager._session_header = "X-Hermes-Session-Key"
        headers = self.manager._headers()
        self.assertEqual("Bearer secret", headers["Authorization"])
        self.assertEqual(
            "agent:default:open-xiaoai-bridge",
            headers["X-Hermes-Session-Key"],
        )

    def test_provider_error_summary_redacts_credentials(self):
        summary = self.manager._error_summary(
            {"error": {"type": "auth", "message": "Bearer secret-token rejected"}}
        )
        self.assertEqual("auth: Bearer <redacted> rejected", summary)

    def test_empty_success_response_has_diagnostic_shape(self):
        summary = self.manager._empty_response_summary(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": ""},
                    }
                ]
            }
        )
        self.assertIn("finish_reason='length'", summary)
        self.assertIn("'content'", summary)


class OpenAITimeoutTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        self.manager = importlib.import_module("core.openai").OpenAIManager
        self.manager._response_events = {}
        self.manager._response_tasks = {}
        self.manager._response_texts = {}
        self.manager._response_tts_speakers = {}
        self.manager.last_error = None

    async def asyncTearDown(self):
        await self.manager.close()

    async def test_tool_timeout_returns_diagnostic_and_cancels_call(self):
        cancelled = asyncio.Event()

        async def slow_tool(_name, _args):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        fake_mcp = types.SimpleNamespace(
            MCPClientManager=types.SimpleNamespace(call_tool=slow_tool)
        )
        self.manager._tool_timeout = 0.01
        tool_call = {
            "function": {"name": "slow_tool", "arguments": "{}"}
        }

        with patch.dict(sys.modules, {"core.mcp_client": fake_mcp}):
            result = await self.manager._run_tool_call(tool_call)

        self.assertIn("工具调用超时", result)
        self.assertTrue(cancelled.is_set())

    async def test_response_timeout_cancels_background_chat_task(self):
        run_id = "slow-run"
        event = asyncio.get_running_loop().create_future()
        cancelled = asyncio.Event()

        async def slow_chat():
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        task = asyncio.create_task(slow_chat())
        self.manager._timeout = 0.01
        self.manager._response_events[run_id] = event
        self.manager._response_tasks[run_id] = task
        self.manager._response_texts[run_id] = ""

        result = await self.manager._wait_response(run_id)

        self.assertIsNone(result)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(task.cancelled())
        self.assertEqual("TimeoutError: response exceeded 0.01s", self.manager.last_error)
        self.assertNotIn(run_id, self.manager._response_tasks)
        self.assertNotIn(run_id, self.manager._response_events)

if __name__ == "__main__":
    unittest.main()
