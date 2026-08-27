"""OpenAI function calling 工具循环测试（桩模式，无需真实后端）"""

import asyncio
import importlib
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeAiohttpSession:
    """aiohttp.ClientSession 桩：作为 async CM 使用"""

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeMCPServer:
    """MCP client 桩：记录 call_tool 调用"""

    def __init__(self, tools=None):
        self.tools = tools or [
            {"type": "function", "function": {"name": "weather", "description": "查天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
        ]
        self.call_history = []

    def get_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        self.call_history.append((name, arguments))
        if name == "weather":
            return "北京晴天 25 度"
        if name == "play_track":
            result = str.__new__(ToolResult, "正在播放：测试歌曲")
            result.terminate_silently = True
            return result
        if name == "play_failed":
            return "[MCPClient] 工具 play_track 返回错误: 播放服务不可用"
        if name == "slow_tool":
            await asyncio.sleep(0.05)
            return "slow result"
        return f"{name} 执行成功"


class ToolResult(str):
    terminate_silently = False


def make_tool_call_response(name, arguments, tool_call_id="call_1"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def make_text_response(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class OpenAIToolLoopTest(unittest.TestCase):
    def setUp(self):
        # 桩掉外部依赖（照抄 test_openai.py 模式）
        sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())
        sys.modules.setdefault(
            "aiohttp",
            types.SimpleNamespace(
                ClientSession=FakeAiohttpSession,
                ClientTimeout=lambda **kwargs: None,
            ),
        )
        # 桩掉 core.mcp_client（openai.py 内部延迟导入）
        self.mcp_stub = FakeMCPServer()
        mcp_mod = types.ModuleType("core.mcp_client")
        mcp_mod.MCPClientManager = self.mcp_stub
        sys.modules["core.mcp_client"] = mcp_mod

        sys.modules.pop("core.openai", None)
        self.manager = importlib.import_module("core.openai").OpenAIManager

        # 重置状态
        self.manager._sessions = {}
        self.manager._model = "test-model"
        self.manager._api_key = ""
        self.manager._session_key = "test-session"
        self.manager._session_header = ""
        self.manager._system_prompt = ""
        self.manager._temperature = None
        self.manager._max_tokens = None
        self.manager._extra_body = {}
        self.manager._history_max_messages = 20
        self.manager._timeout = 30
        self.manager._use_mcp_tools = True
        self.manager._max_tool_rounds = 5

    def run_async(self, coro):
        return asyncio.run(coro)

    def _set_sequential_responses(self, responses):
        self.captured_payloads = []

        async def fake_post(cls, http_session, payload):
            self.captured_payloads.append(json.loads(json.dumps(payload)))
            return responses.pop(0)

        self.manager._post_chat_completion = classmethod(fake_post)

    def test_tool_loop_executes_and_returns_final_text(self):
        """工具调用 → 执行 → 携带 tool 消息重发 → 返回最终文本"""
        self._set_sequential_responses(
            [
                make_tool_call_response("weather", {"city": "北京"}),
                make_text_response("北京晴天 25 度"),
            ]
        )
        result = self.run_async(self.manager._request_chat_completion("今天天气怎么样？"))

        self.assertEqual(result, "北京晴天 25 度")

        # 第一轮 payload 含 tools
        first_payload = self.captured_payloads[0]
        self.assertIn("tools", first_payload)
        self.assertEqual(first_payload["tools"][0]["function"]["name"], "weather")

        # 第二轮 messages 含 assistant(tool_calls) + tool 消息
        second_messages = self.captured_payloads[1]["messages"]
        roles = [m["role"] for m in second_messages]
        self.assertEqual(roles[-3:], ["user", "assistant", "tool"])
        self.assertEqual(second_messages[-2]["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_1")
        self.assertEqual(second_messages[-1]["content"], "北京晴天 25 度")

        # 工具被调用
        self.assertEqual(self.mcp_stub.call_history, [("weather", {"city": "北京"})])

        # history 只存最终文本对（无 tool 消息残留）
        history = self.manager._sessions["test-session"]
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "今天天气怎么样？"},
                {"role": "assistant", "content": "北京晴天 25 度"},
            ],
        )

    def test_multiple_tool_calls_executed_concurrently(self):
        """同轮多个 tool_calls 并发执行"""
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city": "北京"}'},
                                },
                                {
                                    "id": "c2",
                                    "type": "function",
                                    "function": {"name": "slow_tool", "arguments": "{}"},
                                },
                            ],
                        }
                    }
                ]
            },
            make_text_response("都完成了"),
        ]
        self._set_sequential_responses(responses)
        result = self.run_async(self.manager._request_chat_completion("都查一下"))
        self.assertEqual(result, "都完成了")
        self.assertEqual(
            sorted(self.mcp_stub.call_history), sorted([("weather", {"city": "北京"}), ("slow_tool", {})])
        )

    def test_playback_success_skips_final_model_request(self):
        """播放成功信号立即静默终止，不再请求模型生成确认文本"""
        self._set_sequential_responses(
            [make_tool_call_response("play_track", {"query": "测试歌曲"})]
        )

        result = self.run_async(self.manager._request_chat_completion("播放测试歌曲"))

        self.assertIsNone(result)
        self.assertEqual(len(self.captured_payloads), 1)
        self.assertEqual(self.mcp_stub.call_history, [("play_track", {"query": "测试歌曲"})])
        self.assertEqual(self.manager._sessions["test-session"], [])

    def test_playback_failure_still_requests_audible_reply(self):
        """播放失败作为工具错误交给模型，保留最终可播报回复"""
        self._set_sequential_responses(
            [
                make_tool_call_response("play_failed", {}),
                make_text_response("播放失败，请稍后再试"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("播放测试歌曲"))

        self.assertEqual(result, "播放失败，请稍后再试")
        self.assertEqual(len(self.captured_payloads), 2)
        self.assertIn("播放服务不可用", self.captured_payloads[1]["messages"][-1]["content"])

    def test_ordinary_tool_does_not_end_turn_silently(self):
        """普通工具维持工具结果回传与最终模型请求"""
        self._set_sequential_responses(
            [make_tool_call_response("weather", {"city": "北京"}), make_text_response("晴天")]
        )

        result = self.run_async(self.manager._request_chat_completion("查天气"))

        self.assertEqual(result, "晴天")
        self.assertEqual(len(self.captured_payloads), 2)

    def test_bad_arguments_json_falls_back_to_empty(self):
        """坏 JSON 参数兜底为 {}"""
        self._set_sequential_responses(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "weather", "arguments": "{bad json"},
                                    }
                                ],
                            }
                        }
                    ]
                },
                make_text_response("处理完成"),
            ]
        )
        result = self.run_async(self.manager._request_chat_completion("查一下"))
        self.assertEqual(result, "处理完成")
        self.assertEqual(self.mcp_stub.call_history, [("weather", {})])

    def test_tool_loop_max_rounds_guard(self):
        """连续返回 tool_calls 时到达循环上限终止，不死循环"""
        responses = [make_tool_call_response("weather", {})] * 10  # 永远工具调用
        self._set_sequential_responses(responses)
        result = self.run_async(self.manager._request_chat_completion("一直调用"))
        self.assertIsNone(result)
        # 请求次数 = 上限 + 1（初始 + 每轮重发）
        self.assertEqual(len(self.captured_payloads), 6)
        # 工具被调了 5 次
        self.assertEqual(len(self.mcp_stub.call_history), 5)

    def test_no_tools_when_disabled(self):
        """use_mcp_tools=False 时 payload 无 tools 字段（回归）"""
        self.manager._use_mcp_tools = False
        self._set_sequential_responses([make_text_response("普通回复")])
        result = self.run_async(self.manager._request_chat_completion("你好"))
        self.assertEqual(result, "普通回复")
        self.assertNotIn("tools", self.captured_payloads[0])
        self.assertEqual(self.mcp_stub.call_history, [])


if __name__ == "__main__":
    unittest.main()
