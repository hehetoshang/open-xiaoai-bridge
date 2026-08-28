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

from core.tool_result import ToolCallResult


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
        self.results = {}

    def get_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        self.call_history.append((name, arguments))
        if name in self.results:
            return self.results[name]
        if name == "weather":
            return "北京晴天 25 度"
        if name == "play_track":
            return ToolCallResult(
                text="正在播放：测试歌曲",
                silent_end_turn=True,
            )
        if name == "play_failed":
            return ToolCallResult(
                text="[MCPClient] 工具 play_track 返回错误: 播放服务不可用",
                is_error=True,
            )
        if name == "slow_tool":
            await asyncio.sleep(0.05)
            return "slow result"
        return f"{name} 执行成功"


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
        self.manager._tool_timeout = 1
        self.manager._tool_confirmation_text = "好的，正在处理"
        self.manager._tool_confirmation_timeout = 1
        self.manager._use_mcp_tools = True
        self.manager._max_tool_rounds = 5
        self.manager._initialized = True
        self.manager._enabled = True
        self.manager._response_events = {}
        self.manager._response_tasks = {}
        self.manager._response_texts = {}
        self.manager._response_tts_speakers = {}
        self.tts_calls = []

        async def fake_tts(cls, text, **kwargs):
            self.tts_calls.append(text)
            return True

        self.manager._play_response_with_tts = classmethod(fake_tts)

    def run_async(self, coro):
        return asyncio.run(coro)

    def _set_sequential_responses(self, responses):
        self.captured_payloads = []

        async def fake_post(cls, http_session, payload):
            self.captured_payloads.append(json.loads(json.dumps(payload)))
            return responses.pop(0)

        self.manager._post_chat_completion = classmethod(fake_post)

    def test_default_prompt_keeps_music_as_an_optional_capability(self):
        """内置身份是通用助手，音乐工具仅响应明确的操作意图。"""
        prompt = self.manager._resolve_system_prompt("")

        self.assertEqual(prompt, self.manager._resolve_system_prompt(None))
        self.assertIn("通用 AI 语音助手", prompt)
        self.assertIn("外部工具只是可选能力", prompt)
        self.assertIn("音乐理论等知识问题不属于音乐操作", prompt)
        self.assertIn("搜索但不要播放", prompt)
        self.assertIn("不能擅自播放", prompt)
        self.assertIn("不要把自己称为音乐助手或播放器", prompt)
        self.assertIn("不要提及 MCP", prompt)

    def test_default_prompt_is_sent_with_ordinary_questions_and_tools(self):
        """普通问题也携带通用身份约束，但工具是否启用不受影响。"""
        self.manager._system_prompt = self.manager.DEFAULT_SYSTEM_PROMPT
        self._set_sequential_responses([make_text_response("四")])

        result = self.run_async(self.manager._request_chat_completion("二加二等于几？"))

        self.assertEqual(result, "四")
        payload = self.captured_payloads[0]
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(
            payload["messages"][0]["content"],
            self.manager.DEFAULT_SYSTEM_PROMPT,
        )
        self.assertIn("tools", payload)
        self.assertEqual(self.mcp_stub.call_history, [])
        self.assertEqual(self.tts_calls, [])

    def test_identity_question_stays_a_general_assistant_text_reply(self):
        """自我介绍直接回答，音乐只作为可选能力且不触发工具。"""
        self.manager._system_prompt = self.manager.DEFAULT_SYSTEM_PROMPT
        self._set_sequential_responses(
            [make_text_response("我是通用 AI 助手，也可以在需要时帮你播放音乐。")]
        )

        result = self.run_async(self.manager._request_chat_completion("你是谁？"))

        self.assertEqual(result, "我是通用 AI 助手，也可以在需要时帮你播放音乐。")
        self.assertEqual(self.mcp_stub.call_history, [])
        self.assertEqual(self.tts_calls, [])

    def test_music_theory_question_does_not_call_music_tools(self):
        """提到音乐但没有操作意图时正常解释，不调用或确认工具。"""
        self.manager._system_prompt = self.manager.DEFAULT_SYSTEM_PROMPT
        self._set_sequential_responses(
            [make_text_response("五度圈用于表示十二个调之间的关系。")]
        )

        result = self.run_async(self.manager._request_chat_completion("解释一下音乐里的五度圈"))

        self.assertEqual(result, "五度圈用于表示十二个调之间的关系。")
        self.assertEqual(self.mcp_stub.call_history, [])
        self.assertEqual(self.tts_calls, [])

    def test_search_without_playing_calls_search_only(self):
        """“只搜索”严格限制在查询，不会擅自追加播放调用。"""
        self.manager._system_prompt = self.manager.DEFAULT_SYSTEM_PROMPT
        self._set_sequential_responses(
            [
                make_tool_call_response("search_track", {"query": "周杰伦"}),
                make_text_response("找到多首周杰伦的歌曲。"),
            ]
        )

        result = self.run_async(
            self.manager._request_chat_completion("搜索周杰伦，但不要播放")
        )

        self.assertEqual(result, "找到多首周杰伦的歌曲。")
        self.assertEqual(
            self.mcp_stub.call_history,
            [("search_track", {"query": "周杰伦"})],
        )
        self.assertEqual(self.tts_calls, ["好的，正在处理"])

    def test_custom_prompt_can_override_general_assistant_default(self):
        """显式配置仍可完整覆盖内置提示词。"""
        custom_prompt = "你是家庭助理。"

        self.assertEqual(
            self.manager._resolve_system_prompt(f"  {custom_prompt}  "),
            custom_prompt,
        )

        messages = self.manager._build_messages([], "你好")
        self.assertEqual(messages, [{"role": "user", "content": "你好"}])

        self.manager._system_prompt = custom_prompt
        messages = self.manager._build_messages([], "你好")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": "你好"},
            ],
        )

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
        self.assertFalse(self.manager.is_silent_end_turn_result(result))

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
        self.assertEqual(self.tts_calls, ["好的，正在处理"])

    def test_confirmation_finishes_before_first_mcp_call(self):
        """确认语完整播完后才允许触发首个 MCP 调用"""
        events = []
        original_call_tool = self.mcp_stub.call_tool

        async def delayed_confirmation(cls, text, **kwargs):
            events.append("confirmation_started")
            await asyncio.sleep(0.01)
            events.append("confirmation_finished")
            return True

        async def ordered_call_tool(name, arguments):
            events.append("mcp_called")
            return await original_call_tool(name, arguments)

        self.manager._play_response_with_tts = classmethod(delayed_confirmation)
        self.mcp_stub.call_tool = ordered_call_tool
        self._set_sequential_responses(
            [
                make_tool_call_response("weather", {"city": "北京"}),
                make_text_response("北京晴天"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("查天气"))

        self.assertEqual(result, "北京晴天")
        self.assertEqual(
            events,
            ["confirmation_started", "confirmation_finished", "mcp_called"],
        )

    def test_sequential_tool_rounds_confirm_only_once(self):
        """搜索后播放等连续工具轮次只在第一次 MCP 前确认"""
        self._set_sequential_responses(
            [
                make_tool_call_response("weather", {"city": "北京"}, "search-1"),
                make_tool_call_response("play_track", {"query": "测试歌曲"}, "play-1"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("搜索并播放测试歌曲"))

        self.assertTrue(self.manager.is_silent_end_turn_result(result))
        self.assertEqual(self.tts_calls, ["好的，正在处理"])
        self.assertEqual(
            self.mcp_stub.call_history,
            [
                ("weather", {"city": "北京"}),
                ("play_track", {"query": "测试歌曲"}),
            ],
        )

    def test_verified_nonzero_native_tts_confirms_only_once(self):
        """脚本非零但播放完成时放行，同轮连续工具仍只确认一次。"""
        from core.ref import set_stream_player
        from core.services.speaker import CommandResult, SpeakerManager

        set_stream_player(None)
        speaker = SpeakerManager()
        confirmation_calls = []

        async def completed_with_wrapper_failure(_script, timeout=0):
            confirmation_calls.append(timeout)
            return CommandResult(
                stdout='{"code": 0}\n/tmp/tts/tts_dialog_123.mp3\n',
                stderr="miplayer: event(EndReached) is posted\n",
                exit_code=1,
            )

        async def play_verified_confirmation(cls, text, **kwargs):
            return await speaker.play(text=text, blocking=True)

        speaker.run_shell = completed_with_wrapper_failure
        self.manager._play_response_with_tts = classmethod(play_verified_confirmation)
        self._set_sequential_responses(
            [
                make_tool_call_response("weather", {"city": "北京"}, "search-1"),
                make_tool_call_response(
                    "play_track", {"query": "测试歌曲"}, "play-1"
                ),
            ]
        )

        result = self.run_async(
            self.manager._request_chat_completion("搜索并播放测试歌曲")
        )

        self.assertTrue(self.manager.is_silent_end_turn_result(result))
        self.assertEqual(len(confirmation_calls), 1)
        self.assertEqual(
            self.mcp_stub.call_history,
            [
                ("weather", {"city": "北京"}),
                ("play_track", {"query": "测试歌曲"}),
            ],
        )

    def test_new_turn_confirms_again(self):
        """前置确认状态属于单轮，新用户轮次会重新确认"""
        for city in ("北京", "上海"):
            self._set_sequential_responses(
                [
                    make_tool_call_response("weather", {"city": city}),
                    make_text_response(f"{city}晴天"),
                ]
            )
            result = self.run_async(
                self.manager._request_chat_completion(f"查{city}天气")
            )
            self.assertEqual(result, f"{city}晴天")

        self.assertEqual(
            self.tts_calls,
            ["好的，正在处理", "好的，正在处理"],
        )

    def test_confirmation_failure_skips_mcp_and_uses_error_recovery(self):
        """确认播放失败时不调用 MCP，并把可操作错误交给模型"""

        async def failed_confirmation(cls, text, **kwargs):
            self.tts_calls.append(text)
            return False

        self.manager._play_response_with_tts = classmethod(failed_confirmation)
        self._set_sequential_responses(
            [
                make_tool_call_response("weather", {"city": "北京"}),
                make_text_response("语音确认失败，请检查音箱连接后重试"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("查天气"))

        self.assertEqual(result, "语音确认失败，请检查音箱连接后重试")
        self.assertEqual(self.mcp_stub.call_history, [])
        self.assertEqual(self.tts_calls, ["好的，正在处理"])
        self.assertIn("工具未执行", self.captured_payloads[1]["messages"][-1]["content"])

    def test_confirmation_timeout_skips_mcp(self):
        """确认播放超时会取消等待且绝不调用 MCP"""

        async def slow_confirmation(cls, text, **kwargs):
            self.tts_calls.append(text)
            await asyncio.sleep(1)
            return True

        self.manager._tool_confirmation_timeout = 0.01
        self.manager._play_response_with_tts = classmethod(slow_confirmation)
        self._set_sequential_responses(
            [
                make_tool_call_response("weather", {"city": "北京"}),
                make_text_response("语音确认超时，请检查音箱连接后重试"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("查天气"))

        self.assertEqual(result, "语音确认超时，请检查音箱连接后重试")
        self.assertEqual(self.mcp_stub.call_history, [])
        self.assertEqual(self.tts_calls, ["好的，正在处理"])
        self.assertIn("播放超时", self.captured_payloads[1]["messages"][-1]["content"])

    def test_playback_success_skips_final_model_request(self):
        """播放成功信号立即静默终止，不再请求模型生成确认文本"""
        self._set_sequential_responses(
            [make_tool_call_response("play_track", {"query": "测试歌曲"})]
        )

        result = self.run_async(self.manager._request_chat_completion("播放测试歌曲"))

        self.assertTrue(self.manager.is_silent_end_turn_result(result))
        self.assertEqual(len(self.captured_payloads), 1)
        self.assertEqual(self.mcp_stub.call_history, [("play_track", {"query": "测试歌曲"})])
        self.assertEqual(self.manager._sessions["test-session"], [])
        self.assertEqual(self.tts_calls, ["好的，正在处理"])

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

    def test_ordinary_chat_with_tools_enabled_has_no_confirmation(self):
        """工具能力已启用但模型不调用 MCP 时，不插入前置确认"""
        self._set_sequential_responses([make_text_response("你好，有什么可以帮你？")])

        result = self.run_async(self.manager._request_chat_completion("你好"))

        self.assertEqual(result, "你好，有什么可以帮你？")
        self.assertEqual(self.tts_calls, [])
        self.assertEqual(self.mcp_stub.call_history, [])

    def test_playback_success_ends_turn_without_followup_model_call(self):
        self.mcp_stub.results["play_track"] = ToolCallResult(
            text="正在播放：测试歌曲",
            silent_end_turn=True,
        )
        self._set_sequential_responses(
            [make_tool_call_response("play_track", {"query": "测试歌曲"})]
        )

        result = self.run_async(self.manager._request_chat_completion("播放测试歌曲"))

        self.assertTrue(self.manager.is_silent_end_turn_result(result))
        self.assertEqual(len(self.captured_payloads), 1)
        self.assertEqual(self.manager._sessions["test-session"], [])

    def test_playback_failure_is_returned_to_model(self):
        failure = "open-xiaoai-bridge POST /api/stream/play failed (HTTP 400): URL 被拒绝"
        self.mcp_stub.results["play_track"] = ToolCallResult(
            text=failure,
            is_error=True,
        )
        self._set_sequential_responses(
            [
                make_tool_call_response("play_track", {"query": "测试歌曲"}),
                make_text_response("播放失败：URL 被拒绝"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("播放测试歌曲"))

        self.assertEqual(result, "播放失败：URL 被拒绝")
        self.assertEqual(len(self.captured_payloads), 2)
        self.assertEqual(self.captured_payloads[1]["messages"][-1]["content"], failure)

    def test_reasoning_content_is_preserved_for_failure_followup(self):
        self.mcp_stub.results["play_track"] = ToolCallResult(
            text="播放失败：缺少歌曲参数",
            is_error=True,
        )
        first = make_tool_call_response("play_track", {})
        first["choices"][0]["message"]["reasoning_content"] = "opaque-provider-state"
        self._set_sequential_responses(
            [first, make_text_response("播放失败，请告诉我歌曲名称")]
        )

        result = self.run_async(self.manager._request_chat_completion("播放一首歌"))

        self.assertEqual(result, "播放失败，请告诉我歌曲名称")
        assistant_message = self.captured_payloads[1]["messages"][-2]
        self.assertEqual(
            assistant_message["reasoning_content"],
            "opaque-provider-state",
        )

    def test_silent_signal_does_not_swallow_sibling_failure(self):
        self.mcp_stub.results["play_track"] = ToolCallResult(
            text="正在播放：测试歌曲",
            silent_end_turn=True,
        )
        self.mcp_stub.results["status"] = ToolCallResult(
            text="状态查询失败",
            is_error=True,
        )
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
                                        "id": "play-1",
                                        "type": "function",
                                        "function": {
                                            "name": "play_track",
                                            "arguments": "{}",
                                        },
                                    },
                                    {
                                        "id": "status-1",
                                        "type": "function",
                                        "function": {
                                            "name": "status",
                                            "arguments": "{}",
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                },
                make_text_response("状态查询失败，请稍后再试"),
            ]
        )

        result = self.run_async(self.manager._request_chat_completion("播放并查询状态"))

        self.assertEqual(result, "状态查询失败，请稍后再试")
        self.assertEqual(len(self.captured_payloads), 2)

    def test_send_and_play_reply_does_not_tts_silent_directive(self):
        tts_calls = []

        async def fake_send(cls, text):
            return "run-1"

        async def fake_tts(cls, text, **kwargs):
            tts_calls.append(text)

        # Use the real directive without deriving behavior from an empty string.
        from core.openai import OpenAITurnDirective

        async def fake_wait(cls, run_id):
            return OpenAITurnDirective.SILENT_END

        self.manager._send_and_track = classmethod(fake_send)
        self.manager._wait_response = classmethod(fake_wait)
        self.manager._play_response_with_tts = classmethod(fake_tts)
        self.manager._response_tts_speakers = {"run-1": "xiaoai"}

        result = self.run_async(
            self.manager.send_and_play_reply("播放测试歌曲", wait_response=True)
        )

        self.assertTrue(self.manager.is_silent_end_turn_result(result))
        self.assertEqual(tts_calls, [])

    def test_send_and_play_playback_success_has_no_post_tts(self):
        """播放成功时只有前置确认，没有成功后的 AI 语音"""
        self._set_sequential_responses(
            [make_tool_call_response("play_track", {"query": "测试歌曲"})]
        )

        result = self.run_async(
            self.manager.send_and_play_reply("播放测试歌曲", wait_response=True)
        )

        self.assertTrue(self.manager.is_silent_end_turn_result(result))
        self.assertEqual(self.tts_calls, ["好的，正在处理"])
        self.assertEqual(len(self.captured_payloads), 1)

    def test_send_and_play_mcp_failure_still_plays_error_reply(self):
        """MCP 失败时保留前置确认和最终失败播报"""
        self._set_sequential_responses(
            [
                make_tool_call_response("play_failed", {}),
                make_text_response("播放失败，请稍后再试"),
            ]
        )

        result = self.run_async(
            self.manager.send_and_play_reply("播放测试歌曲", wait_response=True)
        )

        self.assertEqual(result, "播放失败，请稍后再试")
        self.assertEqual(
            self.tts_calls,
            ["好的，正在处理", "播放失败，请稍后再试"],
        )


if __name__ == "__main__":
    unittest.main()
