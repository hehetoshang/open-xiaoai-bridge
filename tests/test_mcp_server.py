"""MCP Server 单元测试（桩模式，无需音箱 / 无需真实豆包凭证）"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FakeSpeaker:
    """SpeakerManager 桩：记录调用参数"""

    def __init__(self):
        self.calls = []
        self.playing_status = "idle"

    async def play(self, **kwargs):
        self.calls.append(("play", kwargs))
        return True

    async def play_server_file(self, **kwargs):
        self.calls.append(("play_server_file", kwargs))
        return True

    async def get_playing(self):
        return self.playing_status

    async def stop_device_audio(self):
        self.calls.append(("stop_device_audio", {}))

    async def wake_up(self, **kwargs):
        self.calls.append(("wake_up", kwargs))
        return True


class FakeXiaoAI:
    """XiaoAI 桩"""

    def __init__(self):
        self.stop_calls = 0

    def stop_conversation(self):
        self.stop_calls += 1


class FakeConfig:
    """ConfigManager 桩：注入 tts.doubao 配置，保证测试确定性"""

    def __init__(self, tts_config):
        self._tts = tts_config

    def get_app_config(self, path, default=None):
        if path == "tts.doubao":
            return self._tts
        return default


class RecordingStub:
    """open_xiaoai_server 桩：记录 Rust async 函数调用"""

    def __init__(self):
        self.calls = []

    def _record(self, name):
        async def _fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _fn

    def __getattr__(self, name):
        return self._record(name)


class FakeResponse:
    """aiohttp Response 桩"""

    def __init__(self, session):
        self.status = session.status
        self._body = session.error_body
        self.content = _FakeContent(session.jsonl_lines)

    async def text(self):
        return self._body


class _FakeContent:
    """模拟 aiohttp 流式响应：按 chunk 返回（含跨行切分，验证缓冲逻辑）"""

    def __init__(self, lines):
        chunks = []
        for line in lines:
            encoded = line.encode("utf-8")
            # 把每行切成两半 + 换行符，模拟 chunk 跨行
            if len(encoded) > 2:
                mid = len(encoded) // 2
                chunks.append(encoded[:mid])
                chunks.append(encoded[mid:] + b"\n")
            else:
                chunks.append(encoded + b"\n")
        chunks.append(b"")
        self._chunks = chunks
        self._i = 0

    def __aiter__(self):
        self._i = 0
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


class FakeSession:
    """aiohttp.ClientSession 桩"""

    def __init__(self, jsonl_lines, status=200, error_body=""):
        self.jsonl_lines = jsonl_lines
        self.status = status
        self.error_body = error_body
        self.post_calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.post_calls.append((url, headers, json))
        return _Ctx(self)


class _Ctx:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return FakeResponse(self.session)

    async def __aexit__(self, *exc):
        return False


class MCPServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 桩掉 Rust 扩展
        cls.ext_stub = RecordingStub()
        sys.modules.setdefault("open_xiaoai_server", cls.ext_stub)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        import core.services.mcp_server as mcp_mod
        from core.ref import set_speaker, set_xiaoai

        cls.mcp_mod = mcp_mod
        cls.speaker = FakeSpeaker()
        cls.xiaoai = FakeXiaoAI()
        set_speaker(cls.speaker)
        set_xiaoai(cls.xiaoai)

    def setUp(self):
        self.speaker.calls.clear()
        self.xiaoai.stop_calls = 0
        self.ext_stub.calls.clear()

        # 构造不启动 transport 的 MCPServer（transports=[]）
        server = self.mcp_mod.MCPServer("127.0.0.1", 9093, transports=[])
        server.config = FakeConfig({})
        self.mcp_mod._server = server
        self.server = server

    def run_async(self, coro):
        return asyncio.run(coro)

    # ---------- tts_play ----------

    def test_tts_play_no_credentials_falls_back_to_native_tts(self):
        """无豆包凭证时回退小爱原生 TTS"""
        result = self.run_async(self.mcp_mod._tool_tts_play("你好"))
        self.assertIn("原生 TTS", result)
        self.assertEqual(self.speaker.calls[0][0], "play")
        self.assertEqual(self.speaker.calls[0][1]["text"], "你好")

    def test_tts_play_with_credentials_uses_doubao(self):
        """有豆包凭证时走 open_xiaoai_server.tts_play 并传递参数"""
        self.server.config = FakeConfig(
            {
                "app_id": "config-app",
                "access_key": "config-key",
                "default_speaker": "zh_female_cancan_mars_bigtts",
                "stream": False,
            }
        )
        result = self.run_async(
            self.mcp_mod._tool_tts_play(
                "你好", speaker_id="zh_female_xiaohe_uranus_bigtts", speed=1.5
            )
        )
        self.assertIn("豆包 TTS", result)
        self.assertEqual(len(self.ext_stub.calls), 1)
        name, args, kwargs = self.ext_stub.calls[0]
        self.assertEqual(name, "tts_play")
        self.assertEqual(args[0], "你好")  # text 是位置参数
        self.assertEqual(kwargs["app_id"], "config-app")
        self.assertEqual(kwargs["access_key"], "config-key")
        self.assertEqual(kwargs["speaker"], "zh_female_xiaohe_uranus_bigtts")
        self.assertEqual(kwargs["speed"], 1.5)
        # 凭证覆盖：请求参数优先于 config
        self.server.config = FakeConfig({"app_id": "config-app", "access_key": "config-key", "stream": False})
        self.ext_stub.calls.clear()
        self.run_async(
            self.mcp_mod._tool_tts_play("你好", app_id="req-app", access_key="req-key")
        )
        kwargs = self.ext_stub.calls[0][2]
        self.assertEqual(kwargs["app_id"], "req-app")
        self.assertEqual(kwargs["access_key"], "req-key")

    def test_tts_play_stream_config_uses_stream_play(self):
        """config tts.doubao.stream=True 时走 tts_stream_play"""
        self.server.config = FakeConfig(
            {"app_id": "a", "access_key": "k", "stream": True}
        )
        self.run_async(self.mcp_mod._tool_tts_play("你好"))
        self.assertEqual(self.ext_stub.calls[0][0], "tts_stream_play")

    def test_tts_play_background_uses_background_fn(self):
        """blocking=False 时走 background 变体"""
        self.server.config = FakeConfig(
            {"app_id": "a", "access_key": "k", "stream": False}
        )
        self.run_async(self.mcp_mod._tool_tts_play("你好", blocking=False))
        self.assertEqual(self.ext_stub.calls[0][0], "tts_play_background")

    def test_tts_play_empty_text_raises(self):
        with self.assertRaises(ValueError):
            self.run_async(self.mcp_mod._tool_tts_play("  "))

    # ---------- tts_synthesize ----------

    def test_tts_synthesize_no_credentials_raises(self):
        """无豆包凭证时直接抛错（不播放，不回退）"""
        with self.assertRaises(RuntimeError):
            self.run_async(self.mcp_mod._tool_tts_synthesize("你好"))

    def test_tts_synthesize_returns_base64_audio(self):
        """成功合成并返回 base64 音频数据"""
        audio_bytes = b"audio-bytes-1"
        lines = [
            json.dumps(
                {"code": 0, "data": base64.b64encode(audio_bytes).decode(), "message": ""}
            ),
            json.dumps({"code": 20000000, "message": "", "data": ""}),
        ]
        self.server._session = FakeSession(lines)
        self.server.config = FakeConfig({"app_id": "a", "access_key": "k"})

        # 显式指定 format="mp3" 保证确定性
        result = self.run_async(self.mcp_mod._tool_tts_synthesize("你好", format="mp3"))

        self.assertEqual(result["audio_base64"], base64.b64encode(audio_bytes).decode("ascii"))
        self.assertEqual(result["format"], "mp3")
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(result["bytes"], len(audio_bytes))
        # 校验请求头与 payload（text/format 在嵌套的 req_params 里）
        url, headers, payload = self.server._session.post_calls[0]
        self.assertEqual(headers["X-Api-App-Id"], "a")
        self.assertEqual(headers["X-Api-Access-Key"], "k")
        req_params = payload["req_params"]
        self.assertEqual(req_params["text"], "你好")
        self.assertEqual(req_params["audio_params"]["format"], "mp3")
        self.assertEqual(req_params["audio_params"]["sample_rate"], 24000)

    def test_tts_synthesize_api_error_raises(self):
        """豆包 API 返回错误码时抛错"""
        lines = [
            json.dumps({"code": 500, "message": "internal error", "data": ""})
        ]
        self.server._session = FakeSession(lines)
        self.server.config = FakeConfig({"app_id": "a", "access_key": "k"})
        with self.assertRaises(RuntimeError) as ctx:
            self.run_async(self.mcp_mod._tool_tts_synthesize("你好"))
        self.assertIn("500", str(ctx.exception))

    # ---------- list_voices ----------

    def test_list_voices_all(self):
        result = self.run_async(self.mcp_mod._tool_list_voices("all"))
        self.assertEqual(result["provider"], "doubao")
        self.assertIn("versions", result)
        self.assertIn("1.0", result["versions"])
        self.assertIn("2.0", result["versions"])

    def test_list_voices_version(self):
        result = self.run_async(self.mcp_mod._tool_list_voices("2.0"))
        self.assertEqual(result["version"], "2.0")
        self.assertGreater(result["count"], 0)

    # ---------- play_text / play_url ----------

    def test_play_text_calls_speaker(self):
        result = self.run_async(self.mcp_mod._tool_play_text("播报一下"))
        self.assertTrue(result)
        self.assertEqual(self.speaker.calls[0][0], "play")
        self.assertEqual(self.speaker.calls[0][1]["text"], "播报一下")
        self.assertTrue(self.speaker.calls[0][1]["blocking"])

    def test_play_text_empty_raises(self):
        with self.assertRaises(ValueError):
            self.run_async(self.mcp_mod._tool_play_text(""))

    def test_play_url_calls_speaker(self):
        self.run_async(self.mcp_mod._tool_play_url("https://example.com/a.mp3", blocking=False))
        call = self.speaker.calls[0]
        self.assertEqual(call[0], "play")
        self.assertEqual(call[1]["url"], "https://example.com/a.mp3")
        self.assertFalse(call[1]["blocking"])

    # ---------- play_file ----------

    def test_play_file_valid(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\xff" * 100)
            path = f.name
        try:
            result = self.run_async(self.mcp_mod._tool_play_file(path))
            self.assertTrue(result)
            self.assertEqual(self.speaker.calls[0][0], "play")
            self.assertEqual(self.speaker.calls[0][1]["server_file"], path)
        finally:
            os.unlink(path)

    def test_play_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.run_async(self.mcp_mod._tool_play_file("/no/such/file.mp3"))

    def test_play_file_bad_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"x")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                self.run_async(self.mcp_mod._tool_play_file(path))
        finally:
            os.unlink(path)

    # ---------- get_status / interrupt / wakeup ----------

    def test_get_status(self):
        result = self.run_async(self.mcp_mod._tool_get_status())
        self.assertEqual(result["playing"], "idle")
        self.assertTrue(result["speaker_ready"])

    def test_interrupt(self):
        result = self.run_async(self.mcp_mod._tool_interrupt())
        self.assertTrue(result["interrupted"])
        self.assertEqual(self.speaker.calls[0][0], "stop_device_audio")
        self.assertEqual(self.xiaoai.stop_calls, 1)

    def test_wakeup(self):
        self.run_async(self.mcp_mod._tool_wakeup(silent=True))
        self.assertEqual(self.speaker.calls[0][0], "wake_up")
        self.assertEqual(self.speaker.calls[0][1], {"awake": True, "silent": True})


if __name__ == "__main__":
    unittest.main()
