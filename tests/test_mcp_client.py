"""MCP client 管理器测试：纯函数 + 本地真实 server 连接"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils.config_loader import ensure_config_module_loaded

ensure_config_module_loaded()

import core.mcp_client as mcp_mod
from core.mcp_client import MCPClientManager
from mcp.types import TextContent, Tool

# ---------- 工具：本地 FastMCP server ----------


class LocalMCPServer:
    """用 uvicorn 起一个 FastMCP streamable HTTP server（随机端口）"""

    def __init__(self, name: str = "mcp-test-server"):
        self.name = name
        self.port = None
        self._task = None
        self._server = None

    async def start(self):
        import uvicorn
        from mcp.server.fastmcp import FastMCP

        fastmcp = FastMCP(name=self.name)
        for tool_name, fn in {
            "echo": _echo,
            "add": _add,
            "fail": _fail,
        }.items():
            fastmcp.tool(name=tool_name)(fn)

        config = uvicorn.Config(
            fastmcp.streamable_http_app(),
            host="127.0.0.1",
            # 重启时复用原端口（真实场景端口固定），保证 client 重连可用
            port=self.port or 0,
            log_config=None,
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._server = server
        self._task = asyncio.create_task(server._serve())
        for _ in range(200):
            if self._task.done():
                raise RuntimeError("测试 server 启动失败")
            if server.started:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("测试 server 启动超时")
        self.port = server.servers[0].sockets[0].getsockname()[1]
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    async def stop(self):
        if self._server:
            self._server.should_exit = True
            if self._task:
                try:
                    await self._task
                except Exception:
                    pass
        self._server = None
        self._task = None


def _echo(text: str) -> str:
    return f"echo: {text}"


def _add(a: int, b: int) -> int:
    return a + b


def _fail() -> str:
    raise ValueError("故意失败")


# ---------- 纯函数测试 ----------


class SanitizeTest(unittest.TestCase):
    def test_sanitize_illegal_chars(self):
        self.assertEqual(MCPClientManager._sanitize_name("hello world"), "hello_world")
        self.assertEqual(MCPClientManager._sanitize_name("a.b"), "a_b")
        self.assertEqual(MCPClientManager._sanitize_name("你好"), "tool")  # 全非法字符回退

    def test_sanitize_truncate(self):
        self.assertEqual(
            len(MCPClientManager._sanitize_name("x" * 100)), 64
        )

    def test_sanitize_empty_fallback(self):
        self.assertEqual(MCPClientManager._sanitize_name(""), "tool")
        self.assertEqual(MCPClientManager._sanitize_name("###"), "tool")


class ToolToOpenAITest(unittest.TestCase):
    def test_object_schema(self):
        tool = Tool(
            name="echo",
            description="回显",
            inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
        )
        result = MCPClientManager._tool_to_openai("echo", tool)
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "echo")
        self.assertEqual(result["function"]["description"], "回显")
        self.assertEqual(
            result["function"]["parameters"]["properties"]["text"]["type"], "string"
        )

    def test_non_object_schema_wrapped(self):
        tool = Tool(
            name="calc",
            description="计算",
            inputSchema={"type": "string"},
        )
        result = MCPClientManager._tool_to_openai("calc", tool)
        params = result["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["properties"]["value"]["type"], "string")

    def test_dollar_schema_removed_and_deep_copy(self):
        tool = Tool(
            name="echo",
            description="回显",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        )
        result = MCPClientManager._tool_to_openai("echo", tool)
        self.assertNotIn("$schema", result["function"]["parameters"])
        # 深拷贝：转换不污染原始 Tool
        self.assertIn("$schema", tool.inputSchema)

    def test_description_truncated(self):
        tool = Tool(name="t", description="x" * 2000, inputSchema={"type": "object"})
        result = MCPClientManager._tool_to_openai("t", tool)
        self.assertEqual(len(result["function"]["description"]), 1024)


class RebuildAggregateTest(unittest.TestCase):
    def setUp(self):
        MCPClientManager._tools_by_server = {}
        MCPClientManager._openai_tools = []
        MCPClientManager._name_map = {}

    def _tool(self, name):
        return Tool(name=name, inputSchema={"type": "object"})

    def test_aggregate_single_server(self):
        MCPClientManager._tools_by_server = {
            "weather": [self._tool("get_weather"), self._tool("get_alert")]
        }
        MCPClientManager._rebuild_aggregate()
        names = [t["function"]["name"] for t in MCPClientManager.get_tools()]
        self.assertEqual(names, ["get_weather", "get_alert"])
        self.assertEqual(
            MCPClientManager._name_map["get_weather"], ("weather", "get_weather")
        )

    def test_duplicate_tool_names_get_server_prefix(self):
        MCPClientManager._tools_by_server = {
            "alpha": [self._tool("status")],
            "beta": [self._tool("status")],
        }
        MCPClientManager._rebuild_aggregate()
        names = [t["function"]["name"] for t in MCPClientManager.get_tools()]
        self.assertEqual(names, ["status", "beta_status"])
        self.assertEqual(MCPClientManager._name_map["beta_status"], ("beta", "status"))

    def test_extreme_duplicate_gets_number_suffix(self):
        MCPClientManager._tools_by_server = {
            "a": [self._tool("get")],
            "b": [self._tool("get")],
            "c": [self._tool("get")],
        }
        MCPClientManager._rebuild_aggregate()
        names = [t["function"]["name"] for t in MCPClientManager.get_tools()]
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)  # 全部唯一


class ToolResultTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        MCPClientManager._name_map = {"play_track": ("music", "play_track")}
        MCPClientManager._servers = {
            "music": mcp_mod.MCPServerConfig(name="music", type="http")
        }

    async def _call_with(self, result=None, error=None):
        class FakeSession:
            async def call_tool(self, *_args, **_kwargs):
                if error:
                    raise error
                return result

        MCPClientManager._sessions = {"music": FakeSession()}
        return await MCPClientManager.call_tool("play_track", {})

    async def test_exact_playback_signal_is_preserved(self):
        result = await self._call_with(
            SimpleNamespace(
                content=[TextContent(type="text", text="正在播放：测试歌曲")],
                isError=False,
                structuredContent={
                    "x-open-xiaoai-bridge": {
                        "version": 1,
                        "action": "end_turn_silently",
                        "reason": "playback_started",
                    }
                },
            )
        )

        self.assertEqual(result.text, "正在播放：测试歌曲")
        self.assertFalse(result.is_error)
        self.assertTrue(result.silent_end_turn)

    async def test_mcp_error_never_terminates_silently(self):
        result = await self._call_with(
            SimpleNamespace(
                content=[TextContent(type="text", text="播放失败")],
                isError=True,
                structuredContent={
                    "x-open-xiaoai-bridge": {
                        "version": 1,
                        "action": "end_turn_silently",
                        "reason": "playback_started",
                    }
                },
            )
        )

        self.assertIn("返回错误", result.text)
        self.assertTrue(result.is_error)
        self.assertFalse(result.silent_end_turn)

    async def test_empty_success_is_converted_to_model_facing_error(self):
        result = await self._call_with(
            SimpleNamespace(content=[], isError=False, structuredContent=None)
        )

        self.assertIn("返回空结果", result.text)
        self.assertTrue(result.is_error)
        self.assertFalse(result.silent_end_turn)

    async def test_transport_failure_is_model_facing_and_not_silent(self):
        result = await self._call_with(error=RuntimeError("HTTP 503"))

        self.assertIn("调用失败", result.text)
        self.assertIn("HTTP 503", result.text)
        self.assertTrue(result.is_error)
        self.assertFalse(result.silent_end_turn)


# ---------- 真实连接测试 ----------


class MCPClientConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # 重置管理器状态
        MCPClientManager._sessions = {}
        MCPClientManager._tasks = {}
        MCPClientManager._tools_by_server = {}
        MCPClientManager._openai_tools = []
        MCPClientManager._name_map = {}
        MCPClientManager._stop_event = None
        MCPClientManager._loop = None

    async def asyncTearDown(self):
        await MCPClientManager.stop()

    def _set_servers(self, servers: dict):
        MCPClientManager._servers = {
            name: mcp_mod.MCPServerConfig(name=name, **cfg) for name, cfg in servers.items()
        }

    async def _wait_connected(self, name: str, timeout: float = 10.0):
        async def wait():
            while MCPClientManager._sessions.get(name) is None:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(wait(), timeout=timeout)

    async def test_connect_and_call_tools(self):
        server = await LocalMCPServer().start()
        try:
            self._set_servers({"test": {"type": "http", "url": server.url}})
            await MCPClientManager.start()
            await self._wait_connected("test")

            tools = MCPClientManager.get_tools()
            names = {t["function"]["name"] for t in tools}
            self.assertIn("echo", names)
            self.assertIn("add", names)
            self.assertIn("fail", names)

            result = await MCPClientManager.call_tool("echo", {"text": "你好"})
            self.assertEqual(result.text, "echo: 你好")
            self.assertFalse(result.silent_end_turn)

            result = await MCPClientManager.call_tool("add", {"a": 2, "b": 3})
            self.assertEqual(result.text, "5")

            # isError 工具：返回错误文本而非抛异常
            result = await MCPClientManager.call_tool("fail", {})
            self.assertTrue(result.is_error)
            self.assertIn("返回错误", result.text)
            self.assertIn("故意失败", result.text)
        finally:
            await server.stop()

    async def test_duplicate_tool_names_route_correctly(self):
        server_a = await LocalMCPServer("alpha").start()
        server_b = await LocalMCPServer("beta").start()
        try:
            self._set_servers(
                {
                    "alpha": {"type": "http", "url": server_a.url},
                    "beta": {"type": "http", "url": server_b.url},
                }
            )
            await MCPClientManager.start()
            await self._wait_connected("alpha")
            await self._wait_connected("beta")

            # 两个 server 都有 echo：第二个获得 beta_ 前缀
            names = {t["function"]["name"] for t in MCPClientManager.get_tools()}
            self.assertIn("echo", names)
            self.assertIn("beta_echo", names)

            # 路由到对应 server
            result = await MCPClientManager.call_tool("echo", {"text": "A"})
            self.assertEqual(result.text, "echo: A")
            result = await MCPClientManager.call_tool("beta_echo", {"text": "B"})
            self.assertEqual(result.text, "echo: B")
        finally:
            await server_a.stop()
            await server_b.stop()

    async def test_disconnect_clears_tools_and_reconnects(self):
        server = await LocalMCPServer().start()
        try:
            self._set_servers({"test": {"type": "http", "url": server.url}})
            # 缩短 ping 间隔便于测试检测断开
            mcp_mod._PING_INTERVAL = 0.5
            await MCPClientManager.start()
            await self._wait_connected("test")
            self.assertGreater(len(MCPClientManager.get_tools()), 0)

            # 停止 server → client 应检测到断开并清空该 server 工具
            await server.stop()
            for _ in range(100):
                if MCPClientManager._sessions.get("test") is None:
                    break
                await asyncio.sleep(0.1)
            self.assertIsNone(MCPClientManager._sessions.get("test"))
            self.assertEqual(len(MCPClientManager.get_tools()), 0)

            # 断线期间工具表已清空（模型不会再拿到该工具），调用返回错误文本
            result = await MCPClientManager.call_tool("echo", {"text": "x"})
            self.assertTrue(result.is_error)
            self.assertIn("未知工具", result.text)

            # 重启 server → 指数退避后自动重连
            await server.start()
            await self._wait_connected("test", timeout=15.0)
            result = await MCPClientManager.call_tool("echo", {"text": "恢复"})
            self.assertEqual(result.text, "echo: 恢复")
        finally:
            await server.stop()

    async def test_unknown_tool(self):
        result = await MCPClientManager.call_tool("no_such_tool", {})
        self.assertTrue(result.is_error)
        self.assertIn("未知工具", result.text)

    async def test_playback_signal_is_parsed_from_structured_content(self):
        class SignalSession:
            async def call_tool(self, name, arguments, read_timeout_seconds):
                return type(
                    "Result",
                    (),
                    {
                        "content": [TextContent(type="text", text="正在播放")],
                        "isError": False,
                        "structuredContent": {
                            "x-open-xiaoai-bridge": {
                                "version": 1,
                                "action": "end_turn_silently",
                                "reason": "playback_started",
                            }
                        },
                    },
                )()

        MCPClientManager._name_map = {"play_track": ("music", "play_track")}
        MCPClientManager._sessions = {"music": SignalSession()}
        MCPClientManager._servers = {
            "music": mcp_mod.MCPServerConfig(name="music", type="stdio")
        }

        result = await MCPClientManager.call_tool("play_track", {"query": "test"})

        self.assertEqual(result.text, "正在播放")
        self.assertFalse(result.is_error)
        self.assertTrue(result.silent_end_turn)

    async def test_empty_success_is_a_non_silent_error(self):
        class EmptySession:
            async def call_tool(self, name, arguments, read_timeout_seconds):
                return type(
                    "Result",
                    (),
                    {"content": [], "isError": False, "structuredContent": None},
                )()

        MCPClientManager._name_map = {"play_track": ("music", "play_track")}
        MCPClientManager._sessions = {"music": EmptySession()}
        MCPClientManager._servers = {
            "music": mcp_mod.MCPServerConfig(name="music", type="stdio")
        }

        result = await MCPClientManager.call_tool("play_track", {})

        self.assertTrue(result.is_error)
        self.assertFalse(result.silent_end_turn)
        self.assertIn("返回空结果", result.text)


if __name__ == "__main__":
    unittest.main()
