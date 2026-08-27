"""MCP (Model Context Protocol) 客户端管理器

桥接服务作为 MCP 客户端连接多个外部 MCP server（config.py 的 mcp_servers 段），
聚合外部工具供 OpenAI 兼容后端的 function calling 使用。

约束：
- 所有连接组件（memory streams / httpx client / ClientSession）绑定创建时的
  asyncio loop，必须全部跑在 MainApp.loop 上（start/stop 经 run_coroutine_threadsafe 调度）
- ClientSession 必须显式设置 read_timeout_seconds（默认永不超时）
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

from core.tool_result import ToolCallResult
from core.utils.config import ConfigManager
from core.utils.logger import logger

# OpenAI function name 约束：[a-zA-Z0-9_-]，最长 64 字符
_OPENAI_NAME_MAX = 64
# 保活 ping 间隔（秒）
_PING_INTERVAL = 60
# 重连退避上限（秒）
_MAX_BACKOFF = 60.0
# MCP structuredContent namespace used for bridge-only turn control.
_BRIDGE_CONTROL_KEY = "x-open-xiaoai-bridge"
_SILENT_PLAYBACK_SIGNAL = {
    "version": 1,
    "action": "end_turn_silently",
    "reason": "playback_started",
}


@dataclass
class MCPServerConfig:
    """单个外部 MCP server 的配置快照"""

    name: str
    type: str  # "http" | "sse" | "stdio"
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0  # 会话读超时 / 单次工具调用超时（秒）
    enabled: bool = True


class MCPClientManager:
    """MCP 客户端管理器：连接多个外部 server，聚合工具供 OpenAI function calling 使用"""

    _initialized = False
    _reload_listener_registered = False
    _loop: asyncio.AbstractEventLoop | None = None
    _stop_event: asyncio.Event | None = None

    # 运行时状态（全部只在 MainApp.loop 上读写）
    _servers: dict[str, MCPServerConfig] = {}
    _tasks: dict[str, asyncio.Task] = {}
    _sessions: dict[str, ClientSession | None] = {}
    _tools_by_server: dict[str, list[Tool]] = {}
    # 聚合结果（整体替换式更新，原子无锁）
    _openai_tools: list[dict[str, Any]] = []
    _name_map: dict[str, tuple[str, str]] = {}  # openai 名 -> (server, 原始名)

    # ---------- 生命周期 ----------

    @classmethod
    def initialize_from_config(cls):
        """从 config.py 初始化（主线程同步调用，只读配置）"""
        cls.reload_from_config()
        cls._initialized = True

    @classmethod
    def reload_from_config(cls):
        """刷新 mcp_servers 配置；变更时在 MainApp.loop 上调度重建连接"""
        config_manager = ConfigManager.instance()
        if not cls._reload_listener_registered:
            config_manager.add_reload_listener(lambda _old, _new: cls.reload_from_config())
            cls._reload_listener_registered = True

        raw = config_manager.get_app_config("mcp_servers", {}) or {}
        new_cfgs: dict[str, MCPServerConfig] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            new_cfgs[name] = MCPServerConfig(
                name=name,
                type=str(entry.get("type", "http")).lower(),
                url=entry.get("url"),
                command=entry.get("command"),
                args=list(entry.get("args", []) or []),
                env=dict(entry.get("env", {}) or {}),
                cwd=entry.get("cwd"),
                headers=dict(entry.get("headers", {}) or {}),
                timeout=float(entry.get("timeout", 120)),
                enabled=bool(entry.get("enabled", True)),
            )

        changed = set(new_cfgs) != set(cls._servers) or any(
            new_cfgs[name] != cls._servers[name] for name in new_cfgs if name in cls._servers
        )
        cls._servers = new_cfgs
        if changed and cls._loop is not None and cls._loop.is_running():
            # 配置热重载：在 MainApp.loop 上调度重建全部连接
            cls._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(cls._restart_all()))

    @classmethod
    def is_enabled(cls) -> bool:
        """是否配置了启用的 server"""
        return any(cfg.enabled for cfg in cls._servers.values())

    @classmethod
    def is_connected(cls) -> bool:
        """是否至少一个 server 会话存活"""
        return any(session is not None for session in cls._sessions.values())

    @classmethod
    async def start(cls):
        """启动全部连接任务（必须在 MainApp.loop 上调用）"""
        cls._loop = asyncio.get_running_loop()
        if cls._stop_event is None:
            cls._stop_event = asyncio.Event()
        for name, cfg in cls._servers.items():
            if cfg.enabled and name not in cls._tasks:
                cls._tasks[name] = asyncio.create_task(cls._connection_loop(name, cfg))
        logger.info(f"[MCPClient] started, servers={list(cls._tasks.keys())}", module="MCP")

    @classmethod
    async def stop(cls):
        """停止全部连接任务并清空状态"""
        if cls._stop_event:
            cls._stop_event.set()
        tasks = list(cls._tasks.values())
        cls._tasks.clear()
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        cls._sessions.clear()
        cls._tools_by_server.clear()
        cls._openai_tools = []
        cls._name_map = {}
        cls._loop = None
        cls._stop_event = None
        logger.info("[MCPClient] stopped", module="MCP")

    @classmethod
    async def _restart_all(cls):
        """配置变更后重建全部连接"""
        logger.info("[MCPClient] config changed, restarting connections", module="MCP")
        await cls.stop()
        await cls.start()

    # ---------- 连接 ----------

    @classmethod
    def _open_transport(cls, cfg: MCPServerConfig):
        """按类型构造 transport async context manager"""
        if cfg.type == "http":
            return streamable_http_client(
                cfg.url,
                http_client=httpx.AsyncClient(
                    headers=cfg.headers,
                    timeout=httpx.Timeout(cfg.timeout),
                ),
            )
        if cfg.type == "sse":
            return sse_client(cfg.url, headers=cfg.headers)
        if cfg.type == "stdio":
            return stdio_client(
                StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env=cfg.env or None,
                    cwd=cfg.cwd,
                )
            )
        raise ValueError(f"未知 MCP transport: {cfg.type}")

    @classmethod
    async def _connection_loop(cls, name: str, cfg: MCPServerConfig):
        """单个 server 的长驻连接循环：连接 → 拉工具 → 保活 → 断线重连"""
        backoff = 1.0
        while not (cls._stop_event and cls._stop_event.is_set()):
            try:
                streams = cls._open_transport(cfg)
                async with streams as s:
                    # streamable_http_client yield 3 元组 (read, write, get_session_id)，
                    # sse/stdio yield 2 元组；统一取前两个
                    read, write = s[0], s[1]
                    # 必须 async with 进入（__aenter__ 启动 receive loop，
                    # 否则 list_tools 等请求的响应无人消费而永久挂起）
                    async with ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=cfg.timeout),
                    ) as session:
                        await session.initialize()
                        tools = await cls._list_all_tools(session)
                        cls._sessions[name] = session
                        cls._tools_by_server[name] = tools
                        cls._rebuild_aggregate()
                        backoff = 1.0
                        logger.info(
                            f"[MCPClient] connected server={name}, tools={len(tools)}",
                            module="MCP",
                        )
                        # 常驻：周期 ping 保活；异常/超时/stop 时退出重连
                        while not (cls._stop_event and cls._stop_event.is_set()):
                            await asyncio.sleep(_PING_INTERVAL)
                            try:
                                await session.send_ping()
                            except Exception:
                                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"[MCPClient] server={name} disconnected: {type(exc).__name__}: {exc}",
                    module="MCP",
                )
                cls._sessions[name] = None
                cls._tools_by_server[name] = []
                cls._rebuild_aggregate()
                if cls._stop_event and cls._stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    @classmethod
    async def _list_all_tools(cls, session: ClientSession) -> list[Tool]:
        """拉取工具列表（处理 nextCursor 分页）"""
        tools: list[Tool] = []
        cursor = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                return tools

    # ---------- 工具面 ----------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """OpenAI function name 约束：仅 [a-zA-Z0-9_-]，≤64 字符"""
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        if not cleaned.strip("_"):
            return "tool"  # 全非法字符（如中文名）回退
        return cleaned[:_OPENAI_NAME_MAX]

    @staticmethod
    def _tool_to_openai(name: str, tool: Tool) -> dict[str, Any]:
        """MCP Tool → OpenAI tools schema（深拷贝，不污染共享 Tool 对象）"""
        parameters = tool.inputSchema or {"type": "object", "properties": {}}
        if parameters.get("type") != "object":
            # 某些 server 返回非 object schema（如 {"type":"string"}），OpenAI 会拒绝
            parameters = {"type": "object", "properties": {"value": parameters}}
        params = json.loads(json.dumps(parameters))
        params.pop("$schema", None)
        description = (tool.description or "").strip() or f"MCP tool {name}"
        if len(description) > 1024:
            description = description[:1024]
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": params},
        }

    @classmethod
    def _rebuild_aggregate(cls):
        """重新聚合全部工具：sanitize → 冲突加 {server}_ 前缀 → 整体替换"""
        openai_tools: list[dict[str, Any]] = []
        name_map: dict[str, tuple[str, str]] = {}
        used: set[str] = set()
        for server, tools in cls._tools_by_server.items():
            prefix = cls._sanitize_name(server)
            for tool in tools:
                candidate = cls._sanitize_name(tool.name)
                if candidate in used:
                    candidate = f"{prefix}_{candidate}"[:_OPENAI_NAME_MAX]
                if candidate in used:
                    i = 2
                    while f"{candidate}_{i}"[:_OPENAI_NAME_MAX] in used:
                        i += 1
                    candidate = f"{candidate}_{i}"[:_OPENAI_NAME_MAX]
                used.add(candidate)
                name_map[candidate] = (server, tool.name)
                openai_tools.append(cls._tool_to_openai(candidate, tool))
        cls._openai_tools, cls._name_map = openai_tools, name_map

    @classmethod
    def get_tools(cls) -> list[dict[str, Any]]:
        """返回聚合后的 OpenAI 格式工具列表（同步、无 IO）"""
        return cls._openai_tools

    @classmethod
    async def call_tool(cls, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """调用外部 MCP 工具并保留显式的错误/回合控制元数据。"""
        mapping = cls._name_map.get(name)
        if not mapping:
            return ToolCallResult(
                text=f"[MCPClient] 未知工具: {name}",
                is_error=True,
            )
        server, original = mapping
        session = cls._sessions.get(server)
        cfg = cls._servers.get(server)
        if session is None:
            return ToolCallResult(
                text=f"[MCPClient] 工具 {name} 不可用：server {server} 未连接",
                is_error=True,
            )
        try:
            result = await session.call_tool(
                original,
                arguments or {},
                read_timeout_seconds=timedelta(seconds=cfg.timeout if cfg else 120),
            )
        except Exception as exc:
            return ToolCallResult(
                text=f"[MCPClient] 工具 {name} 调用失败: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        text = "\n".join(
            c.text for c in result.content if isinstance(c, TextContent)
        )
        if result.isError:
            return ToolCallResult(
                text=f"[MCPClient] 工具 {name} 返回错误: {text or '未知错误'}",
                is_error=True,
            )
        if not text.strip():
            return ToolCallResult(
                text=f"[MCPClient] 工具 {name} 返回空结果，无法确认执行成功",
                is_error=True,
            )

        structured = getattr(result, "structuredContent", None)
        silent_end_turn = (
            isinstance(structured, dict)
            and structured.get(_BRIDGE_CONTROL_KEY) == _SILENT_PLAYBACK_SIGNAL
        )
        return ToolCallResult(text=text, silent_end_turn=silent_end_turn)
