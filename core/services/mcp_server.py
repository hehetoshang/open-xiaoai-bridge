"""MCP (Model Context Protocol) 服务器：把音箱能力暴露为 MCP 工具

支持三种 transport：
- stdio：供 Claude Desktop / Claude Code 等本地 MCP 客户端连接
- sse：SSE (Server-Sent Events) transport
- http：streamable HTTP transport

工具运行时通过全局引用（core.ref）访问 SpeakerManager / XiaoAI，
与 api_server.py 保持一致，不持有 MainApp 引用。
"""

import asyncio
import base64
import json
import os
import sys
from typing import Any

from core.ref import get_speaker, get_xiaoai
from core.services.tts.doubao import DoubaoTTS
from core.utils.config import ConfigManager
from core.utils.logger import logger, set_log_stream

# 播放文件安全限制
PLAY_FILE_EXTENSIONS = {".mp3", ".wav", ".opus", ".flac", ".aac", ".m4a", ".ogg"}
MAX_PLAY_FILE_SIZE = 20 * 1024 * 1024  # 20MB
DEFAULT_SPEAKER = "zh_female_cancan_mars_bigtts"

# 模块级 server 引用（工具函数通过它访问实例状态，与 ref.py 全局引用思路一致）
_server: "MCPServer | None" = None


class MCPServer:
    """MCP 服务器，嵌入 MainApp 进程，支持 stdio / SSE / streamable HTTP"""

    def __init__(
        self,
        host: str,
        port: int,
        transports: list[str],
        name: str = "open-xiaoai-bridge",
    ):
        from mcp.server.fastmcp import FastMCP

        self.host = host
        self.port = port
        self.transports = transports
        self.config = ConfigManager.instance()
        self._fastmcp = FastMCP(name=name)
        self._register_tools()
        self._uvicorn: Any = None  # uvicorn.Server（延迟导入）
        self._http_task: asyncio.Task | None = None
        self._stdio_task: asyncio.Task | None = None
        self._session: Any = None  # aiohttp.ClientSession（tts_synthesize 用）

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        """启动 MCP 服务器（在 MainApp 事件循环上执行）"""
        global _server
        _server = self
        try:
            if "stdio" in self.transports:
                # stdio 模式下 stdout 属于 MCP 协议帧，日志必须切到 stderr
                set_log_stream(sys.stderr)
                self._stdio_task = asyncio.create_task(self._fastmcp.run_stdio_async())
                logger.info("[MCPServer] stdio transport started", module="MCP")

            http_transports = [t for t in self.transports if t in ("http", "sse")]
            if http_transports:
                await self._start_http_transports(http_transports)

            if not self._session:
                import aiohttp

                self._session = aiohttp.ClientSession()

            logger.info(
                f"[MCPServer] MCP server started: transports={self.transports}, "
                f"host={self.host}, port={self.port}",
                module="MCP",
            )
        except Exception as exc:
            logger.error(f"[MCPServer] Start failed: {exc}", module="MCP")

    async def _start_http_transports(self, transports: list[str]) -> None:
        """启动 SSE / streamable HTTP transport（合并路由到单个 uvicorn）"""
        import uvicorn
        from starlette.applications import Starlette

        routes = []
        lifespan = None
        if "sse" in transports:
            routes.extend(self._fastmcp.sse_app().routes)
            logger.debug("[MCPServer] SSE transport routes added", module="MCP")
        if "http" in transports:
            http_app = self._fastmcp.streamable_http_app()
            routes.extend(http_app.routes)
            # streamable HTTP 的任务组在 lifespan 中初始化（session_manager.run()），
            # 合并路由时必须保留，否则请求会报 "Task group is not initialized"
            # （FastMCP 内部 lifespan 即 lambda app: self._session_manager.run()）
            lifespan = lambda app: self._fastmcp._session_manager.run()
            logger.debug("[MCPServer] Streamable HTTP transport routes added", module="MCP")

        combined = Starlette(routes=routes, lifespan=lifespan)
        config = uvicorn.Config(
            combined,
            host=self.host,
            port=self.port,
            log_config=None,
            access_log=False,
        )
        self._uvicorn = uvicorn.Server(config)
        # _serve() 内部创建 lifespan 并依次执行 startup → main_loop → shutdown，
        # 不直接调用 serve()（其 capture_signals 依赖主线程）
        self._http_task = asyncio.create_task(self._uvicorn._serve())
        # 等待启动完成（端口绑定成功），失败（如端口占用）时抛出异常
        for _ in range(200):
            if self._http_task.done():
                exc = self._http_task.exception()
                if exc:
                    raise exc
                break
            if self._uvicorn.started:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("uvicorn 启动超时")
        logger.debug(
            f"[MCPServer] HTTP transports listening on {self.host}:{self.port}",
            module="MCP",
        )

    async def stop(self) -> None:
        """停止 MCP 服务器"""
        try:
            if self._stdio_task:
                self._stdio_task.cancel()
                try:
                    await self._stdio_task
                except asyncio.CancelledError:
                    pass
                self._stdio_task = None
                logger.info("[MCPServer] stdio transport stopped", module="MCP")

            if self._uvicorn:
                # _serve() 内部在 main_loop 退出后会自动执行 shutdown
                self._uvicorn.should_exit = True
                if self._http_task:
                    try:
                        await self._http_task
                    except Exception:
                        pass
                    self._http_task = None
                self._uvicorn = None
                logger.info("[MCPServer] HTTP transports stopped", module="MCP")

            if self._session:
                await self._session.close()
                self._session = None
        except Exception as exc:
            logger.error(f"[MCPServer] Stop failed: {exc}", module="MCP")

    # ---------- 工具注册 ----------

    def _register_tools(self) -> None:
        """注册 MCP 工具（模块级函数，避免 self 混入 schema）"""
        for name, fn in {
            "tts_play": _tool_tts_play,
            "tts_synthesize": _tool_tts_synthesize,
            "list_voices": _tool_list_voices,
            "play_text": _tool_play_text,
            "play_url": _tool_play_url,
            "play_file": _tool_play_file,
            "get_status": _tool_get_status,
            "interrupt": _tool_interrupt,
            "wakeup": _tool_wakeup,
            "pause": _tool_pause,
            "resume": _tool_resume,
            "stream_play": _tool_stream_play,
            "stream_pause": _tool_stream_pause,
            "stream_resume": _tool_stream_resume,
            "stream_seek": _tool_stream_seek,
            "stream_stop": _tool_stream_stop,
            "stream_status": _tool_stream_status,
        }.items():
            self._fastmcp.tool(name=name)(fn)

    # ---------- 配置辅助 ----------

    @staticmethod
    def _resolve_tts_credentials(
        app_id: str | None, access_key: str | None, resource_id: str | None
    ) -> tuple[str, str, str | None]:
        """解析豆包 TTS 凭证：请求参数优先，fallback 到 config.py 的 tts.doubao 段"""
        tts_config = _server.config.get_app_config("tts.doubao", {})
        resolved_app_id = app_id or tts_config.get("app_id")
        resolved_access_key = access_key or tts_config.get("access_key")
        resolved_resource_id = resource_id or tts_config.get("resource_id")
        if not resolved_app_id or not resolved_access_key:
            raise RuntimeError(
                "豆包 TTS 凭证未配置：请在请求中提供 app_id/access_key，"
                "或在 config.py 的 tts.doubao 段配置"
            )
        return resolved_app_id, resolved_access_key, resolved_resource_id

    @staticmethod
    def _resolve_speaker_id(speaker_id: str | None) -> str:
        """解析音色：请求参数优先，fallback 到 config 默认音色"""
        if speaker_id:
            return speaker_id
        return _server.config.get_app_config(
            "tts.doubao.default_speaker", DEFAULT_SPEAKER
        )


# ---------- MCP 工具（模块级函数，docstring 会成为客户端可见的 tool description） ----------


async def _tool_tts_play(
    text: str,
    speaker_id: str | None = None,
    speed: float = 1.0,
    emotion: str | None = None,
    context_texts: list[str] | None = None,
    app_id: str | None = None,
    access_key: str | None = None,
    resource_id: str | None = None,
    blocking: bool = True,
) -> str:
    """豆包 TTS 合成并直接在小爱音箱上播放。

    如果未配置豆包凭证（app_id/access_key），自动回退到小爱系统原生 TTS 播报。
    播放失败时同样回退到小爱原生 TTS。

    参数:
        text: 要播报的文字内容
        speaker_id: 音色 ID（可用 list_voices 查询），默认取 config.py 的 tts.doubao.default_speaker
        speed: 语速，范围 0.8-2.0，默认 1.0
        emotion: 情感（仅 2.0 音色支持）：happy/sad/angry/surprised/lovey-dovey/tender/storytelling/news/advertising/magnetic
        context_texts: 上下文指令列表（仅 2.0 音色支持，仅第一条生效）
        app_id: 豆包应用 ID（可选，默认用 config.py 配置）
        access_key: 豆包访问密钥（可选，默认用 config.py 配置）
        resource_id: 资源 ID（可选，默认按音色自动检测）
        blocking: True 等待播放完成返回；False 后台播放立即返回
    """
    if not text or not text.strip():
        raise ValueError("text 不能为空")
    server = _server
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")

    tts_config = server.config.get_app_config("tts.doubao", {})
    resolved_app_id = app_id or tts_config.get("app_id")
    resolved_access_key = access_key or tts_config.get("access_key")

    if not resolved_app_id or not resolved_access_key:
        logger.warning(
            "[MCP] Doubao TTS credentials not configured, falling back to xiaoai native tts",
            module="MCP",
        )
        await speaker.play(text=text, blocking=True)
        return "已使用小爱系统原生 TTS 播报"

    try:
        resolved_speaker_id = server._resolve_speaker_id(speaker_id)
        tts = DoubaoTTS(
            app_id=resolved_app_id,
            access_key=resolved_access_key,
            resource_id=resource_id or tts_config.get("resource_id"),
            speaker=resolved_speaker_id,
        )
        resolved_format = tts.resolve_audio_format(text)
        logger.info(
            f"[MCP] Doubao TTS: speaker={resolved_speaker_id}, "
            f"resource_id={tts.resource_id}, format={resolved_format}, blocking={blocking}",
            module="MCP",
        )

        import open_xiaoai_server

        if tts_config.get("stream", False):
            play_fn = (
                open_xiaoai_server.tts_stream_play
                if blocking
                else open_xiaoai_server.tts_stream_play_background
            )
        else:
            play_fn = (
                open_xiaoai_server.tts_play
                if blocking
                else open_xiaoai_server.tts_play_background
            )
        await play_fn(
            text,
            app_id=resolved_app_id,
            access_key=resolved_access_key,
            resource_id=tts.resource_id,
            speaker=resolved_speaker_id,
            speed=speed,
            format=resolved_format,
            sample_rate=24000,
            emotion=emotion,
            context_texts=context_texts,
        )
        return (
            f"已通过豆包 TTS 播放完成（speaker={resolved_speaker_id}）"
            if blocking
            else f"已提交豆包 TTS 后台播放（speaker={resolved_speaker_id}）"
        )
    except Exception as exc:
        logger.error(f"[MCP] Doubao TTS failed, falling back to xiaoai native tts: {exc}", module="MCP")
        await speaker.play(text=text, blocking=True)
        return f"豆包 TTS 失败（{exc}），已使用小爱系统原生 TTS 播报"


async def _tool_tts_synthesize(
    text: str,
    speaker_id: str | None = None,
    speed: float = 1.0,
    format: str | None = None,
    emotion: str | None = None,
    context_texts: list[str] | None = None,
    app_id: str | None = None,
    access_key: str | None = None,
    resource_id: str | None = None,
) -> dict:
    """豆包 TTS 合成音频并返回 base64 编码的音频数据（不在音箱播放）。

    返回的音频可由调用方保存为文件或自行处理。需要豆包凭证（app_id/access_key），
    未配置时抛出错误（不回退到小爱原生 TTS，因为本工具不播放）。

    参数:
        text: 要合成的文字内容
        speaker_id: 音色 ID（可用 list_voices 查询），默认取 config.py 的 tts.doubao.default_speaker
        speed: 语速，范围 0.8-2.0，默认 1.0
        format: 音频格式 pcm/mp3/ogg_opus（默认取 config.py 的 tts.doubao.audio_format，默认 mp3）
        emotion: 情感（仅 2.0 音色支持）
        context_texts: 上下文指令列表（仅 2.0 音色支持，仅第一条生效）
        app_id: 豆包应用 ID（可选，默认用 config.py 配置）
        access_key: 豆包访问密钥（可选，默认用 config.py 配置）
        resource_id: 资源 ID（可选，默认按音色自动检测）

    返回:
        {"audio_base64": ..., "format": ..., "sample_rate": 24000,
         "speaker_id": ..., "resource_id": ..., "bytes": 音频字节数}
    """
    if not text or not text.strip():
        raise ValueError("text 不能为空")
    server = _server
    resolved_app_id, resolved_access_key, resolved_resource_id = (
        server._resolve_tts_credentials(app_id, access_key, resource_id)
    )
    resolved_speaker_id = server._resolve_speaker_id(speaker_id)

    tts = DoubaoTTS(
        app_id=resolved_app_id,
        access_key=resolved_access_key,
        resource_id=resolved_resource_id,
        speaker=resolved_speaker_id,
    )
    # 未显式指定时按 config 解析（"auto" 会按文本长度自动选择 pcm/mp3）
    audio_format = format or tts.resolve_audio_format(text)
    payload = tts._build_payload(
        text,
        format=audio_format,
        sample_rate=24000,
        speed=speed,
        context_texts=context_texts,
        emotion=emotion,
    )
    headers = {
        "X-Api-App-Id": tts.app_id,
        "X-Api-Access-Key": tts.access_key,
        "X-Api-Resource-Id": tts.resource_id,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }

    if not server._session:
        raise RuntimeError("MCPServer 会话未初始化（start() 未成功调用）")

    encoded_audio = bytearray()
    async with server._session.post(
        tts.api_url, headers=headers, json=payload, timeout=60
    ) as response:
        if response.status >= 400:
            body = (await response.text())[:500]
            raise RuntimeError(
                f"TTS 请求失败: HTTP {response.status}, resource_id={tts.resource_id}, "
                f"speaker={tts.speaker}, body={body}"
            )
        # 按行解析 JSONL 流式响应（chunk 可能跨行，需累积缓冲）
        buffer = b""
        async for chunk in response.content:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                await _handle_tts_response_line(line, encoded_audio)
        if buffer.strip():
            await _handle_tts_response_line(buffer, encoded_audio)

    if not encoded_audio:
        raise RuntimeError("TTS 合成未返回音频数据")

    return {
        "audio_base64": base64.b64encode(bytes(encoded_audio)).decode("ascii"),
        "format": audio_format,
        "sample_rate": 24000,
        "speaker_id": resolved_speaker_id,
        "resource_id": tts.resource_id,
        "bytes": len(encoded_audio),
    }


async def _handle_tts_response_line(line: bytes, encoded_audio: bytearray) -> None:
    """处理豆包 TTS 响应的单行 JSONL"""
    line = line.strip()
    if not line:
        return
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        logger.warning(f"[MCP] 忽略无法解析的 TTS 响应行: {line[:200]}", module="MCP")
        return
    code = data.get("code", 0)
    if code == 0 and data.get("data"):
        encoded_audio.extend(base64.b64decode(data["data"]))
    elif code == 20000000:
        return  # 正常结束标记
    elif code > 0:
        raise RuntimeError(f"TTS API 错误 {code}: {data.get('message')}")


async def _tool_list_voices(version: str = "all") -> dict:
    """查询豆包 TTS 可用音色列表。

    参数:
        version: "1.0"（语音合成模型 1.0）、"2.0"（支持情感/指令遵循）、"all"（全部），默认 all
    """
    if version == "2.0":
        voices = DoubaoTTS.VOICES_2_0
        return {"provider": "doubao", "version": version, "count": len(voices), "voices": voices}
    if version == "1.0":
        voices = DoubaoTTS.VOICES_1_0
        return {"provider": "doubao", "version": version, "count": len(voices), "voices": voices}
    return {
        "provider": "doubao",
        "version": "all",
        "total_voices": len(DoubaoTTS.list_voices()),
        "versions": {
            "1.0": {
                "count": len(DoubaoTTS.VOICES_1_0),
                "description": "豆包语音合成模型 1.0",
                "voices": DoubaoTTS.VOICES_1_0,
            },
            "2.0": {
                "count": len(DoubaoTTS.VOICES_2_0),
                "description": "豆包语音合成模型 2.0 - 支持情感变化、指令遵循、ASMR",
                "voices": DoubaoTTS.VOICES_2_0,
            },
        },
    }


async def _tool_play_text(text: str, blocking: bool = True, timeout_ms: int = 600000) -> bool:
    """使用小爱音箱系统原生 TTS 播报文字（设备端合成，无需豆包凭证）。

    参数:
        text: 要播报的文字内容
        blocking: True 等待播报完成返回；False 后台播报立即返回
        timeout_ms: 超时时间（毫秒），默认 600000（10 分钟）
    """
    if not text or not text.strip():
        raise ValueError("text 不能为空")
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    return await speaker.play(text=text, blocking=blocking, timeout=timeout_ms)


async def _tool_play_url(url: str, blocking: bool = True, timeout_ms: int = 600000) -> bool:
    """在小爱音箱上播放网络音频 URL（音乐、电台等，音箱端解码）。

    参数:
        url: 音频文件 URL（http/https）
        blocking: True 等待播放完成返回；False 后台播放立即返回
        timeout_ms: 超时时间（毫秒），默认 600000（10 分钟）
    """
    if not url:
        raise ValueError("url 不能为空")
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    return await speaker.play(url=url, blocking=blocking, timeout=timeout_ms)


async def _tool_play_file(
    file_path: str, blocking: bool = True, sample_rate: int = 24000
) -> bool:
    """播放桥接服务所在机器上的本地音频文件（服务端解码后推流到音箱）。

    参数:
        file_path: 本地音频文件路径
        blocking: True 等待播放完成返回；False 后台播放立即返回
        sample_rate: PCM 采样率，默认 24000
    """
    if not file_path:
        raise ValueError("file_path 不能为空")
    real_path = os.path.realpath(file_path)
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    ext = os.path.splitext(real_path)[1].lower()
    if ext not in PLAY_FILE_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式 {ext or '(无扩展名)'}，允许: {sorted(PLAY_FILE_EXTENSIONS)}"
        )
    file_size = os.path.getsize(real_path)
    if file_size > MAX_PLAY_FILE_SIZE:
        raise ValueError(f"文件过大（{file_size} 字节），超过上限 {MAX_PLAY_FILE_SIZE} 字节")
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    return await speaker.play(
        server_file=real_path, blocking=blocking, sample_rate=sample_rate
    )


async def _tool_get_status() -> dict:
    """查询小爱音箱当前播放状态。"""
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    playing = await speaker.get_playing()
    return {"playing": playing, "speaker_ready": True}


async def _tool_interrupt() -> dict:
    """打断小爱音箱当前播放（TTS/音乐/流式音频）并停止连续对话。"""
    speaker = get_speaker()
    xiaoai = get_xiaoai()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    await speaker.stop_device_audio()
    if xiaoai:
        xiaoai.stop_conversation()
    return {"interrupted": True}


async def _tool_pause() -> bool:
    """暂停音箱当前播放（mediaplayer 媒体播放，如 play_url 播放的音乐）。"""
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    return await speaker.pause_playback()


async def _tool_resume() -> bool:
    """恢复音箱暂停的媒体播放（mediaplayer）。"""
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    return await speaker.resume_playback()


def _get_stream_player():
    from core.ref import get_stream_player

    player = get_stream_player()
    if not player:
        raise RuntimeError("StreamPlayer 未初始化")
    return player


async def _tool_stream_play(
    url: str,
    headers: dict[str, str] | None = None,
    start_ms: int = 0,
    loop: bool = False,
) -> dict:
    """中转播放音频 URL：下载 → 解码 → 推流到音箱（支持后续暂停/恢复/seek/循环）。

    与 play_url 的区别：本工具由桥接服务下载并解码音频后推流播放，
    播放控制（暂停/恢复/跳进度）完全由桥接服务管理，不受音箱设备限制，
    且可自定义请求头（headers）绕过防盗链。

    参数:
        url: 音频文件 URL（http/https）
        headers: 下载请求的自定义头（如 User-Agent），用于防盗链
        start_ms: 起始播放位置（毫秒），默认 0
        loop: True 单曲循环播放
    """
    if not url:
        raise ValueError("url 不能为空")
    player = _get_stream_player()
    return await player.play(
        url, headers=headers or None, start_ms=int(start_ms), loop=bool(loop)
    )


async def _tool_stream_pause() -> dict:
    """暂停中转播放（保留位置，可恢复续播）。"""
    return await _get_stream_player().pause()


async def _tool_stream_resume() -> dict:
    """恢复暂停的中转播放。"""
    return await _get_stream_player().resume()


async def _tool_stream_seek(position_ms: int) -> dict:
    """跳转到中转播放的指定位置（如 60000 = 1:00，50% 需先查 stream_status 计算）。

    参数:
        position_ms: 目标位置（毫秒）
    """
    if position_ms < 0:
        raise ValueError("position_ms 不能为负")
    return await _get_stream_player().seek(int(position_ms))


async def _tool_stream_stop() -> dict:
    """停止中转播放并清空状态。"""
    return await _get_stream_player().stop()


async def _tool_stream_status() -> dict:
    """查询中转播放状态（state/position_ms/duration_ms/loop）。"""
    return _get_stream_player().get_status()


async def _tool_wakeup(silent: bool = True) -> bool:
    """唤醒小爱音箱（触发"小爱同学"唤醒流程）。

    参数:
        silent: True 静默唤醒（无提示音）；False 带提示音，默认 True
    """
    speaker = get_speaker()
    if not speaker:
        raise RuntimeError("Speaker 未初始化")
    return await speaker.wake_up(awake=True, silent=silent)
