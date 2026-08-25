"""通用流媒体中转播放器：下载音频 → 解码 PCM → 推流到音箱

中转推流模式下音频解码在本进程内完成，因此暂停/恢复/seek 全部由
本模块控制（PCM 缓冲 + 播放偏移），不受音箱设备能力限制。

推流通过 `open_xiaoai_server.on_output_data()` 分块发送，Rust 侧
负责节流（设备最多超前 1500ms）与 aplay 通道管理。
"""

import asyncio
import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import open_xiaoai_server

from core.ref import set_stream_player
from core.services.audio.aec import AEC
from core.utils.logger import logger

# 推流分块大小：24kHz * 2B * 0.06s ≈ 2880B（60ms，与 Rust 侧一致）
CHUNK_BYTES = 2880
# PCM 参数：16bit 单声道 24kHz
BYTES_PER_SECOND = 48000
MAX_DOWNLOAD_BYTES = int(
    os.environ.get("STREAM_PLAYER_MAX_DOWNLOAD_BYTES", 128 * 1024 * 1024)
)

# 支持的音频格式（URL 扩展名 / Content-Type 映射）
_FORMAT_BY_EXT = {
    ".mp3": "mp3",
    ".flac": "flac",
    ".wav": "wav",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".m4a": "m4a",
    ".aac": "aac",
}
_FORMAT_BY_CONTENT_TYPE = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
}


def _infer_format(url: str, content_type: str | None) -> str:
    """从 URL 扩展名 / Content-Type 推断音频格式"""
    path = url.split("?", 1)[0].lower()
    for ext, fmt in _FORMAT_BY_EXT.items():
        if path.endswith(ext):
            return fmt
    if content_type:
        content_type_lower = content_type.lower()
        for ct, fmt in _FORMAT_BY_CONTENT_TYPE.items():
            if content_type_lower.startswith(ct):
                return fmt
    return "mp3"  # 兜底（大多数流媒体默认 mp3）


async def _validate_stream_url(url: str) -> None:
    """阻止控制面被用作访问本机、内网或云元数据的 SSRF 代理。"""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("播放 URL 仅支持 http/https")
    if parsed.username or parsed.password:
        raise ValueError("播放 URL 不允许嵌入凭据")
    if os.environ.get("STREAM_PLAYER_ALLOW_PRIVATE_URLS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    loop = asyncio.get_running_loop()
    addresses = await loop.run_in_executor(
        None,
        lambda: socket.getaddrinfo(
            parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
        ),
    )
    if not addresses:
        raise ValueError("播放 URL 主机无法解析")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("播放 URL 不允许访问本机或私有网络地址")


class StreamPlayer:
    """中转流媒体播放器：下载 → 解码 PCM → 推流（支持暂停/恢复/seek/循环）"""

    def __init__(self):
        self.state = "idle"  # idle / playing / paused
        self.file_path: str | None = None
        self.pcm: bytes = b""
        self.offset: int = 0  # PCM 字节偏移
        self.duration_ms: int = 0
        self.loop: bool = False
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        set_stream_player(self)

    # ---------- 生命周期 ----------

    async def close(self) -> None:
        """释放资源（应用关闭时调用）"""
        await self.stop()
        if self._session:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    # ---------- 播放控制 ----------

    async def play(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        start_ms: int = 0,
        loop: bool = False,
    ) -> dict:
        """下载并播放音频 URL（替换当前播放，并打断其他播放通道）"""
        await self.stop()
        await _validate_stream_url(url)
        # 打断设备其他播放通道（mediaplayer / TTS），保证同一时刻只有一路声音
        from core.ref import get_speaker

        speaker = get_speaker()
        if speaker:
            await speaker.stop_device_audio()
        logger.info(
            f"[StreamPlayer] play host={urlsplit(url).hostname}, start_ms={start_ms}, loop={loop}",
            module="StreamPlayer",
        )

        # 1) 下载到临时文件
        session = self._get_session()
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"下载失败: HTTP {resp.status}")
            content_type = resp.headers.get("Content-Type")
            ext = _infer_format(url, content_type)
            # 读入内存（流式读取）
            data = bytearray()
            async for chunk in resp.content:
                data.extend(chunk)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("下载内容超过大小限制")
        encoded = bytes(data)
        if not encoded:
            raise RuntimeError("下载内容为空")

        # 2) 解码为 PCM
        pcm = open_xiaoai_server.decode_audio(encoded, format=ext, sample_rate=24000)
        if not pcm:
            raise RuntimeError(f"解码失败: format={ext}")

        self.file_path = url
        self.pcm = pcm
        self.duration_ms = int(len(pcm) / BYTES_PER_SECOND * 1000)
        self.loop = loop
        self.offset = max(0, min(int(start_ms / 1000 * BYTES_PER_SECOND), len(pcm)))
        self.state = "playing"
        self._start_pump()
        logger.info(
            f"[StreamPlayer] started: duration_ms={self.duration_ms}, loop={loop}",
            module="StreamPlayer",
        )
        return self.get_status()

    async def pause(self) -> dict:
        """暂停播放（保留位置，可 resume 续播）"""
        if self.state != "playing":
            return self.get_status()
        await self._cancel_pump()
        self.state = "paused"
        logger.info(f"[StreamPlayer] paused at {self._offset_ms()}ms", module="StreamPlayer")
        return self.get_status()

    async def resume(self) -> dict:
        """从暂停位置恢复播放"""
        if self.state != "paused":
            return self.get_status()
        if not self.pcm:
            self.state = "idle"
            return self.get_status()
        self.state = "playing"
        self._start_pump()
        logger.info(f"[StreamPlayer] resumed at {self._offset_ms()}ms", module="StreamPlayer")
        return self.get_status()

    async def seek(self, position_ms: int) -> dict:
        """跳到指定位置（毫秒）"""
        if not self.pcm:
            raise RuntimeError("当前没有可 seek 的播放内容")
        was_playing = self.state == "playing"
        await self._cancel_pump()
        self.offset = max(0, min(int(position_ms / 1000 * BYTES_PER_SECOND), len(self.pcm)))
        if was_playing:
            self.state = "playing"
            self._start_pump()
        else:
            self.state = "paused"
        logger.info(f"[StreamPlayer] seeked to {self._offset_ms()}ms", module="StreamPlayer")
        return self.get_status()

    async def stop(self) -> dict:
        """停止播放并清空状态"""
        await self._cancel_pump()
        self.state = "idle"
        self.file_path = None
        self.pcm = b""
        self.offset = 0
        self.duration_ms = 0
        self.loop = False
        return self.get_status()

    def get_status(self) -> dict:
        """当前播放状态（同步，无 IO）"""
        return {
            "state": self.state,
            "position_ms": self._offset_ms(),
            "duration_ms": self.duration_ms,
            "loop": self.loop,
            "playing": self.state == "playing",
            "paused": self.state == "paused",
        }

    # ---------- 内部 ----------

    def _offset_ms(self) -> int:
        return int(self.offset / BYTES_PER_SECOND * 1000) if self.pcm else 0

    def _start_pump(self) -> None:
        """启动推流任务（同一时刻只有一个）"""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._pump())

    async def _cancel_pump(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        await open_xiaoai_server.stop_playing()
        AEC.reset()

    async def _pump(self) -> None:
        """推流循环：从 offset 分块发送 PCM，末尾按 loop 回绕"""
        try:
            while self.pcm:
                chunk = self.pcm[self.offset : self.offset + CHUNK_BYTES]
                if not chunk:
                    if self.loop:
                        self.offset = 0
                        continue
                    break
                AEC.feed_reference(chunk, sample_rate=24000, channels=1)
                await open_xiaoai_server.on_output_data(chunk)
                self.offset += len(chunk)
            if not self.loop:
                # 播放完成
                self.state = "idle"
                self.offset = 0
                logger.info("[StreamPlayer] playback finished", module="StreamPlayer")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[StreamPlayer] pump failed: {exc}", module="StreamPlayer")
            self.state = "idle"
