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
# Rust stop_playing() 最终等待设备 RPC；链路异常时不能无限阻塞唤醒/TTS。
STOP_PLAYING_TIMEOUT_SECONDS = float(
    os.environ.get("STREAM_PLAYER_STOP_TIMEOUT_SECONDS", "1.5")
)


def _trusted_stream_hosts() -> set[str]:
    """Return explicitly trusted media hosts/suffixes from the environment.

    Some transparent proxies resolve public media CDNs to RFC 2544 benchmark
    addresses (198.18.0.0/15).  Those addresses must remain blocked by default,
    but an operator can trust a controlled CDN suffix without disabling SSRF
    protection for every private destination.
    """
    return {
        host.strip().lower().strip(".")
        for host in os.environ.get("STREAM_PLAYER_TRUSTED_HOSTS", "").split(",")
        if host.strip().strip(".")
    }


def _is_trusted_stream_host(hostname: str) -> bool:
    normalized = hostname.lower().strip(".")
    return any(
        normalized == trusted or normalized.endswith(f".{trusted}")
        for trusted in _trusted_stream_hosts()
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
    trusted_host = _is_trusted_stream_host(parsed.hostname)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global and not trusted_host:
            raise ValueError("播放 URL 不允许访问本机或私有网络地址")


class StreamPlayer:
    """中转流媒体播放器：下载 → 解码 PCM → 推流（支持暂停/恢复/seek/循环）"""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self._loop = loop
        self.state = "idle"  # idle / playing / paused
        self.file_path: str | None = None
        self.pcm: bytes = b""
        self.offset: int = 0  # PCM 字节偏移
        self.duration_ms: int = 0
        self.loop: bool = False
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        # 临时抢占（TTS / 提示音）只暂停，不清空歌曲。token 使并发/重复
        # 抢占只有最后一个持有者释放时才恢复，并避免恢复用户主动停止的音乐。
        self._preemption_serial = 0
        self._preemption_tokens: set[int] = set()
        self._preemption_should_resume = False
        set_stream_player(self)

    # ---------- 生命周期 ----------

    async def close(self) -> None:
        """释放资源（应用关闭时调用）"""
        await self._run_on_owner_loop(self._close())

    async def _close(self) -> None:
        await self._stop()
        if self._session:
            await self._session.close()
            self._session = None

    async def _run_on_owner_loop(self, coroutine):
        """Run mutable player state on MainApp.loop even for XiaoAI callbacks."""
        current_loop = asyncio.get_running_loop()
        if self._loop is None or self._loop is current_loop:
            return await coroutine
        if not self._loop.is_running():
            coroutine.close()
            raise RuntimeError("StreamPlayer owner loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return await asyncio.wrap_future(future)

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
        return await self._run_on_owner_loop(
            self._play(url, headers=headers, start_ms=start_ms, loop=loop)
        )

    async def _play(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        start_ms: int = 0,
        loop: bool = False,
    ) -> dict:
        await self._stop()
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
        return await self._run_on_owner_loop(self._pause())

    async def _pause(self) -> dict:
        # 即使当前已被临时抢占暂停，用户再次发出 pause 也应取消自动恢复。
        if self._preemption_tokens:
            self._preemption_should_resume = False
        if self.state != "playing":
            return self.get_status()
        await self._cancel_pump()
        self.state = "paused"
        logger.info(f"[StreamPlayer] paused at {self._offset_ms()}ms", module="StreamPlayer")
        return self.get_status()

    async def resume(self) -> dict:
        """从暂停位置恢复播放"""
        return await self._run_on_owner_loop(self._resume())

    async def _resume(self) -> dict:
        # 用户显式恢复取得状态所有权，后续旧 token 不得再次操作播放状态。
        self._preemption_tokens.clear()
        self._preemption_should_resume = False
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
        return await self._run_on_owner_loop(self._seek(position_ms))

    async def _seek(self, position_ms: int) -> dict:
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
        return await self._run_on_owner_loop(self._stop())

    async def _stop(self) -> dict:
        # stop/next 等用户操作会使所有旧临时抢占 token 失效。
        self._preemption_tokens.clear()
        self._preemption_should_resume = False
        await self._cancel_pump()
        self.state = "idle"
        self.file_path = None
        self.pcm = b""
        self.offset = 0
        self.duration_ms = 0
        self.loop = False
        return self.get_status()

    async def acquire_temporary_pause(self, reason: str = "transient_audio") -> int:
        """为短暂音频抢占创建可幂等释放的暂停 token。

        首个 token 负责暂停当前歌曲；嵌套或重复唤醒只增加持有者，不会
        重复 pause。原本已经 paused/idle 的播放器不会被错误自动恢复。
        """
        return await self._run_on_owner_loop(
            self._acquire_temporary_pause(reason)
        )

    async def _acquire_temporary_pause(self, reason: str) -> int:
        self._preemption_serial += 1
        token = self._preemption_serial

        if not self._preemption_tokens:
            self._preemption_should_resume = self.state == "playing"
            if self._preemption_should_resume:
                await self._cancel_pump()
                self.state = "paused"
                logger.info(
                    f"[StreamPlayer] temporarily paused at {self._offset_ms()}ms "
                    f"for {reason}",
                    module="StreamPlayer",
                )

        self._preemption_tokens.add(token)
        return token

    async def release_temporary_pause(self, token: int) -> dict:
        """释放临时暂停；仅恢复仍由这组 token 拥有的 paused 状态。"""
        return await self._run_on_owner_loop(
            self._release_temporary_pause(token)
        )

    async def _release_temporary_pause(self, token: int) -> dict:
        if token not in self._preemption_tokens:
            return self.get_status()

        self._preemption_tokens.remove(token)
        if self._preemption_tokens:
            return self.get_status()

        should_resume = self._preemption_should_resume
        self._preemption_should_resume = False
        if should_resume and self.state == "paused" and self.pcm:
            self.state = "playing"
            self._start_pump()
            logger.info(
                f"[StreamPlayer] automatically resumed at {self._offset_ms()}ms",
                module="StreamPlayer",
            )
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
            "temporarily_paused": bool(self._preemption_tokens),
        }

    def has_resumable_media(self) -> bool:
        """播放会话是否仍由临时抢占持有，结束后可自动续播。"""
        return bool(
            self.pcm
            and (
                self.state == "playing"
                or (
                    self._preemption_tokens
                    and self._preemption_should_resume
                )
            )
        )

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
        try:
            await asyncio.wait_for(
                open_xiaoai_server.stop_playing(),
                timeout=STOP_PLAYING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[StreamPlayer] stop_playing RPC timed out; continuing with "
                "local pump stopped",
                module="StreamPlayer",
            )
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
