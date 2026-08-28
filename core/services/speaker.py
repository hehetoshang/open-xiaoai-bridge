import asyncio
import os
import re
from typing import Literal

import open_xiaoai_server

from core.ref import get_xiaoai, set_speaker
from core.utils.base import json_decode, json_encode
from core.utils.logger import logger


class CommandResult:
    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class SpeakerManager:
    _NATIVE_TTS_RESULT_CODE_PATTERN = re.compile(r'"code"\s*:\s*0(?:\D|$)')
    _NATIVE_TTS_FILE_PATTERN = re.compile(r"/tmp/tts/[^\s'\"]+\.mp3")
    _NATIVE_TTS_END_MARKER = "event(EndReached) is posted"

    status: Literal["playing", "paused", "idle"] = "idle"

    def __init__(self):
        set_speaker(self)

    async def get_playing(self, sync=False):
        """获取播放状态"""
        if sync:
            # 同步远端最新状态
            res = await self.run_shell("mphelper mute_stat")
            if "1" in res.stdout:
                self.status = "playing"
            elif "2" in res.stdout:
                self.status = "paused"
        return self.status

    async def set_playing(self, playing=True):
        """播放/暂停"""
        command = "mphelper play" if playing else "mphelper pause"
        res = await self.run_shell(command)
        return '"code": 0' in res.stdout

    async def play(
        self,
        text=None,
        url=None,
        buffer=None,
        server_file=None,
        blocking=True,
        timeout=10 * 60 * 1000,
    ):
        """
        播放文字、音频链接、音频流

        参数:
            text: 文字内容
            url: 音频链接
            buffer: 音频流
            server_file: 服务端本地音频文件路径
            timeout: 超时时长（毫秒），默认10分钟
            blocking: 是否阻塞运行
        """
        # 阻塞式 TTS / 提示音属于短暂抢占：保留中转歌曲、位置和循环状态，
        # 并在成功、失败、超时或取消后释放本次 token。非阻塞播放没有可靠的
        # 完成事件，仍沿用替换语义，避免两路 PCM 长期重叠。
        from core.ref import get_stream_player

        stream_player = get_stream_player()
        pause_token = None
        if stream_player:
            if blocking and hasattr(stream_player, "acquire_temporary_pause"):
                pause_token = await stream_player.acquire_temporary_pause(
                    reason="speaker_play"
                )
            else:
                await stream_player.stop()

        try:
            if server_file is not None:
                return await self.play_server_file(
                    file_path=server_file,
                    blocking=blocking,
                )

            if buffer is not None:
                result = get_xiaoai().on_output_data(buffer)
                if blocking:
                    # PCM buffer 为 24kHz / int16 / mono；阻塞语义应覆盖实际
                    # 播放窗口，使临时抢占 token 不会提前恢复音乐造成混音。
                    await asyncio.sleep(len(buffer) / (24000 * 2))
                return result

            if blocking:
                native_text = text or "你好"
                escaped_text = native_text.replace("'", "'\\''")
                command = (
                    f"miplayer -f '{url}'"
                    if url
                    else f"/usr/sbin/tts_play.sh '{escaped_text}'"
                )
                res = await self.run_shell(command, timeout=timeout)
                if not url:
                    return self._native_tts_completed(res)
                return res.exit_code == 0

            if url:
                data = json_encode({"url": url, "type": 1})
                command = f"ubus call mediaplayer player_play_url '{data}'"
            else:
                data = json_encode({"text": text or "你好", "save": 0})
                command = f"ubus call mibrain text_to_speech '{data}'"

            res = await self.run_shell(command, timeout=timeout)
            return '"code": 0' in res.stdout if res else False
        finally:
            if pause_token is not None:
                await stream_player.release_temporary_pause(pause_token)

    @classmethod
    def _native_tts_completed(cls, result: CommandResult) -> bool:
        """Accept only an exit zero or a fully observed native playback.

        Some XiaoAI firmware leaves ``tts_play.sh`` with the result of a final
        shell condition used to resume previous media. When nothing was playing
        before TTS, that condition is false and the script exits 1 even though
        synthesis and ``miplayer`` both completed. The device's ``EndReached``
        event is the playback-complete signal; the successful synthesis response
        and generated file path tie it to this invocation.
        """
        if result.exit_code == 0:
            return True

        synthesis_ok = bool(
            cls._NATIVE_TTS_RESULT_CODE_PATTERN.search(result.stdout)
        )
        generated_file = bool(cls._NATIVE_TTS_FILE_PATTERN.search(result.stdout))
        playback_ended = cls._NATIVE_TTS_END_MARKER in result.stderr
        completed = synthesis_ok and generated_file and playback_ended

        if completed:
            logger.warning(
                "Blocking native TTS completed with a non-zero wrapper exit "
                f"code={result.exit_code}; accepted verified EndReached signal",
                module="Speaker",
            )
        else:
            logger.warning(
                "Blocking native TTS failed "
                f"exit_code={result.exit_code}, synthesis_ok={synthesis_ok}, "
                f"generated_file={generated_file}, playback_ended={playback_ended}",
                module="Speaker",
            )
        return completed

    async def play_server_file(
        self,
        file_path: str,
        blocking: bool = True,
        sample_rate: int = 24000,
    ) -> bool:
        """播放服务端本地音频文件（解码为 PCM 后推流到音箱）"""
        if not file_path:
            raise ValueError("file_path is required")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        logger.info(
            f"[Speaker] Playing local file via Rust audio pipeline: {file_path}, "
            f"sample_rate={sample_rate}"
        )

        if blocking:
            await open_xiaoai_server.play_audio_file(file_path, sample_rate=sample_rate)
            return True

        asyncio.create_task(
            open_xiaoai_server.play_audio_file(file_path, sample_rate=sample_rate)
        )
        return True

    async def stop_device_audio(self) -> None:
        """
        停止设备上的全部播放链路。
        aplay 不立即重启，由 Rust 侧 ensure_player_ready() 在首次
        发送音频数据时按需启动，避免空 buffer 导致 underrun。
        """
        await self.run_shell(
            "killall tts_play.sh miplayer 2>/dev/null; mphelper pause"
        )
        await open_xiaoai_server.stop_playing()

    async def pause_playback(self) -> bool:
        """暂停媒体播放器（mediaplayer）的当前播放"""
        res = await self.run_shell(
            "ubus call mediaplayer player_play_operation '{\"action\":\"pause\"}'"
        )
        return '"code": 0' in res.stdout if res else False

    async def resume_playback(self) -> bool:
        """恢复媒体播放器（mediaplayer）的暂停播放"""
        res = await self.run_shell(
            "ubus call mediaplayer player_play_operation '{\"action\":\"play\"}'"
        )
        return '"code": 0' in res.stdout if res else False

    async def wake_up(self, awake=True, silent=True):
        """
        （取消）唤醒小爱

        参数:
            awake: 是否唤醒
            silent: 是否静默唤醒
        """

        if awake:
            if silent:
                command = 'ubus call pnshelper event_notify \'{"src":1,"event":0}\''
            else:
                command = 'ubus call pnshelper event_notify \'{"src":0,"event":0}\''
        else:
            command = """
                ubus call pnshelper event_notify '{"src":3, "event":7}'
                sleep 0.1
                ubus call pnshelper event_notify '{"src":3, "event":8}'
            """
        res = await self.run_shell(command)
        return '"code": 0' in res.stdout

    async def ask_xiaoai(self, text: str, silent=False):
        """
        把文字指令交给原来的小爱执行

        参数:
            text: 文字指令
            silent: 是否静默执行
        """

        data = {"nlp": 1, "nlp_text": text}
        if not silent:
            data["tts"] = 1

        command = f"ubus call mibrain ai_service '{json_encode(data)}'"
        res = await self.run_shell(command)
        return '"code": 0' in res.stdout

    async def abort_xiaoai(self):
        """
        中断原来小爱的运行

        注意：重启需要大约 1-2s 的时间，在此期间无法使用小爱音箱自带的 TTS 服务
        """
        # Stop current audio playback first, then restart xiaoai voice service
        res = await self.run_shell("/etc/init.d/mico_aivs_lab restart >/dev/null 2>&1")
        return res.exit_code == 0

    async def get_boot(self):
        """获取启动分区"""
        res = await self.run_shell("echo $(fw_env -g boot_part)")
        return res.stdout.strip()

    async def set_boot(self, boot_part: Literal["boot0", "boot1"]):
        """设置启动分区"""
        command = f"fw_env -s boot_part {boot_part} >/dev/null 2>&1 && echo $(fw_env -g boot_part)"
        res = await self.run_shell(command)
        return boot_part in res.stdout

    async def get_device(self):
        """获取设备型号、序列号信息"""
        res = await self.run_shell("echo $(micocfg_model) $(micocfg_sn)")
        info = res.stdout.strip().split(" ")
        return {
            "model": info[0] if len(info) > 0 else "unknown",
            "sn": info[1] if len(info) > 1 else "unknown",
        }

    async def get_mic(self):
        """获取麦克风状态"""
        res = await self.run_shell("[ ! -f /tmp/mipns/mute ] && echo on || echo off")
        status = "off"
        if "on" in res.stdout:
            status = "on"
        return status

    async def set_mic(self, on=True):
        """打开/关闭麦克风"""
        if on:
            command = (
                'ubus -t1 -S call pnshelper event_notify \'{"src":3, "event":7}\' 2>&1'
            )
        else:
            command = (
                'ubus -t1 -S call pnshelper event_notify \'{"src":3, "event":8}\' 2>&1'
            )
        res = await self.run_shell(command)
        return '"code":0' in res.stdout

    async def run_shell(self, script: str, timeout=10000):
        """
        执行脚本

        参数:
            script: 脚本内容
            timeout: 超时时间（毫秒）
        """
        res = "unknown"
        try:
            res = await get_xiaoai().run_shell(script, timeout=timeout)
            data = json_decode(res)
            if isinstance(data, dict):
                return CommandResult(
                    data.get("stdout", ""),
                    data.get("stderr", ""),
                    data.get("exit_code", 0),
                )
            logger.warning(
                "Remote shell returned an invalid result",
                module="Speaker",
            )
            return CommandResult("", "invalid remote shell result", -1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"Remote shell failed (error_type={type(exc).__name__})",
                module="Speaker",
            )
            return CommandResult("", "remote shell failed", -1)
