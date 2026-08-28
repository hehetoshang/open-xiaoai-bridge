"""StreamPlayer（中转推流播放器）单元测试（桩模式，无需音箱）"""

import asyncio
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils.config_loader import ensure_config_module_loaded

ensure_config_module_loaded()


class _FakeAEC:
    @staticmethod
    def feed_reference(*args, **kwargs):
        return None

    @staticmethod
    def reset():
        return None


# StreamPlayer tests exercise transport ownership, not the DSP implementation.
# Keep them runnable in the lightweight unit-test environment without NumPy/SciPy.
sys.modules.setdefault(
    "core.services.audio.aec",
    types.SimpleNamespace(AEC=_FakeAEC()),
)


class FakeExt:
    """open_xiaoai_server 桩：记录 on_output_data / stop_playing 调用"""

    def __init__(self, pcm_data: bytes):
        self.pcm_data = pcm_data
        self.sent = bytearray()
        self.stop_calls = 0

    def decode_audio(self, encoded, format, sample_rate):
        return self.pcm_data

    async def on_output_data(self, data):
        self.sent.extend(data)
        # Rust 端会按设备缓冲节流；桩也保留极短节流，避免整首歌在一个调度片内送完。
        await asyncio.sleep(0.001)

    async def stop_playing(self):
        self.stop_calls += 1


class FakeSession:
    """aiohttp 桩：GET 返回音频字节"""

    def __init__(self, data: bytes = b"test", content_type: str = "audio/mpeg"):
        self.data = data
        self.content_type = content_type

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, headers=None, timeout=None):
        return _Ctx(self)

    async def close(self):
        pass


class _Ctx:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return _Resp(self.session)

    async def __aexit__(self, *exc):
        return False


class _Resp:
    def __init__(self, session):
        self.status = 200
        self.headers = {"Content-Type": session.content_type}
        self.content = _Content(session.data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Content:
    def __init__(self, data):
        self._chunks = [data[:100], data[100:], b""]
        self._i = 0

    def __aiter__(self):
        self._i = 0
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._i]
        self._i += 1
        return c


PCM = bytes(range(256)) * 200  # 51200 字节 ≈ 1.07 秒
# 共享桩：stream_player 模块顶部 import 绑定的是首次导入时的对象，
# 不能每测试新建 FakeExt（否则 pump 推送到旧对象，断言读到新对象）
EXT = FakeExt(PCM)


class StreamPlayerTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules["open_xiaoai_server"] = EXT

        import core.services.stream_player as sp

        sp.aiohttp.ClientSession = FakeSession
        sp.aiohttp.ClientTimeout = lambda **kw: None

        cls.sp_mod = sp
        cls.pcm = PCM
        cls.ext = EXT

    async def asyncSetUp(self):
        os.environ["STREAM_PLAYER_ALLOW_PRIVATE_URLS"] = "1"
        os.environ.pop("STREAM_PLAYER_TRUSTED_HOSTS", None)
        EXT.sent.clear()
        EXT.stop_calls = 0
        # 清掉可能被其他测试设置的 speaker 桩，避免污染
        from core.ref import set_speaker

        set_speaker(None)
        self.sp = self.sp_mod.StreamPlayer()

    async def asyncTearDown(self):
        await self.sp.close()
        os.environ.pop("STREAM_PLAYER_ALLOW_PRIVATE_URLS", None)
        os.environ.pop("STREAM_PLAYER_TRUSTED_HOSTS", None)

    async def test_play_decodes_and_pumps(self):
        status = await self.sp.play("http://example.com/song.mp3")
        self.assertEqual(status["state"], "playing")
        self.assertEqual(status["duration_ms"], int(len(self.pcm) / 48000 * 1000))
        # 等推流任务跑几轮
        for _ in range(200):
            if self.ext.sent:
                break
            await asyncio.sleep(0.01)
        self.assertGreater(len(self.ext.sent), 0)
        # 播放完成后（非循环）状态回到 idle
        for _ in range(500):
            if self.sp.state == "idle":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.sp.state, "idle")
        self.assertEqual(bytes(self.ext.sent), self.pcm)

    async def test_pause_resume_preserves_position(self):
        await self.sp.play("http://example.com/song.mp3")
        # 等推送一部分
        for _ in range(100):
            if self.ext.sent:
                break
            await asyncio.sleep(0.01)
        paused_pos = self.sp.offset
        self.assertGreater(paused_pos, 0)

        status = await self.sp.pause()
        self.assertEqual(status["state"], "paused")
        self.assertEqual(self.sp.offset, paused_pos)  # 位置保留
        sent_at_pause = len(self.ext.sent)

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.ext.sent), sent_at_pause)  # 暂停后不再推送

        status = await self.sp.resume()
        self.assertEqual(status["state"], "playing")
        for _ in range(100):
            if len(self.ext.sent) > sent_at_pause:
                break
            await asyncio.sleep(0.01)
        self.assertGreater(len(self.ext.sent), sent_at_pause)

    async def test_temporary_pause_preserves_song_and_resumes_at_position(self):
        await self.sp.play("http://example.com/song.mp3", loop=True)
        for _ in range(100):
            if self.sp.offset:
                break
            await asyncio.sleep(0.01)
        original_url = self.sp.file_path
        original_pcm = self.sp.pcm
        original_position = self.sp.offset

        token = await self.sp.acquire_temporary_pause("test_tts")
        self.assertEqual(self.sp.state, "paused")
        self.assertEqual(self.sp.file_path, original_url)
        self.assertIs(self.sp.pcm, original_pcm)
        self.assertEqual(self.sp.offset, original_position)

        status = await self.sp.release_temporary_pause(token)
        self.assertEqual(status["state"], "playing")
        self.assertEqual(self.sp.file_path, original_url)

    async def test_nested_temporary_pauses_resume_only_after_last_release(self):
        await self.sp.play("http://example.com/song.mp3", loop=True)
        first = await self.sp.acquire_temporary_pause("first_wakeup")
        second = await self.sp.acquire_temporary_pause("repeated_wakeup")

        await self.sp.release_temporary_pause(first)
        self.assertEqual(self.sp.state, "paused")
        await self.sp.release_temporary_pause(second)
        self.assertEqual(self.sp.state, "playing")

    async def test_explicit_pause_during_preemption_prevents_auto_resume(self):
        await self.sp.play("http://example.com/song.mp3", loop=True)
        token = await self.sp.acquire_temporary_pause("test_tts")

        await self.sp.pause()
        await self.sp.release_temporary_pause(token)

        self.assertEqual(self.sp.state, "paused")
        self.assertTrue(self.sp.pcm)

    async def test_explicit_stop_invalidates_preemption_token(self):
        await self.sp.play("http://example.com/song.mp3", loop=True)
        token = await self.sp.acquire_temporary_pause("test_tts")

        await self.sp.stop()
        status = await self.sp.release_temporary_pause(token)

        self.assertEqual(status["state"], "idle")
        self.assertEqual(self.sp.pcm, b"")

    async def test_temporary_pause_bounds_unresponsive_stop_rpc(self):
        async def blocked_stop_playing():
            await asyncio.Future()

        self.sp.pcm = self.pcm
        self.sp.state = "playing"
        self.sp._task = asyncio.create_task(asyncio.sleep(60))
        with (
            patch.object(self.ext, "stop_playing", blocked_stop_playing),
            patch.object(self.sp_mod, "STOP_PLAYING_TIMEOUT_SECONDS", 0.01),
        ):
            token = await asyncio.wait_for(
                self.sp.acquire_temporary_pause("test_timeout"),
                timeout=0.2,
            )

        self.assertEqual(self.sp.state, "paused")
        await self.sp.release_temporary_pause(token)
        self.assertEqual(self.sp.state, "playing")

    async def test_next_song_invalidates_old_preemption_token(self):
        await self.sp.play("http://example.com/first.mp3", loop=True)
        token = await self.sp.acquire_temporary_pause("test_tts")

        await self.sp.play("http://example.com/next.mp3", loop=True)
        await self.sp.release_temporary_pause(token)

        self.assertEqual(self.sp.state, "playing")
        self.assertEqual(self.sp.file_path, "http://example.com/next.mp3")

    async def test_seek_jumps_position(self):
        await self.sp.play("http://example.com/song.mp3")
        target = int(len(self.pcm) / 2)
        status = await self.sp.seek(int(target / 48000 * 1000))
        self.assertGreaterEqual(self.sp.offset, target - 480)  # 误差 < 10ms
        self.assertEqual(status["position_ms"], int(self.sp.offset / 48000 * 1000))

    async def test_seek_without_content_raises(self):
        with self.assertRaises(RuntimeError):
            await self.sp.seek(1000)

    async def test_loop_rewinds(self):
        await self.sp.play("http://example.com/song.mp3", loop=True)
        # 等完整播完一轮并回绕
        for _ in range(600):
            if self.sp.state == "playing" and self.sp.offset < 480 and len(self.ext.sent) > len(self.pcm):
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.sp.state, "playing")  # 循环中不结束
        self.assertGreater(len(self.ext.sent), len(self.pcm))  # 已推送超过一轮

    async def test_stop_clears_state(self):
        await self.sp.play("http://example.com/song.mp3")
        status = await self.sp.stop()
        self.assertEqual(status["state"], "idle")
        self.assertEqual(self.sp.pcm, b"")
        self.assertEqual(self.sp.duration_ms, 0)

    async def test_stop_from_foreign_loop_runs_on_owner_loop(self):
        """XiaoAI 回调跨 loop 停止 MainApp.loop 上的 pump 不会抛错。"""
        owner_loop = asyncio.new_event_loop()
        owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        owner_thread.start()
        player = self.sp_mod.StreamPlayer(loop=owner_loop)

        async def start_pump():
            player.pcm = self.pcm
            player.state = "playing"
            player._task = asyncio.create_task(asyncio.sleep(60))

        try:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(start_pump(), owner_loop)
            )
            status = await player.stop()
            self.assertEqual("idle", status["state"])
            self.assertIsNone(player._task)
        finally:
            await player.close()
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            owner_thread.join(timeout=2)
            owner_loop.close()

    async def test_start_ms_skips_beginning(self):
        await self.sp.play("http://example.com/song.mp3", start_ms=500)
        expected_offset = int(500 / 1000 * 48000)
        self.assertGreaterEqual(self.sp.offset, expected_offset - 480)

    async def test_play_interrupts_device_audio(self):
        """新播放请求打断设备其他通道（stop_device_audio 被调用）"""
        from core.ref import set_speaker

        calls = []

        class FakeSpeaker:
            async def stop_device_audio(self):
                calls.append("stop_device_audio")

        set_speaker(FakeSpeaker())
        await self.sp.play("http://example.com/song.mp3")
        self.assertIn("stop_device_audio", calls)
        self.assertEqual(self.sp.state, "playing")

    async def test_stream_url_rejects_private_network_and_unsupported_scheme(self):
        os.environ.pop("STREAM_PLAYER_ALLOW_PRIVATE_URLS", None)
        with self.assertRaisesRegex(ValueError, "http/https"):
            await self.sp_mod._validate_stream_url("file:///etc/passwd")
        with self.assertRaisesRegex(ValueError, "私有网络"):
            await self.sp_mod._validate_stream_url("http://127.0.0.1/audio.mp3")

    async def test_stream_url_allows_configured_cdn_suffix_with_proxy_fake_ip(self):
        os.environ.pop("STREAM_PLAYER_ALLOW_PRIVATE_URLS", None)
        os.environ["STREAM_PLAYER_TRUSTED_HOSTS"] = "music.126.net"
        fake_ip = [(2, 1, 6, "", ("198.18.0.61", 443))]
        with patch.object(self.sp_mod.socket, "getaddrinfo", return_value=fake_ip):
            await self.sp_mod._validate_stream_url(
                "https://m801.music.126.net/song.mp3"
            )

    async def test_stream_url_does_not_trust_unrelated_fake_ip_host(self):
        os.environ.pop("STREAM_PLAYER_ALLOW_PRIVATE_URLS", None)
        os.environ["STREAM_PLAYER_TRUSTED_HOSTS"] = "music.126.net"
        fake_ip = [(2, 1, 6, "", ("198.18.0.61", 443))]
        with patch.object(self.sp_mod.socket, "getaddrinfo", return_value=fake_ip):
            with self.assertRaisesRegex(ValueError, "私有网络"):
                await self.sp_mod._validate_stream_url(
                    "https://attacker.example/song.mp3"
                )

    async def test_blocking_speaker_play_temporarily_pauses_stream_player(self):
        """阻塞式 TTS 只取得临时暂停 token，结束后自动释放。"""
        from core.ref import set_stream_player

        calls = []

        class FakeStreamPlayer:
            async def acquire_temporary_pause(self, reason):
                calls.append(("acquire", reason))
                return 7

            async def release_temporary_pause(self, token):
                calls.append(("release", token))

            async def stop(self):
                calls.append(("stop", None))

        set_stream_player(FakeStreamPlayer())

        from core.services.speaker import CommandResult, SpeakerManager

        sm = SpeakerManager()
        sm.run_shell = self._fake_run_shell

        await sm.play(text="你好")
        self.assertEqual(
            calls,
            [("acquire", "speaker_play"), ("release", 7)],
        )

    async def test_blocking_speaker_failure_still_releases_temporary_pause(self):
        from core.ref import set_stream_player

        calls = []

        class FakeStreamPlayer:
            async def acquire_temporary_pause(self, reason):
                calls.append("acquire")
                return 11

            async def release_temporary_pause(self, token):
                calls.append(("release", token))

        async def fail_run_shell(*args, **kwargs):
            raise RuntimeError("device unavailable")

        set_stream_player(FakeStreamPlayer())
        from core.services.speaker import SpeakerManager

        sm = SpeakerManager()
        sm.run_shell = fail_run_shell
        with self.assertRaisesRegex(RuntimeError, "device unavailable"):
            await sm.play(text="你好")
        self.assertEqual(calls, ["acquire", ("release", 11)])

    async def test_cancelled_speaker_play_still_releases_temporary_pause(self):
        from core.ref import set_stream_player

        calls = []
        started = asyncio.Event()

        class FakeStreamPlayer:
            async def acquire_temporary_pause(self, reason):
                calls.append("acquire")
                return 13

            async def release_temporary_pause(self, token):
                calls.append(("release", token))

        async def blocked_run_shell(*args, **kwargs):
            started.set()
            await asyncio.Future()

        set_stream_player(FakeStreamPlayer())
        from core.services.speaker import SpeakerManager

        sm = SpeakerManager()
        sm.run_shell = blocked_run_shell
        task = asyncio.create_task(sm.play(text="你好"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(calls, ["acquire", ("release", 13)])

    async def _fake_run_shell(self, script, timeout=0):
        from core.services.speaker import CommandResult

        return CommandResult(stdout='"code": 0', stderr="", exit_code=0)


if __name__ == "__main__":
    unittest.main()
