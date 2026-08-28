"""SpeakerManager native blocking TTS completion tests."""

import asyncio
import sys
import types
import unittest

_previous_open_xiaoai_server = sys.modules.get("open_xiaoai_server")
sys.modules["open_xiaoai_server"] = types.SimpleNamespace()

from core.ref import set_stream_player, set_xiaoai
from core.services.speaker import CommandResult, SpeakerManager

if _previous_open_xiaoai_server is None:
    sys.modules.pop("open_xiaoai_server", None)
else:
    sys.modules["open_xiaoai_server"] = _previous_open_xiaoai_server


class SpeakerManagerNativeTTSTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        set_stream_player(None)

    async def test_nonzero_exit_with_verified_end_reached_is_success(self):
        speaker = SpeakerManager()

        async def completed_with_wrapper_failure(_script, timeout=0):
            return CommandResult(
                stdout=(
                    '{"code": 0}\n'
                    "/tmp/tts/tts_dialog_123.mp3\n"
                ),
                stderr="miplayer: event(EndReached) is posted\n",
                exit_code=1,
            )

        speaker.run_shell = completed_with_wrapper_failure

        self.assertTrue(
            await speaker.play(text="好的，正在处理", blocking=True, timeout=1000)
        )

    async def test_nonzero_exit_without_complete_evidence_is_failure(self):
        incomplete_results = (
            CommandResult(
                stdout='/tmp/tts/tts_dialog_123.mp3\n',
                stderr="miplayer: event(EndReached) is posted\n",
                exit_code=1,
            ),
            CommandResult(
                stdout='{"code": 0}\n',
                stderr="miplayer: event(EndReached) is posted\n",
                exit_code=1,
            ),
            CommandResult(
                stdout='{"code": 0}\n/tmp/tts/tts_dialog_123.mp3\n',
                stderr="miplayer: playback error\n",
                exit_code=1,
            ),
            CommandResult(
                stdout='{"code": 0}\n/tmp/tts/tts_dialog_123.mp3\n',
                stderr="miplayer terminated by signal\n",
                exit_code=143,
            ),
        )

        for result in incomplete_results:
            with self.subTest(result=result.__dict__):
                self.assertFalse(SpeakerManager._native_tts_completed(result))

    async def test_remote_timeout_remains_failure(self):
        class TimedOutXiaoAI:
            async def run_shell(self, _script, timeout=0):
                raise asyncio.TimeoutError

        set_xiaoai(TimedOutXiaoAI())
        speaker = SpeakerManager()

        self.assertFalse(
            await speaker.play(text="好的，正在处理", blocking=True, timeout=1)
        )

    async def test_cancellation_propagates_as_interruption(self):
        class InterruptedXiaoAI:
            async def run_shell(self, _script, timeout=0):
                raise asyncio.CancelledError

        set_xiaoai(InterruptedXiaoAI())
        speaker = SpeakerManager()

        with self.assertRaises(asyncio.CancelledError):
            await speaker.play(text="好的，正在处理", blocking=True, timeout=1000)


if __name__ == "__main__":
    unittest.main()
