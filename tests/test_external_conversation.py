"""Conversation behavior for explicit silent-end backend directives."""

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault(
    "open_xiaoai_server",
    types.SimpleNamespace(decode_audio=lambda *args, **kwargs: b""),
)

from core.external_conversation import ExternalConversationController
from core.openai import OpenAITurnDirective


class FakeConfig:
    def get_app_config(self, key, default=None):
        if key.endswith("exit_keywords"):
            return []
        return default


class FakeBackend:
    _rule_prompt = ""
    _session_key = "test-session"

    def __init__(self, response):
        self.response = response

    async def send(self, text, wait_response):
        return self.response

    @staticmethod
    def is_silent_end_turn_result(result):
        return result is OpenAITurnDirective.SILENT_END


def make_controller(response):
    controller = object.__new__(ExternalConversationController)
    controller.config = FakeConfig()
    controller.backend = FakeBackend(response)
    controller.active = True
    controller._loop = None
    controller._playback_token = None
    controller._vad_future = None
    controller._xiaoai_asr_future = None
    return controller


class SilentEndConversationTest(unittest.IsolatedAsyncioTestCase):
    async def test_xiaoai_asr_turn_silently_exits_without_tts(self):
        controller = make_controller(OpenAITurnDirective.SILENT_END)
        tts_calls = []

        async def recognized_text():
            return "播放测试歌曲"

        async def record_tts(text):
            tts_calls.append(text)

        controller._wait_for_xiaoai_asr_text = recognized_text
        controller._play_tts = record_tts

        result = await controller._run_one_turn_with_xiaoai_asr()

        self.assertEqual(result, "silent_exit")
        self.assertEqual(tts_calls, [])

    async def test_failure_text_is_spoken_and_turn_continues(self):
        controller = make_controller("播放失败：URL 被拒绝")
        tts_calls = []

        async def recognized_text():
            return "播放测试歌曲"

        async def no_op(*args, **kwargs):
            return None

        async def record_tts(text):
            tts_calls.append(text)

        controller._wait_for_xiaoai_asr_text = recognized_text
        controller._stop_recording = no_op
        controller._play_tts = record_tts
        controller._play_notify = no_op
        controller._start_recording = no_op

        result = await controller._run_one_turn_with_xiaoai_asr()

        self.assertEqual(result, "continue")
        self.assertEqual(tts_calls, ["播放失败：URL 被拒绝"])

    async def test_silent_exit_skips_after_wakeup_and_next_turn(self):
        controller = make_controller(OpenAITurnDirective.SILENT_END)
        turns = 0
        after_wakeup_calls = 0

        async def no_op(*args, **kwargs):
            return None

        async def silent_turn():
            nonlocal turns
            turns += 1
            return "silent_exit"

        async def after_wakeup():
            nonlocal after_wakeup_calls
            after_wakeup_calls += 1

        controller.uses_xiaoai_asr = lambda: False
        controller._stop_recording = no_op
        controller._play_notify = no_op
        controller._start_recording = no_op
        controller._run_one_turn_with_local_asr = silent_turn
        controller._call_after_wakeup = after_wakeup

        await controller._conversation_loop()

        self.assertEqual(turns, 1)
        self.assertEqual(after_wakeup_calls, 0)


if __name__ == "__main__":
    unittest.main()
