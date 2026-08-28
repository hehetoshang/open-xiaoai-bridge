"""Wakeup session media-ownership regressions."""

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())

from core.ref import set_xiaozhi
from core.wakeup_session import WakeupSessionManager
from config import before_wakeup


class WakeupSessionResetTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        set_xiaozhi(None)
        self.manager = WakeupSessionManager()

    async def test_normal_session_reset_does_not_stop_unowned_media(self):
        stop_calls = []

        async def destructive_global_stop():
            stop_calls.append("stop")

        self.manager._stop_device_playback = destructive_global_stop
        fake_xiaoai = types.SimpleNamespace(
            XiaoAI=types.SimpleNamespace(stop_conversation=lambda: None)
        )
        with patch.dict(sys.modules, {"core.xiaoai": fake_xiaoai}):
            await self.manager.reset_all_sessions()

        self.assertEqual(stop_calls, [])

    async def test_default_kws_routes_do_not_wait_for_spoken_confirmation(self):
        class FailOnPlayback:
            async def play(self, **kwargs):
                raise AssertionError("KWS route must not block on TTS")

        speaker = FailOnPlayback()
        expected_routes = {
            "你好龙虾": "openclaw",
            "你好小黑": "openai",
            "你好小爪": "qwenpaw",
            "你好小智": "xiaozhi",
        }
        for text, expected in expected_routes.items():
            with self.subTest(text=text):
                route = await before_wakeup(speaker, text, "kws", None)
                self.assertEqual(route, expected)

    async def test_default_xiaoai_routes_do_not_restart_native_service(self):
        class FailOnAbort:
            async def abort_xiaoai(self):
                raise AssertionError("XiaoAI route must not restart mico_aivs_lab")

        speaker = FailOnAbort()
        expected_routes = {
            "召唤龙虾": "openclaw",
            "召唤小黑": "openai",
            "召唤小爪": "qwenpaw",
            "召唤小智": "xiaozhi",
        }
        for text, expected in expected_routes.items():
            with self.subTest(text=text):
                route = await before_wakeup(speaker, text, "xiaoai", None)
                self.assertEqual(route, expected)

    async def test_reset_cancels_previous_external_task_before_replacement(self):
        started = asyncio.Event()

        async def previous_session():
            started.set()
            await asyncio.Future()

        task = asyncio.create_task(previous_session())
        await started.wait()
        controller = types.SimpleNamespace(
            is_active=lambda: True,
            stop=lambda: None,
        )
        self.manager._openai_controller = controller
        self.manager._openai_task = task

        fake_xiaoai = types.SimpleNamespace(
            XiaoAI=types.SimpleNamespace(stop_conversation=lambda: None)
        )
        with patch.dict(sys.modules, {"core.xiaoai": fake_xiaoai}):
            await self.manager.reset_all_sessions()

        self.assertTrue(task.cancelled())


if __name__ == "__main__":
    unittest.main()
