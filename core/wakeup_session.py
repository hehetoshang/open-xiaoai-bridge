import asyncio
import time

from core.ref import (
    get_app,
    get_kws,
    get_speaker,
    get_xiaozhi,
)
from core.services.protocols.typing import AbortReason
from core.utils.config import ConfigManager
from core.utils.logger import logger


class WakeupSessionManager:
    """Dispatches wakeup events to XiaoZhi or external backend controllers."""

    def __init__(self):
        self.config = ConfigManager.instance()
        self._openclaw_controller = None
        self._openclaw_task: asyncio.Task | None = None
        self._openai_controller = None
        self._openai_task: asyncio.Task | None = None
        self._qwenpaw_controller = None
        self._qwenpaw_task: asyncio.Task | None = None
        self._xiaozhi_future: asyncio.Future | None = None

    def _get_loop(self):
        app = get_app()
        if app:
            return app.loop
        from core.xiaoai import XiaoAI
        return XiaoAI.async_loop

    async def _stop_device_playback(self):
        """Stop all audio playback on the device and restart recording.

        - killall tts_play.sh miplayer: stop blocking TTS (tts_play.sh + child miplayer)
        - mphelper pause: stop non-blocking TTS (mibrain text_to_speech via mediaplayer)
        - stop_playing: kill aplay (our PCM channel)
        - start_playing / start_recording: restart audio streams
        """
        speaker = get_speaker()
        if speaker:
            await speaker.stop_device_audio()
            import open_xiaoai_server
            await open_xiaoai_server.start_recording()
            return

        import open_xiaoai_server
        await open_xiaoai_server.stop_playing()
        await open_xiaoai_server.start_recording()

    def on_interrupt(self):
        logger.info("[Wakeup] XiaoAI wakeup — interrupting active sessions")

        loop = self._get_loop()

        # Stop XiaoZhi wakeup session (cancels futures + aborts server audio)
        if self._xiaozhi_future and not self._xiaozhi_future.done():
            self._xiaozhi_future.cancel()
        self._xiaozhi_future = None
        xiaozhi = get_xiaozhi()
        if xiaozhi:
            xiaozhi.stop_wakeup_session()

        # Stop external backend conversations (cancels VAD + stops TTS stream + kills aplay)
        if self._openclaw_controller and self._openclaw_controller.is_active():
            self._openclaw_controller.stop()
        if self._openclaw_task and not self._openclaw_task.done():
            loop.call_soon_threadsafe(self._openclaw_task.cancel)
        if self._openai_controller and self._openai_controller.is_active():
            self._openai_controller.stop()
        if self._openai_task and not self._openai_task.done():
            loop.call_soon_threadsafe(self._openai_task.cancel)
        if self._qwenpaw_controller and self._qwenpaw_controller.is_active():
            self._qwenpaw_controller.stop()
        if self._qwenpaw_task and not self._qwenpaw_task.done():
            loop.call_soon_threadsafe(self._qwenpaw_task.cancel)

        asyncio.run_coroutine_threadsafe(self._stop_device_playback(), loop)

        from core.xiaoai import XiaoAI
        XiaoAI.stop_conversation()

    def on_wakeup(self):
        logger.info("[Wakeup] Wakeup session started")
        xiaozhi = get_xiaozhi()
        if xiaozhi:
            xiaozhi._is_first_round = True
            future = asyncio.run_coroutine_threadsafe(
                xiaozhi.start_wakeup_session(), self._get_loop()
            )
            self._xiaozhi_future = future

            def _clear_future(done_future):
                if self._xiaozhi_future is done_future:
                    self._xiaozhi_future = None

            future.add_done_callback(_clear_future)

    def on_speech(self, speech_buffer: bytes):
        """Called by VAD when speech is detected."""
        pass

    def on_silence(self):
        """Called by VAD when silence is detected."""
        pass

    def consume_xiaoai_asr_result(
        self,
        dialog_id: str,
        text: str,
        is_final,
        is_vad_begin,
    ) -> bool:
        """Route XiaoAI native ASR results to the active external backend controller."""
        for controller in (
            self._openclaw_controller,
            self._openai_controller,
            self._qwenpaw_controller,
        ):
            if controller and controller.is_active():
                return controller.consume_xiaoai_recognize_result(
                    dialog_id=dialog_id,
                    text=text,
                    is_final=is_final,
                    is_vad_begin=is_vad_begin,
                )
        return False

    async def wakeup(self, text, source, on_route_resolved=None):
        wakeup_started_at = time.monotonic()
        before_wakeup = self.config.get_app_config("wakeup.before_wakeup")
        kws = get_kws()
        logger.debug(f"[Wakeup] Received wakeup request from {source}: {text}")

        # Reset session_key to config default before each wakeup,
        # so paths that don't call set_openclaw_session_key() always use the default.
        from core.openclaw import OpenClawManager
        default_session_key = self.config.get_app_config("openclaw", {}).get(
            "session_key", "agent:main:open-xiaoai-bridge"
        )
        OpenClawManager._session_key = default_session_key
        from core.openai import OpenAIManager
        default_openai_session_key = self.config.get_app_config(
            "openai", {}
        ).get("session_key", "agent:default:open-xiaoai-bridge")
        OpenAIManager._session_key = default_openai_session_key
        from core.qwenpaw import QwenPawManager
        default_qwenpaw_session_key = self.config.get_app_config(
            "qwenpaw", {}
        ).get("session_key", "agent:default:open-xiaoai-bridge")
        QwenPawManager._session_key = default_qwenpaw_session_key

        if kws:
            kws.pause()
        try:
            should_wakeup = await before_wakeup(
                get_speaker(),
                text,
                source,
                get_app(),
            )
        finally:
            if kws:
                kws.resume()
        logger.info(
            f"[Wakeup] phase=route_resolved elapsed_ms="
            f"{(time.monotonic() - wakeup_started_at) * 1000:.0f} "
            f"route={should_wakeup}"
        )
        if on_route_resolved is not None:
            callback_result = on_route_resolved(should_wakeup)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        if should_wakeup is not None:
            await self.reset_all_sessions()
            logger.debug(
                f"[Wakeup] phase=sessions_reset elapsed_ms="
                f"{(time.monotonic() - wakeup_started_at) * 1000:.0f}"
            )

        if should_wakeup == "openclaw":
            await self._start_openclaw_conversation(wakeup_started_at)
        elif should_wakeup == "openai":
            await self._start_openai_conversation(wakeup_started_at)
        elif should_wakeup == "qwenpaw":
            await self._start_qwenpaw_conversation(wakeup_started_at)
        elif should_wakeup == "xiaozhi":
            logger.info(
                f"[Wakeup] phase=xiaozhi_dispatched elapsed_ms="
                f"{(time.monotonic() - wakeup_started_at) * 1000:.0f}"
            )
            self.on_wakeup()

    async def _start_openclaw_conversation(self, wakeup_started_at=None):
        """Start an OpenClaw continuous conversation session.

        This runs independently of the XiaoZhi session state machine.
        KWS is paused during the conversation and resumed when done.
        """
        from core.openclaw_conversation import OpenClawConversationController

        kws = get_kws()
        task = None
        if kws:
            kws.pause()
        try:
            controller = OpenClawConversationController()
            controller.wakeup_started_at = wakeup_started_at
            task = asyncio.create_task(controller.start())
            self._openclaw_controller = controller
            self._openclaw_task = task
            await task
        except asyncio.CancelledError:
            pass  # interrupted cleanly by on_interrupt
        except Exception as exc:
            logger.error(
                f"[Wakeup] OpenClaw conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            if task is not None and self._openclaw_task is task:
                self._openclaw_controller = None
                self._openclaw_task = None
            if kws:
                kws.resume()

    async def _start_openai_conversation(self, wakeup_started_at=None):
        """Start an OpenAI-compatible continuous conversation session."""
        from core.openai_conversation import OpenAIConversationController

        kws = get_kws()
        task = None
        if kws:
            kws.pause()
        try:
            controller = OpenAIConversationController()
            controller.wakeup_started_at = wakeup_started_at
            task = asyncio.create_task(controller.start())
            self._openai_controller = controller
            self._openai_task = task
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"[Wakeup] OpenAI conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            if task is not None and self._openai_task is task:
                self._openai_controller = None
                self._openai_task = None
            if kws:
                kws.resume()

    async def _start_qwenpaw_conversation(self, wakeup_started_at=None):
        """Start a QwenPaw continuous conversation session."""
        from core.qwenpaw_conversation import QwenPawConversationController

        kws = get_kws()
        task = None
        if kws:
            kws.pause()
        try:
            controller = QwenPawConversationController()
            controller.wakeup_started_at = wakeup_started_at
            task = asyncio.create_task(controller.start())
            self._qwenpaw_controller = controller
            self._qwenpaw_task = task
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"[Wakeup] QwenPaw conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            if task is not None and self._qwenpaw_task is task:
                self._qwenpaw_controller = None
                self._qwenpaw_task = None
            if kws:
                kws.resume()

    async def reset_all_sessions(self):
        """Reset all active sessions before starting a new one.

        Stops XiaoAI continuous conversation, interrupts any active XiaoZhi
        session, and stops any external backend continuous conversation.
        """
        from core.xiaoai import XiaoAI
        from core.ref import get_xiaozhi

        # Stop XiaoAI continuous conversation
        XiaoAI.stop_conversation()

        # Interrupt active XiaoZhi session
        xiaozhi = get_xiaozhi()
        if xiaozhi and xiaozhi.is_connected():
            try:
                await xiaozhi.send_abort_speaking(AbortReason.ABORT)
            except Exception:
                pass

        # Stop OpenClaw continuous conversation (also stops its TTS stream)
        if self._openclaw_controller and self._openclaw_controller.is_active():
            self._openclaw_controller.stop()
        if self._openai_controller and self._openai_controller.is_active():
            self._openai_controller.stop()
        if self._qwenpaw_controller and self._qwenpaw_controller.is_active():
            self._qwenpaw_controller.stop()

        active_tasks = [
            task
            for task in (
                self._openclaw_task,
                self._openai_task,
                self._qwenpaw_task,
            )
            if task and not task.done()
        ]
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        # 不在普通 KWS 唤醒路径执行全局 stop。外部 controller 使用自己的
        # playback_token 停止会话 TTS；XiaoZhi 使用协议 abort。全局 stop 会
        # 错把不属于会话的音乐/队列当作清理对象，并阻塞进入监听。

        logger.debug("[Wakeup] All sessions reset")


EventManager = WakeupSessionManager()
