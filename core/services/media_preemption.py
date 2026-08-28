"""短暂音频抢占的状态所有权辅助函数。"""

from contextlib import asynccontextmanager

from core.ref import get_stream_player


@asynccontextmanager
async def temporary_stream_pause(reason: str):
    """暂停全局中转播放器，并保证成功、失败或取消时释放本次 token。"""
    player = get_stream_player()
    token = None
    if player and hasattr(player, "acquire_temporary_pause"):
        token = await player.acquire_temporary_pause(reason=reason)
    try:
        yield
    finally:
        if token is not None:
            await player.release_temporary_pause(token)
