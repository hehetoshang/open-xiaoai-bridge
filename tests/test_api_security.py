"""HTTP API 安全策略测试。"""

import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import ClientSession, web

from core.services.api_security import (
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    bearer_middleware,
    build_server_ssl_context,
    load_or_create_token,
    validate_stream_request,
)


class TokenStorageTest(unittest.TestCase):
    def test_token_is_generated_once_with_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "api-token"
            with patch.dict(
                os.environ,
                {TOKEN_FILE_ENV: str(path)},
                clear=False,
            ):
                os.environ.pop(TOKEN_ENV, None)
                first, source = load_or_create_token()
                second, _ = load_or_create_token()
            self.assertEqual(first, second)
            self.assertEqual(path, source)
            self.assertGreaterEqual(len(first), 32)
            self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))

    def test_short_explicit_token_is_rejected(self):
        with patch.dict(os.environ, {TOKEN_ENV: "short"}, clear=False):
            with self.assertRaisesRegex(ValueError, "at least 32"):
                load_or_create_token()

    def test_remote_plaintext_listener_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(build_server_ssl_context("127.0.0.1"))
            with self.assertRaisesRegex(ValueError, "non-loopback plaintext"):
                build_server_ssl_context("0.0.0.0")

    def test_malformed_and_oversized_stream_commands_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "object"):
            validate_stream_request([])
        with self.assertRaisesRegex(ValueError, "2048"):
            validate_stream_request({"url": "https://example.com/" + "x" * 2048})
        with self.assertRaisesRegex(ValueError, "headers"):
            validate_stream_request(
                {"url": "https://example.com/a.mp3", "headers": {"Authorization": "secret"}}
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_stream_request({"url": "https://example.com/a.mp3", "start_ms": -1})
        self.assertEqual(
            ("https://example.com/a.mp3", None, 0, False),
            validate_stream_request({"url": "https://example.com/a.mp3"}),
        )


class BearerMiddlewareTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.token = "a" * 43
        app = web.Application(middlewares=[bearer_middleware(self.token)])

        async def control(_request):
            return web.json_response({"success": True})

        app.router.add_post("/control", control)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        server = self.site._server
        self.port = server.sockets[0].getsockname()[1]
        self.session = ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def test_missing_and_wrong_credentials_are_rejected(self):
        base = f"http://127.0.0.1:{self.port}/control"
        async with self.session.post(base) as response:
            self.assertEqual(401, response.status)
        async with self.session.post(
            base, headers={"Authorization": "Bearer wrong"}
        ) as response:
            self.assertEqual(401, response.status)

    async def test_valid_credential_reaches_handler(self):
        base = f"http://127.0.0.1:{self.port}/control"
        async with self.session.post(
            base, headers={"Authorization": f"Bearer {self.token}"}
        ) as response:
            self.assertEqual(200, response.status)
            self.assertEqual("no-store", response.headers["Cache-Control"])


if __name__ == "__main__":
    unittest.main()
