"""HTTP 控制面的凭据、监听地址与 TLS 安全策略。"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import ssl
from pathlib import Path

from aiohttp import web


TOKEN_ENV = "OPENXIAOAI_API_TOKEN"
TOKEN_FILE_ENV = "OPENXIAOAI_API_TOKEN_FILE"
DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024


def default_token_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "open-xiaoai-bridge" / "api-token"


def token_path() -> Path:
    configured = os.environ.get(TOKEN_FILE_ENV)
    return Path(configured).expanduser() if configured else default_token_path()


def load_or_create_token() -> tuple[str, Path | None]:
    """返回 API token；环境变量优先，否则从 0600 文件读取/首次生成。"""
    explicit = os.environ.get(TOKEN_ENV, "").strip()
    if explicit:
        if len(explicit) < 32:
            raise ValueError(f"{TOKEN_ENV} must contain at least 32 characters")
        return explicit, None

    path = token_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        value = path.read_text(encoding="utf-8").strip()
        if len(value) < 32:
            raise ValueError(f"API token file is empty or too short: {path}")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return value, path

    value = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    return value, path


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def build_server_ssl_context(host: str) -> ssl.SSLContext | None:
    """非 loopback 监听必须启用 TLS；可选 CA 会进一步启用 mTLS。"""
    cert = os.environ.get("API_SERVER_TLS_CERT", "").strip()
    key = os.environ.get("API_SERVER_TLS_KEY", "").strip()
    client_ca = os.environ.get("API_SERVER_CLIENT_CA", "").strip()

    if not cert and not key:
        if is_loopback_host(host) or os.environ.get(
            "OPENXIAOAI_ALLOW_INSECURE_HTTP", ""
        ).lower() in {"1", "true", "yes", "on"}:
            return None
        raise ValueError(
            "Refusing non-loopback plaintext API listener; configure "
            "API_SERVER_TLS_CERT and API_SERVER_TLS_KEY"
        )
    if not cert or not key:
        raise ValueError("API_SERVER_TLS_CERT and API_SERVER_TLS_KEY must be set together")

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=cert, keyfile=key)
    if client_ca:
        context.load_verify_locations(cafile=client_ca)
        context.verify_mode = ssl.CERT_REQUIRED
    return context


def bearer_middleware(expected_token: str):
    """为整个控制面强制固定时序 Bearer token 校验。"""

    @web.middleware
    async def authenticate(request: web.Request, handler):
        header = request.headers.get("Authorization", "")
        scheme, _, supplied = header.partition(" ")
        valid = (
            scheme.lower() == "bearer"
            and bool(supplied)
            and hmac.compare_digest(supplied, expected_token)
        )
        if not valid:
            return web.json_response(
                {"success": False, "error": "Unauthorized"},
                status=401,
                headers={
                    "WWW-Authenticate": "Bearer",
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return authenticate


def validate_stream_request(data: object) -> tuple[str, dict[str, str] | None, int, bool]:
    """严格校验中转播放命令，拒绝隐式类型转换和任意头转发。"""
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    url = data.get("url")
    if not isinstance(url, str) or not url or len(url.encode("utf-8")) > 2048:
        raise ValueError("url must be a non-empty string up to 2048 bytes")
    headers = data.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("headers must be an object")
    allowed_headers = {"user-agent", "referer"}
    if len(headers) > 2 or any(
        not isinstance(key, str)
        or key.lower() not in allowed_headers
        or not isinstance(value, str)
        or len(value.encode("utf-8")) > 512
        for key, value in headers.items()
    ):
        raise ValueError("only short User-Agent and Referer headers are allowed")
    start_ms = data.get("start_ms", 0)
    if isinstance(start_ms, bool) or not isinstance(start_ms, int) or start_ms < 0:
        raise ValueError("start_ms must be a non-negative integer")
    loop = data.get("loop", False)
    if not isinstance(loop, bool):
        raise ValueError("loop must be a boolean")
    return url, headers or None, start_ms, loop
