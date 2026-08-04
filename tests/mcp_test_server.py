"""MCP 测试 server：供 tests/test_mcp_client.py 真实连接测试使用

支持 stdio 运行（python mcp_test_server.py）或作为模块被测试代码以
streamable HTTP 方式启动。
"""

import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="mcp-test-server")


@mcp.tool()
def echo(text: str) -> str:
    """回显输入文本"""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b


@mcp.tool()
def fail() -> str:
    """总是失败的测试工具（返回 isError）"""
    raise ValueError("故意失败")


if __name__ == "__main__":
    # 默认 stdio transport
    mcp.run()
