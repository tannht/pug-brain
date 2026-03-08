"""MCP (Model Context Protocol) server for PugBrain.

This module provides an MCP server that exposes PugBrain tools
to Claude Code, Claude Desktop, and other MCP-compatible clients.
"""

from neural_memory.mcp.server import create_mcp_server, main, run_mcp_server

__all__ = ["create_mcp_server", "main", "run_mcp_server"]
