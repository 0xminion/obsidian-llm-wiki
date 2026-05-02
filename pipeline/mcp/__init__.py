"""MCP server package."""
try:
    from pipeline.mcp.server import WikiMCPServer, create_server, run_stdio_server  # noqa: F401
except ImportError:
    WikiMCPServer = None
    create_server = None
    run_stdio_server = None
