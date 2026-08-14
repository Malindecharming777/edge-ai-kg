"""MCP server exposing the {{KG_NAME}} KG. Run: python -m mcp_server.server"""
from fastmcp import FastMCP
mcp = FastMCP("{{KG_SLUG}}-kg")

@mcp.tool()
def example_query(limit: int = 5) -> list[dict]:
    """Replace with a real domain query against the Samyama graph engine."""
    return []

if __name__ == "__main__":
    mcp.run()
