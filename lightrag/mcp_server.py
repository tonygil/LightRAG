"""LightRAG MCP server — wraps the REST API as MCP tools.

Exposes two tools:
- query: full LLM-synthesised answer from the knowledge graph
- search: raw entities, relations, and chunks without LLM synthesis

Configure via environment variables:
  LIGHTRAG_API_URL  Base URL of a running lightrag-server (default: http://localhost:9621)
  LIGHTRAG_API_KEY  Optional API key for lightrag-server authentication

Usage:
  lightrag-mcp                          # stdio transport (Claude Desktop / Cursor)
  lightrag-mcp --transport sse          # SSE transport (remote clients)
  lightrag-mcp --transport sse --port 8001 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

API_URL: str = os.environ.get("LIGHTRAG_API_URL", "http://localhost:9621").rstrip("/")
API_KEY: str = os.environ.get("LIGHTRAG_API_KEY", "")

mcp = FastMCP("LightRAG")

_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        headers: dict[str, str] = {}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        _http_client = httpx.AsyncClient(
            base_url=API_URL,
            headers=headers,
            timeout=120.0,
        )
    return _http_client


@mcp.tool()
async def query(
    question: str,
    mode: str = "mix",
    top_k: int = 60,
) -> str:
    """Query the LightRAG knowledge graph and return an LLM-synthesised answer.

    LightRAG understands entities and their relationships, giving richer context
    than plain vector search. Use this when you want a direct answer.

    Args:
        question: The question to answer.
        mode: Retrieval mode — local, global, hybrid, naive, or mix (default).
              'mix' combines knowledge-graph and vector search and is recommended.
        top_k: Number of top entities/relations to retrieve (default 60).
    """
    payload: dict[str, Any] = {
        "query": question,
        "mode": mode,
        "top_k": top_k,
        "stream": False,
        "include_references": True,
    }
    try:
        resp = await _client().post("/query", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # The server already appends a **Sources:** section with clickable URLs
        # to the response text via _build_sources_section — return it as-is.
        return data.get("response", "")
    except httpx.HTTPStatusError as e:
        return f"LightRAG error {e.response.status_code}: {e.response.text[:500]}"
    except httpx.RequestError as e:
        return (
            f"Could not reach LightRAG server at {API_URL}. "
            f"Is lightrag-server running? Details: {e}"
        )


@mcp.tool()
async def search(
    question: str,
    mode: str = "mix",
    top_k: int = 60,
) -> str:
    """Search the LightRAG knowledge graph and return raw entities, relations, and chunks.

    Returns the retrieved context directly with no LLM synthesis, giving full
    visibility into the graph relationships and source text LightRAG found.
    Useful when you want to reason over the raw evidence yourself.

    Args:
        question: The search query.
        mode: Retrieval mode — local, global, hybrid, naive, or mix (default).
        top_k: Number of top entities/relations to retrieve (default 60).
    """
    payload: dict[str, Any] = {
        "query": question,
        "mode": mode,
        "top_k": top_k,
    }
    try:
        resp = await _client().post("/query/data", json=payload)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2, ensure_ascii=False)
    except httpx.HTTPStatusError as e:
        return f"LightRAG error {e.response.status_code}: {e.response.text[:500]}"
    except httpx.RequestError as e:
        return (
            f"Could not reach LightRAG server at {API_URL}. "
            f"Is lightrag-server running? Details: {e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LightRAG MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for SSE transport (default: 8001)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
