from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class RetrievedChunk:
    doc_id: str
    doc_type: str
    date: str
    title: str
    chunk_id: int
    score: float
    text: str
    source_path: str


class MCPRAGClient:
    """
    Spawns the MCP server as a subprocess and calls tools over stdio.
    """

    def __init__(self, server_command: List[str]) -> None:
        self.server_command = server_command

    async def __aenter__(self) -> "MCPRAGClient":
        params = StdioServerParameters(command=self.server_command[0], args=self.server_command[1:])
        self._stdio = stdio_client(params)
        self._read, self._write = await self._stdio.__aenter__()
        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.session.__aexit__(exc_type, exc, tb)
        await self._stdio.__aexit__(exc_type, exc, tb)

    async def search(self, query: str, *, top_k: int = 6, index_type: str = "hnsw", hnsw_ef_search: int = 64, ivf_nprobe: int = 5) -> List[RetrievedChunk]:
        # Tool name must match what your server registers.
        # In our earlier server: rag_search(...)
        res = await self.session.call_tool(
            "rag_search",
            {
                "query": query,
                "top_k": top_k,
                "index_type": index_type,
                "hnsw_ef_search": hnsw_ef_search,
                "ivf_nprobe": ivf_nprobe,
            },
        )

        # res.content is MCP structured content; most servers return JSON-serializable content
        # We'll handle the common case: first content item is JSON.
        items: Any = None
        if hasattr(res, "content") and res.content:
            # content items are often objects with .text or .data depending on SDK version
            c0 = res.content[0]
            items = getattr(c0, "data", None) or getattr(c0, "text", None) or c0
        else:
            items = res

        # If it's a string, try to parse as JSON. If it's already a list, use it.
        if isinstance(items, str):
            import json
            items = json.loads(items)

        chunks: List[RetrievedChunk] = []
        for it in items:
            chunks.append(
                RetrievedChunk(
                    doc_id=it["doc_id"],
                    doc_type=it["doc_type"],
                    date=it["date"],
                    title=it["title"],
                    chunk_id=int(it["chunk_id"]),
                    score=float(it.get("score", 0.0)),
                    text=it["text"],
                    source_path=it.get("source_path", ""),
                )
            )
        return chunks