from __future__ import annotations
import os
os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"
from pathlib import Path
from fastmcp import FastMCP
from src.mcp_server.store import RAGStore

mcp = FastMCP("fomc-rag")
_store: RAGStore | None = None

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_store() -> RAGStore:
    global _store
    if _store is None:
        _store = RAGStore(repo_root=repo_root())
    return _store

@mcp.tool()
def rag_search(query: str, top_k: int = 5, index_type: str = "hnsw", hnsw_ef_search: int = 64, ivf_nprobe: int = 5):
    store = get_store()
    hits = store.search(
        query,
        top_k=top_k,
        index_type=index_type,
        hnsw_ef_search=hnsw_ef_search,
        ivf_nprobe=ivf_nprobe,
    )
    return [
        {
            "doc_id": h.doc_id,
            "doc_type": h.doc_type,
            "date": h.date,
            "title": h.title,
            "chunk_id": h.chunk_id,
            "score": h.score,
            "text": h.text,
            "source_path": h.source_path,
        }
        for h in hits
    ]

@mcp.tool()
def rag_get_chunk(doc_id: str, chunk_id: int):
    store = get_store()
    hit = store.get_chunk(doc_id, chunk_id)
    if hit is None:
        return None
    return {
        "doc_id": hit.doc_id,
        "doc_type": hit.doc_type,
        "date": hit.date,
        "title": hit.title,
        "chunk_id": hit.chunk_id,
        "text": hit.text,
        "source_path": hit.source_path,
    }

if __name__ == "__main__":
    mcp.run(show_banner=False)