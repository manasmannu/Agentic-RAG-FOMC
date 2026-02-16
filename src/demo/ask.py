from __future__ import annotations
from typing import Literal
import asyncio
import json
import typer

from src.agent.rag_agent import answer_question

app = typer.Typer(help="Demo CLI for the agentic RAG assistant (MCP + hosted LLM).")

IndexType = Literal["hnsw", "ivf"]

@app.command()
def run(
    question: str = typer.Argument(...),
    index: IndexType = typer.Option("hnsw", "-i", "--index", help="Which FAISS index to use for retrieval"),
    top_k: int = typer.Option(6, help="How many chunks to retrieve"),
    hnsw_ef_search: int = typer.Option(64, help="HNSW efSearch (higher = better recall, slower)"),
    ivf_nprobe: int = typer.Option(5, help="IVF nprobe (higher = better recall, slower)"),
    debug: bool = typer.Option(False, "-d", "--debug", help="Enable debug output"),
) -> None:
    # MCP server command (stdio). This spawns the server locally.
    mcp_cmd = ["python", "-m", "src.mcp_server.server"]

    out = asyncio.run(
        answer_question(
            question,
            mcp_server_cmd=mcp_cmd,
            index_type=index,
            top_k_default=top_k,
            hnsw_ef_search=hnsw_ef_search,
            ivf_nprobe=ivf_nprobe,
            debug=debug,
        )
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()