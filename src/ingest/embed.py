from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import typer
from huggingface_hub import logging as hfhub_logging
hfhub_logging.set_verbosity_error()
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

app = typer.Typer(help="Embed chunks.parquet into index/embeddings.npy (float32, normalized).")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command()
def run(
    chunks_path: Optional[Path] = typer.Option(None, help="Path to chunks.parquet (default index/chunks.parquet)"),
    out_path: Optional[Path] = typer.Option(None, help="Output .npy for embeddings (default index/embeddings.npy)"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model"),
    batch_size: int = typer.Option(32, help="Batch size for embedding"),
    normalize: bool = typer.Option(True, help="L2-normalize embeddings (recommended for cosine/IP search)"),
) -> None:
    root = repo_root()
    chunks_path = chunks_path or (root / "index" / "chunks.parquet")
    out_path = out_path or (root / "index" / "embeddings.npy")

    if not chunks_path.exists():
        typer.echo(f"Missing chunks file: {chunks_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(chunks_path)
    texts = df["text"].tolist()

    typer.echo(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    typer.echo(f"Embedding {len(texts)} chunks (batch_size={batch_size}) ...")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    emb = emb.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)

    typer.echo("\nDone.")
    typer.echo(f"Embeddings shape: {emb.shape}")
    typer.echo(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    app()