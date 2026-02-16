from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import typer
import faiss

app = typer.Typer(help="Build FAISS HNSW and IVF indexes from embeddings.npy")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command()
def run(
    embeddings_path: Optional[Path] = typer.Option(None, help="Path to embeddings.npy (default index/embeddings.npy)"),
    chunks_path: Optional[Path] = typer.Option(None, help="Path to chunks.parquet (default index/chunks.parquet)"),
    out_dir: Optional[Path] = typer.Option(None, help="Output dir for FAISS indexes (default index/)"),

    # HNSW knobs
    hnsw_m: int = typer.Option(32, help="HNSW M (graph degree). Common: 16-48"),
    hnsw_ef_construction: int = typer.Option(200, help="HNSW efConstruction (build quality)"),

    # IVF knobs
    ivf_nlist: int = typer.Option(32, help="IVF nlist (num clusters). For small corpora 32-256 is fine."),
    ivf_metric: str = typer.Option("ip", help="Similarity metric: ip (cosine via normalized vectors) or l2"),

) -> None:
    root = repo_root()
    embeddings_path = embeddings_path or (root / "index" / "embeddings.npy")
    chunks_path = chunks_path or (root / "index" / "chunks.parquet")
    out_dir = out_dir or (root / "index")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        typer.echo(f"Missing embeddings: {embeddings_path}")
        raise typer.Exit(code=1)
    if not chunks_path.exists():
        typer.echo(f"Missing chunks: {chunks_path}")
        raise typer.Exit(code=1)

    X = np.load(embeddings_path).astype(np.float32)
    df = pd.read_parquet(chunks_path)
    n, d = X.shape

    typer.echo(f"Loaded embeddings: n={n}, d={d}")
    typer.echo(f"Loaded chunks: {len(df)} rows")

    metric = faiss.METRIC_INNER_PRODUCT if ivf_metric.lower() == "ip" else faiss.METRIC_L2


    # Build HNSW (IP)
    typer.echo("\nBuilding HNSW index...")
    t0 = time.time()
    # For cosine similarity, we use IP on normalized embeddings.
    # IndexHNSWFlat supports METRIC_L2 by default; for IP we wrap with IndexHNSWFlat + IndexIDMap is not needed if we keep implicit ids.
    hnsw = faiss.IndexHNSWFlat(d, hnsw_m, metric)
    hnsw.hnsw.efConstruction = hnsw_ef_construction
    hnsw.add(X)
    hnsw_path = out_dir / "faiss_hnsw_ip.index"
    faiss.write_index(hnsw, str(hnsw_path))
    t1 = time.time()
    typer.echo(f"Saved: {hnsw_path} (build_sec={t1 - t0:.2f})")


    # Build IVF (IP)
    typer.echo("\nBuilding IVF index...")
    # Quantizer must match metric
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(d)
    else:
        quantizer = faiss.IndexFlatL2(d)

    ivf = faiss.IndexIVFFlat(quantizer, d, ivf_nlist, metric)

    # IVF requires training
    t0 = time.time()
    ivf.train(X)
    ivf.add(X)
    ivf_path = out_dir / "faiss_ivf_ip.index"
    faiss.write_index(ivf, str(ivf_path))
    t1 = time.time()
    typer.echo(f"Saved: {ivf_path} (build_sec={t1 - t0:.2f})")

    # Save config for reproducibility
    cfg = {
        "embeddings_path": str(embeddings_path),
        "chunks_path": str(chunks_path),
        "n_vectors": int(n),
        "dim": int(d),
        "metric": "ip" if metric == faiss.METRIC_INNER_PRODUCT else "l2",
        "hnsw": {"M": hnsw_m, "efConstruction": hnsw_ef_construction},
        "ivf": {"nlist": ivf_nlist},
        "outputs": {"hnsw_index": str(hnsw_path), "ivf_index": str(ivf_path)},
    }
    cfg_path = out_dir / "index_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    typer.echo(f"\nWrote config: {cfg_path.resolve()}")
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()