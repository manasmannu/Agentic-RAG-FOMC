from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import typer
import faiss

app = typer.Typer(help="Benchmark FAISS HNSW vs IVF vs Flat (Recall@K + latency).")

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def recall_at_k(gt: np.ndarray, pred: np.ndarray) -> float:
    # gt, pred are (nq, k) of ids
    hits = 0
    total = gt.shape[0] * gt.shape[1]
    gt_sets = [set(row) for row in gt]
    for i in range(pred.shape[0]):
        for x in pred[i]:
            if x in gt_sets[i]:
                hits += 1
    return hits / total

def search_with_timing(index, Q: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    D, I = index.search(Q, k)
    t1 = time.perf_counter()
    return I, (t1 - t0) * 1000.0  # ms

@app.command()
def run(
    embeddings_path: Optional[Path] = typer.Option(None, help="index/embeddings.npy"),
    hnsw_index_path: Optional[Path] = typer.Option(None, help="index/faiss_hnsw_ip.index"),
    ivf_index_path: Optional[Path] = typer.Option(None, help="index/faiss_ivf_ip.index"),
    k: int = typer.Option(5, help="K for recall@K"),
    nq: int = typer.Option(50, help="Number of query vectors to sample"),
) -> None:
    root = repo_root()
    embeddings_path = embeddings_path or (root / "index" / "embeddings.npy")
    hnsw_index_path = hnsw_index_path or (root / "index" / "faiss_hnsw_ip.index")
    ivf_index_path = ivf_index_path or (root / "index" / "faiss_ivf_ip.index")

    X = np.load(embeddings_path).astype(np.float32)
    n, d = X.shape

    # sample queries from the corpus (simple + works)
    nq = min(nq, n)
    rng = np.random.default_rng(42)
    q_idx = rng.choice(n, size=nq, replace=False)
    Q = X[q_idx]

    # Ground truth
    flat = faiss.IndexFlatIP(d)
    flat.add(X)
    gt_I, gt_ms = search_with_timing(flat, Q, k)

    # HNSW
    hnsw = faiss.read_index(str(hnsw_index_path))
    # IVF
    ivf = faiss.read_index(str(ivf_index_path))

    print(f"n={n}, d={d}, nq={nq}, k={k}")
    print(f"FlatIP: {gt_ms/nq:.3f} ms/query (ground truth)")

    # HNSW sweep
    for ef in [32, 64, 128]:
        # Set efSearch if available
        if hasattr(hnsw, "hnsw"):
            hnsw.hnsw.efSearch = ef
        pred_I, ms = search_with_timing(hnsw, Q, k)
        r = recall_at_k(gt_I, pred_I)
        print(f"HNSW efSearch={ef:3d}: {ms/nq:.3f} ms/query | recall@{k}={r:.3f}")

    # IVF sweep
    for nprobe in [1, 3, 5, 10]:
        if hasattr(ivf, "nprobe"):
            ivf.nprobe = nprobe
        pred_I, ms = search_with_timing(ivf, Q, k)
        r = recall_at_k(gt_I, pred_I)
        print(f"IVF nprobe={nprobe:2d}:     {ms/nq:.3f} ms/query | recall@{k}={r:.3f}")

if __name__ == "__main__":
    app()