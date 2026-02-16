from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd
import faiss
from huggingface_hub import logging as hfhub_logging
hfhub_logging.set_verbosity_error()
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

IndexType = Literal["hnsw", "ivf"]

@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    doc_type: str
    date: str
    title: str
    chunk_id: int
    score: float
    text: str
    source_path: str

class RAGStore:
    """
    Loads:
      - chunks.parquet metadata
      - SentenceTransformer embedder (same model as ingestion)
      - FAISS indexes (HNSW + IVF)
    Exposes:
      - search(query, top_k, index_type, tuning knobs)
      - get_chunk(doc_id, chunk_id)
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunks_path: Optional[Path] = None,
        hnsw_index_path: Optional[Path] = None,
        ivf_index_path: Optional[Path] = None,
    ) -> None:
        
        self.repo_root = repo_root
        self.model_name = model_name
        self.chunks_path = chunks_path or (repo_root / "index" / "chunks.parquet")
        self.hnsw_index_path = hnsw_index_path or (repo_root / "index" / "faiss_hnsw_ip.index")
        self.ivf_index_path = ivf_index_path or (repo_root / "index" / "faiss_ivf_ip.index")

        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks.parquet at {self.chunks_path}")
        if not self.hnsw_index_path.exists():
            raise FileNotFoundError(f"Missing HNSW index at {self.hnsw_index_path}")
        if not self.ivf_index_path.exists():
            raise FileNotFoundError(f"Missing IVF index at {self.ivf_index_path}")

        # Load metadata
        self.df = pd.read_parquet(self.chunks_path).reset_index(drop=True)

        # Map (doc_id, chunk_id) -> row index
        self._key_to_row: Dict[str, int] = {}
        for i, r in self.df.iterrows():
            key = f"{r['doc_id']}::{int(r['chunk_id'])}"
            self._key_to_row[key] = int(i)

        self.model = SentenceTransformer(self.model_name)

        # Load indexes
        self.hnsw = faiss.read_index(str(self.hnsw_index_path))
        self.ivf = faiss.read_index(str(self.ivf_index_path))

        # Dimension sanity check
        self.dim = self.hnsw.d

    def embed_query(self, query: str) -> np.ndarray:
        v = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        v = v.astype(np.float32)
        return v

    def search(self, query: str, *, top_k: int = 5, index_type: IndexType = "hnsw", hnsw_ef_search: int = 64, ivf_nprobe: int = 5) -> List[SearchHit]:
        q = self.embed_query(query)

        if index_type == "hnsw":
            if hasattr(self.hnsw, "hnsw"):
                self.hnsw.hnsw.efSearch = int(hnsw_ef_search)
            D, I = self.hnsw.search(q, top_k)

        elif index_type == "ivf":
            if hasattr(self.ivf, "nprobe"):
                self.ivf.nprobe = int(ivf_nprobe)
            D, I = self.ivf.search(q, top_k)

        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        hits: List[SearchHit] = []
        scores = D[0].tolist()
        ids = I[0].tolist()

        for score, idx in zip(scores, ids):
            if idx == -1:
                continue
            row = self.df.iloc[int(idx)]
            hits.append(
                SearchHit(
                    doc_id=str(row["doc_id"]),
                    doc_type=str(row["doc_type"]),
                    date=str(row["date"]),
                    title=str(row["title"]),
                    chunk_id=int(row["chunk_id"]),
                    score=float(score),
                    text=str(row["text"]),
                    source_path=str(row.get("source_path", "")),
                )
            )
        return hits

    def get_chunk(self, doc_id: str, chunk_id: int) -> Optional[SearchHit]:
        key = f"{doc_id}::{int(chunk_id)}"
        row_idx = self._key_to_row.get(key)
        if row_idx is None:
            return None
        row = self.df.iloc[int(row_idx)]
        return SearchHit(
            doc_id=str(row["doc_id"]),
            doc_type=str(row["doc_type"]),
            date=str(row["date"]),
            title=str(row["title"]),
            chunk_id=int(row["chunk_id"]),
            score=1.0,
            text=str(row["text"]),
            source_path=str(row.get("source_path", "")),
        )