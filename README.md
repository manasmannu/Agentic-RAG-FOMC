# Agentic RAG for FOMC Minutes & Statements

This project implements an Agentic Retrieval-Augmented Generation (RAG) research assistant over Federal Reserve FOMC documents.

It answers questions like:

- “What risks did the FOMC highlight about inflation?”

By retrieving evidence from:

- FOMC Minutes → reasoning, outlook, risks (analyst-style)
- FOMC Press Statements → official decisions & policy language

The assistant produces a grounded answer with verifiable citations only.

⸻

## What makes this agentic?

Instead of simple retrieve-then-generate:

1. The LLM plans the retrieval strategy
2. Chooses index (HNSW vs IVF)
3. Adjusts top-k based on question type
4. Reranks documents by intent (policy vs analysis)
5. Generates grounded answer
6. System validates citations programmatically

No hallucinated evidence can pass validation.

⸻

## Features

- End-to-end ingestion pipeline (download → clean → chunk → embed → index)
- FAISS retrieval (HNSW & IVF comparison)
- Intent-aware retrieval behavior
- MCP server exposing retrieval as a tool
- Strict citation enforcement
- Benchmark script (latency + recall)

⸻

    agentic-rag-fomc/
      README.md
      requirements.txt
      .env

      data/
        raw/                       # downloaded HTML (Fed pages)
        cleaned/                   # cleaned .txt (one file per doc)

      index/
        chunks.parquet # produced by src/ingest/chunk.py
        embeddings.npy # produced by embedding step
        faiss_hnsw_ip.index # HNSW index
        faiss_ivf_ip.index # IVF (optionally IVF-PQ)

      src/
        agent/
          rag_agent.py            # planner + retrieval + grounded answering
          llm_openai.py           # llm_json wrapper (OpenAI)
          mcp_tools.py            # MCP client (stdio) to call server tool

        mcp_server/
          server.py               # FastMCP tool registration + run()
          store.py                # loads chunks+embeddings+faiss indexes, search()

        ingest/
          clean.py                # raw html -> cleaned txt (optional)
          chunk.py                # cleaned txt -> chunks.parquet
          embed.py                # chunks.parquet -> embeddings.npy
          build_index.py          # embeddings.npy -> faiss indexes

        demo/
          ask.py                  # CLI: python -m src.demo.ask

        vectordb/
          build_index.py          # Build indexes
          benchmark.py            # benchmark script (your snippet)

## Demo
https://github.com/user-attachments/assets/f1cf15e5-359c-4109-a013-c5c88bd025ad



## Setup

1. Create environment

git clone https://github.com/manasmannu/Agentic-RAG-FOMC.git

```bash
cd agentic-rag-fomc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure API key

Create .env

OPENAI_API_KEY=your_key_here

⸻

## Build the knowledge base

1. Download FOMC documents

```bash
python -m src.ingest.download --start-date 2021 --end-date 2025
```

2. Clean HTML → text

```bash
python -m src.ingest.clean
```

3. Chunk documents

```bash
python -m src.ingest.chunk
```

4. Create embeddings

```bash
python -m src.ingest.embed
```

5. Build FAISS indexes

```bash
python -m src.ingest.build_index
```

You should now have:

index/

- chunks.parquet
- embeddings.npy
- faiss_hnsw_ip.index
- faiss_ivf_ip.index

⸻

## Ask questions

Default (HNSW)

```bash
python -m src.demo.ask "What did the FOMC say about inflation risks?"
```

IVF retrieval

```bash
python -m src.demo.ask -i ivf "What did the FOMC say about inflation risks?"
```

Debug mode
Shows planner decisions and retrieved chunks

```bash
python -m src.demo.ask -d "Why was inflation considered persistent?"
```

⸻

## Example Output

The FOMC repeatedly noted inflation risks skewed to the upside, particularly due to persistent price pressures and supply shocks.
In the June 18, 2025 minutes, staff warned inflation could be more persistent than expected. Similar concerns appeared in the Nov 1, 2023 and Sep 20, 2023 meetings.

Citations:
(minutes_2025_06_18)
(minutes_2023_11_01)
(minutes_2023_09_20)

⸻

## Benchmark Retrieval Quality

Compares recall vs latency between Flat, HNSW, and IVF.

```bash
python -m src.demo.bench_faiss
```

⸻

## Design Choices

- Chunking strategy
  Minutes and statements behave differently:

- Document Strategy Reason
- Minutes smaller chunks many topics & reasoning chains
- Statements larger chunks short dense policy text

Overlap is used to avoid losing meaning across boundaries.

⸻

## Retrieval behavior

1. Planner decides if retrieval needed
2. Generates optimized semantic query
3. Retrieves with FAISS 4. Reranks by intent:

Question type Priority
Policy decision Statements first
Risk / Why / Outlook Minutes first

⸻

## Grounding & Citation Enforcement

Layer 1 — Prompt grounding

Model must:

- Use only retrieved passages
- Admit uncertainty if missing
- Never fabricate citations

Layer 2 — Hard validation

After generation:

```bash
If citation not in retrieved set → answer rejected
```

This guarantees factual grounding.

⸻

## HNSW vs IVF (Understanding)

HNSW
Graph-based nearest neighbor search

Best for:

- Highest recall
- Real-time Q&A
- Medium-scale datasets

Tradeoff: High memory usage
Tune:

- efSearch ↑ → accuracy ↑ latency ↑

IVF
Cluster-based search

Best for:

- Large datasets
- Memory-efficient retrieval
- Scalable systems

Tradeoff: Lower recall if poorly tuned
Tune:

- nprobe ↑ → accuracy ↑ latency ↑

⸻

When to use which

Use case Index
Best answer quality HNSW
Huge dataset IVF
Billion-scale IVF-PQ

⸻

## Scaling to Billions of Documents

Approach:

1. IVF coarse search
2. rerank on candidates
3. shard by date/topic
4. compress vectors (IVF-PQ)

This mirrors production search architectures used in real vector databases.

⸻

## Summary

This project demonstrates:

- Retrieval-aware agent behavior
- Reliable grounded answering
- ANN index tradeoffs
- Scalable vector search design
