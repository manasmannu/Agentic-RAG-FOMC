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
- Create an OpenAI API key: https://platform.openai.com/api-keys
- Add the key to a '.env' file in the project root (agentic-rag-fomc directory):
```bash
OPENAI_API_KEY=<your_key_here>
```
- The project requires OpenAI usage credits. If your free trial quota is exhausted, you may need to purchase a minimum of $5 in API credits to run the system.

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
python -m src.vectordb.build_index
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
python -m src.demo.ask "What did the FOMC say about inflation risks?" -i ivf
```

Debug mode
Shows planner decisions and retrieved chunks

```bash
python -m src.demo.ask "Why was inflation considered persistent?" -d
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
python -m src.vectordb.benchmark
```

⸻

## Design Choices

## Chunking strategy
  Minutes and statements behave differently:

- Document | Strategy | Reason
- Minutes | smaller chunks | many topics & reasoning chains
- Statements | larger chunks | short dense policy text

Overlap is used to avoid losing meaning across boundaries.

- Why not semantic chunking?
- This system is designed for evidence-grounded retrieval, not summarization.
- Semantic chunking tries to group text by meaning, but it introduces two risks for a research assistant:
1. The boundaries become model-dependent (non-deterministic)
2. A single chunk may mix multiple claims → weak citations

Instead, we use deterministic paragraph-group chunking with overlap because:
- Each chunk corresponds to a real passage in the document
- Citations remain verifiable
- Retrieval behavior stays stable across re-indexing
- Analysts can trace the answer back to the source text


## Embeddings
The system uses the following embedding model:
```bash
sentence-transformers/all-MiniLM-L6-v2
```
Why this model?
- This project prioritizes retrieval correctness and grounding over generative richness, so the embedding choice was made based on retrieval behavior rather than raw semantic fluency.
- Fast + lightweight
	•	~384 dimensional vectors
	•	Very low latency
	•	Small memory footprint
This keeps FAISS search fast and reproducible locally.

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

## LLM
OpenAI chat model (via API) for the agent reasoning steps:
- Planner — decides whether retrieval is needed and constructs the search query
- Answerer — produces a grounded response strictly from retrieved passages

The model is not used as a knowledge source.
It only performs reasoning over retrieved evidence, ensuring answers come from FOMC documents rather than model memory.

⸻

## Framework Server
FastMCP

FastMCP provides a structured tool-calling interface between the LLM agent and external systems.
Instead of allowing the model to directly access the vector database, the agent communicates with a small MCP tool server using a standardized protocol.

This separates reasoning from execution and makes the system safer and more production-like:
- The model decides what to do
- The server performs the action

Conceptual analogy
- Agent → brain (reasoning & planning)
- FastMCP → communication layer (how the brain talks to tools)
- MCP server → hands (executes retrieval)
- FAISS → memory/storage (vector database)

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
