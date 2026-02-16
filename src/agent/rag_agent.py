from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Literal, Tuple

from src.agent.llm_openai import DEFAULT_MODEL, llm_json
from src.agent.mcp_tools import MCPRAGClient, RetrievedChunk

IndexType = Literal["hnsw", "ivf"]

PLAN_SCHEMA: Dict[str, Any] = {
    "name": "rag_plan",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "needs_retrieval": {"type": "boolean"},
            "search_query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "index_type": {"type": "string", "enum": ["hnsw", "ivf"]},
        },
        "required": ["needs_retrieval", "search_query", "top_k", "index_type"],
    },
}

ANSWER_SCHEMA: Dict[str, Any] = {
    "name": "rag_answer",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "final_answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "doc_id": {"type": "string"},
                        "chunk_id": {"type": "integer"},
                        "snippet": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["doc_id", "chunk_id", "snippet", "source"],
                },
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "uncertainty": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_uncertain": {"type": "boolean"},
                    "missing_information": {"type": "array", "items": {"type": "string"}},
                    "suggested_next_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["is_uncertain", "missing_information", "suggested_next_queries"],
            },
        },
        "required": ["final_answer", "citations", "confidence", "uncertainty"],
    },
}

SYSTEM_PLAN = """You are a RAG agent planner for FOMC documents (minutes + press statements).
Decide if retrieval is needed. If the user asks about FOMC content, retrieval is needed.
Return a search_query optimized for semantic retrieval (keywords, remove fluff).
For analytical questions, include synonyms like ‘eased’, ‘decline’, ‘softening’, ‘inflation expectations’, ‘core inflation’, ‘price pressures’ in the search query.
Use top_k 6 by default; use 8–10 only for synthesis/why/risk questions
Choose index_type: "hnsw" for best recall; "ivf" only if asked to compare IVF or memory/scale topics.
"""

SYSTEM_ANSWER = """You are a grounded research assistant for asset management analysts.
You MUST answer ONLY using the provided retrieved passages.
When possible, mention the meeting dates for key claims (e.g., "In the Nov 1, 2023 statement...").
If the passages do not contain the answer, say so clearly and set uncertainty.is_uncertain=true.
If the retrieved evidence is limited to one short statement, keep confidence at medium unless multiple passages corroborate.
If you cannot support an answer with retrieved evidence, do not attempt to fabricate a citation. Instead, express uncertainty.
"""


def build_context(chunks: List[RetrievedChunk]) -> str:
    max_chars_per_chunk = 1200  # keeps prompts smaller + cheaper
    lines = []
    for c in chunks:
        snippet = c.text[:max_chars_per_chunk]
        lines.append(
            f"[{c.doc_id} | {c.title} | {c.date} | chunk {c.chunk_id} | score {c.score:.3f}]\n{snippet}"
        )
    return "\n\n---\n\n".join(lines)


def detect_information_intent(question: str) -> str:
    q = question.lower()

    analytical = ["why", "risk", "concern", "uncertainty", "outlook", "reason", "drivers", "factors"]
    decision = ["decide", "decision", "rate", "raise", "cut", "hold", "announce", "target range"]

    if any(w in q for w in analytical):
        return "analysis"
    if any(w in q for w in decision):
        return "policy"
    return "general"


def rerank_by_intent(intent: str, retrieved: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Reorder retrieved evidence based on question intent. This does NOT change what was retrieved; it only changes ordering.
    """
    if intent in ("analysis", "general"):
        # analyst-style reasoning/synthesis: minutes usually carry the "why"
        return sorted(retrieved, key=lambda x: 0 if x.doc_type == "minutes" else 1)
    if intent == "policy":
        # decision questions: statements contain official wording
        return sorted(retrieved, key=lambda x: 0 if x.doc_type == "statement" else 1)
    return retrieved


def adjust_top_k_for_intent(intent: str, top_k: int) -> int:
    """
    Broaden retrieval slightly for synthesis. No retry loop.
    """
    if intent == "analysis":
        return max(top_k, 8)
    return top_k


def validate_citations(answer: Dict[str, Any], retrieved: List[RetrievedChunk]) -> Tuple[bool, List[str]]:
    allowed = {(c.doc_id, c.chunk_id) for c in retrieved}
    problems: List[str] = []

    for cit in answer.get("citations", []):
        key = (cit.get("doc_id"), int(cit.get("chunk_id")))
        if key not in allowed:
            problems.append(f"Invalid citation not in retrieved set: {key}")

    return (len(problems) == 0), problems


async def answer_question(
    question: str,
    *,
    mcp_server_cmd: List[str],
    index_type: IndexType = "hnsw",
    model: str = DEFAULT_MODEL,
    top_k_default: int = 6,
    hnsw_ef_search: int = 64,
    ivf_nprobe: int = 5,
    debug: bool = False,
) -> Dict[str, Any]:

    # 1) Plan
    plan = llm_json(model=model, system=SYSTEM_PLAN, user=f"User question: {question}\nDefault index_type requested by caller: {index_type}", schema_wrapper=PLAN_SCHEMA)

    needs = bool(plan["needs_retrieval"])
    search_query = plan["search_query"].strip() or question
    top_k = int(plan["top_k"] or top_k_default)
    chosen_index = plan["index_type"] if plan["index_type"] in ("hnsw", "ivf") else index_type

    retrieved: List[RetrievedChunk] = []

    # Compute intent ONCE (used for both top_k adjustment and reranking)
    intent = detect_information_intent(question)

    if needs:
        top_k = adjust_top_k_for_intent(intent, top_k)

        async with MCPRAGClient(mcp_server_cmd) as mcp:
            retrieved = await mcp.search(
                search_query,
                top_k=top_k,
                index_type=chosen_index,
                hnsw_ef_search=hnsw_ef_search,
                ivf_nprobe=ivf_nprobe,
            )

        retrieved = rerank_by_intent(intent, retrieved)

    # 2) Answer (grounded)
    context = build_context(retrieved) if retrieved else "NO_RETRIEVED_PASSAGES"
    user_prompt = f"""Question:
{question}

Retrieved passages:
{context}

Return JSON only, following the schema.
"""
    out = llm_json(model=model, system=SYSTEM_ANSWER, user=user_prompt, schema_wrapper=ANSWER_SCHEMA)

    # 3) Validate citations
    ok, problems = validate_citations(out, retrieved)
    if not ok:
        out["final_answer"] = (
            "I attempted to answer, but the citations produced were not consistent with retrieved evidence. "
            "Please rerun with a higher top_k or rephrase the query."
        )
        out["citations"] = []
        out["confidence"] = "low"
        out["uncertainty"] = {
            "is_uncertain": True,
            "missing_information": problems,
            "suggested_next_queries": [search_query],
        }

    # Optional: attach debug info for your demo
    if debug:
        out["_debug"] = {
            "plan": plan,
            "intent": intent,
            "retrieved": [asdict(c) for c in retrieved],
        }

    return out