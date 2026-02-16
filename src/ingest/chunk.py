from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import typer
import pyarrow as pa
import pyarrow.parquet as pq

app = typer.Typer(help="Chunk cleaned FOMC text files into index/chunks.parquet")

DOC_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<dtype>minutes|statement)\.txt$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class ChunkRow:
    doc_id: str
    doc_type: str
    date: str
    title: str
    source_path: str
    chunk_id: int
    text: str
    char_start: int
    char_end: int


def normalize_paragraphs(text: str) -> list[str]:
    t = text.replace("\r", "").strip()
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def make_doc_metadata(filename: str) -> Optional[tuple[str, str, str, str]]:
    """
    From 'YYYY-MM-DD_minutes.txt' return:
      (date, doc_type, doc_id, title)
    """
    m = DOC_RE.match(filename)
    if not m:
        return None
    date = m.group("date")
    doc_type = m.group("dtype")
    doc_id = f"{doc_type}_{date.replace('-', '_')}"
    title = "FOMC Minutes" if doc_type == "minutes" else "FOMC Press Statement"
    return date, doc_type, doc_id, title


def chunk_paragraphs(
    paragraphs: list[str], *, target_chars: int = 3000, max_chars: int = 3800, overlap_chars: int = 400) -> list[tuple[str, int, int]]:
    """
    Linear-time chunking.
    Returns list of (chunk_text, char_start, char_end) relative to the reconstructed doc text.
    We reconstruct the doc as paragraphs joined with '\n\n'.
    """
    if not paragraphs:
        return []

    # Precompute paragraph spans in the reconstructed full text
    spans: list[tuple[int, int]] = []
    pos = 0
    for p in paragraphs:
        start = pos
        end = start + len(p)
        spans.append((start, end))
        pos = end + 2  # for '\n\n'

    chunks: list[tuple[str, int, int]] = []
    current_parts: list[str] = []
    current_start = 0
    current_len = 0  # length of "\n\n".join(current_parts) without constructing it repeatedly

    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_start, p_end = spans[i]
        
        # Guard: if a single paragraph is longer than max_chars, it can cause an infinite loop
        # (because we keep retrying the same paragraph with overlap tail).
        # Fix: flush current chunk (if any) and then emit the long paragraph as its own chunk.
        if len(p) > max_chars:
            # Flush current chunk first (no overlap here)
            if current_parts:
                chunk_text = "\n\n".join(current_parts).strip()
                chunk_end = spans[i - 1][1]
                chunks.append((chunk_text, current_start, chunk_end))
                current_parts = []
                current_len = 0

            # Emit the long paragraph as its own chunk (may exceed max_chars, but safe and complete)
            chunks.append((p.strip(), p_start, p_end))
            i += 1
            continue

        if current_parts and len(current_parts) == 1 and current_len <= overlap_chars:
            if (current_len + add_len) > max_chars:
                current_parts = []
                current_len = 0
                current_start = p_start
                continue

        # cost to add this paragraph into current chunk
        add_len = len(p) + (2 if current_parts else 0)  # +2 for '\n\n' join

        # If adding would exceed max_chars, flush current (if not empty), and retry same paragraph
        if current_parts and (current_len + add_len) > max_chars:
            chunk_text = "\n\n".join(current_parts).strip()
            chunk_end = spans[i - 1][1]
            chunks.append((chunk_text, current_start, chunk_end))

            if overlap_chars > 0 and chunk_text:
                tail = chunk_text[-overlap_chars:]
                current_parts = [tail]
                # approximate overlap start
                current_start = max(current_start, chunk_end - overlap_chars)
                current_len = len(tail)
            else:
                current_parts = []
                current_len = 0

            continue  # retry paragraph i into the new chunk

        # Add paragraph
        current_parts.append(p)
        current_len += add_len

        # If we reached target size, flush
        if current_len >= target_chars:
            chunk_text = "\n\n".join(current_parts).strip()
            chunk_end = p_end
            chunks.append((chunk_text, current_start, chunk_end))

            if overlap_chars > 0 and chunk_text:
                tail = chunk_text[-overlap_chars:]
                current_parts = [tail]
                current_start = max(current_start, chunk_end - overlap_chars)
                current_len = len(tail)
            else:
                current_parts = []
                current_len = 0

        i += 1

    # Flush remainder
    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        last_end = spans[-1][1]
        chunks.append((chunk_text, current_start, last_end))

    return chunks


@app.command()
def run(
    cleaned_dir: Optional[Path] = typer.Option(None, help="Input dir (default repo_root/data/cleaned)"),
    out_path: Optional[Path] = typer.Option(None, help="Output parquet (default repo_root/index/chunks.parquet)"),
    batch_size: int = typer.Option(500, help="Rows per parquet write batch (lower if memory is tight)"),
) -> None:
    root = repo_root()
    cleaned_dir = cleaned_dir or (root / "data" / "cleaned")
    out_path = out_path or (root / "index" / "chunks.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(cleaned_dir.glob("*.txt"))
    if not files:
        typer.echo(f"No cleaned .txt files found in {cleaned_dir}")
        raise typer.Exit(code=1)

    typer.echo(f"Chunking {len(files)} cleaned files from {cleaned_dir}")
    start_all = time.time()

    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    total_chunks = 0
    total_docs = 0

    try:
        for f in files:
            meta = make_doc_metadata(f.name)
            if not meta:
                typer.echo(f"  ! SKIP (unrecognized name) {f.name}")
                continue

            date, doc_type, doc_id, title = meta

            typer.echo(f"\n  * Reading {f.name} ...")
            t0 = time.time()

            text = f.read_text(encoding="utf-8", errors="ignore")
            paragraphs = normalize_paragraphs(text)

            typer.echo(f"    chars={len(text)} paragraphs={len(paragraphs)} doc_type={doc_type}")

            # Doc-type-specific chunking strategy
            if doc_type == "minutes":
                t_chars, m_chars, o_chars = 1800, 2000, 200
            else:  # statement
                t_chars, m_chars, o_chars = 2200, 2500, 200

            chunk_tuples = chunk_paragraphs(
                paragraphs,
                target_chars=t_chars,
                max_chars=m_chars,
                overlap_chars=o_chars,
            )

            typer.echo(f"    produced_chunks={len(chunk_tuples)} (target={t_chars} max={m_chars} overlap={o_chars})")

            for chunk_id, (chunk_text, cs, ce) in enumerate(chunk_tuples):
                row = ChunkRow(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    date=date,
                    title=title,
                    source_path=str(f),
                    chunk_id=chunk_id,
                    text=chunk_text,
                    char_start=int(cs),
                    char_end=int(ce),
                )
                buffer.append(asdict(row))

                if len(buffer) >= batch_size:
                    table = pa.Table.from_pylist(buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                    buffer.clear()

            total_chunks += len(chunk_tuples)
            total_docs += 1
            typer.echo(f"  + Done {f.name}: {len(chunk_tuples)} chunks (sec={time.time() - t0:.2f})")

        # Flush remainder
        if buffer:
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            buffer.clear()

    finally:
        if writer is not None:
            writer.close()

    typer.echo("\nDone.")
    typer.echo(f"Docs processed: {total_docs}")
    typer.echo(f"Total chunks:  {total_chunks}")
    typer.echo(f"Saved:         {out_path.resolve()}")
    typer.echo(f"Total time:    {time.time() - start_all:.2f} sec")


if __name__ == "__main__":
    app()