from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin
import requests
import typer
from bs4 import BeautifulSoup

app = typer.Typer(help="Download FOMC press statements and meeting minutes into data/raw/")

FED_BASE = "https://www.federalreserve.gov"
CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

DATE_YYYYMMDD_RE = re.compile(r"(?P<date>\d{8})")

def is_minutes_link(url: str) -> bool:
    u = url.lower()
    return ("fomcminutes" in u) and u.endswith((".htm", ".html", ".pdf"))

def is_statement_link(url: str) -> bool:
    u = url.lower()
    return ("/pressreleases/monetary" in u) and u.endswith((".htm", ".html", ".pdf"))

def prefer_html(urls: Iterable[str]) -> list[str]:
    urls = list(dict.fromkeys(urls))  # stable de-dupe
    html = [u for u in urls if u.lower().endswith((".htm", ".html"))]
    pdf = [u for u in urls if u.lower().endswith(".pdf")]
    return html if html else pdf

def extract_date_from_url(url: str) -> Optional[str]:
    m = DATE_YYYYMMDD_RE.search(url)
    if not m:
        return None
    yyyymmdd = m.group("date")
    try:
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

@dataclass(frozen=True)
class DocLink:
    date: str       # YYYY-MM-DD
    doc_type: str   # "minutes" or "statement"
    url: str

def fetch(url: str, *, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "agentic-rag-fomc/1.0"})
    r.raise_for_status()
    return r.content

def parse_master_calendar_links() -> list[DocLink]:
    """
    Download the master FOMC calendar page and extract statement + minutes links. Dates are parsed from URLs (YYYYMMDD).
    """
    html = fetch(CAL_URL).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    anchors = soup.find_all("a", href=True)
    minutes_urls: list[str] = []
    statement_urls: list[str] = []

    for a in anchors:
        href = a.get("href", "").strip()
        if not href:
            continue
        abs_url = urljoin(FED_BASE, href)

        if is_minutes_link(abs_url):
            minutes_urls.append(abs_url)
        elif is_statement_link(abs_url):
            statement_urls.append(abs_url)

    minutes_urls = prefer_html(minutes_urls)
    statement_urls = prefer_html(statement_urls)

    links: list[DocLink] = []
    for u in minutes_urls:
        d = extract_date_from_url(u)
        if d:
            links.append(DocLink(date=d, doc_type="minutes", url=u))

    for u in statement_urls:
        d = extract_date_from_url(u)
        if d:
            links.append(DocLink(date=d, doc_type="statement", url=u))

    # De-dupe by (date, type)
    seen = set()
    unique: list[DocLink] = []
    for link in sorted(links, key=lambda x: (x.date, x.doc_type, x.url)):
        key = (link.date, link.doc_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(link)

    return unique

def repo_root() -> Path:
    # src/ingest/download.py -> parents[2] should be repo root
    return Path(__file__).resolve().parents[2]

def save_file(raw_dir: Path, link: DocLink, content: bytes) -> Path:
    ext = ".pdf" if link.url.lower().endswith(".pdf") else ".html"
    out = raw_dir / f"{link.date}_{link.doc_type}{ext}"
    out.write_bytes(content)
    return out

@app.command()
def run(
    start_year: int = typer.Option(..., help="Start year (inclusive), e.g. 2022"),
    end_year: int = typer.Option(..., help="End year (inclusive), e.g. 2023"),
    raw_dir: Optional[Path] = typer.Option(None, help="Directory to store raw downloads (defaults to repo_root/data/raw)"),
    manifest_path: Optional[Path] = typer.Option(None, help="Manifest path (defaults to repo_root/data/raw/manifest.jsonl). Set to '' to disable."),
    overwrite: bool = typer.Option(False, help="If true, re-download even if file exists"),
) -> None:
    if end_year < start_year:
        raise typer.BadParameter("end_year must be >= start_year")

    root = repo_root()
    raw_dir = raw_dir or (root / "data" / "raw")

    if manifest_path is None:
        manifest_path = root / "data" / "raw" / "manifest.jsonl"

    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = None
    if manifest_path and str(manifest_path).strip():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_path.open("a", encoding="utf-8")

    typer.echo(f"Downloading FOMC documents {start_year}..{end_year} into {raw_dir}/")
    typer.echo(f"Calendar source: {CAL_URL}")

    all_links = parse_master_calendar_links()

    # Filter to requested years
    links = []
    for link in all_links:
        y = int(link.date.split("-")[0])
        if start_year <= y <= end_year:
            links.append(link)

    total = len(links)
    downloaded = 0
    skipped = 0

    typer.echo(f"Found {total} docs after year-filter")

    for link in links:
        ext = ".pdf" if link.url.lower().endswith(".pdf") else ".html"
        out_path = raw_dir / f"{link.date}_{link.doc_type}{ext}"

        if out_path.exists() and not overwrite:
            skipped += 1
            typer.echo(f"  - SKIP {out_path.name}")
            continue

        try:
            content = fetch(link.url)
            saved = save_file(raw_dir, link, content)
            downloaded += 1
            typer.echo(f"  + DOWN {saved.name}")

            if manifest_file:
                record = {
                    "date": link.date,
                    "doc_type": link.doc_type,
                    "url": link.url,
                    "path": str(saved),
                    "downloaded_at": datetime.utcnow().isoformat() + "Z",
                }
                manifest_file.write(json.dumps(record) + "\n")
                manifest_file.flush()

        except Exception as e:
            typer.echo(f"  ! FAIL {link.date} {link.doc_type}: {e}")

    if manifest_file:
        manifest_file.close()

    typer.echo("\nDone.")
    typer.echo(f"Downloaded: {downloaded}")
    typer.echo(f"Skipped:    {skipped}")
    typer.echo(f"Raw dir:    {raw_dir.resolve()}")
    if manifest_path and str(manifest_path).strip():
        typer.echo(f"Manifest:   {Path(manifest_path).resolve()}")

if __name__ == "__main__":
    app()