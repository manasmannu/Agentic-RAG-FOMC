from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import typer
from bs4 import BeautifulSoup
import html2text

app = typer.Typer(help="Clean downloaded FOMC HTML into plain text for chunking.")

WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")

@dataclass(frozen=True)
class CleanResult:
    text: str
    removed_reason: Optional[str] = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    """
    Remove obvious non-content elements from Federal Reserve pages. This is intentionally conservative (don’t delete too much).
    """
    # Remove scripts/styles/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove header/footer/nav if present
    for selector in [
        "header", "footer", "nav", ".header", ".footer", ".nav", "#header", "#footer", "#nav", ".share", ".social", ".breadcrumbs",
        ".breadcrumb", ".related", ".relatedLinks", ".side-nav", ".sidenav", ".lastUpdate", ".page-footer", ".pageFooter"
    ]:
        for tag in soup.select(selector):
            tag.decompose()


def _extract_main_content_html(soup: BeautifulSoup) -> str:
    """
    Try to find the main content container first; fallback to body. Fed pages vary, so we attempt a few common containers.
    """
    candidates = [
        soup.select_one("main"),
        soup.select_one("#content"),
        soup.select_one(".content"),
        soup.select_one(".col-xs-12"),
        soup.select_one("article"),
        soup.body
    ]
    for c in candidates:
        if c and c.get_text(strip=True):
            return str(c)
    return str(soup)


def clean_fed_html_to_text(html: str) -> CleanResult:
    soup = BeautifulSoup(html, "lxml")
    _strip_boilerplate(soup)
    main_html = _extract_main_content_html(soup)

    # html2text keeps headings/links reasonably and produces paragraph-like output
    h = html2text.HTML2Text()
    h.ignore_links = True         # citations come from doc_id/chunk_id, not URLs
    h.ignore_images = True
    h.body_width = 0              # don’t hard-wrap
    h.single_line_break = False

    text = h.handle(main_html)

    # Normalize whitespace
    text = text.replace("\r", "")
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text).strip()

    # Safety: If text is too short, something went wrong (maybe container selection)
    if len(text) < 400:
        return CleanResult(text=text, removed_reason="too_short_after_cleaning")

    return CleanResult(text=text)


def output_name_from_raw(raw_path: Path) -> Path:
    # e.g., 2023-12-13_minutes.html -> 2023-12-13_minutes.txt
    return raw_path.with_suffix(".txt").name


@app.command()
def run(
    raw_dir: Optional[Path] = typer.Option(None, help="Input dir of raw downloads (default repo_root/data/raw)"),
    out_dir: Optional[Path] = typer.Option(None, help="Output dir for cleaned text (default repo_root/data/cleaned)"),
    overwrite: bool = typer.Option(False, help="Overwrite existing cleaned files"),
    fail_on_short: bool = typer.Option(False, help="Fail if a cleaned doc becomes too short"),
) -> None:
    root = repo_root()
    raw_dir = raw_dir or (root / "data" / "raw")
    out_dir = out_dir or (root / "data" / "cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(list(raw_dir.glob("*.html")))
    if not raw_files:
        typer.echo(f"No .html files found in {raw_dir}")
        raise typer.Exit(code=1)

    typer.echo(f"Cleaning {len(raw_files)} HTML files from {raw_dir} -> {out_dir}")

    cleaned = 0
    skipped = 0
    warned = 0

    for raw_path in raw_files:
        out_path = out_dir / output_name_from_raw(raw_path)

        if out_path.exists() and not overwrite:
            skipped += 1
            typer.echo(f"  - SKIP {out_path.name}")
            continue

        html = raw_path.read_text(encoding="utf-8", errors="ignore")
        result = clean_fed_html_to_text(html)

        if result.removed_reason:
            warned += 1
            msg = f"  ! WARN {raw_path.name}: {result.removed_reason} (len={len(result.text)})"
            typer.echo(msg)
            if fail_on_short:
                raise typer.Exit(code=2)

        out_path.write_text(result.text, encoding="utf-8")
        cleaned += 1
        typer.echo(f"  + OUT  {out_path.name}  (chars={len(result.text)})")

    typer.echo("\nDone.")
    typer.echo(f"Cleaned: {cleaned}")
    typer.echo(f"Skipped: {skipped}")
    typer.echo(f"Warned:  {warned}")
    typer.echo(f"Output:  {out_dir.resolve()}")


if __name__ == "__main__":
    app()