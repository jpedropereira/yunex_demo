"""Index an EDGAR filing: validate, fetch, split, embed, and store.

The page and the command line both call ``ingest_filing``. Pre-index before the talk
with ``make index-all``, or directly:

    python -m yunex_demo.sec.ingest --all
    python -m yunex_demo.sec.ingest <edgar document url>
"""

import logging
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import chromadb
import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ollama import ResponseError

from yunex_demo.config import settings
from yunex_demo.sec import filing, store

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 30

# EDGAR files open with an XML declaration, but we read them as HTML on purpose: it
# copes with the markup as filed, so the warning about this is only noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass(frozen=True)
class SampleFiling:
    name: str
    url: str


# The 10-Ks offered as one-click buttons on the page, and indexed by `make index-all`.
SAMPLE_FILINGS = [
    SampleFiling(
        "NVIDIA",
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm",
    ),
    SampleFiling(
        "Apple",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
    ),
    SampleFiling(
        "Microsoft",
        "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm",
    ),
    SampleFiling(
        "Alphabet",
        "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm",
    ),
    SampleFiling(
        "Amazon",
        "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm",
    ),
]


class FetchError(Exception):
    """The filing could not be downloaded from the SEC."""


# What can go wrong for reasons outside this code, shown as a message rather than a crash.
# The ollama client turns "server unreachable" into a builtin ConnectionError.
EXPECTED_ERRORS = (ValueError, FetchError, ConnectionError, httpx.HTTPError, ResponseError)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    already_indexed: bool
    pages_indexed: int
    pages_skipped: int = 0
    company: str = ""
    form: str = ""
    fiscal_year: str = ""


def fetch_filing(url: str) -> str:
    """Download a filing's HTML. Only ever called with a validated EDGAR URL."""
    filing.parse_edgar_url(url)

    if not settings.sec_user_agent:
        raise FetchError('Set SEC_USER_AGENT="Your Name you@example.com" in .env to fetch filings.')

    # Don't follow redirects: they could lead away from the host we just validated.
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": settings.sec_user_agent},
            timeout=FETCH_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise FetchError(f"Could not reach the SEC: {type(exc).__name__}.") from exc

    if response.status_code == 403:
        raise FetchError("The SEC returned 403: check that SEC_USER_AGENT has a name and email.")
    if response.status_code != 200:
        raise FetchError(f"The SEC returned HTTP {response.status_code}.")
    return response.text


def chunk_id(accession: str, chunk: filing.Chunk, page_was_split: bool) -> str:
    base_id = f"{accession}-p{chunk.page:04d}"
    return f"{base_id}-{chunk.part}" if page_was_split else base_id


def indexed_page_count(collection: chromadb.Collection, doc_id: str) -> int:
    stored = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    return len({metadata["page"] for metadata in stored["metadatas"]})


def ingest_filing(
    url: str,
    collection: chromadb.Collection,
    on_progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    """Index one filing, or do nothing if it is already there.

    ``on_progress`` is called with (chunks embedded, total chunks) after each batch.
    """
    url = filing.canonical_edgar_url(url)
    cik, accession = filing.parse_edgar_url(url)

    # The accession number identifies the filing, so re-running costs nothing.
    pages_already_indexed = indexed_page_count(collection, accession)
    if pages_already_indexed:
        return IngestResult(accession, already_indexed=True, pages_indexed=pages_already_indexed)

    soup = BeautifulSoup(fetch_filing(url), "lxml")
    metadata = filing.read_metadata(soup)
    chunks, page_count = filing.split_pages(soup)
    if not chunks:
        raise ValueError("No readable text found in this filing.")

    title = f"{metadata['company']} {metadata['form']} {metadata['fiscal_year']}"
    prompts = [store.document_prompt(title, chunk.text) for chunk in chunks]

    embeddings = []
    for start in range(0, len(prompts), settings.embed_batch_size):
        embeddings.extend(store.embed_texts(prompts[start : start + settings.embed_batch_size]))
        logger.info("Embedded %d of %d chunks", len(embeddings), len(prompts))
        if on_progress:
            on_progress(len(embeddings), len(prompts))

    # Store only once everything is embedded. A failure halfway must not leave part of
    # a filing behind, because the check above would then report it as already indexed.
    split_page_numbers = {chunk.page for chunk in chunks if chunk.part > 1}
    collection.upsert(
        ids=[chunk_id(accession, c, c.page in split_page_numbers) for c in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                **metadata,
                "doc_id": accession,
                "cik": cik,
                "source_url": url,
                "page": chunk.page,
                "part": chunk.part,
            }
            for chunk in chunks
        ],
    )

    pages_indexed = len({chunk.page for chunk in chunks})
    return IngestResult(
        doc_id=accession,
        already_indexed=False,
        pages_indexed=pages_indexed,
        pages_skipped=page_count - pages_indexed,
        company=metadata["company"],
        form=metadata["form"],
        fiscal_year=metadata["fiscal_year"],
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m yunex_demo.sec.ingest <edgar document url> | --all",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    urls = [sample.url for sample in SAMPLE_FILINGS] if sys.argv[1] == "--all" else sys.argv[1:]

    # One bad filing shouldn't stop the others: they are separate, and re-running is cheap.
    failures = 0
    for url in urls:
        print(url)
        try:
            result = ingest_filing(url, store.get_collection())
        except EXPECTED_ERRORS as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        if result.already_indexed:
            print(f"Already indexed: {result.pages_indexed} pages")
        else:
            print(
                f"Indexed {result.company} {result.form} FY{result.fiscal_year}: "
                f"{result.pages_indexed} pages, {result.pages_skipped} skipped"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
