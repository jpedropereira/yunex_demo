"""Index an EDGAR filing: validate, fetch, split, embed, and store.

The collection holds one 10-K for each company in ``settings.agent_filings``, and nothing
else, so the agent can only ever cite those. The page and the command line both call
``ingest_filing``. Pre-index before the talk with ``make index-all``, or directly:

    python -m yunex_demo.sec.ingest --all
    python -m yunex_demo.sec.ingest --rebuild
    python -m yunex_demo.sec.ingest <edgar document url>
"""

import logging
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

import chromadb
import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ollama import ResponseError

from yunex_demo.config import AgentFiling, settings
from yunex_demo.sec import filing, store

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 30

# EDGAR files open with an XML declaration, but we read them as HTML on purpose: it
# copes with the markup as filed, so the warning about this is only noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class FetchError(Exception):
    """The filing could not be downloaded from the SEC."""


# What can go wrong for reasons outside this code, shown as a message rather than a crash.
# The ollama client turns "server unreachable" into a builtin ConnectionError. KeyError is
# an SEC response without the fields we read; bad JSON is already a ValueError.
EXPECTED_ERRORS = (
    ValueError,
    KeyError,
    FetchError,
    ConnectionError,
    httpx.HTTPError,
    ResponseError,
)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    already_indexed: bool
    pages_indexed: int
    pages_skipped: int = 0
    replaced_older: bool = False
    company: str = ""
    form: str = ""
    fiscal_year: str = ""


def _sec_get(url: str) -> httpx.Response:
    """GET from the SEC, declaring who we are. Only called with a URL built or validated here."""
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
    return response


def fetch_filing(url: str) -> str:
    """Download a filing's HTML. Only ever called with a validated EDGAR URL."""
    filing.parse_edgar_url(url)
    return _sec_get(url).text


def latest_10k_url(cik: int) -> str:
    """Ask the SEC for this company's most recent 10-K, and return its document URL."""
    # The one request that leaves www.sec.gov, and the CIK is an int, so nothing a user
    # typed can reach this URL.
    recent = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"][
        "recent"
    ]

    # Newest first, and an exact match: a 10-K/A amends a filing, it doesn't replace it.
    if "10-K" not in recent["form"]:
        raise FetchError(f"The SEC lists no recent 10-K for CIK {cik}.")
    position = recent["form"].index("10-K")
    accession = recent["accessionNumber"][position].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
        f"{recent['primaryDocument'][position]}"
    )
    # This came from a response, so it gets the same check as a pasted URL.
    filing.parse_edgar_url(url)
    return url


def _agent_filing(cik: str) -> AgentFiling:
    for agent_filing in settings.agent_filings:
        if agent_filing.cik == int(cik):
            return agent_filing
    companies = ", ".join(agent_filing.company for agent_filing in settings.agent_filings)
    raise ValueError(f"Only filings from these companies can be indexed: {companies}.")


def _period_end(period_end: str) -> date:
    return datetime.strptime(period_end, "%B %d, %Y").date()


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
    ticker = _agent_filing(cik).ticker

    # The accession number identifies the filing, so re-running costs nothing.
    pages_already_indexed = indexed_page_count(collection, accession)
    if pages_already_indexed:
        return IngestResult(accession, already_indexed=True, pages_indexed=pages_already_indexed)

    soup = BeautifulSoup(fetch_filing(url), "lxml")
    metadata = filing.read_metadata(soup)
    chunks, page_count = filing.split_pages(soup)
    if not chunks:
        raise ValueError("No readable text found in this filing.")

    # A 10-Q has a newer period, so without this check it would replace the 10-K.
    if metadata["form"] != "10-K":
        raise ValueError(f"Only 10-K filings can be indexed, and this is a {metadata['form']}.")

    # One 10-K per company. Pasting an old filing's URL must not push out the latest one.
    period_end = _period_end(metadata["period_end"])
    indexed = collection.get(where={"ticker": ticker}, include=["metadatas"])["metadatas"]
    if any(_period_end(chunk["period_end"]) > period_end for chunk in indexed):
        raise ValueError(f"A newer 10-K for {ticker} is already indexed.")
    older_doc_ids = {chunk["doc_id"] for chunk in indexed}

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
                "ticker": ticker,
                "doc_id": accession,
                "cik": cik,
                "source_url": url,
                "page": chunk.page,
                "part": chunk.part,
                "printed_page": chunk.printed_page,
            }
            for chunk in chunks
        ],
    )

    # Only now that the new filing is safely stored, so a failure never leaves no filing.
    for older_doc_id in older_doc_ids:
        collection.delete(where={"doc_id": older_doc_id})

    pages_indexed = len({chunk.page for chunk in chunks})
    return IngestResult(
        doc_id=accession,
        already_indexed=False,
        pages_indexed=pages_indexed,
        pages_skipped=page_count - pages_indexed,
        replaced_older=bool(older_doc_ids),
        company=metadata["company"],
        form=metadata["form"],
        fiscal_year=metadata["fiscal_year"],
    )


def _print_result(result: IngestResult) -> None:
    if result.already_indexed:
        print(f"Already indexed: {result.pages_indexed} pages")
    else:
        replaced = ", replacing the older one" if result.replaced_older else ""
        print(
            f"Indexed {result.company} {result.form} FY{result.fiscal_year}: "
            f"{result.pages_indexed} pages, {result.pages_skipped} skipped{replaced}"
        )


def rebuild_index() -> int:
    """Index every filing beside the current index, and swap only if all of them worked.

    So a missing SEC_USER_AGENT or bad Wi-Fi can't leave the talk with no filings at all.
    """
    try:
        rebuilt = store.empty_rebuild_collection()
        for agent_filing in settings.agent_filings:
            url = latest_10k_url(agent_filing.cik)
            print(url)
            _print_result(ingest_filing(url, rebuilt))
    except EXPECTED_ERRORS as exc:
        print(f"FAILED: {exc}\nThe current index was not changed.", file=sys.stderr)
        return 1

    store.replace_index(rebuilt)
    print("The search index was rebuilt.")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m yunex_demo.sec.ingest <edgar document url> | --all | --rebuild",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if sys.argv[1] == "--rebuild":
        return rebuild_index()

    # One bad filing shouldn't stop the others: they are separate, and re-running is cheap.
    failures = 0
    targets = settings.agent_filings if sys.argv[1] == "--all" else sys.argv[1:]
    for target in targets:
        try:
            url = target if isinstance(target, str) else latest_10k_url(target.cik)
            print(url)
            result = ingest_filing(url, store.get_collection())
        except EXPECTED_ERRORS as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue

        _print_result(result)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
