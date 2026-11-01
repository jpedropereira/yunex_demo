"""Turn an EDGAR filing into chunks of text: plain parsing, with no network and no model.

Three things here protect the demo from bad input or silent data loss: only EDGAR
document URLs are accepted, metadata comes from the filing's own XBRL tags, and no
chunk is too long for the embedding model to read in full.
"""

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# fullmatch against this leaves no room for another host, a user:pass@ prefix, a
# query string or "..": the app must never fetch an arbitrary URL someone types (SSRF).
EDGAR_URL = re.compile(
    r"https://www\.sec\.gov/Archives/edgar/data/"
    r"(?P<cik>\d{1,10})/(?P<accession>\d{18})/[A-Za-z0-9][A-Za-z0-9_-]*\.htm"
)

METADATA_TAGS = {
    "company": "dei:EntityRegistrantName",
    "ticker": "dei:TradingSymbol",
    "form": "dei:DocumentType",
    "period_end": "dei:DocumentPeriodEndDate",
    "fiscal_year": "dei:DocumentFiscalYearFocus",
}

# EDGAR HTML has no real pages, only CSS page breaks, and the spacing varies.
PAGE_BREAK_STYLE = re.compile(r"page-break-(before|after)\s*:\s*always", re.IGNORECASE)
PAGE_MARKER = "\x00PAGE-BREAK\x00"

# Marks where tables and rows sit in the extracted text, so that a long page can be split
# without cutting through them. A number means little once it's apart from its row.
TABLE_START = "\x00TABLE-START\x00"
TABLE_END = "\x00TABLE-END\x00"
ROW_END = "\x00ROW-END\x00"
TABLE_PATTERN = re.compile(f"{TABLE_START}(.*?){TABLE_END}", re.DOTALL)

MIN_WORDS_PER_PAGE = 30
FALLBACK_WORDS_PER_PAGE = 500

# Measured on this model: prose is about 1.1 tokens per word, but financial tables
# reach 4.75, because every digit and comma is its own token. A word limit would
# either split most prose pages or truncate the tables, so count what costs tokens.
TOKEN_PATTERN = re.compile(r"\d|\w+|[^\w\s]")

# The model accepts 2048 tokens. The estimate above can run low, by up to 1.8x on
# acronym-heavy text (measured), so 2048 / 1.8 leaves room for that and the title
# prefix. Typical prose pages still fit whole, and embedding fails loudly if not.
MAX_TOKENS_PER_CHUNK = 1100


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a filing. ``part`` is above 1 only on a split page."""

    page: int
    part: int
    text: str


def parse_edgar_url(url: str) -> tuple[str, str]:
    """Return (cik, accession) for an EDGAR document URL, or raise ValueError."""
    match = EDGAR_URL.fullmatch(url.strip())
    if match is None:
        raise ValueError(
            "Not an EDGAR document URL. Expected "
            "https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<file>.htm "
            "(the plain document link, not the /ix?doc= viewer link)."
        )
    return match["cik"], match["accession"]


def canonical_edgar_url(url: str) -> str:
    """Drop leading zeros from the CIK of a valid EDGAR URL, or raise ValueError.

    EDGAR links often pad the CIK, and the SEC answers those with a redirect, which we
    deliberately don't follow. Rewriting it keeps the same validated host and path.
    """
    cik, _ = parse_edgar_url(url)
    return url.strip().replace(f"/data/{cik}/", f"/data/{int(cik)}/")


def read_metadata(soup: BeautifulSoup) -> dict[str, str]:
    """Read the filing's facts from its XBRL tags. Call this before ``split_pages``.

    Look these up by ``name``, across the whole document: the HTML parser lowercases
    tag names, so searching for "ix:nonNumeric" would find nothing. A missing fact is
    "", not None, because Chroma metadata values cannot be None.
    """
    metadata = {}
    for field, tag_name in METADATA_TAGS.items():
        element = soup.find(attrs={"name": tag_name})
        # No separator: a value split across spans must read "January 25, 2026", and a
        # separator would make it "January 25 , 2026".
        metadata[field] = _clean(element.get_text()) if element else ""
    return metadata


def estimate_tokens(text: str) -> int:
    """Roughly how many tokens the embedding model will see in this text."""
    return len(TOKEN_PATTERN.findall(text))


def split_pages(soup: BeautifulSoup) -> tuple[list[Chunk], int]:
    """Split a filing into chunks, one per page unless a page is too long to embed.

    Returns the chunks and the number of pages found, including skipped ones.
    Modifies ``soup``, removing the hidden XBRL header, scripts and styles.
    """
    for tag_name in ("ix:header", "script", "style"):
        for element in soup.find_all(tag_name):
            element.decompose()

    # Page breaks can be nested at any depth, so mark them in place and split the
    # finished text, instead of walking the HTML tree.
    markers = soup.find_all(style=PAGE_BREAK_STYLE)
    for element in markers:
        if PAGE_BREAK_STYLE.search(element["style"]).group(1).lower() == "before":
            element.insert_before(PAGE_MARKER)
        else:
            element.insert_after(PAGE_MARKER)

    for table in soup.find_all("table"):
        if table.find_parent("table") is None:
            for row in table.find_all("tr"):
                row.insert_after(ROW_END)
            table.insert_before(TABLE_START)
            table.insert_after(TABLE_END)

    text = soup.get_text(" ")
    if markers:
        pages = text.split(PAGE_MARKER)
    else:
        logger.warning("No page breaks found; using %d-word windows.", FALLBACK_WORDS_PER_PAGE)
        pages = _word_windows(_plain_text(text))

    chunks = []
    # Number by position in the document, so a skipped page still takes up a number.
    for page_number, page_text in enumerate(pages, start=1):
        if len(_plain_text(page_text).split()) < MIN_WORDS_PER_PAGE:
            continue
        for part, part_text in enumerate(_split_long_page(page_text), start=1):
            chunks.append(Chunk(page=page_number, part=part, text=part_text))
    return chunks, len(pages)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _word_windows(text: str) -> list[str]:
    words = text.split()
    return [
        " ".join(words[start : start + FALLBACK_WORDS_PER_PAGE])
        for start in range(0, len(words), FALLBACK_WORDS_PER_PAGE)
    ]


def _plain_text(text: str) -> str:
    """The text as it will be stored: without our markers, and with tidy whitespace."""
    for marker in (TABLE_START, TABLE_END, ROW_END):
        text = text.replace(marker, " ")
    return _clean(text)


def _unbreakable_blocks(page: str) -> list[str]:
    """The pieces a page may be split between: words, whole tables, or rows of long tables."""
    blocks = []
    # With one capture group, re.split alternates: text outside tables, table, outside...
    for index, piece in enumerate(TABLE_PATTERN.split(page)):
        is_table = index % 2 == 1
        if not is_table:
            blocks.extend(_plain_text(piece).split())
        elif estimate_tokens(_plain_text(piece)) <= MAX_TOKENS_PER_CHUNK:
            blocks.append(_plain_text(piece))
        else:
            rows = [_plain_text(row) for row in piece.split(ROW_END)]
            blocks.extend(row for row in rows if row)
    return blocks


def _split_long_page(page: str) -> list[str]:
    """Split a page into evenly sized parts that fit within the token limit.

    Even sizes matter: filling each part to the limit can leave a tail of a few words,
    too little to embed meaningfully, which then takes up a search result.
    """
    total_tokens = estimate_tokens(_plain_text(page))
    if total_tokens <= MAX_TOKENS_PER_CHUNK:
        return [_plain_text(page)]
    target_tokens = total_tokens / (total_tokens // MAX_TOKENS_PER_CHUNK + 1)

    parts = []
    current_blocks = []
    current_tokens = 0
    for block in _unbreakable_blocks(page):
        block_tokens = estimate_tokens(block)
        reached_target = current_tokens >= target_tokens
        would_overflow = current_tokens + block_tokens > MAX_TOKENS_PER_CHUNK
        if current_blocks and (reached_target or would_overflow):
            parts.append(" ".join(current_blocks))
            current_blocks = []
            current_tokens = 0
        current_blocks.append(block)
        current_tokens += block_tokens
    parts.append(" ".join(current_blocks))
    return parts
