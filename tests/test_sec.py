"""SEC filing search: URL safety, parsing, ingest and search. No network and no Ollama."""

import httpx
import pytest
from bs4 import BeautifulSoup
from conftest import ALPHA_ACCESSION, BETA_ACCESSION, filing_html, filing_url

from yunex_demo.config import settings
from yunex_demo.sec import filing, ingest, search, store

NVIDIA_URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"

# Long enough not to be skipped as a blank page.
WORDS = "word " * 40


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# --- URL validation -----------------------------------------------------------


def test_accepts_the_nvidia_url():
    assert filing.parse_edgar_url(NVIDIA_URL) == ("1045810", "000104581026000021")


@pytest.mark.parametrize(
    "url",
    [
        NVIDIA_URL.replace("https://", "http://"),
        "https://evil.example/Archives/edgar/data/1045810/000104581026000021/nvda.htm",
        "https://www.sec.gov.evil.example/Archives/edgar/data/1/000104581026000021/nvda.htm",
        "https://www.sec.gov@evil.example/Archives/edgar/data/1/000104581026000021/nvda.htm",
        "https://www.sec.gov/cgi-bin/browse-edgar",
        "https://www.sec.gov/Archives/edgar/data/1045810/short/nvda.htm",
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/../../secret.htm",
        NVIDIA_URL + "?next=https://evil.example",
        NVIDIA_URL + "#fragment",
        "https://www.sec.gov/ix?doc=/Archives/edgar/data/1045810/000104581026000021/nvda.htm",
        "http://localhost:8000/nvda.htm",
        "https://localhost/Archives/edgar/data/1/000104581026000021/nvda.htm",
        "",
    ],
)
def test_rejects_everything_else(url):
    with pytest.raises(ValueError, match="Not an EDGAR document URL"):
        filing.parse_edgar_url(url)


def test_every_sample_filing_passes_validation():
    accessions = [filing.parse_edgar_url(sample.url)[1] for sample in ingest.SAMPLE_FILINGS]
    assert len(set(accessions)) == len(ingest.SAMPLE_FILINGS)


def test_a_padded_cik_is_rewritten_because_the_sec_would_redirect_it():
    padded = "https://www.sec.gov/Archives/edgar/data/0000789019/000119312526323660/msft.htm"
    assert filing.canonical_edgar_url(padded) == padded.replace("0000789019", "789019")
    assert filing.canonical_edgar_url(NVIDIA_URL) == NVIDIA_URL


def test_canonicalising_still_rejects_other_urls():
    with pytest.raises(ValueError):
        filing.canonical_edgar_url("https://evil.example/Archives/edgar/data/0001/x.htm")


# --- Metadata -----------------------------------------------------------------


def test_reads_metadata_inside_and_outside_the_hidden_header():
    metadata = filing.read_metadata(_soup(filing_html("ALPHA CORP", [WORDS])))
    assert metadata == {
        "company": "ALPHA CORP",
        "ticker": "TICK",
        "form": "10-K",
        "period_end": "January 25, 2026",
        "fiscal_year": "2026",
    }


def test_a_value_split_across_spans_reads_as_written():
    html = (
        '<html><body><ix:nonNumeric name="dei:DocumentPeriodEndDate">'
        "<span>January 25</span><span>, 2026</span></ix:nonNumeric></body></html>"
    )
    assert filing.read_metadata(_soup(html))["period_end"] == "January 25, 2026"


def test_a_missing_fact_is_an_empty_string():
    assert filing.read_metadata(_soup("<html><body>No tags</body></html>"))["company"] == ""


# --- Splitting into pages -----------------------------------------------------


def test_three_breaks_give_four_pages_with_the_right_text():
    html = (
        f"<html><body><div>one {WORDS}</div>"
        f'<hr style="page-break-after: always"/><div>two {WORDS}</div>'
        f'<div><hr style="PAGE-BREAK-AFTER:always"/></div><div>three {WORDS}</div>'
        f'<div style="page-break-before:always">four {WORDS}</div></body></html>'
    )
    chunks, page_count = filing.split_pages(_soup(html))
    assert page_count == 4
    assert [chunk.page for chunk in chunks] == [1, 2, 3, 4]
    assert [chunk.text.split()[0] for chunk in chunks] == ["one", "two", "three", "four"]


def test_hidden_header_scripts_and_styles_are_not_indexed():
    html = (
        "<html><body><ix:header>hidden-fact</ix:header>"
        f"<div>{WORDS}</div>"
        "<script>var secret;</script><style>.x{}</style></body></html>"
    )
    chunks, _ = filing.split_pages(_soup(html))
    assert chunks[0].text == WORDS.strip()


def test_short_pages_are_skipped_but_keep_their_number():
    html = filing_html("ALPHA CORP", [f"first {WORDS}", "12 13 14", f"third {WORDS}"])
    chunks, page_count = filing.split_pages(_soup(html))
    assert page_count == 3
    assert [chunk.page for chunk in chunks] == [1, 3]


def test_without_page_breaks_it_falls_back_to_word_windows():
    chunks, page_count = filing.split_pages(_soup(f"<html><body>{'word ' * 1200}</body></html>"))
    assert page_count == 3
    assert [chunk.page for chunk in chunks] == [1, 2, 3]
    assert len(chunks[0].text.split()) == filing.FALLBACK_WORDS_PER_PAGE


def test_a_long_prose_page_is_split_into_parts_of_the_same_page():
    chunks, _ = filing.split_pages(
        _soup(
            f"<html><body><div>{'word ' * 1500}</div>"
            '<hr style="page-break-after:always"/>'
            f"<div>{WORDS}</div></body></html>"
        )
    )
    assert [(chunk.page, chunk.part) for chunk in chunks] == [(1, 1), (1, 2), (2, 1)]


def test_a_page_just_over_the_limit_leaves_no_tiny_tail():
    page = "word " * (filing.MAX_TOKENS_PER_CHUNK + 20)
    html = filing_html("ALPHA CORP", [page, WORDS])
    chunks = [chunk for chunk in filing.split_pages(_soup(html))[0] if chunk.page == 1]

    sizes = [filing.estimate_tokens(chunk.text) for chunk in chunks]
    assert len(sizes) == 2
    assert abs(sizes[0] - sizes[1]) <= 25, "the cover-page facts add a few tokens"


def test_a_table_heavy_page_splits_on_tokens_not_words():
    """Under 1,000 words, but every digit is a token: a word limit would truncate this."""
    table = "1,234 5,678 (910) 11.2 % $ 3,456 7,890 12,345 678 " * 100
    assert len(table.split()) == 1000

    chunks, _ = filing.split_pages(_soup(f"<html><body><div>{table}</div></body></html>"))

    assert len(chunks) > 2
    assert all(
        filing.estimate_tokens(chunk.text) <= filing.MAX_TOKENS_PER_CHUNK for chunk in chunks
    )
    assert " ".join(chunk.text for chunk in chunks).split() == table.split(), "text was lost"


def _table(rows: int, label: str = "row") -> str:
    cells = "".join(
        f"<tr><td>{label}{n}</td><td>1,234</td><td>5,678</td></tr>" for n in range(rows)
    )
    return f"<table>{cells}</table>"


def test_a_table_that_fits_stays_whole_when_its_page_is_split():
    html = (
        f"<html><body><div>{'word ' * 900}</div>"
        f"{_table(40)}<div>{'word ' * 300}</div></body></html>"
    )
    html = html.replace(
        "</body>", f'<hr style="page-break-after:always"/><div>{WORDS}</div></body>'
    )

    chunks, _ = filing.split_pages(_soup(html))
    page_one = [chunk for chunk in chunks if chunk.page == 1]

    assert len(page_one) > 1, "this page must be long enough to split"
    holding_rows = [chunk for chunk in page_one if "row0" in chunk.text or "row39" in chunk.text]
    assert len(holding_rows) == 1
    assert "row0" in holding_rows[0].text and "row39" in holding_rows[0].text


def test_a_table_too_long_for_one_chunk_splits_only_between_rows():
    table = _table(300)
    html = (
        f"<html><body><div>{table}</div>"
        f'<hr style="page-break-after:always"/><div>{WORDS}</div></body></html>'
    )

    chunks, _ = filing.split_pages(_soup(html))
    page_one = [chunk for chunk in chunks if chunk.page == 1]

    assert len(page_one) > 1
    assert all(filing.estimate_tokens(c.text) <= filing.MAX_TOKENS_PER_CHUNK for c in page_one)
    text = " ".join(chunk.text for chunk in page_one)
    for n in range(300):
        row = f"row{n} 1,234 5,678"
        assert row in text, f"row {n} was cut or lost"
        assert sum(row in chunk.text for chunk in page_one) == 1


def test_our_markers_never_reach_the_stored_text():
    long_page = f"<div>{'word ' * 1500}</div>{_table(300)}"
    with_breaks = (
        f'<html><body>{long_page}<hr style="page-break-after:always"/>{_table(40)}</body></html>'
    )
    without_breaks = f"<html><body>{long_page}</body></html>"

    for html in (with_breaks, without_breaks):
        chunks, _ = filing.split_pages(_soup(html))
        assert chunks
        assert not any("\x00" in chunk.text for chunk in chunks)


def test_a_page_with_only_a_tiny_table_is_still_skipped_as_short():
    html = (
        f"<html><body><div>{WORDS}</div>"
        f'<hr style="page-break-after:always"/>{_table(1)}</body></html>'
    )
    chunks, page_count = filing.split_pages(_soup(html))
    assert page_count == 2
    assert [chunk.page for chunk in chunks] == [1]


# --- Fetching -----------------------------------------------------------------


def test_fetch_declares_the_user_agent_and_does_not_follow_redirects(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, text="<html></html>")

    monkeypatch.setattr(settings, "sec_user_agent", "Test Name test@example.com")
    monkeypatch.setattr(ingest.httpx, "get", fake_get)

    assert ingest.fetch_filing(NVIDIA_URL) == "<html></html>"
    url, kwargs = calls[0]
    assert url == NVIDIA_URL
    assert kwargs["headers"]["User-Agent"] == "Test Name test@example.com"
    assert kwargs["follow_redirects"] is False


def test_fetch_refuses_an_unvalidated_url_before_any_request(monkeypatch):
    def must_not_run(url, **kwargs):
        raise AssertionError("no request should be made")

    monkeypatch.setattr(settings, "sec_user_agent", "Test Name test@example.com")
    monkeypatch.setattr(ingest.httpx, "get", must_not_run)
    with pytest.raises(ValueError):
        ingest.fetch_filing("https://evil.example/x.htm")


def test_fetch_without_a_user_agent_says_what_to_set(monkeypatch):
    monkeypatch.setattr(settings, "sec_user_agent", "")
    with pytest.raises(ingest.FetchError, match="SEC_USER_AGENT"):
        ingest.fetch_filing(NVIDIA_URL)


@pytest.mark.parametrize("status", [301, 403, 500])
def test_fetch_errors_explain_the_status_without_revealing_the_contact(monkeypatch, status):
    contact = "Test Name test@example.com"
    monkeypatch.setattr(settings, "sec_user_agent", contact)
    monkeypatch.setattr(ingest.httpx, "get", lambda url, **kwargs: httpx.Response(status))

    with pytest.raises(ingest.FetchError) as error:
        ingest.fetch_filing(NVIDIA_URL)
    assert str(status) in str(error.value)
    assert "test@example.com" not in str(error.value)


# --- Ingest -------------------------------------------------------------------


def _serve(monkeypatch, html: str) -> list[str]:
    """Answer fetches with this HTML, and record them so a test can count them."""
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return html

    monkeypatch.setattr(ingest, "fetch_filing", fake_fetch)
    return fetched


def test_ingest_stores_one_chunk_per_page_with_ids_and_metadata(monkeypatch, sec_collection):
    url = filing_url(ALPHA_ACCESSION)
    html = filing_html("ALPHA CORP", [f"first {WORDS}", "too short", f"third {WORDS}"])
    _serve(monkeypatch, html)

    result = ingest.ingest_filing(url, sec_collection)

    assert (result.pages_indexed, result.pages_skipped) == (2, 1)
    stored = sec_collection.get(include=["documents", "metadatas"])
    assert sorted(stored["ids"]) == [f"{ALPHA_ACCESSION}-p0001", f"{ALPHA_ACCESSION}-p0003"]

    page_one = stored["ids"].index(f"{ALPHA_ACCESSION}-p0001")
    assert "first" in stored["documents"][page_one]
    assert "title:" not in stored["documents"][page_one], "the prefix must not be stored"
    assert stored["metadatas"][page_one] == {
        "company": "ALPHA CORP",
        "ticker": "TICK",
        "form": "10-K",
        "period_end": "January 25, 2026",
        "fiscal_year": "2026",
        "doc_id": ALPHA_ACCESSION,
        "cik": "1234",
        "source_url": url,
        "page": 1,
        "part": 1,
    }


def test_ingest_fetches_and_stores_the_canonical_url(monkeypatch, sec_collection):
    fetched = _serve(monkeypatch, filing_html("ALPHA CORP", [WORDS, WORDS]))
    padded = f"https://www.sec.gov/Archives/edgar/data/0001234/{ALPHA_ACCESSION}/filing.htm"

    ingest.ingest_filing(padded, sec_collection)

    assert fetched == [filing_url(ALPHA_ACCESSION)]
    metadata = sec_collection.get(include=["metadatas"])["metadatas"][0]
    assert metadata["source_url"] == filing_url(ALPHA_ACCESSION)
    assert metadata["cik"] == "1234"


def test_ingest_embeds_documents_with_the_title_prefix(monkeypatch, sec_collection):
    embedded = []

    def recording_embedder(texts):
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    _serve(monkeypatch, filing_html("ALPHA CORP", [WORDS]))
    monkeypatch.setattr(store, "embed_texts", recording_embedder)

    ingest.ingest_filing(filing_url(ALPHA_ACCESSION), sec_collection)

    assert embedded[0].startswith("title: ALPHA CORP 10-K 2026 | text: ")


def test_split_pages_get_part_numbers_in_their_ids(monkeypatch, sec_collection):
    _serve(monkeypatch, filing_html("ALPHA CORP", ["word " * 1500, WORDS]))

    ingest.ingest_filing(filing_url(ALPHA_ACCESSION), sec_collection)

    assert sorted(sec_collection.get()["ids"]) == [
        f"{ALPHA_ACCESSION}-p0001-1",
        f"{ALPHA_ACCESSION}-p0001-2",
        f"{ALPHA_ACCESSION}-p0002",
    ]


def test_ingesting_twice_neither_duplicates_nor_fetches_again(monkeypatch, sec_collection):
    url = filing_url(ALPHA_ACCESSION)
    fetched = _serve(monkeypatch, filing_html("ALPHA CORP", [WORDS, WORDS]))

    ingest.ingest_filing(url, sec_collection)
    second = ingest.ingest_filing(url, sec_collection)

    assert second.already_indexed
    assert second.pages_indexed == 2
    assert len(fetched) == 1
    assert sec_collection.count() == 2


def test_a_failed_embed_stores_nothing_so_a_retry_really_retries(monkeypatch, sec_collection):
    """A partial filing would otherwise be reported as "Already indexed" forever."""
    url = filing_url(ALPHA_ACCESSION)
    _serve(monkeypatch, filing_html("ALPHA CORP", [WORDS] * 3))
    monkeypatch.setattr(settings, "embed_batch_size", 2)
    real_embedder = store.embed_texts
    calls = []

    def fails_on_second_batch(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise ConnectionError("Ollama went away")
        return real_embedder(texts)

    monkeypatch.setattr(store, "embed_texts", fails_on_second_batch)
    with pytest.raises(ConnectionError):
        ingest.ingest_filing(url, sec_collection)
    assert sec_collection.count() == 0

    monkeypatch.setattr(store, "embed_texts", real_embedder)
    assert not ingest.ingest_filing(url, sec_collection).already_indexed
    assert sec_collection.count() == 3


def test_ingest_reports_progress_per_batch(monkeypatch, sec_collection):
    _serve(monkeypatch, filing_html("ALPHA CORP", [WORDS] * 5))
    monkeypatch.setattr(settings, "embed_batch_size", 2)
    progress = []

    ingest.ingest_filing(
        filing_url(ALPHA_ACCESSION),
        sec_collection,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_the_command_line_indexes_every_sample_and_survives_a_failure(
    monkeypatch, sec_collection, capsys
):
    bad_url = ingest.SAMPLE_FILINGS[1].url

    def fake_fetch(url):
        if url == bad_url:
            raise ingest.FetchError("The SEC returned HTTP 500.")
        return filing_html("ALPHA CORP", [WORDS, WORDS])

    monkeypatch.setattr(ingest, "fetch_filing", fake_fetch)
    monkeypatch.setattr(store, "get_collection", lambda: sec_collection)
    monkeypatch.setattr("sys.argv", ["ingest", "--all"])

    assert ingest.main() == 1, "a failure must show in the exit code"
    assert "HTTP 500" in capsys.readouterr().err
    assert len(search.list_documents(sec_collection)) == len(ingest.SAMPLE_FILINGS) - 1


# --- Search -------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 10])
def test_search_returns_exactly_k_hits_best_first(two_filings, k):
    hits = search.search("export controls", k, two_filings)
    assert len(hits) == k
    scores = [hit.similarity for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_scores_are_cosine_similarity(monkeypatch, sec_collection):
    """A vector identical to the query is 1.0, an orthogonal one is 0.0."""
    metadata = {
        "company": "ALPHA CORP",
        "form": "10-K",
        "fiscal_year": "2026",
        "source_url": filing_url(ALPHA_ACCESSION),
        "page": 1,
        "part": 1,
        "doc_id": ALPHA_ACCESSION,
    }
    sec_collection.upsert(
        ids=["same", "orthogonal"],
        documents=["same", "orthogonal"],
        embeddings=[[3.0, 0.0], [0.0, 3.0]],
        metadatas=[metadata, metadata],
    )
    monkeypatch.setattr(store, "embed_texts", lambda texts: [[1.0, 0.0]])

    hits = search.search("anything", 2, sec_collection)

    assert [(hit.text, hit.similarity) for hit in hits] == [("same", 1.0), ("orthogonal", 0.0)]


def test_search_embeds_the_question_with_the_query_prefix(monkeypatch, two_filings):
    embedded = []
    real_embedder = store.embed_texts

    def recording_embedder(texts):
        embedded.extend(texts)
        return real_embedder(texts)

    monkeypatch.setattr(store, "embed_texts", recording_embedder)
    search.search("export controls", 1, two_filings)
    assert embedded == ["task: search result | query: export controls"]


def test_the_document_filter_is_applied_before_ranking(two_filings):
    """Still k results, all from the chosen filing, even though the other scores the same."""
    for accession, company in ((ALPHA_ACCESSION, "ALPHA CORP"), (BETA_ACCESSION, "BETA CORP")):
        hits = search.search("export controls", 10, two_filings, doc_id=accession)
        assert len(hits) == 10
        assert {hit.company for hit in hits} == {company}


def test_list_documents_groups_chunks_by_filing(two_filings):
    documents = search.list_documents(two_filings)
    assert [document.label for document in documents] == [
        "ALPHA CORP · 10-K · FY2026",
        "BETA CORP · 10-K · FY2026",
    ]
    assert [document.pages for document in documents] == [12, 12]
    assert documents[0].source_url == filing_url(ALPHA_ACCESSION)


def test_list_documents_counts_a_split_page_once(monkeypatch, sec_collection):
    _serve(monkeypatch, filing_html("ALPHA CORP", ["word " * 1500, WORDS]))
    ingest.ingest_filing(filing_url(ALPHA_ACCESSION), sec_collection)
    assert search.list_documents(sec_collection)[0].pages == 2
