"""Stand-ins for the models, shared by the workflow, SEC and page tests.

The fake model returns the correct answer for each sample PDF, so these tests need no
Ollama, while the validators still check those answers against the real PDF text. The
SEC stand-ins need no Ollama, Chroma or network either.
"""

import hashlib
import json
import uuid

import chromadb
import pytest

from yunex_demo.agent import loop, quotes
from yunex_demo.sec import ingest, store
from yunex_demo.workflow import pipeline
from yunex_demo.workflow.schemas import Classification

SOLARIS = {
    "reference": "TC-2026-10418",
    "trade_date": "2026-10-14",
    "settlement_date": "2026-10-16",
    "side": "buy",
    "security_name": "Solaris Parks Ltd",
    "ticker": "SOLR",
    "quantity": 200,
    "price": "42.50",
    "currency": "USD",
    "gross_amount": "8500.00",
    "commission": "12.75",
    "total_amount": "8512.75",
}

HELIOS = {
    "reference": "TC-2026-10422",
    "trade_date": "2026-10-15",
    "settlement_date": "2026-10-19",
    "side": "buy",
    "security_name": "Helios Storage Ltd",
    "ticker": "HELS",
    "quantity": 300,
    "price": "18.20",
    "currency": "USD",
    "gross_amount": "5460.00",
    "commission": "8.19",
    "total_amount": "5460.00",
}

ACME_CALL = {
    "fund_name": "ACME Green Infrastructure Fund I",
    "call_number": 7,
    "notice_date": "2026-10-20",
    "due_date": "2026-11-30",
    "total_commitment": "2000000.00",
    "call_pct": "5.0",
    "amount_due": "100000.00",
    "currency": "USD",
    "called_before": "800000.00",
    "unfunded_after": "1100000.00",
}


def fake_call_model(prompt: str, schema: type) -> str:
    """Answer as the model should, recognising each sample by text only it contains."""
    if "TC-2026-10418" in prompt:
        doc_type, answer = "trade_confirmation", SOLARIS
    elif "TC-2026-10422" in prompt:
        doc_type, answer = "trade_confirmation", HELIOS
    elif "Capital Call Notice" in prompt:
        doc_type, answer = "capital_call", ACME_CALL
    else:
        doc_type, answer = "other", {}

    if schema is Classification:
        return json.dumps({"doc_type": doc_type})
    return json.dumps(answer)


@pytest.fixture(autouse=True)
def no_prompt_rendering(monkeypatch):
    """The agent asks Ollama for each call's final prompt, which no test may really do."""
    monkeypatch.setattr(loop, "render_prompt", lambda request: "")


def quote_the_whole_page(prompt: str) -> str:
    """The relevance step at its most generous, so a test can script just the agent."""
    page = prompt.split("<document>\n")[1].split("\n</document>")[0]
    return json.dumps({"quotes": [page]})


@pytest.fixture(autouse=True)
def no_relevance_model(monkeypatch):
    monkeypatch.setattr(quotes, "ask_model", quote_the_whole_page)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(pipeline, "call_model", fake_call_model)


# --- SEC filings ---

ALPHA_ACCESSION = "000000000000000001"
BETA_ACCESSION = "000000000000000002"


def fake_embed_texts(texts: list[str]) -> list[list[float]]:
    """Count hashed words into a vector: repeatable, and shared words score higher."""
    vectors = []
    for text in texts:
        vector = [1.0] + [0.0] * 63
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode()).digest()
            vector[digest[0] % 64] += 1.0
        vectors.append(vector)
    return vectors


def filing_html(company: str, pages: list[str]) -> str:
    """A small inline XBRL filing: a hidden header, cover-page facts, and page breaks."""
    body = '<hr style="page-break-after:always"/>'.join(f"<div>{page}</div>" for page in pages)
    return f"""<html><body>
    <div style="display:none"><ix:header><ix:hidden>
      <ix:nonNumeric name="dei:DocumentFiscalYearFocus">2026</ix:nonNumeric>
    </ix:hidden></ix:header></div>
    <ix:nonNumeric name="dei:EntityRegistrantName"><span>{company}</span></ix:nonNumeric>
    <ix:nonNumeric name="dei:TradingSymbol">TICK</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentType">10-K</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentPeriodEndDate">January 25, 2026</ix:nonNumeric>
    {body}
    </body></html>"""


NVIDIA_CIK = 1045810
APPLE_CIK = 320193


def filing_url(accession: str, cik: int = NVIDIA_CIK) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/filing.htm"


def fake_latest_10k_url(cik: int) -> str:
    """What the SEC would answer: a different filing for each company."""
    return filing_url(f"{cik:018d}", cik)


@pytest.fixture
def sec_collection(monkeypatch):
    """A fake embedder and an empty collection. Ephemeral clients share state, so unique name."""
    monkeypatch.setattr(store, "embed_texts", fake_embed_texts)
    client = chromadb.EphemeralClient()
    name = f"test-{uuid.uuid4().hex[:8]}"
    yield client.create_collection(
        name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    client.delete_collection(name)


@pytest.fixture
def two_filings(monkeypatch, sec_collection):
    """Two companies indexed, with twelve pages each, so k=10 and the filter can be tested."""
    filler = "revenue customers products markets suppliers employees " * 6
    pages = [f"Page {number} export controls sales China {filler}" for number in range(1, 13)]
    html_by_url = {
        filing_url(ALPHA_ACCESSION): filing_html("ALPHA CORP", pages),
        filing_url(BETA_ACCESSION, APPLE_CIK): filing_html("BETA CORP", pages),
    }
    monkeypatch.setattr(ingest, "fetch_filing", html_by_url.__getitem__)
    for url in html_by_url:
        ingest.ingest_filing(url, sec_collection)
    return sec_collection
