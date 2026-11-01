"""A stand-in for the model, shared by the workflow and page tests.

It returns the correct answer for each sample PDF, so these tests need no Ollama,
while the validators still check those answers against the real PDF text.
"""

import json

import pytest

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


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(pipeline, "call_model", fake_call_model)
