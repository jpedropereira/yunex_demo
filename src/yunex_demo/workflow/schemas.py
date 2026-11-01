"""What a valid model answer looks like, for each LLM step of the workflow.

Ollama receives each class's JSON schema and constrains the model's output to
it. Pydantic then validates the answer anyway, because model output is untrusted.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, WithJsonSchema

# Pydantic's default Decimal schema allows a string, and given a string this model
# writes "USD 18.20". Declaring a number makes it write 18.2, which Pydantic still
# parses into a Decimal. max_digits rejects values like 1e999999, which would
# overflow the arithmetic in the validators.
Money = Annotated[Decimal, Field(max_digits=18), WithJsonSchema({"type": "number"})]

# The limit keeps a value inside SQLite's 64-bit integers. It is left out of the
# schema sent to Ollama, which only needs to know that this is a whole number.
WholeNumber = Annotated[int, Field(lt=10**15), WithJsonSchema({"type": "integer"})]

CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]

# These become database keys, and grounding ignores case and spacing. Without one
# spelling, "tc-1 " would pass grounding and then load again as a different trade from "TC-1".
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True)]

# Grounding ignores case and spacing here too, and the database key ignores case.
FundName = Annotated[str, StringConstraints(strip_whitespace=True)]


class Classification(BaseModel):
    # A closed set of labels: the model cannot invent a new type of document.
    doc_type: Literal["trade_confirmation", "capital_call", "factsheet", "other"]


class TradeConfirmation(BaseModel):
    reference: Identifier
    trade_date: date
    settlement_date: date
    side: Literal["buy", "sell"]
    security_name: str
    ticker: Identifier
    quantity: WholeNumber
    price: Money
    currency: CurrencyCode
    gross_amount: Money
    commission: Money
    total_amount: Money


class CapitalCall(BaseModel):
    fund_name: FundName
    call_number: WholeNumber
    notice_date: date
    due_date: date
    total_commitment: Money
    call_pct: Money
    amount_due: Money
    currency: CurrencyCode
    called_before: Money
    unfunded_after: Money
