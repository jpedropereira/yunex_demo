"""Demo beat 2: one document in, structured data out, and what can go wrong.

Deliberately thin: prompts, schema and the model call live in ``yunex_demo.extract``.

Upload a PDF (the examples in example_docs/one_off/ hide an instruction in
their text layer), pick a system prompt and a scenario, run the extraction, and
compare: does the model obey the document, and is the answer even valid JSON?
"""

import inspect

import httpx
import streamlit as st
from ollama import ResponseError
from pdfplumber.utils.exceptions import PdfminerException

from yunex_demo import extract
from yunex_demo.config import settings

# --- 1. Page header -----------------------------------------------------------
st.set_page_config(page_title="One-off", page_icon=":material/bolt:", layout="wide")

st.title("One-off extraction")
st.caption(f"`{settings.model_main}` reads a fund factsheet and returns its key facts as JSON.")

st.markdown(
    """
Three separate defences are on show here:

- **Against prompt injection:** fence the document off from your instructions,
  and tell the model (in the system prompt) that document text is data, never orders.
- **Against malformed output:** don't just *ask* for JSON. Have Ollama *enforce*
  the schema, then validate the result with Pydantic anyway.
- **Against wrong values:** check in plain code that each value actually
  appears in the document.
"""
)

# --- 2. Pick a document and show what the model will actually read ----------
uploaded = st.file_uploader("Upload a PDF", type="pdf", max_upload_size=extract.MAX_UPLOAD_MB)
if uploaded is None:
    st.info("Upload a PDF to start.", icon=":material/upload_file:")
    st.stop()

try:
    document_text = extract.pdf_to_text(uploaded)
# An upload with a .pdf name is not necessarily a valid PDF.
except PdfminerException:
    st.error("Could not read this file as a PDF.")
    st.stop()

if not document_text.strip():
    st.warning("This PDF has no text layer (is it a scan?), so there is nothing to extract.")
    st.stop()
if len(document_text) > extract.MAX_DOCUMENT_CHARS:
    st.error(
        f"This PDF has {len(document_text):,} characters of text, more than the "
        f"{extract.MAX_DOCUMENT_CHARS:,} that fit in the model's context window."
    )
    st.stop()

# The PDF viewer and the model see different things: the text layer can hold
# text a human never notices. Scroll to the end of this.
with st.expander("What the model reads (the PDF's text layer)"):
    st.code(document_text, language="text")

# --- 3. Choose the system prompt and the scenario ----------------------------
left, right = st.columns(2)

with left:
    system_label = st.radio("System prompt", list(extract.SYSTEM_PROMPTS))
    system_prompt = extract.SYSTEM_PROMPTS[system_label]
    st.code(system_prompt, language="text", wrap_lines=True)

with right:
    scenario_label = st.radio("Scenario", list(extract.SCENARIOS))
    scenario = extract.SCENARIOS[scenario_label]
    if scenario.enforce_schema:
        # Show the schema as it is written: a Pydantic class. The JSON Schema
        # Ollama receives is generated from it, and is much noisier to read.
        st.caption("Output is constrained to this schema by Ollama:")
        schema_source = "\n\n".join(
            inspect.getsource(model) for model in (extract.LargestHolding, extract.FundFacts)
        )
        st.code(schema_source, language="python")
        with st.expander("What Ollama actually receives (generated JSON Schema)"):
            st.json(extract.OLLAMA_SCHEMA)
    else:
        st.caption("No schema enforcement: the model is only *asked* for JSON.")

with st.expander("User prompt template"):
    st.code(scenario.template, language="text")

# --- 4. Run the extraction ------------------------------------------------------
if not st.button("Run extraction", icon=":material/play_arrow:", type="primary"):
    st.stop()

try:
    with st.spinner("Extracting..."):
        result = extract.extract(scenario, system_prompt, document_text)
# The ollama client turns "server unreachable" into a builtin ConnectionError.
except (ConnectionError, httpx.HTTPError, ResponseError) as exc:
    st.error(f"Model call failed: {exc}. Check the Stack Health page.")
    st.stop()

# --- 5. Show the raw answer, then whether it survived validation -------------
# Model output is untrusted, so it is shown as plain text: st.code never
# renders HTML.
st.subheader("Raw model output")
st.code(result.raw or "(empty)", language="json", wrap_lines=True)

st.subheader("Validation")
if result.facts is None:
    st.error("The output does not match the FundFacts schema.", icon=":material/close:")
    st.code(result.error, language="text")
else:
    st.success("The output matches the FundFacts schema.", icon=":material/check:")
    if result.facts.suspicious_content:
        st.warning("The model flagged instructions inside the document.", icon=":material/flag:")
    st.json(result.facts.model_dump())

# --- 6. Grounding: is every value actually in the document? ------------------
if result.grounding is not None:
    st.subheader("Grounding check")
    found_label = {True: "yes", False: "NO", None: "(null, not checked)"}
    rows = [
        {"field": field, "found in document": found_label[found]}
        for field, found in result.grounding.items()
    ]
    st.table(rows)

    missing = [field for field, found in result.grounding.items() if found is False]
    if missing:
        st.error(f"Not found in the document: {', '.join(missing)}", icon=":material/close:")
    else:
        st.success("Every extracted value appears in the document.", icon=":material/check:")

# --- 7. What each check does and does not prove ------------------------------
st.caption(
    "A valid result has the right *shape*. A grounded result uses values that "
    "appear in the document. Neither proves a value is the *right* one: an "
    "injected phrase is in the document too, and a small number like 1 appears "
    "almost everywhere."
)
