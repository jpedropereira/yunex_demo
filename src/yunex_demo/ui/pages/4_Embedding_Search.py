"""Demo beat 4: search a real 10-K by meaning, with no LLM generating anything.

Deliberately thin: fetching, splitting, embedding and querying live in ``yunex_demo.sec``.
"""

from functools import partial

import streamlit as st

from yunex_demo.config import settings
from yunex_demo.sec import ingest, search, store

PREVIEW_WORDS = 60
EXAMPLE_URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"

# Worded differently from the filings, so they only work by meaning: "How many people work
# there?" has to find "42,000 employees". Chosen by reading the top 5 results on all five
# 10-Ks, and keeping those that find relevant pages at most of the companies.
SUGGESTED_QUESTIONS = [
    "What happens if the US limits sales to China?",
    "How many people work there?",
    "What if hackers break in?",
    "Can rivals steal their customers?",
    "Who are they being sued by?",
    "What legislation risks affect them?",
]
SUGGESTIONS_PER_ROW = 3

# Fixed text only: nothing from a filing ever goes through unsafe_allow_html.
PROJECTOR_CSS = """
<style>
[data-testid="stText"] { font-size: 1.2rem; }
[data-testid="stTextInput"] input { font-size: 1.2rem; }
</style>
"""

# --- 1. Page header -----------------------------------------------------------
st.set_page_config(
    page_title="Embedding search", page_icon=":material/scatter_plot:", layout="wide"
)
st.markdown(PROJECTOR_CSS, unsafe_allow_html=True)

st.title("Embedding search")
st.subheader("Embeddings turn text into vectors; similar meaning lands close together.")

try:
    collection = store.get_collection()
except ingest.EXPECTED_ERRORS as exc:
    st.error(f"Could not reach ChromaDB: {exc} Check the Stack Health page.")
    st.stop()

# --- 2. Ingest: one click for a company's latest 10-K, or paste a filing's URL -----
st.header("Ingest")

if not settings.sec_user_agent:
    st.warning(
        'Fetching a new filing needs SEC_USER_AGENT="Your Name you@example.com" in .env. '
        "Filings that are already indexed can still be searched."
    )

try:
    documents = search.list_documents(collection)
except ingest.EXPECTED_ERRORS as exc:
    st.error(
        "Could not read the search index. If an older version built it, rebuild it with "
        "`make reindex`. Otherwise, check the Stack Health page."
    )
    st.text(str(exc))
    st.stop()

pages_by_ticker = {document.ticker: document.pages for document in documents}
targets_to_index = []

# The buttons stay clickable, because a company may have filed since. Only a click asks
# the SEC: drawing this page must work offline.
*company_columns, index_all_column = st.columns(len(settings.agent_filings) + 1)
for column, agent_filing in zip(company_columns, settings.agent_filings, strict=True):
    pages = pages_by_ticker.get(agent_filing.ticker)
    with column:
        clicked = st.button(
            f"{agent_filing.company} · {pages} pages" if pages else agent_filing.company,
            key=f"latest_{agent_filing.ticker}",
            icon=":material/check:" if pages else ":material/download:",
            width="stretch",
        )
    if clicked:
        targets_to_index = [agent_filing.cik]

with index_all_column:
    if st.button("Index all", key="index_all", type="primary", width="stretch"):
        targets_to_index = [agent_filing.cik for agent_filing in settings.agent_filings]

pasted_url = st.text_input(
    "Or paste the URL of a 10-K from one of these companies",
    placeholder=EXAMPLE_URL,
    key="url",
)
if st.button("Index", icon=":material/download:", key="index"):
    targets_to_index = [pasted_url]

if targets_to_index:
    progress_bar = st.progress(0.0, text="Fetching the filing...")

    def show_progress(position: str, done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"{position}: embedded {done} of {total}")

    outcomes = []
    for number, target in enumerate(targets_to_index, start=1):
        position = f"Filing {number} of {len(targets_to_index)}"
        progress_bar.progress(0.0, text=f"{position}: fetching...")

        # A failed filing is reported, and the rest still get their turn.
        try:
            url = target if isinstance(target, str) else ingest.latest_10k_url(target)
            outcomes.append(
                ingest.ingest_filing(url, collection, on_progress=partial(show_progress, position))
            )
        except ingest.EXPECTED_ERRORS as exc:
            outcomes.append(f"Not indexed: {exc}")

    # Rerun so the buttons and table below show the new filings. Session state carries
    # the outcomes across that rerun, which would otherwise wipe them.
    st.session_state.ingest_outcomes = outcomes
    st.rerun()

for outcome in st.session_state.pop("ingest_outcomes", []):
    if isinstance(outcome, str):
        st.error(outcome)
    elif outcome.already_indexed:
        st.success(f"Already indexed: {outcome.pages_indexed} pages")
    else:
        replaced = " Replaced the older 10-K." if outcome.replaced_older else ""
        st.success(
            f"Indexed {outcome.pages_indexed} pages, skipped {outcome.pages_skipped}.{replaced}"
        )
        # Text from the filing, so never markdown.
        st.text(f"{outcome.company} · {outcome.form} · FY{outcome.fiscal_year}")

if documents:
    st.dataframe(
        [
            {
                "Company": document.company,
                "Form": document.form,
                "Year": document.fiscal_year,
                "Period end": document.period_end,
                "Pages": document.pages,
                "URL": document.source_url,
            }
            for document in documents
        ],
        hide_index=True,
    )

# --- 3. Search: the question goes in, the nearest pages come out --------------
st.header("Search")

if not documents:
    st.info("Nothing indexed yet. Index a filing above, or run `make index-all`.")
    st.stop()

question = st.text_input(
    "Question", placeholder="What happens if the US limits sales to China?", key="question"
)


def use_suggested_question(suggestion: str) -> None:
    # A callback, because Streamlit forbids changing a widget's value once it has been
    # drawn on this run. The question box fills in, so it is clear what was searched.
    st.session_state.question = suggestion


st.markdown("##### Or try one of these")
suggestion_columns = st.columns(SUGGESTIONS_PER_ROW)
for number, suggestion in enumerate(SUGGESTED_QUESTIONS):
    with suggestion_columns[number % SUGGESTIONS_PER_ROW]:
        st.button(
            suggestion,
            key=f"suggestion_{number}",
            on_click=use_suggested_question,
            args=(suggestion,),
            width="stretch",
        )

left, right = st.columns(2)
with left:
    k = st.slider("Results (k)", min_value=1, max_value=10, value=5, key="k")
with right:
    labels = {"": "All documents", **{document.doc_id: document.label for document in documents}}
    selected_doc_id = st.selectbox(
        "Document", options=list(labels), format_func=labels.get, key="document"
    )

if question:
    try:
        hits = search.search(question, k, collection, doc_id=selected_doc_id or None)
    except ingest.EXPECTED_ERRORS as exc:
        st.error(f"Search failed: {exc}. Check the Stack Health page.")
        st.stop()

    # --- 4. The results, straight from the database, best first ---------------
    for rank, hit in enumerate(hits, start=1):
        with st.container(border=True):
            st.markdown(f"#### {rank}. Similarity {hit.similarity:.2f}")
            # Cosine similarity can be negative, which st.progress rejects.
            st.progress(max(0.0, min(1.0, hit.similarity)))

            # Filing text is untrusted, so it goes through st.text, never markdown.
            st.text(f"{hit.company} · {hit.form} · FY{hit.fiscal_year} · {hit.page_label}")

            words = hit.text.split()
            preview = " ".join(words[:PREVIEW_WORDS])
            st.text(f"{preview}..." if len(words) > PREVIEW_WORDS else preview)
            with st.expander("Full text"):
                st.text(hit.text)

            st.link_button("Open the filing", hit.source_url, icon=":material/open_in_new:")
