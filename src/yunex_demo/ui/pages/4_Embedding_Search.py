"""Demo beat 4: search a real 10-K by meaning, with no LLM generating anything.

Deliberately thin: fetching, splitting, embedding and querying live in ``yunex_demo.sec``.
"""

from functools import partial

import streamlit as st

from yunex_demo.config import settings
from yunex_demo.sec import filing, ingest, search, store

PREVIEW_WORDS = 60

# Worded differently from the filings, so they only work by meaning: "How many people work
# there?" has to find "42,000 employees". Chosen by reading the top 5 results on all five
# sample 10-Ks, and keeping those that find relevant pages at most of the companies.
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

# --- 2. Ingest: one click for a sample filing, or paste any EDGAR URL ----------
st.header("Ingest")

if not settings.sec_user_agent:
    st.warning(
        'Fetching a new filing needs SEC_USER_AGENT="Your Name you@example.com" in .env. '
        "Filings that are already indexed can still be searched."
    )

documents = search.list_documents(collection)
pages_by_doc_id = {document.doc_id: document.pages for document in documents}
urls_to_index = []
missing_urls = []

*sample_columns, index_all_column = st.columns(len(ingest.SAMPLE_FILINGS) + 1)
for column, sample in zip(sample_columns, ingest.SAMPLE_FILINGS, strict=True):
    _, accession = filing.parse_edgar_url(sample.url)
    pages = pages_by_doc_id.get(accession)
    if not pages:
        missing_urls.append(sample.url)
    with column:
        clicked = st.button(
            f"{sample.name} · {pages} pages" if pages else sample.name,
            key=f"sample_{sample.name}",
            icon=":material/check:" if pages else ":material/download:",
            disabled=bool(pages),
            width="stretch",
        )
    if clicked:
        urls_to_index = [sample.url]

with index_all_column:
    if st.button(
        "Index all",
        key="index_all",
        type="primary",
        disabled=not missing_urls,
        width="stretch",
    ):
        urls_to_index = missing_urls

pasted_url = st.text_input(
    "Or paste another EDGAR document URL",
    placeholder=ingest.SAMPLE_FILINGS[0].url,
    key="url",
)
if st.button("Index", icon=":material/download:", key="index"):
    urls_to_index = [pasted_url]

if urls_to_index:
    progress_bar = st.progress(0.0, text="Fetching the filing...")

    def show_progress(position: str, done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"{position}: embedded {done} of {total}")

    outcomes = []
    for number, url in enumerate(urls_to_index, start=1):
        position = f"Filing {number} of {len(urls_to_index)}"
        progress_bar.progress(0.0, text=f"{position}: fetching...")

        # A failed filing is reported, and the rest still get their turn.
        try:
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
        st.success(f"Indexed {outcome.pages_indexed} pages, skipped {outcome.pages_skipped}.")
        # Text from the filing, so never markdown.
        st.text(f"{outcome.company} · {outcome.form} · FY{outcome.fiscal_year}")

if documents:
    st.dataframe(
        [
            {
                "Company": document.company,
                "Form": document.form,
                "Year": document.fiscal_year,
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

            page_label = f"page {hit.page}"
            if hit.part > 1:
                page_label += f", part {hit.part}"
            # Filing text is untrusted, so it goes through st.text, never markdown.
            st.text(f"{hit.company} · {hit.form} · FY{hit.fiscal_year} · {page_label}")

            words = hit.text.split()
            preview = " ".join(words[:PREVIEW_WORDS])
            st.text(f"{preview}..." if len(words) > PREVIEW_WORDS else preview)
            with st.expander("Full text"):
                st.text(hit.text)

            st.link_button("Open the filing", hit.source_url, icon=":material/open_in_new:")
