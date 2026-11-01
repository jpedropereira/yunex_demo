"""Demo beat 3: a document inbox, where the model fills in steps and the code decides.

Deliberately thin: the steps, rules and database live in ``yunex_demo.workflow``, and
the flowchart and stage text in ``yunex_demo.workflow.explain``.

Each PDF takes the same fixed path: classify, route, extract, validate, load. The
flowchart shows that path before anything runs, then lights up the current stage.
"""

import time
from datetime import date
from decimal import Decimal

import httpx
import streamlit as st
from ollama import ResponseError

from yunex_demo import extract
from yunex_demo.config import settings
from yunex_demo.workflow import db, explain, pipeline
from yunex_demo.workflow.schemas import Classification

# Code steps take milliseconds, so on stage they would flash past unseen.
CODE_STEP_PAUSE_SECONDS = 1

ACTORS = {stage.name: stage.actor for stage in explain.STAGES}

# --- 1. Page header and the flowchart -----------------------------------------
st.set_page_config(page_title="Workflow", page_icon=":material/account_tree:", layout="wide")

st.title("Workflow")
st.subheader("The model fills in steps; the code decides.")
st.caption(
    f"Ellipses are LLM steps, run by `{settings.model_main}`. Boxes are plain Python. "
    "Amber is running, green passed, red failed, grey skipped."
)

# Filled in further down, once we know which document's path to show. The live panel
# sits right below the flowchart, so both stay on screen during a run.
chart_slot = st.empty()
live_slot = st.empty()

# One database and one list of results per browser session. Results are kept here
# because any widget interaction reruns this script, and must not re-call the model.
if "workflow_db" not in st.session_state:
    st.session_state.workflow_db = db.connect()
    st.session_state.workflow_results = []
conn = st.session_state.workflow_db

# --- 2. Controls --------------------------------------------------------------
left, right = st.columns([1, 2])

with left:
    run_samples = st.button(
        "Load sample inbox", icon=":material/play_arrow:", type="primary", key="run_samples"
    )
    slow_down = st.toggle(
        "Slow down for presenting",
        value=True,
        key="slow_down",
        help=f"Holds each code step for {CODE_STEP_PAUSE_SECONDS}s so it can be seen.",
    )
    reset = st.button("Reset", icon=":material/restart_alt:", key="reset")

with right:
    uploads = st.file_uploader(
        "Or run your own PDFs",
        type="pdf",
        accept_multiple_files=True,
        max_upload_size=extract.MAX_UPLOAD_MB,
    )
    run_uploads = st.button("Run uploaded PDFs", key="run_uploads", disabled=not uploads)

if reset:
    db.seed(conn)
    st.session_state.workflow_results = []


def show_stage_columns(result: pipeline.DocumentResult, stage_name: str) -> None:
    """Went in / who decides / came out. All untrusted text, so never markdown."""
    panel = explain.describe_stage(result, stage_name)
    went_in, who_decides, came_out = st.columns(3)
    with went_in:
        st.markdown("##### Went in")
        st.text(panel.went_in)
    with who_decides:
        st.markdown(f"##### Who decides: {ACTORS[stage_name]}")
        st.text(panel.who_decides)
    with came_out:
        st.markdown("##### Came out")
        # st.code never renders HTML, and keeps JSON and the list of checks aligned.
        st.code(panel.came_out, language=None, wrap_lines=True)


# --- 3. Run the pipeline, redrawing the flowchart as each stage starts --------
documents = []
if run_samples:
    documents = [(name, pipeline.SAMPLE_INBOX_DIR / name) for name in pipeline.SAMPLE_FILES]
elif run_uploads:
    documents = [(upload.name, upload) for upload in uploads]

if documents:
    st.session_state.workflow_results = []

    def show_progress(result: pipeline.DocumentResult, position: str) -> None:
        chart_slot.graphviz_chart(explain.pipeline_dot(result), width="stretch")
        running = [name for name, step in result.steps().items() if step.status == "running"]
        is_code_step = bool(running) and ACTORS[running[0]] == "Code"
        holding = slow_down and is_code_step

        with live_slot.container():
            st.markdown(f"#### {position}")
            st.text(result.file_name)
            if running:
                st.markdown(f"##### Now running: {running[0]}")
                show_stage_columns(result, running[0])
            if holding:
                st.caption(
                    f"Held for {CODE_STEP_PAUSE_SECONDS}s so you can see it; "
                    "this step really takes milliseconds."
                )
        if holding:
            time.sleep(CODE_STEP_PAUSE_SECONDS)

    try:
        with st.spinner("Running the pipeline..."):
            for number, (file_name, pdf_file) in enumerate(documents, start=1):
                position = f"Document {number} of {len(documents)}"
                result = pipeline.process_document(
                    file_name,
                    pdf_file,
                    conn,
                    on_stage_start=lambda result, position=position: show_progress(
                        result, position
                    ),
                )
                st.session_state.workflow_results.append(result)

                # Let the audience see where this document ended up before the next starts.
                show_progress(result, position)
                if slow_down:
                    time.sleep(CODE_STEP_PAUSE_SECONDS)
    # The ollama client turns "server unreachable" into a builtin ConnectionError.
    except (ConnectionError, httpx.HTTPError, ResponseError) as exc:
        st.error(f"Model call failed: {exc}. Check the Stack Health page.")
    live_slot.empty()

# --- 4. Compare the documents: one row each, one column per stage -------------
results = st.session_state.workflow_results
selected = None

# Open before a run, to talk through the design. Afterwards it folds away to leave
# room for the results, and each stage's tab shows its own explanation.
with st.expander("What each step does", expanded=not results):
    for number, stage in enumerate(explain.STAGES, start=1):
        st.markdown(f"**{number}. {stage.name} ({stage.actor}).** {stage.explanation}")

if results:
    st.subheader("Results")
    st.caption("Select a row to see that document's path above and its stages below.")
    # st.dataframe keeps file names, which are untrusted, away from markdown.
    grid = st.dataframe(
        explain.results_grid(results),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="workflow_grid",
    )
    selected_rows = [row for row in grid.selection.rows if row < len(results)]
    # With nothing selected, show the last document: in the sample inbox, the failure.
    selected = results[selected_rows[0]] if selected_rows else results[-1]

chart_slot.graphviz_chart(explain.pipeline_dot(selected), width="stretch")

# --- 5. How one document went through each stage ------------------------------
if selected:
    st.text(f"Selected: {selected.file_name}")
    if selected.review_reason:
        st.text(f"Needs review: {selected.review_reason}")

    steps = selected.steps()
    labels = [f"{explain.STATUS_ICONS[step.status]} {name}" for name, step in steps.items()]
    # Open on the stage that stopped the document, if one did.
    failed = [
        label for label, step in zip(labels, steps.values(), strict=True) if step.status == "failed"
    ]
    tabs = st.tabs(labels, default=failed[0] if failed else None)

    for tab, stage_name in zip(tabs, steps, strict=True):
        with tab:
            stage = next(stage for stage in explain.STAGES if stage.name == stage_name)
            st.markdown(f"**{stage.question}** {stage.explanation}")
            show_stage_columns(selected, stage_name)

            # The LLM stages' real input, for anyone who wants to read it.
            prompt = {
                "Classify": selected.classify_prompt,
                "Extract": selected.extract_prompt,
            }.get(stage_name)
            if prompt:
                with st.expander("The exact prompt and schema"):
                    st.code(prompt, language="text", wrap_lines=True)
                    if stage_name == "Classify":
                        schema = Classification
                    else:
                        schema = pipeline.ROUTES[selected.doc_type].schema
                    st.json(schema.model_json_schema())

# --- 6. The raw output of every step, for every document ----------------------
if results:
    st.subheader("Raw outputs")
    st.caption(
        "Exactly what each step produced, before anything was shown or summarised: the "
        "model's unparsed text for LLM steps, the return value as JSON for code steps."
    )
    for result in results:
        with st.container(border=True):
            st.text(result.file_name)
            steps = result.steps()
            # Tab labels render markdown, so these hold only fixed stage names.
            labels = [f"{explain.STATUS_ICONS[step.status]} {name}" for name, step in steps.items()]
            for tab, name in zip(st.tabs(labels), steps, strict=True):
                with tab:
                    # st.code shows model output as plain text and never renders HTML.
                    raw = steps[name].raw or "(this step did not run)"
                    st.code(raw, language="json", wrap_lines=True)

# --- 7. What the workflow changed ---------------------------------------------
st.divider()

st.subheader("Positions")
st.dataframe(
    [
        {
            "Portfolio": position["portfolio"],
            "Ticker": position["ticker"],
            "Security": position["security_name"],
            "Before → after": f"{position['before']:,} → {position['after']:,}",
        }
        for position in db.read_positions(conn)
    ],
    hide_index=True,
)

st.subheader("Cash calls")
cash_calls = db.read_cash_calls(conn)
if cash_calls:
    rows = []
    for call in cash_calls:
        due_date = date.fromisoformat(call["due_date"])
        rows.append(
            {
                "Fund": call["fund_name"],
                "Call": call["call_number"],
                "Amount due": f"{call['currency']} {Decimal(call['amount_due']):,.2f}",
                "Due": f"{due_date.day} {due_date:%b %Y}",
                "Status": call["status"],
            }
        )
    st.dataframe(rows, hide_index=True)
else:
    st.text("None yet.")

st.subheader("Needs review")
review_queue = db.read_review_queue(conn)
if review_queue:
    st.dataframe(
        [
            {"File": row["file_name"], "Type": row["doc_type"], "Reason": row["reason"]}
            for row in review_queue
        ],
        hide_index=True,
    )
else:
    st.text("Nothing to review.")
