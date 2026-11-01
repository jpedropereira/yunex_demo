"""Turn a pipeline run into something the audience can see.

Draws the flowchart and writes the "went in / who decides / came out" text for each
stage. It only reads a DocumentResult, and never changes what the pipeline does.

The flowchart is built only from fixed stage names and validated statuses. File names
and model output never go into it, so a document can't inject anything into the DOT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from yunex_demo.workflow.pipeline import ROUTES, DocumentResult, Step

STATUS_ICONS = {"running": "⏳", "passed": "✅", "failed": "❌", "skipped": "⏭️"}

# Explicit text colour on every fill, so nodes stay readable in Streamlit's dark theme.
_TEXT = "#111827"
_IDLE_FILL = "#f1f5f9"
_STATUS_FILLS = {
    "running": "#fcd34d",
    "passed": "#86efac",
    "failed": "#fca5a5",
    "skipped": "#d1d5db",
}
_ACTOR_BORDERS = {"LLM": "#7c3aed", "Code": "#2563eb"}
_IDLE_EDGE = "#94a3b8"
_SUCCESS_EDGE = "#16a34a"
_FAILURE_EDGE = "#dc2626"


@dataclass(frozen=True)
class Stage:
    name: str
    actor: str  # "LLM" or "Code"
    question: str
    explanation: str  # written to be read aloud; keep it in step with the code it describes


STAGES = [
    Stage(
        "Classify",
        "LLM",
        "What kind of document is this?",
        "The model reads the document and answers one narrow question: what kind of document "
        "is this? The schema limits its answer to four labels, so it can't invent a new one, "
        "and Pydantic checks the answer again. The document sits inside tags it can't close, "
        "and the model is told that everything inside them is data, not instructions.",
    ),
    Stage(
        "Route",
        "Code",
        "Which path does this type take?",
        "Plain Python looks the label up in a table. Trade confirmations and capital calls each "
        "have their own schema, rules and loader. Factsheets, and anything else without an "
        "entry, go straight to Needs review. The model's label is only an input to this lookup: "
        "it can't choose a step that isn't in the table, or skip one that is.",
    ),
    Stage(
        "Extract",
        "LLM",
        "What are this type's fields?",
        "The model reads the document again and fills in the fields of the schema Route chose, "
        "using only values stated in the document. Pydantic then parses the answer, turning "
        "money into exact decimals and dates into real dates. An invalid answer gets one retry "
        "with the error attached, and a second failure goes to Needs review. Passing here only "
        "proves the answer has the right shape, not that the values are right.",
    ),
    Stage(
        "Validate",
        "Code",
        "Do the values add up, and are they in the document?",
        "No model here, just rules. The amounts must add up to within one cent, and the dates "
        "must be in the right order. Grounding then confirms that every value appears in the "
        "document, which catches an invented number that happens to add up; it proves a value "
        "is in the document, not that it is the right one. Every check runs and is reported, "
        "and any failure sends the document to Needs review with the reason. This is where "
        "Helios stops: its total leaves out the commission, even though every value was copied "
        "correctly from the document.",
    ),
    Stage(
        "Load",
        "Code",
        "May this change the database?",
        "Only a document that passed every check gets here, and only this code writes to the "
        "database. A buy adds to the position and a sell subtracts from it, while a sell larger "
        "than the holding goes to Needs review. A capital call becomes a pending cash call. "
        "Each trade reference and call number is loaded only once, so running the inbox again "
        "changes nothing.",
    ),
]


# --------------------------------------------------------------------------- #
# The flowchart
# --------------------------------------------------------------------------- #


def _stage_node(number: int, stage: Stage, status: str | None) -> str:
    """LLM stages are ellipses and code stages are boxes, whatever their status."""
    shape = "ellipse" if stage.actor == "LLM" else "box"
    fill = _STATUS_FILLS[status] if status else _IDLE_FILL
    border_width = 5 if status == "running" else 2
    return (
        f'{stage.name} [label="{number}. {stage.name}\\n{stage.actor}" shape={shape} '
        f'fillcolor="{fill}" color="{_ACTOR_BORDERS[stage.actor]}" penwidth={border_width}]'
    )


def _edge(source: str, target: str, colour: str | None, extras: str = "") -> str:
    """colour is set for an edge the document took; None draws it as a quiet path."""
    if colour:
        return f'{source} -> {target} [color="{colour}" penwidth=4 {extras}]'
    return f'{source} -> {target} [color="{_IDLE_EDGE}" penwidth=1.5 {extras}]'


def pipeline_dot(result: DocumentResult | None = None) -> str:
    """Draw the pipeline as DOT. With a result, also show where that document went."""
    steps = result.steps() if result else {}
    statuses = {name: step.status for name, step in steps.items()}
    loaded = statuses.get("Load") == "passed"
    in_review = result is not None and result.review_reason is not None

    database_fill = _STATUS_FILLS["passed"] if loaded else _IDLE_FILL
    review_fill = _STATUS_FILLS["failed"] if in_review else _IDLE_FILL

    lines = [
        "digraph pipeline {",
        # Streamlit's CSS forces chart text to 9.6px, whatever fontsize says, then scales
        # the whole chart to the page width. So fontsize only sets the size of the nodes:
        # a value near 9.6 keeps them compact, which is what makes the text large on screen.
        'rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.25;',
        f'node [style=filled fontname="Helvetica" fontsize=10 fontcolor="{_TEXT}" '
        'margin="0.08,0.05"];',
        f'PDF [label="PDF" shape=note fillcolor="{_IDLE_FILL}"];',
        f'Database [shape=cylinder fillcolor="{database_fill}"];',
        f'Review [label="Needs review" shape=octagon fillcolor="{review_fill}"];',
    ]

    for number, stage in enumerate(STAGES, start=1):
        lines.append(_stage_node(number, stage, statuses.get(stage.name)))

    # The happy path, left to right. A stage that passed always hands on to the next.
    reached_classify = statuses.get("Classify", "skipped") != "skipped"
    lines.append(_edge("PDF", "Classify", _SUCCESS_EDGE if reached_classify else None))
    for current, following in zip(STAGES, STAGES[1:], strict=False):
        taken = statuses.get(current.name) == "passed"
        lines.append(_edge(current.name, following.name, _SUCCESS_EDGE if taken else None))
    lines.append(_edge("Load", "Database", _SUCCESS_EDGE if loaded else None))

    # Failure paths don't set the layout (constraint=false): Needs review sits below.
    failure_style = "style=dashed constraint=false"
    # Only drawn when taken: an unreadable file is rare, and the chart has enough arrows.
    if in_review and not reached_classify:
        lines.append(_edge("PDF", "Review", _FAILURE_EDGE, failure_style))
    for stage in STAGES:
        failed = statuses.get(stage.name) == "failed"
        colour = _FAILURE_EDGE if failed else None
        lines.append(_edge(stage.name, "Review", colour, failure_style))

    # Same column as Extract, the middle stage, so the failure arrows stay short.
    # The invisible edge is what puts Needs review below the row rather than above it.
    lines.append("{rank=same; Extract; Review}")
    lines.append("Extract -> Review [style=invis]")
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The stage panels
# --------------------------------------------------------------------------- #


@dataclass
class StagePanel:
    went_in: str
    who_decides: str
    came_out: str


def _outcome_text(step: Step, finished_text: str) -> str:
    if step.status == "running":
        return "Working..."
    if step.status == "skipped" and not step.detail:
        return "Not reached."
    return finished_text


def _record_name(result: DocumentResult) -> str:
    """Which record reached the Load stage: a trade reference, or a fund's call number."""
    extracted = result.extracted or {}
    if "reference" in extracted:
        return f"Trade {extracted['reference']}"
    if "call_number" in extracted:
        return f"{extracted['fund_name']}, call no. {extracted['call_number']}"
    return "Nothing: the document did not get this far."


def describe_stage(result: DocumentResult, stage_name: str) -> StagePanel:
    """Say what went into a stage, who decided, and what came out, with real values."""
    text_size = f"{len(result.document_text):,} characters of document text"

    if stage_name == "Classify":
        step = result.classify
        return StagePanel(
            went_in=text_size,
            who_decides="The LLM, but the schema limits its answer to a closed set of 4 labels.",
            came_out=_outcome_text(step, step.detail),
        )

    if stage_name == "Route":
        step = result.route
        return StagePanel(
            went_in=f"doc_type = {result.doc_type}",
            who_decides=(
                f"Code: a lookup table. Supported types: {', '.join(ROUTES)}. "
                "Anything else goes to review."
            ),
            came_out=_outcome_text(step, step.detail),
        )

    if stage_name == "Extract":
        step = result.extract
        extracted = json.dumps(result.extracted, indent=2) if result.extracted else step.detail
        return StagePanel(
            went_in=f"{text_size}, and the schema chosen by Route",
            who_decides=(
                "The LLM fills in the values. The schema enforces the shape, Pydantic "
                "validates it, and an invalid answer gets one retry."
            ),
            came_out=_outcome_text(step, extracted),
        )

    if stage_name == "Validate":
        step = result.validate
        checks = "\n".join(
            f"{'✅' if check.passed else '❌'} {check.name}: {check.detail}"
            for check in result.checks
        )
        return StagePanel(
            went_in="The extracted values, and the original document text",
            who_decides="Code: arithmetic, date order, and grounding in the document.",
            came_out=_outcome_text(step, checks),
        )

    step = result.load
    return StagePanel(
        went_in=_record_name(result),
        who_decides="Code: only if every check passed, and only once per document.",
        came_out=_outcome_text(step, step.detail),
    )


def results_grid(results: list[DocumentResult]) -> list[dict[str, str]]:
    """One row per document and one column per stage, so the paths can be compared."""
    rows = []
    for result in results:
        if result.review_reason:
            outcome = "Needs review"
        elif result.load.status == "passed":
            outcome = "Loaded"
        else:
            outcome = "Already loaded"

        row = {"File": result.file_name, "Type": result.doc_type}
        for name, step in result.steps().items():
            row[name] = STATUS_ICONS[step.status]
        row["Outcome"] = outcome
        rows.append(row)
    return rows
