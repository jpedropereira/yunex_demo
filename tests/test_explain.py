"""Tests for what the audience sees: the flowchart and the stage panels."""

from yunex_demo.workflow import db, explain, pipeline
from yunex_demo.workflow.pipeline import DocumentResult, Step

SOLARIS_PDF = pipeline.SAMPLE_INBOX_DIR / "trade_confirmation_solaris.pdf"
HELIOS_PDF = pipeline.SAMPLE_INBOX_DIR / "trade_confirmation_helios.pdf"

SUCCESS_EDGE_TO_DATABASE = 'Load -> Database [color="#16a34a"'
FAILURE_EDGE_FROM_VALIDATE = 'Validate -> Review [color="#dc2626"'


def _node_line(dot: str, name: str) -> str:
    return next(line for line in dot.splitlines() if line.startswith(f"{name} ["))


def test_the_idle_chart_shows_every_stage_and_who_does_it():
    dot = explain.pipeline_dot()
    assert '1. Classify\\nLLM" shape=ellipse' in dot
    assert '2. Route\\nCode" shape=box' in dot
    assert '3. Extract\\nLLM" shape=ellipse' in dot
    assert '4. Validate\\nCode" shape=box' in dot
    assert '5. Load\\nCode" shape=box' in dot


def test_the_idle_chart_takes_no_path():
    dot = explain.pipeline_dot()
    assert 'color="#16a34a"' not in dot
    assert 'color="#dc2626"' not in dot


def test_the_running_stage_is_highlighted():
    result = DocumentResult("doc.pdf", classify=Step("passed"), route=Step("running"))
    dot = explain.pipeline_dot(result)
    assert 'fillcolor="#fcd34d"' in _node_line(dot, "Route")
    assert 'fillcolor="#86efac"' in _node_line(dot, "Classify")
    assert 'fillcolor="#fcd34d"' not in _node_line(dot, "Extract")
    assert 'Classify -> Route [color="#16a34a"' in dot


def test_a_loaded_document_ends_at_the_database(fake_model):
    result = pipeline.process_document("solaris.pdf", SOLARIS_PDF, db.connect())
    dot = explain.pipeline_dot(result)
    assert SUCCESS_EDGE_TO_DATABASE in dot
    assert FAILURE_EDGE_FROM_VALIDATE not in dot


def test_a_failed_document_ends_in_review_from_the_stage_that_stopped_it(fake_model):
    result = pipeline.process_document("helios.pdf", HELIOS_PDF, db.connect())
    dot = explain.pipeline_dot(result)
    assert 'fillcolor="#fca5a5"' in _node_line(dot, "Validate")
    assert 'fillcolor="#fca5a5"' in _node_line(dot, "Review")
    assert FAILURE_EDGE_FROM_VALIDATE in dot
    assert SUCCESS_EDGE_TO_DATABASE not in dot


def test_an_unreadable_file_goes_straight_from_the_pdf_to_review():
    result = DocumentResult("scan.pdf", review_reason="no text layer")
    assert 'PDF -> Review [color="#dc2626"' in explain.pipeline_dot(result)


def test_untrusted_text_never_reaches_the_chart():
    """A file name or model value in the DOT could rewrite the diagram."""
    hostile = '"]; evil [label="pwned'
    result = DocumentResult(
        hostile,
        classify=Step("passed", hostile),
        extract=Step("failed", hostile),
        extracted={"reference": hostile},
        review_reason=hostile,
        document_text=hostile,
    )
    assert "pwned" not in explain.pipeline_dot(result)


def test_the_validate_panel_gives_the_helios_failure(fake_model):
    result = pipeline.process_document("helios.pdf", HELIOS_PDF, db.connect())
    panel = explain.describe_stage(result, "Validate")
    assert panel.who_decides.startswith("Code")
    assert "❌ gross + commission = total" in panel.came_out
    assert "5,468.19" in panel.came_out
    assert "✅ Grounding" in panel.came_out


def test_stages_after_the_failure_say_they_were_not_reached(fake_model):
    result = pipeline.process_document("helios.pdf", HELIOS_PDF, db.connect())
    assert explain.describe_stage(result, "Load").came_out == "Not reached."


def test_the_extract_panel_shows_the_json_and_who_decided(fake_model):
    result = pipeline.process_document("solaris.pdf", SOLARIS_PDF, db.connect())
    panel = explain.describe_stage(result, "Extract")
    assert panel.who_decides.startswith("The LLM")
    assert '"reference": "TC-2026-10418"' in panel.came_out


def test_a_running_stage_says_it_is_working():
    result = DocumentResult("doc.pdf", classify=Step("running"))
    assert explain.describe_stage(result, "Classify").came_out == "Working..."


def test_the_grid_gives_each_document_its_path_and_outcome(fake_model):
    conn = db.connect()
    solaris = pipeline.process_document("solaris.pdf", SOLARIS_PDF, conn)
    helios = pipeline.process_document("helios.pdf", HELIOS_PDF, conn)
    solaris_again = pipeline.process_document("solaris.pdf", SOLARIS_PDF, conn)

    first, second, third = explain.results_grid([solaris, helios, solaris_again])

    assert first["Load"] == "✅"
    assert first["Outcome"] == "Loaded"
    assert second["Validate"] == "❌"
    assert second["Load"] == "⏭️"
    assert second["Outcome"] == "Needs review"
    assert third["Load"] == "⏭️"
    assert third["Outcome"] == "Already loaded"
