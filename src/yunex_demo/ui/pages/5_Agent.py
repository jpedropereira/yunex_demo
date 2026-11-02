"""Demo beat 5: an agent. The model picks the tools; code runs and guards every one.

Deliberately thin: the guards, tools and loop live in ``yunex_demo.agent``.

Buttons use callbacks because a callback runs before the script does, so a click has
already changed the portfolio by the time anything on the page is drawn.
"""

import copy
import json
import os
from dataclasses import asdict

import httpx
import streamlit as st
from ollama import ResponseError

from yunex_demo.agent import (
    approval,
    citations,
    context_usage,
    explain,
    loop,
    market,
    portfolio,
    quotes,
    tools,
)
from yunex_demo.agent.tools import AgentContext
from yunex_demo.config import settings
from yunex_demo.sec import ingest, search, store

DECISION_CHIPS = {"allow": "✅ allow", "deny": "⛔ deny", "ask": "✋ ask"}
# Never "verified": matching words can't tell that a claim means what the page means.
EVIDENCE_CHIPS = {
    citations.PASSAGE_FOUND: "🔎 Closest passage",
    citations.NO_CLOSE_PASSAGE: "⚠️ No close passage on this page",
    citations.NOT_RETRIEVED: "⛔ The agent never read this page",
}
RESULT_PREVIEW_CHARS = 400
# Long text scrolls inside its own block, so one filing page can't push the rest off screen.
LONG_TEXT_CHARS = 1500
LONG_TEXT_HEIGHT = 320

# The bar's colors, validated for color blindness in each mode, by part. A part this small
# has no room for its share to be printed on it, and hovering still gives its numbers.
BAR_COLORS = {
    "light": {
        context_usage.SYSTEM: "#2a78d6",
        context_usage.TOOL_DEFINITIONS: "#eb6834",
        context_usage.USER: "#1baf7a",
        context_usage.TOOL_RESULTS: "#eda100",
        context_usage.MODEL: "#e87ba4",
    },
    "dark": {
        context_usage.SYSTEM: "#3987e5",
        context_usage.TOOL_DEFINITIONS: "#d95926",
        context_usage.USER: "#199e70",
        context_usage.TOOL_RESULTS: "#c98500",
        context_usage.MODEL: "#d55181",
    },
}
FREE_SPACE_COLOR = "rgba(128, 128, 128, 0.22)"
# Streamlit's own page colors. The sticky bar needs one, or answers show through it.
PAGE_BACKGROUND = {"light": "#ffffff", "dark": "#0e1117"}
SMALLEST_LABELLED_SHARE = 0.05
NEARLY_FULL_SHARE = 0.85

# Fixed text only: nothing from a filing or the model ever goes through unsafe_allow_html.
PROJECTOR_CSS = """
<style>
[data-testid="stText"] { font-size: 1.1rem; }
[data-testid="stChatMessage"] p { font-size: 1.2rem; }
/* Wide enough to show the portfolio table's last column, Weight, without scrolling. */
[data-testid="stSidebar"] { min-width: 28rem !important; max-width: 28rem !important; }
.context-bar { display: flex; gap: 2px; height: 28px; margin-bottom: 0.4rem; }
.context-part {
    border-radius: 4px; color: #ffffff; font-size: 0.85rem; line-height: 28px;
    padding-left: 6px; overflow: visible; white-space: nowrap; min-width: 2px;
}
.context-legend { display: flex; flex-wrap: wrap; gap: 0.2rem 1rem; font-size: 0.9rem; }
.context-swatch {
    display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 0.35rem;
}
/* Stays in view while a long answer scrolls, because this is what the audience should watch. */
[data-testid="stVerticalBlock"] > div:has(.context-bar-sticky) {
    position: sticky; top: 3.75rem; z-index: 50; padding: 0.4rem 0;
}
/* The input lives in the Chat tab, so a long answer would otherwise push it out of view. */
[data-testid="stElementContainer"]:has([data-testid="stChatInput"]) {
    position: sticky; bottom: 0.75rem; z-index: 40;
}
</style>
"""


def theme() -> str:
    return "dark" if st.context.theme.type == "dark" else "light"


def markdown_safe(text: str) -> str:
    # Streamlit reads "$...$" as a formula, which garbles any answer with two prices in it.
    return text.replace("$", r"\$")


# --- 1. Page header -----------------------------------------------------------
st.set_page_config(page_title="Agent", page_icon=":material/smart_toy:", layout="wide")
st.markdown(PROJECTOR_CSS, unsafe_allow_html=True)
st.markdown(
    '<style>[data-testid="stVerticalBlock"] > div:has(.context-bar-sticky) '
    f"{{ background: {PAGE_BACKGROUND[theme()]}; }}</style>",
    unsafe_allow_html=True,
)

st.title("Agent")
st.subheader("The model chooses the tools; code runs them, guards them, and asks a human.")

# --- 2. What survives between reruns ------------------------------------------
# The model remembers nothing, so the page keeps the history and resends it all each turn.
if "agent_messages" not in st.session_state:
    st.session_state.agent_conn = portfolio.connect()
    st.session_state.agent_snapshot = market.load_snapshot()
    st.session_state.agent_messages = loop.new_conversation()
    st.session_state.agent_chat = []
    st.session_state.agent_calls = []
    st.session_state.agent_retrieved_pages = {}

# Measured once, because these never change. Asked again on each run until Ollama answers,
# and until then the bar says it is estimating, so the page still opens without Ollama.
if st.session_state.get("agent_fixed_costs") is None:
    try:
        with st.spinner("Measuring the context..."):
            st.session_state.agent_fixed_costs = context_usage.measure_fixed_costs(
                st.session_state.agent_messages[0]
            )
    # The ollama client turns "server unreachable" into a builtin ConnectionError.
    except (ConnectionError, httpx.HTTPError, ResponseError):
        st.session_state.agent_fixed_costs = None

fixed_costs = st.session_state.agent_fixed_costs
conn = st.session_state.agent_conn
snapshot = st.session_state.agent_snapshot
messages = st.session_state.agent_messages
chat = st.session_state.agent_chat
calls = st.session_state.agent_calls


def refresh_prices() -> None:
    st.session_state.agent_snapshot = market.load_snapshot()


def reset_portfolio() -> None:
    portfolio.reset(conn)


def new_conversation() -> None:
    st.session_state.agent_messages = loop.new_conversation()
    st.session_state.agent_chat = []
    st.session_state.agent_calls = []
    st.session_state.agent_retrieved_pages = {}
    st.session_state.agent_turns_dropped = 0


def record_decision(outcome: approval.ApprovalOutcome) -> None:
    # Into the history, so the model knows on its next turn. It isn't called now: the
    # click decides the trade, and asking the model to comment would only add a wait.
    if outcome.note:
        messages.append({"role": "user", "content": outcome.note})
        chat.append({"role": "note", "text": outcome.note})


def approve(trade_id: int) -> None:
    record_decision(approval.approve_trade(conn, trade_id, st.session_state.agent_snapshot))


def reject(trade_id: int) -> None:
    record_decision(approval.reject_trade(conn, trade_id))


# --- 3. How each piece is drawn. Tool results and filing text are never markdown ---
def context_bar_html(usage: context_usage.ContextUsage) -> str:
    """Fixed names and whole numbers only, so nothing from a filing or the model gets in."""
    colors = BAR_COLORS[theme()]
    blocks = []
    legend = []
    summary = []
    for name, tokens in usage.parts.items():
        if tokens == 0:
            continue
        share = usage.share(name)
        is_free = name == context_usage.FREE
        color = FREE_SPACE_COLOR if is_free else colors[name]
        # Near-black on the yellow, where white is hard to read. Free space sits on the page.
        ink = "inherit" if is_free else "#0b0b0b" if name == context_usage.TOOL_RESULTS else "#fff"
        label = f"{share:.0%}" if share >= SMALLEST_LABELLED_SHARE else ""
        summary.append(f"{name}: {tokens:,} tokens, {share:.0%}")
        blocks.append(
            f'<div class="context-part" style="flex: {tokens} 1 0; background: {color}; '
            f'color: {ink};" title="{summary[-1]}">{label}</div>'
        )
        legend.append(
            f'<span><span class="context-swatch" style="background: {color};"></span>{name}</span>'
        )
    return (
        f'<div class="context-bar" role="img" aria-label="Context used. {"; ".join(summary)}">'
        f'{"".join(blocks)}</div><div class="context-legend">{"".join(legend)}</div>'
    )


def context_summary(usage: context_usage.ContextUsage) -> str:
    free = usage.parts[context_usage.FREE]
    summary = f"{usage.used:,} of {usage.capacity:,} tokens used · {free:,} free"
    if usage.reply_tokens:
        summary += f" · last reply took {usage.reply_tokens:,}, thinking included"
    if not usage.measured:
        summary += " · estimated, because Ollama couldn't be asked"
    return summary


def show_context_usage(usage: context_usage.ContextUsage) -> None:
    # The marker is what the sticky rule in PROJECTOR_CSS finds this block by.
    st.markdown(
        f'<div class="context-bar-sticky"></div>{context_bar_html(usage)}',
        unsafe_allow_html=True,
    )
    st.caption(context_summary(usage))
    if usage.done_reason == "length":
        st.warning("The last reply was cut short: it ran out of room in the context.")
    elif usage.used / usage.capacity >= NEARLY_FULL_SHARE:
        st.caption(
            f"⚠️ Nearly full. The oldest turns are dropped once the prompt passes about "
            f"{loop.context_budget():,} tokens, leaving room for the reply."
        )
    with st.expander("Context breakdown"):
        st.dataframe(
            [
                {"Part": name, "Tokens": tokens, "Share": f"{usage.share(name):.1%}"}
                for name, tokens in usage.parts.items()
            ],
            hide_index=True,
        )
        st.caption(
            "Real counts from Ollama. It reports only each prompt's total, so a message costs "
            "what the prompt grew by when it arrived, and messages arriving together are "
            "divided by estimate. Model includes earlier replies, tool calls and the last reply."
        )


def show_steps(steps: list[dict]) -> None:
    if not steps:
        return
    with st.expander(f"Steps ({len(steps)})"):
        for step in steps:
            with st.container(border=True):
                st.markdown(f"**{DECISION_CHIPS[step['decision']]}**  ·  {step['seconds']:.1f}s")
                st.text(step["tool"])
                st.code(json.dumps(step["arguments"], indent=2), language="json")
                if step["hook"]:
                    st.text(f"{step['hook']}: {step['reason']}")
                for warning in step["warnings"]:
                    st.text(f"⚠️ {warning}")
                for review in step["page_reviews"]:
                    st.text(f"{review['cite']}: {review['outcome']}")
                st.code(step["result"][:RESULT_PREVIEW_CHARS], language="json", wrap_lines=True)
                if len(step["result"]) > RESULT_PREVIEW_CHARS:
                    st.caption("Cut short here. The Context tab shows all of it.")


def show_filing_link(source_url: str, passage: str, key: str) -> None:
    link = citations.filing_link(source_url, passage)
    if link:
        st.link_button("Open in the filing", link, icon=":material/open_in_new:", key=key)


def show_evidence_flags(evidence: citations.Evidence) -> None:
    for number in evidence.missing_numbers:
        st.text(f"⚠️ The claim says {number}, and that number isn't on this page.")
    for misquote in evidence.misquotes:
        st.text(f"⚠️ In quotation marks, but not word for word on this page: {misquote}")
    if evidence.better_cite:
        st.text(f"The closest wording is on {evidence.better_cite}: {evidence.better_passage}")


def show_evidence_card(evidence: citations.Evidence, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{EVIDENCE_CHIPS[evidence.status]}**")
        st.text(evidence.cite)
        show_evidence_flags(evidence)
        says_column, filing_column = st.columns(2)
        with says_column:
            st.markdown("##### The agent says")
            st.text(evidence.claim)
        with filing_column:
            st.markdown("##### The filing says")
            if evidence.status == citations.NOT_RETRIEVED:
                st.text("Nothing: this page was not among the search results.")
                return
            # Tabs, because an expander can't go inside the one this card is in.
            passage_tab, page_tab = st.tabs(["Passage", "Whole page"])
            with passage_tab:
                show_long_text(evidence.passage or "Nothing on this page comes close.")
            with page_tab:
                show_long_text(evidence.page_text)
            show_filing_link(evidence.source_url, evidence.passage, key)


def show_evidence(evidence_list: list[citations.Evidence], item_id: int) -> None:
    flagged = [evidence for evidence in evidence_list if evidence.needs_attention]
    close = sum(evidence.status == citations.PASSAGE_FOUND for evidence in evidence_list)
    with st.expander(f"Check the sources ({len(evidence_list)})", expanded=bool(flagged)):
        st.caption(
            f"{close} of {len(evidence_list)} citations have a close passage. "
            "Matching words don't prove the meaning, so read them."
        )
        # Sorted on False before True, so what needs a look comes first.
        ordered = sorted(evidence_list, key=lambda evidence: not evidence.needs_attention)
        for number, evidence in enumerate(ordered):
            show_evidence_card(evidence, key=f"evidence_{item_id}_{number}")


def show_uncited_pages(pages: list[tools.RetrievedPage], item_id: int) -> None:
    st.warning("This answer cites no page, but the agent had sentences from these. Check it.")
    with st.expander(f"Pages its sentences came from ({len(pages)})"):
        for number, page in enumerate(pages):
            with st.container(border=True):
                st.text(page.cite)
                show_long_text(page.text)
                show_filing_link(page.source_url, "", key=f"read_{item_id}_{number}")


def show_chat_item(item: dict) -> None:
    if item["role"] == "note":
        with st.chat_message("note", avatar=":material/gavel:"):
            st.text(item["text"])
        return

    with st.chat_message(item["role"]):
        st.markdown(markdown_safe(item["text"]))
        if item.get("evidence"):
            show_evidence(item["evidence"], item["id"])
        if item.get("uncited_pages"):
            show_uncited_pages(item["uncited_pages"], item["id"])
        show_steps(item.get("steps", []))


def show_pending_trade(trade: dict, total_value) -> None:
    with st.container(border=True):
        st.markdown(f"#### ✋ Trade #{trade['id']} needs your approval")
        value = float(trade["value"])
        share = f"{value / float(total_value) * 100:.2f}%" if total_value else "unknown"
        st.text(
            f"{trade['side']} {trade['quantity']} {trade['ticker']} "
            f"at USD {float(trade['price']):,.2f} = USD {value:,.2f} ({share} of the portfolio)"
        )
        left, right, _ = st.columns([1, 1, 4])
        with left:
            st.button(
                "Approve",
                key=f"approve_{trade['id']}",
                type="primary",
                icon=":material/check:",
                on_click=approve,
                args=(trade["id"],),
                disabled=snapshot is None,
            )
        with right:
            st.button(
                "Reject",
                key=f"reject_{trade['id']}",
                icon=":material/close:",
                on_click=reject,
                args=(trade["id"],),
            )


def tool_row(tool: dict) -> dict:
    """One tool, read from the same definition the model receives, so the two can't disagree."""
    definition = tool["function"]
    parameters = definition["parameters"]
    arguments = []
    for name, schema in parameters["properties"].items():
        argument = f"`{name}`"
        if "enum" in schema:
            argument += f" ({', '.join(schema['enum'])})"
        if name not in parameters["required"]:
            argument += " (optional)"
        arguments.append(argument)
    effect = "Only reads" if definition["name"] in tools.READ_TOOLS else "Waits for a human"
    return {
        "Tool": f"`{definition['name']}`",
        "What the model is told": definition["description"],
        "Arguments": ", ".join(arguments) or "None",
        "Effect": effect,
    }


def show_long_text(text: str) -> None:
    height = LONG_TEXT_HEIGHT if len(text) > LONG_TEXT_CHARS else "content"
    st.code(text, language=None, wrap_lines=True, height=height)


def call_label(call: loop.ModelCall) -> str:
    tool_calls = call.response.get("message", {}).get("tool_calls", [])
    names = ", ".join(tool_call["function"]["name"] for tool_call in tool_calls)
    return (
        f"Call {call.number} · turn {call.turn} · "
        f"{call.response.get('prompt_eval_count', 0):,} prompt tokens · "
        f"{f'asked for {names}' if names else 'answered'}"
    )


def show_rendered_prompt(call: loop.ModelCall, previous_prompt: str) -> None:
    prompt = call.rendered_prompt
    if not prompt:
        st.info("Ollama didn't return the final text for this call. See Message by message.")
        return
    st.caption(
        "Ollama's final text for this call, special tokens included: this is all the model "
        "gets. Filing quotes have no prompt of their own: they are search_filing's tool "
        "result, as JSON, so a line break in a quote reads `\\n`. Page checks, on the call "
        "that asked for the search, shows how they were chosen."
    )
    # Split where the text first differs, so this also holds after old turns are dropped.
    resent = os.path.commonprefix([previous_prompt, prompt])
    if resent:
        st.markdown("##### Resent from the last call")
        show_long_text(resent)
    st.markdown("##### New in this call")
    show_long_text(prompt[len(resent) :])


def show_tool_result(message: dict) -> None:
    st.text(message["tool_name"])
    # Exactly the string that was sent, not reformatted: this is how quotes reach the model.
    show_long_text(message["content"])


def show_sent_message(message: dict, is_new: bool) -> None:
    with st.container(border=True):
        label = f"{message['role']} · {'new' if is_new else 'resent'}"
        st.badge(label, color="green" if is_new else "gray")
        if message["role"] == "tool":
            show_tool_result(message)
        elif message["content"]:
            show_long_text(message["content"])
        for tool_call in message.get("tool_calls", []):
            st.text(tool_call["function"]["name"])
            st.code(json.dumps(tool_call["function"]["arguments"], indent=2), language="json")


def show_reply(response: dict) -> None:
    reply = response.get("message", {})
    if reply.get("thinking"):
        st.markdown("##### Thinking")
        st.caption("It counts towards this call's output, but isn't sent back on later calls.")
        show_long_text(reply["thinking"])
    if reply.get("content"):
        st.markdown("##### Reply")
        show_long_text(reply["content"])
    for tool_call in reply.get("tool_calls", []):
        st.markdown("##### Tool call")
        st.text(tool_call["function"]["name"])
        st.code(json.dumps(tool_call["function"]["arguments"], indent=2), language="json")
    st.text(
        f"done_reason: {response.get('done_reason')} · "
        f"prompt tokens: {response.get('prompt_eval_count', 0):,} "
        f"of {settings.ollama_num_ctx:,} · "
        f"output tokens: {response.get('eval_count', 0):,}"
    )


def show_page_review(review: quotes.PageReview) -> None:
    with st.container(border=True):
        st.text(f"{review.cite} · similarity {review.similarity} · {review.seconds:.1f}s")
        st.text(review.outcome)
        for heading, sentences in (("Kept", review.kept), ("Discarded", review.discarded)):
            if sentences:
                st.markdown(f"##### {heading}")
                show_long_text("\n\n".join(sentences))
        if review.prompt:
            st.markdown("##### Prompt, exactly as sent")
            show_long_text(review.prompt)
            st.markdown("##### Raw reply")
            show_long_text(review.reply)


def show_model_call(call: loop.ModelCall, previous_prompt: str) -> None:
    labels = ["Prompt as the model reads it", "Message by message", "What came back", "API JSON"]
    if call.page_reviews:
        labels.append("Page checks")
    prompt_tab, messages_tab, reply_tab, json_tab, *checks_tab = st.tabs(labels)
    if call.page_reviews:
        with checks_tab[0]:
            st.caption(
                "Code ran these after this call asked to search, and only the kept sentences "
                "went back to the agent. Each is a model call in a context of its own, so "
                "none of this counts in the bar above."
            )
            for review in call.page_reviews:
                show_page_review(review)
    with prompt_tab:
        show_rendered_prompt(call, previous_prompt)
    with messages_tab:
        st.caption("A readable copy, with quotes unescaped. The first tab is the literal text.")
        for position, message in enumerate(call.request["messages"]):
            show_sent_message(message, position in call.new_messages)
            # Where the model's template puts them: in the system turn, before any other message.
            if position == 0:
                with st.container(border=True):
                    st.badge("tool definitions · resent on every call", color="gray")
                    st.json(call.request["tools"], expanded=False)
    with reply_tab:
        show_reply(call.response)
    with json_tab:
        st.json({"request": call.request, "response": call.response}, expanded=False)


def show_section_text(section: explain.Section) -> None:
    st.markdown(section.text)
    st.caption(f"Read the code: {section.read_the_code}")


def show_how_it_works() -> None:
    st.markdown(explain.INTRODUCTION)
    st.graphviz_chart(explain.flow_dot(), width="stretch")

    with st.expander(f"1. {explain.LOOP.title}"):
        show_section_text(explain.LOOP)

    with st.expander(f"2. {explain.HOOKS.title}"):
        show_section_text(explain.HOOKS)
        all_steps = [step for item in chat for step in item.get("steps", [])]
        last_step = all_steps[-1] if all_steps else None
        # st.dataframe and st.text, because the model chooses a tool call's name.
        st.dataframe(explain.hook_rows(last_step["checks"] if last_step else None), hide_index=True)
        if last_step:
            st.text(f"Last tool call: {last_step['tool']}")

    with st.expander(f"3. {explain.APPROVAL.title}"):
        show_section_text(explain.APPROVAL)

    with st.expander(f"4. {explain.SHORT_RESULTS.title}"):
        show_section_text(explain.SHORT_RESULTS)
        st.markdown("##### The prompt for each page, and the only reply shape allowed")
        show_long_text(explain.relevance_prompt_example())
        st.code(json.dumps(quotes.Quotes.model_json_schema(), indent=2), language="json")
        searches = [call.page_reviews for call in calls if call.page_reviews]
        if searches:
            st.markdown("##### Your last search")
            st.text(explain.search_funnel(searches[-1]))

    with st.expander(f"5. {explain.FIT_CONTEXT.title}"):
        show_section_text(explain.FIT_CONTEXT)

    with st.expander(f"6. {explain.CHECK_ANSWER.title}"):
        show_section_text(explain.CHECK_ANSWER)

    with st.expander(explain.ASKED_OR_ENFORCED.title):
        show_section_text(explain.ASKED_OR_ENFORCED)
        # st.table wraps long text, where st.dataframe would cut it off.
        st.table(explain.ASKED_OR_ENFORCED_ROWS, hide_index=True)


# --- 4. How full the context is. Filled in last, once this run's turn has happened -
context_slot = st.container()

# --- 5. The tabs: the conversation, what the model may ask for, what it received, and why
chat_tab, tools_tab, context_tab, how_it_works_tab = st.tabs(
    ["Chat", "Tools", "Context", "How it works"]
)

# In the Chat tab, not pinned to the page: a pinned input makes Streamlit follow new content to
# the bottom, so the other tabs would open scrolled past their first lines.
with chat_tab:
    conversation = st.container()
    question = st.chat_input("Ask about the five companies, or propose a trade")

with conversation:
    if snapshot is None:
        st.error("No market data, live or cached. Trading is disabled until prices load.")

    for item in chat:
        show_chat_item(item)

    if question:
        chat.append({"role": "user", "text": question})
        show_chat_item(chat[-1])

        # Restored if the model call fails, so a retry doesn't send the question twice.
        history_before = copy.deepcopy(messages)
        messages.append({"role": "user", "content": question})
        context = AgentContext(conn, snapshot, st.session_state.agent_retrieved_pages)

        try:
            with st.spinner("Thinking..."):
                result = loop.run_turn(
                    messages,
                    context,
                    turn_number=sum(item["role"] == "user" for item in chat),
                    previous_messages=calls[-1].request["messages"] if calls else None,
                    first_call_number=len(calls) + 1,
                )
        # The ollama client turns "server unreachable" into a builtin ConnectionError.
        except (ConnectionError, httpx.HTTPError, ResponseError) as exc:
            messages[:] = history_before
            chat.pop()
            st.error(f"Model call failed: {exc}. Check the Stack Health page.")
        else:
            calls.extend(result.calls)
            evidence = citations.check_answer(result.answer, context.retrieved_pages)
            chat.append(
                {
                    "role": "assistant",
                    # Its position in the chat, which keeps each answer's widgets apart.
                    "id": len(chat),
                    "text": result.answer,
                    "steps": [asdict(step) for step in result.steps],
                    "evidence": evidence,
                    "uncited_pages": []
                    if evidence
                    else citations.pages_read(result.steps, context.retrieved_pages),
                }
            )
            show_chat_item(chat[-1])
            if result.turns_dropped:
                st.session_state.agent_turns_dropped = (
                    st.session_state.get("agent_turns_dropped", 0) + result.turns_dropped
                )

    # Read after the turn above, so a trade the model just proposed shows straight away.
    positions = portfolio.read_positions(conn)
    prices = snapshot.latest_prices() if snapshot else {}
    total_value = portfolio.total_value(positions, prices)

    for trade in portfolio.read_pending_trades(conn):
        show_pending_trade(dict(trade), total_value)

    if calls:
        st.caption(f"messages sent this turn: {len(calls[-1].request['messages'])}")
    if st.session_state.get("agent_turns_dropped"):
        st.caption(
            f"The {st.session_state.agent_turns_dropped} oldest turns were dropped from the "
            "history to fit the context, so the model no longer sees them."
        )

with tools_tab:
    st.markdown(
        f"These {len(tools.TOOLS)} tools are all the agent can ask for, and asking is all it can "
        "do: code checks each request, runs it, and denies any other tool name."
    )
    # st.table wraps long text, where st.dataframe would cut it off.
    st.table([tool_row(tool) for tool in tools.TOOLS], hide_index=True)
    if fixed_costs:
        st.caption(f"Sent on every model call, costing {fixed_costs.tool_definitions:,} tokens.")
    with st.expander("Exactly as the model receives them"):
        st.json(tools.TOOLS)

# Worked out here, after the turn above, so it includes this run's calls.
usages = context_usage.usage_of_calls(calls, fixed_costs)

with context_tab:
    st.markdown(
        "One question can take several model calls, because each tool result goes back in a new "
        "call. The API has no memory, so every call carries everything before it."
    )
    if not calls:
        st.text("No model calls yet. Ask something in the Chat tab.")
    else:
        # No key, so the choice returns to the newest call whenever a new one is added.
        chosen = st.selectbox(
            "Model call",
            range(len(calls)),
            index=len(calls) - 1,
            format_func=lambda position: call_label(calls[position]),
        )
        st.caption(context_summary(usages[chosen]))
        previous_prompt = calls[chosen - 1].rendered_prompt if chosen else ""
        show_model_call(calls[chosen], previous_prompt)
        st.download_button(
            "Download the call log as JSON",
            data=json.dumps([asdict(call) for call in calls], indent=2),
            file_name="agent_call_log.json",
            mime="application/json",
            icon=":material/download:",
        )

with how_it_works_tab:
    show_how_it_works()

with context_slot:
    if usages:
        show_context_usage(usages[-1])
    else:
        show_context_usage(context_usage.usage_before_any_call(messages, fixed_costs))

# --- 6. The sidebar, drawn last so it shows anything the turn above changed ------
with st.sidebar:
    st.subheader(portfolio.PORTFOLIO_NAME)

    if snapshot:
        st.dataframe(
            [
                {
                    "Ticker": ticker,
                    "Shares": quantity,
                    "Price": f"{prices[ticker]:,.2f}",
                    "Value": f"{quantity * prices[ticker]:,.2f}",
                    "Weight": f"{quantity * prices[ticker] / total_value * 100:.1f}%",
                }
                for ticker, quantity in positions.items()
            ],
            hide_index=True,
        )
        st.text(f"Total: USD {total_value:,.2f}")
        st.text(
            f"Max buy ({settings.trade_max_pct}%): "
            f"USD {portfolio.max_trade_value(total_value):,.2f}"
        )
        fetched_at = f"{snapshot.fetched_at:%H:%M}"
        st.text(f"Prices cached at {fetched_at}" if snapshot.from_cache else f"Prices {fetched_at}")
    else:
        st.dataframe(
            [{"Ticker": ticker, "Shares": quantity} for ticker, quantity in positions.items()],
            hide_index=True,
        )

    st.text(f"Model: {settings.agent_model}")

    st.button("Refresh prices", icon=":material/refresh:", on_click=refresh_prices, width="stretch")
    st.button(
        "Reset portfolio", icon=":material/restart_alt:", on_click=reset_portfolio, width="stretch"
    )
    st.button("New conversation", icon=":material/add:", on_click=new_conversation, width="stretch")

    st.subheader("Indexed filings")
    try:
        documents = search.list_documents(store.get_collection())
    except ingest.EXPECTED_ERRORS as exc:
        st.warning("Could not reach ChromaDB, so filing search won't work.")
        st.text(str(exc))
    else:
        if documents:
            st.dataframe(
                [
                    {
                        "Ticker": document.ticker,
                        "Form": document.form,
                        "Year": document.fiscal_year,
                        "Period end": document.period_end,
                    }
                    for document in documents
                ],
                hide_index=True,
            )
        else:
            st.info("Nothing indexed yet. Run `make index-all`.")

    st.subheader("Recent trades")
    trades = portfolio.read_trades(conn)
    if trades:
        # st.dataframe, because a denied request can hold anything the model sent.
        st.dataframe(
            [
                {
                    "#": trade["id"],
                    "Status": trade["status"],
                    "Trade": f"{trade['side']} {trade['quantity'] or '?'} {trade['ticker']}",
                    "Reason": trade["reason"],
                }
                for trade in trades
            ],
            hide_index=True,
        )
    else:
        st.text("None yet.")
