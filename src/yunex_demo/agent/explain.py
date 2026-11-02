"""Explain the agent to the audience: one question's path, and why each step exists.

It only reads, and never changes what the agent does. Numbers come from the constants the
agent really uses, so the text can't quote a limit the code has since changed.

The flowchart is fixed text, so nothing from a filing or the model can get into the DOT.
"""

from dataclasses import dataclass

from yunex_demo.agent import citations, hooks, loop, quotes, tools
from yunex_demo.config import settings
from yunex_demo.sec.filing import estimate_tokens

CANDIDATE_PAGES = tools.SEARCH_RESULTS + tools.MOST_FRONT_MATTER_HITS

# Explicit text colour on every fill, so nodes stay readable in Streamlit's dark theme.
_TEXT = "#111827"
_FILL = "#f1f5f9"
_MODEL_BORDER = "#7c3aed"
_CODE_BORDER = "#2563eb"
_HUMAN_BORDER = "#d97706"
_EDGE = "#94a3b8"


@dataclass(frozen=True)
class Section:
    title: str
    text: str  # written to be read aloud; keep it in step with the code it describes
    read_the_code: str


INTRODUCTION = (
    "The model can only ask for things. Everything that happens is done by code, which can say "
    "no. Follow one question through the chart: ellipses are the model, boxes are code, and the "
    "numbers match the sections below."
)

LOOP = Section(
    "The loop: the model asks, and code acts",
    "The model never runs anything. Its reply is either an answer or a request for a tool. Code "
    "runs the request and sends the result back in a new call, along with everything said so "
    "far, because the API has no memory. One question can take several calls, and code stops "
    f"the loop after {settings.agent_max_turns}, so a model that keeps asking can't run forever.",
    "agent/loop.py",
)

HOOKS = Section(
    "Hooks: the guardrails on every tool call",
    "A **guardrail** is anything that keeps the model in bounds. A **hook** is the kind used "
    "here: a small function that runs before every tool call and can stop it. Each one returns "
    "**deny** (the tool doesn't run, and the model is told why), **ask** (wait for a human), "
    "**warn** (note it, and carry on) or nothing at all. They run in this order, because each "
    "relies on the ones above: the size limit can only do its sums once the arguments are known "
    "to be sensible. The first deny wins. The model has no say in any of this.",
    "agent/hooks.py",
)

APPROVAL = Section(
    "Human approval: the only way to change the portfolio",
    "A trade that passes the hooks stops here, and `propose_trade` has no code that executes it, "
    "so no reply from the model can reach a position. Clicking Approve runs the checks again, "
    "because prices or holdings may have changed since the trade was proposed. Only then does "
    "this code change the portfolio. The model hears the outcome on its next turn.",
    "agent/approval.py",
)

SHORT_RESULTS = Section(
    "Keeping tool results short: only the sentences that help",
    "A filing page costs hundreds of tokens, and every call resends every result, so whole pages "
    "would soon fill the context. So `search_filing` never gives the agent a page. It gives it "
    "sentences:\n\n"
    f"1. **Find pages (code).** The question is embedded and compared with the filing, up to "
    f"{CANDIDATE_PAGES} pages. Code filters by company, so the pages can't come from another "
    "filing, however similar their text.\n"
    f"2. **Skip the front pages (code).** Covers and contents can rank well, but never help. The "
    f"first {tools.SEARCH_RESULTS} other pages go on.\n"
    "3. **Copy the sentences (model).** Each page goes to the model in a call of its own, with "
    "the prompt below. A schema forces the reply to be a list, and the page is fenced as data, "
    "so that it isn't read as instructions.\n"
    "4. **Check each sentence (code).** A sentence is kept only if it is really on the page, so "
    "the model can't slip in its own wording.\n"
    "5. **Send only those (code).** The agent's history gets the kept sentences and their "
    "citations, and the rest of each page is left out.\n\n"
    # Measured on ten searches across all five companies: 11% overall, from 6% to 21%. Each
    # search took 3 to 7 seconds. The citations and JSON around the sentences cost a little more.
    "**The cost and the limits.** This keeps about a tenth of the text, for a few seconds more "
    "per search. Copying is still a judgment, so a kept sentence may not really help. These "
    "calls have a context of their own, so they don't show in the bar above, and the Page checks "
    "tab shows every one.",
    "agent/tools.py and agent/quotes.py",
)

FIT_CONTEXT = Section(
    "Fitting the context: drop the oldest, and never overflow",
    f"The model holds {settings.ollama_num_ctx:,} tokens, and Ollama fits a longer prompt by "
    "dropping messages with no error at all. So before every call, code counts first. Past "
    f"{loop.context_budget():,} tokens or {loop.MAX_HISTORY_MESSAGES} messages it drops the "
    f"oldest whole turns, which keeps {loop.CONTEXT_RESERVE_TOKENS:,} tokens free for the reply "
    "and never separates a tool result from its request. If the current question alone doesn't "
    "fit, the agent stops and says so instead of answering with half its evidence.",
    "agent/loop.py and agent/context_usage.py",
)

CHECK_ANSWER = Section(
    "Checking the answer: find the passage, and let a person read it",
    "A model can cite a page it never read, or a real page that says something else. So code "
    "compares each citation with the page, finds the closest passage, and shows the two side by "
    f"side. It flags a score below {citations.MATCH_THRESHOLD}, a page the agent wasn't given, a "
    "number that isn't on the page, and quoted text that isn't word for word. It never says "
    '"verified": matching words can\'t judge meaning, so "none" in place of "all" still matches. '
    "These checks flag the answer and don't change it. A person decides.",
    "agent/citations.py",
)

ASKED_OR_ENFORCED = Section(
    "Asked or enforced? What the model could ignore",
    "A system prompt is a request, and a hook is a fact. So ask of each rule: what happens if "
    "the model ignores it? Anything that must never go wrong belongs in code.",
    "agent/loop.py for the requests, agent/hooks.py for what is enforced",
)

ASKED_OR_ENFORCED_ROWS = [
    {
        "Rule": "Stay on topic",
        "Where it lives": "System prompt",
        "If the model ignores it": "It answers anyway. It has written a Python function.",
    },
    {
        "Rule": "Cite every sentence",
        "Where it lives": "System prompt",
        "If the model ignores it": "Gaps in the answer. The page warns only if nothing is cited.",
    },
    {
        "Rule": "Treat filings as data",
        "Where it lives": "System prompt and fence",
        "If the model ignores it": "It may be misled, but every tool call still meets the hooks.",
    },
    {
        "Rule": "Only allowed tickers",
        "Where it lives": "Hook",
        "If the model ignores it": "Denied, and the tool never runs.",
    },
    {
        "Rule": f"No buy above {settings.trade_max_pct}% of the portfolio",
        "Where it lives": "Hook",
        "If the model ignores it": "Denied, and checked again when a human approves.",
    },
    {
        "Rule": "Every trade needs a human",
        "Where it lives": "Hook, and no code to execute it",
        "If the model ignores it": "Nothing happens. It has no way to execute a trade.",
    },
    {
        "Rule": "Pages come from the company searched",
        "Where it lives": "Code filter on the search",
        "If the model ignores it": "It can't: it never handles the filter.",
    },
]


@dataclass(frozen=True)
class HookExplanation:
    checks: str
    # Each of these fired that hook on the real model. Empty when no question was seen to.
    try_asking: str = ""


HOOK_EXPLANATIONS = {
    "known_tool_hook": HookExplanation("Is this a tool that exists?"),
    "arguments_hook": HookExplanation("Are the arguments the right shape?"),
    "allowlist_hook": HookExplanation("Is this one of the five companies?", "Buy 10 TSLA"),
    "ticker_mention_hook": HookExplanation("Warns if it searched a company nobody asked about."),
    "prices_hook": HookExplanation("Is there market data to price a trade with?"),
    "quantity_hook": HookExplanation("Is it a positive whole number of shares?", "Buy 0 AAPL"),
    "holdings_hook": HookExplanation("Do we hold enough to sell?", "Sell 1000 AAPL"),
    "size_limit_hook": HookExplanation(
        f"Is a buy within {settings.trade_max_pct}% of the portfolio?", "Buy 5000 NVDA"
    ),
    "approval_hook": HookExplanation("Every trade waits for a human.", "Buy 1 AAPL"),
}

_HOOK_SAID = {None: "no objection", "deny": "⛔ deny", "ask": "✋ ask", "warn": "⚠️ warn"}


def hook_rows(last_checks: list[dict] | None = None) -> list[dict]:
    """The hooks in running order. With a tool call's checks, also what each one said then."""
    said_by_hook = {check["hook"]: check["decision"] for check in last_checks or []}
    rows = []
    for number, hook in enumerate(hooks.PRE_TOOL_HOOKS, start=1):
        name = hook.__name__
        explanation = HOOK_EXPLANATIONS[name]
        row = {
            "#": number,
            "Hook": name,
            "What it checks": explanation.checks,
            "Try asking": explanation.try_asking or "—",
        }
        if last_checks is not None:
            # The first deny ends the list, so a hook missing from it never ran.
            row["Last tool call"] = _HOOK_SAID[said_by_hook[name]] if name in said_by_hook else "—"
        rows.append(row)
    return rows


def relevance_prompt_example() -> str:
    """The real prompt, with stand-ins for the two things that change."""
    return quotes.review_prompt("The user's question", "The whole page goes here.")


def search_funnel(reviews: list[quotes.PageReview]) -> str:
    """What one reply's searches did, in numbers only, so no filing text gets into the line."""
    read = [review for review in reviews if review.prompt]
    skipped = len(reviews) - len(read)
    kept = [sentence for review in read for sentence in review.kept]
    discarded = sum(len(review.discarded) for review in read)
    tokens_in = sum(estimate_tokens(review.prompt) for review in read)
    tokens_out = sum(estimate_tokens(sentence) for sentence in kept)
    return (
        f"Front pages skipped: {skipped} → pages read: {len(read)} → sentences kept: {len(kept)} "
        f"(not on the page, so discarded: {discarded}) · "
        f"about {tokens_in:,} tokens in → about {tokens_out:,} out"
    )


def flow_dot() -> str:
    """One question's path as DOT: ellipses are the model, boxes are code."""
    return "\n".join(
        [
            "digraph agent {",
            # Streamlit forces chart text to 9.6px and scales the chart to the page, so a
            # fontsize near that keeps nodes compact, which makes the text large on screen.
            'rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.35;',
            f'node [style=filled fontname="Helvetica" fontsize=10 fontcolor="{_TEXT}" '
            f'fillcolor="{_FILL}" margin="0.08,0.05" shape=box color="{_CODE_BORDER}" penwidth=2];',
            f'edge [color="{_EDGE}" fontcolor="{_EDGE}" fontname="Helvetica" fontsize=9 '
            "penwidth=1.5];",
            'Question [label="Your question" shape=note color="#64748b"];',
            'Fit [label="5. Fit the\\ncontext"];',
            f'Model [label="1. Model" shape=ellipse color="{_MODEL_BORDER}"];',
            'Hooks [label="2. Hooks"];',
            f'Human [label="3. Human\\napproves" shape=octagon color="{_HUMAN_BORDER}"];',
            'Portfolio [label="Portfolio" shape=cylinder];',
            'Tool [label="Run the tool"];',
            f'Relevance [label="4. Copy the\\nsentences" shape=ellipse color="{_MODEL_BORDER}"];',
            'Sentences [label="4. Check they\\nare on the page"];',
            'Check [label="6. Check the\\ncitations"];',
            'Answer [label="Your answer" shape=note color="#64748b"];',
            "Question -> Fit -> Model;",
            'Model -> Check [label="answers"];',
            "Check -> Answer;",
            'Model -> Hooks [label="asks for a tool"];',
            'Hooks -> Human [label="ask"];',
            "Human -> Portfolio;",
            'Hooks -> Tool [label="allow"];',
            'Tool -> Relevance [label="search_filing"];',
            "Relevance -> Sentences;",
            # These go back to the start, because every result means another call to the
            # model. constraint=false keeps them from pulling the chart out of shape.
            'Hooks -> Fit [label="deny, and why" style=dashed constraint=false];',
            'Tool -> Fit [label="result" constraint=false];',
            'Sentences -> Fit [label="sentences" constraint=false];',
            # The invisible edge puts the way out above Hooks, clear of the arrows going back.
            "{rank=same; Check; Hooks}",
            "Check -> Hooks [style=invis];",
            "}",
        ]
    )
