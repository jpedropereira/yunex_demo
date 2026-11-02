"""How full the model's context is, and what is filling it.

Ollama has no tokenizer endpoint, and reports one number: the size of the whole prompt. But
the model resends everything on each call, so the difference between two calls' prompts is
exactly what was added between them. Subtracting gives real counts, for no extra model calls:

- The system prompt and tool definitions never change, so they are measured once.
- Each later message costs what the prompt grew by when it first arrived.
- Messages that arrive together, like a tool call and its result, share one difference, and
  are divided by estimate. The small one is only tens of tokens, so the error is small.
"""

import json
from dataclasses import dataclass

from ollama import Client

from yunex_demo.agent import tools
from yunex_demo.agent.loop import ModelCall
from yunex_demo.config import settings
from yunex_demo.sec.filing import estimate_tokens

# The estimate counts every bracket and quote of this JSON, but the model is shown a compact
# form: tool definitions measured 489 tokens against 718 estimated, and tool calls a similar
# ratio. Used for dividing messages that arrive together, and if Ollama can't be asked.
JSON_COST = 489 / 718

SYSTEM = "System prompt"
TOOL_DEFINITIONS = "Tool definitions"
USER = "User"
TOOL_RESULTS = "Tool results"
MODEL = "Model"
FREE = "Free"

PART_BY_ROLE = {"system": SYSTEM, "user": USER, "tool": TOOL_RESULTS, "assistant": MODEL}


@dataclass(frozen=True)
class FixedCosts:
    """What every call starts with, in real tokens."""

    system_prompt: int
    tool_definitions: int


@dataclass(frozen=True)
class ContextUsage:
    # Tokens by part, in display order. FREE is what is left of ``capacity``.
    parts: dict[str, int]
    capacity: int
    # False when Ollama couldn't be asked, so all of this is only an estimate.
    measured: bool
    # The last reply, within parts[MODEL]. It counts because generating it uses the context too.
    reply_tokens: int = 0
    # "length" means the reply ran out of room or hit a limit, and was cut short.
    done_reason: str = ""

    @property
    def used(self) -> int:
        return self.capacity - self.parts[FREE]

    def share(self, part: str) -> float:
        return self.parts[part] / self.capacity


def _count_prompt_tokens(messages: list[dict], tool_definitions: list[dict] | None) -> int:
    """Ask Ollama how big this prompt really is. A one-token reply is the least it takes."""
    response = Client(host=settings.ollama_host, timeout=300).chat(
        model=settings.agent_model,
        messages=messages,
        tools=tool_definitions,
        keep_alive=settings.ollama_keep_alive,
        options={
            "num_ctx": settings.ollama_num_ctx,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_predict": 1,
        },
    )
    return response.prompt_eval_count


def measure_fixed_costs(system_message: dict) -> FixedCosts:
    """Measure the system prompt and tool definitions. Raises if Ollama can't be reached."""
    without_tools = _count_prompt_tokens([system_message], None)
    with_tools = _count_prompt_tokens([system_message], tools.TOOLS)
    return FixedCosts(system_prompt=without_tools, tool_definitions=with_tools - without_tools)


def _estimate_message(message: dict) -> int:
    tokens = estimate_tokens(message["content"])
    if message.get("tool_calls"):
        tokens += round(estimate_tokens(json.dumps(message["tool_calls"])) * JSON_COST)
    return tokens


def _divide(total: int, weights: list[int]) -> list[int]:
    """Split ``total`` in proportion to ``weights``, adding up to exactly ``total``."""
    if not weights:
        return []
    weight_sum = sum(weights) or 1
    shares = [total * weight // weight_sum for weight in weights]
    shares[weights.index(max(weights))] += total - sum(shares)
    return shares


def _message_costs(calls: list[ModelCall], fixed: FixedCosts) -> list[list[int]]:
    """The real token cost of every message in every call, by subtracting between calls."""
    costs_by_call = []
    previous = []
    for call in calls:
        messages = call.request["messages"]
        first_new = call.new_messages[0] if call.new_messages else len(messages)

        # What was sent before costs what it cost before. Dropping old turns takes them from
        # the front, so what remains of the last call is its system prompt and its tail.
        if first_new == 0:
            carried = [fixed.system_prompt]
            first_new = 1
        else:
            kept = first_new - 1
            carried = [previous[0], *previous[len(previous) - kept :]]

        prompt_tokens = call.response.get("prompt_eval_count", 0)
        growth = max(0, prompt_tokens - fixed.tool_definitions - sum(carried))
        new_messages = messages[first_new:]
        previous = carried + _divide(growth, [_estimate_message(m) for m in new_messages])
        costs_by_call.append(previous)
    return costs_by_call


def _with_free_space(parts: dict[str, int], capacity: int) -> dict[str, int]:
    return {**parts, FREE: max(0, capacity - sum(parts.values()))}


def _empty_parts(tool_definitions: int) -> dict[str, int]:
    return {
        SYSTEM: 0,
        TOOL_DEFINITIONS: tool_definitions,
        USER: 0,
        TOOL_RESULTS: 0,
        MODEL: 0,
    }


def _estimated_tool_definitions() -> int:
    return round(estimate_tokens(json.dumps(tools.TOOLS)) * JSON_COST)


def usage_before_any_call(messages: list[dict], fixed: FixedCosts | None) -> ContextUsage:
    """What the first call will start from."""
    capacity = settings.ollama_num_ctx
    if fixed is None:
        parts = _empty_parts(_estimated_tool_definitions())
        for message in messages:
            parts[PART_BY_ROLE[message["role"]]] += _estimate_message(message)
    else:
        parts = _empty_parts(fixed.tool_definitions)
        parts[SYSTEM] = fixed.system_prompt
    return ContextUsage(_with_free_space(parts, capacity), capacity, measured=fixed is not None)


def usage_of_calls(calls: list[ModelCall], fixed: FixedCosts | None) -> list[ContextUsage]:
    """What each model call really occupied, including the reply it generated."""
    measured = fixed is not None and all(c.response.get("prompt_eval_count") for c in calls)
    if measured:
        costs_by_call = _message_costs(calls, fixed)
        tool_definitions = fixed.tool_definitions
    else:
        costs_by_call = [[_estimate_message(m) for m in c.request["messages"]] for c in calls]
        tool_definitions = _estimated_tool_definitions()

    usages = []
    for call, costs in zip(calls, costs_by_call, strict=True):
        parts = _empty_parts(tool_definitions)
        for message, cost in zip(call.request["messages"], costs, strict=True):
            parts[PART_BY_ROLE[message["role"]]] += cost

        reply_tokens = call.response.get("eval_count", 0)
        parts[MODEL] += reply_tokens
        capacity = call.request["options"]["num_ctx"]
        usages.append(
            ContextUsage(
                _with_free_space(parts, capacity),
                capacity,
                measured,
                reply_tokens,
                call.response.get("done_reason", ""),
            )
        )
    return usages
