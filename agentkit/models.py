"""Model backends: Anthropic (official SDK), OpenAI (official SDK) and a scripted fake for tests.

Every backend exposes ``query(system, messages) -> Reply``.  ``messages`` is a list of
``{"role": "user"|"assistant", "content": str}``; consecutive same-role messages are merged
before sending so callers can append freely.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# USD per 1M tokens: (input, output, cache_read, cache_write_5m).
PRICES = {
    # Claude Haiku 5.5: prompts <=100K tokens; longer prompts are billed at the "long" card.
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-haiku-5-5:long": (0.50, 2.50, 0.05, 0.625),
    "gpt-5-nano": (0.05, 0.40, 0.005, 0.05),
    "gpt-5-mini": (0.25, 2.00, 0.025, 0.25),
}


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    stop_reason: str | None = None


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0

    def add(self, r: Reply) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cache_read_tokens += r.cache_read_tokens
        self.cache_write_tokens += r.cache_write_tokens
        self.cost += r.cost

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def merge_roles(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(start)"})
    return out


def _price(model: str, prompt_tokens: int) -> tuple[float, float, float, float]:
    if model == "claude-haiku-5-5" and prompt_tokens > 100_000:
        return PRICES["claude-haiku-5-5:long"]
    return PRICES.get(model, (0.0, 0.0, 0.0, 0.0))


class AnthropicModel:
    """Claude via the official ``anthropic`` SDK.

    Thinking is left at the model default (adaptive) and effort at ``medium``.  The agent
    protocol is plain text (one bash block per reply), so only the text blocks are kept in
    the conversation history; thinking blocks are never replayed.  That keeps history
    edits (compaction, masking) valid.
    """

    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 8000, effort: str = "medium"):
        import anthropic

        self.anthropic = anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        self.client = anthropic.Anthropic(api_key=key, max_retries=6, timeout=300.0)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def query(self, system: str, messages: list[dict]) -> Reply:
        msgs = merge_roles(messages)
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=msgs,
                    output_config={"effort": self.effort},
                    cache_control={"type": "ephemeral"},
                )
                break
            except self.anthropic.RateLimitError:
                time.sleep(20 * (attempt + 1))
            except self.anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except self.anthropic.APIConnectionError:
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError("Anthropic API kept failing")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        prompt = u.input_tokens + cr + cw
        pin, pout, pcr, pcw = _price(self.model, prompt)
        cost = (u.input_tokens * pin + u.output_tokens * pout + cr * pcr + cw * pcw) / 1e6
        return Reply(text, u.input_tokens, u.output_tokens, cr, cw, cost, resp.stop_reason)


class OpenAIModel:
    """OpenAI via the official ``openai`` SDK (Chat Completions)."""

    def __init__(self, model: str = "gpt-5-nano", max_tokens: int = 32000, effort: str = "medium"):
        # gpt-5 models count hidden reasoning tokens against max_completion_tokens; with 8000 the
        # reasoning alone often used the whole budget and the visible reply came back empty.
        import openai

        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            max_retries=6,
            timeout=300.0,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def query(self, system: str, messages: list[dict]) -> Reply:
        msgs = [{"role": "system", "content": system}] + merge_roles(messages)
        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs, max_completion_tokens=self.max_tokens,
            reasoning_effort=self.effort,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        pin, pout, pcr, _ = _price(self.model, u.prompt_tokens)
        cost = ((u.prompt_tokens - cached) * pin + cached * pcr + u.completion_tokens * pout) / 1e6
        return Reply(text, u.prompt_tokens - cached, u.completion_tokens, cached, 0, cost,
                     resp.choices[0].finish_reason)


@dataclass
class FakeModel:
    """Returns scripted replies in order; records every request for assertions."""

    replies: list[str]
    requests: list[tuple[str, list[dict]]] = field(default_factory=list)
    model: str = "fake"

    def query(self, system: str, messages: list[dict]) -> Reply:
        self.requests.append((system, [dict(m) for m in messages]))
        text = self.replies.pop(0) if self.replies else "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
        chars = len(system) + sum(len(m["content"]) for m in messages)
        return Reply(text, chars // 4, len(text) // 4, 0, 0, 0.0, "end_turn")


def make_model(name: str):
    if name.startswith("claude"):
        return AnthropicModel(name)
    if name.startswith("gpt") or name.startswith("o"):
        return OpenAIModel(name)
    raise ValueError(f"unknown model {name}")
