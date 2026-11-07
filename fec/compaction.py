"""Threshold-triggered compaction inside the agent loop, and the CliffCompaction baseline.

View sent to the model:  [instance] + [summary] + history[cut:]   (append-only between compactions)

When the view exceeds ``threshold_tokens``, everything before the last ``keep_recent`` assistant
turns is replaced by a summary. Subclasses decide what the summary contains:

* ``CliffCompaction``: mechanical summary of the turns being cut (thought, command, short outputs);
  a re-compaction drops the previous summary and summarises only the newly cut turns.
* ``FECCompaction`` (``fec/fec.py``): the cumulative evidence ledger with FRESH / STALE / SUPERSEDED
  marks, rebuilt over all compacted turns at every compaction.
"""

from __future__ import annotations

import re

from agentkit.agent import ACTION_RE, Hook, estimate_tokens

from .ledger import classify, paths_in

THOUGHT_CHARS, CMD_CHARS, RESULT_CHARS = 300, 150, 500
OBS_RE = re.compile(r"<returncode>(-?\d+)</returncode>\n<output>\n(.*)\n</output>", re.S)


def _thought(text: str) -> str:
    t = ACTION_RE.sub("", text).replace("THOUGHT:", "").strip()
    return re.sub(r"\s+", " ", t)[:THOUGHT_CHARS]


def _cmd(text: str) -> str:
    b = ACTION_RE.findall(text)
    return re.sub(r"\s+", " ", b[0].strip())[:CMD_CHARS] if len(b) == 1 else "(no command)"


class ThresholdCompaction(Hook):
    name = "compaction"

    def __init__(self, threshold_tokens: int = 12000, keep_recent: int = 3):
        self.threshold = threshold_tokens
        self.keep_recent = keep_recent
        self.cut: int | None = None
        self.summary: str | None = None
        self.n_compactions = 0
        self.summary_tokens: list[int] = []
        self.compaction_steps: list[int] = []

    def _view(self, agent, messages):
        if self.cut is None:
            return messages
        return [messages[0], {"role": "user", "content": self.summary}] + messages[self.cut:]

    def _tokens(self, agent, view) -> int:
        return estimate_tokens(agent.system) + sum(estimate_tokens(m["content"]) for m in view)

    def _summarise(self, messages: list[dict], new_cut: int, upto_step: int) -> str:
        """Summary replacing messages[1:new_cut]; ``upto_step`` = number of steps being compacted."""
        raise NotImplementedError

    def before_query(self, agent, messages):
        view = self._view(agent, messages)
        if self._tokens(agent, view) <= self.threshold:
            return view
        # new cut: keep the last `keep_recent` assistant turns (and what follows them) verbatim
        a_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if len(a_idx) <= self.keep_recent:
            return view
        new_cut = a_idx[-self.keep_recent]
        if self.cut is not None and new_cut <= self.cut:
            return view
        upto = sum(1 for i in a_idx if i < new_cut)
        self.summary = self._summarise(messages, new_cut, upto)
        self.cut = new_cut
        self.n_compactions += 1
        self.compaction_steps.append(len(agent.steps))
        self.summary_tokens.append(estimate_tokens(self.summary))
        return self._view(agent, messages)

    def stats(self):
        return {"n_compactions": self.n_compactions, "summary_tokens": self.summary_tokens,
                "compaction_steps": self.compaction_steps}

    def on_end(self, agent, result):
        result.setdefault("diagnostics", {})["redundant_rereads_after_compaction"] = \
            redundant_rereads(agent.steps, self.compaction_steps[0] if self.compaction_steps else None)


class CliffCompaction(ThresholdCompaction):
    name = "cliff"

    def _summarise(self, messages, new_cut, upto_step):
        start = 1 if self.cut is None else self.cut   # only the newly cut turns
        parts = ["[CONTEXT COMPACTED - mechanical summary of earlier turns; recent turns follow verbatim]"]
        for m in messages[start:new_cut]:
            if m["role"] == "assistant":
                parts.append(f"ASSISTANT: {_thought(m['content'])}\n  $ {_cmd(m['content'])}")
            else:
                o = OBS_RE.search(m["content"])
                if o:
                    rc, out = o.group(1), o.group(2)
                    parts.append(f"  -> rc={rc}: {out.strip()}" if len(out) <= RESULT_CHARS
                                 else f"  -> rc={rc}: [output dropped, {len(out)} chars]")
                else:
                    parts.append("  USER: " + re.sub(r"\s+", " ", m["content"])[:RESULT_CHARS])
        return "\n".join(parts)


def redundant_rereads(steps, first_compaction_step: int | None) -> int:
    """READs, after the first compaction, of a file already read before and not edited since."""
    if first_compaction_step is None:
        return 0
    last_read: dict[str, int] = {}
    last_edit: dict[str, int] = {}
    n = 0
    for s in steps:
        if s.kind != "action" or not s.action:
            continue
        kind, _ = classify(s.action)
        ps = paths_in(s.action)
        if kind == "EDIT":
            for p in ps:
                last_edit[p] = s.index
        elif kind == "READ":
            for p in ps:
                if s.index >= first_compaction_step and p in last_read and last_edit.get(p, -1) < last_read[p]:
                    n += 1
                last_read[p] = s.index
    return n
