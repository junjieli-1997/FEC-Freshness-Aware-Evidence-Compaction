"""A hookable bash-only agent loop (mini-swe-agent style: one ```bash block per reply).

Hooks let a method (e.g. FEC compaction) change one thing without forking the loop:

* ``on_start(agent)``                         - once, before the first query
* ``before_query(agent, messages) -> messages`` - rewrite what the model sees (compaction)
* ``before_action(agent, action) -> str|None``  - veto a command (returns the message shown)
* ``on_observation(agent, action, result) -> str|None`` - replace the observation text
* ``after_step(agent, step) -> list[str]``      - extra user messages (critic, warnings)
* ``on_submit(agent) -> str|None``              - reject a submission (returns the reason)
* ``on_end(agent, result)``                     - once, after the loop
* ``stats() -> dict``                           - merged into the run result

``agent.history`` always holds the full, unmodified conversation; hooks that rewrite the
context do so only in ``before_query``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .models import Reply, Usage

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
ACTION_RE = re.compile(r"```(?:bash|sh)\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class Step:
    index: int
    response: str
    action: str | None = None
    output: str = ""
    returncode: int | None = None
    kind: str = "action"  # action | format_error | vetoed | submit | rejected_submit
    context_tokens: int = 0
    elapsed_s: float = 0.0       # seconds since the run started, when the reply arrived
    input_tokens: int = 0        # this call's prompt tokens (uncached + cache read + cache write)
    output_tokens: int = 0


class Hook:
    name = "hook"

    def on_start(self, agent: "Agent") -> None: ...

    def before_query(self, agent: "Agent", messages: list[dict]) -> list[dict]:
        return messages

    def before_action(self, agent: "Agent", action: str) -> str | None:
        return None

    def on_observation(self, agent: "Agent", action: str, result) -> str | None:
        return None

    def after_step(self, agent: "Agent", step: Step) -> list[str]:
        return []

    def on_submit(self, agent: "Agent") -> str | None:
        return None

    def on_end(self, agent: "Agent", result: dict) -> None: ...

    def stats(self) -> dict:
        return {}


@dataclass
class AgentConfig:
    system_template: str
    instance_template: str
    step_limit: int = 60
    cost_limit: float = 1.0
    wall_limit: float = 1800.0
    command_timeout: float = 120.0
    max_obs_chars: int = 10000
    max_rejected_submits: int = 2


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def format_observation(rc: int, output: str, limit: int) -> str:
    if len(output) > limit:
        half = limit // 2
        output = (output[:half] + f"\n\n[... {len(output) - limit} characters elided; use more "
                  f"selective commands such as head, tail, grep or sed -n ...]\n\n" + output[-half:])
    return f"<returncode>{rc}</returncode>\n<output>\n{output}\n</output>"


@dataclass
class Agent:
    model: object
    env: object
    config: AgentConfig
    hooks: list[Hook] = field(default_factory=list)
    role: str = "main"
    history: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    aux_usage: dict[str, Usage] = field(default_factory=dict)
    vars: dict = field(default_factory=dict)
    peak_context_tokens: int = 0
    rejected_submits: int = 0
    system: str = ""
    started: float = 0.0

    # ------------------------------------------------------------------ helpers
    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def aux_query(self, label: str, system: str, messages: list[dict]) -> Reply:
        """Model call made by a hook (critic, sub-agent ...); billed separately."""
        r = self.model.query(system, messages)
        self.aux_usage.setdefault(label, Usage()).add(r)
        return r

    def total_cost(self) -> float:
        return self.usage.cost + sum(u.cost for u in self.aux_usage.values())

    def actions(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "action"]

    # ------------------------------------------------------------------ loop
    def run(self, task: str, **template_vars) -> dict:
        self.started = time.time()
        self.vars = {"task": task, **template_vars}
        self.system = self.config.system_template.format(**self.vars)
        self.add("user", self.config.instance_template.format(**self.vars))
        for h in self.hooks:
            h.on_start(self)
        exit_status = "step_limit"
        while True:
            if len(self.steps) >= self.config.step_limit:
                exit_status = "step_limit"
                break
            if self.total_cost() >= self.config.cost_limit:
                exit_status = "cost_limit"
                break
            if time.time() - self.started >= self.config.wall_limit:
                exit_status = "wall_limit"
                break
            messages = list(self.history)
            for h in self.hooks:
                messages = h.before_query(self, messages)
            ctx = estimate_tokens(self.system) + sum(estimate_tokens(m["content"]) for m in messages)
            self.peak_context_tokens = max(self.peak_context_tokens, ctx)
            try:
                reply = self.model.query(self.system, messages)
            except Exception as e:  # noqa: BLE001 - surface API failures as a run outcome
                exit_status = f"model_error: {type(e).__name__}: {str(e)[:200]}"
                break
            self.usage.add(reply)
            self.add("assistant", reply.text)
            step = Step(len(self.steps), reply.text, context_tokens=ctx,
                        elapsed_s=round(time.time() - self.started, 1),
                        input_tokens=reply.input_tokens + reply.cache_read_tokens + reply.cache_write_tokens,
                        output_tokens=reply.output_tokens)
            self.steps.append(step)
            if self._handle(step):
                exit_status = "submitted"
                break
            for h in self.hooks:
                for msg in h.after_step(self, step):
                    self.add("user", msg)
        result = {
            "role": self.role,
            "exit_status": exit_status,
            "n_steps": len(self.steps),
            "n_actions": len(self.actions()),
            "usage": self.usage.as_dict(),
            "aux_usage": {k: v.as_dict() for k, v in self.aux_usage.items()},
            "total_cost": round(self.total_cost(), 6),
            "peak_context_tokens": self.peak_context_tokens,
            "rejected_submits": self.rejected_submits,
            "wall_time": round(time.time() - self.started, 1),
            "last_response": self.steps[-1].response if self.steps else "",
        }
        for h in self.hooks:
            h.on_end(self, result)
            st = h.stats()
            if st:
                result.setdefault("hooks", {})[h.name] = st
        return result

    def _handle(self, step: Step) -> bool:
        """Execute the action of one reply. Returns True when the run is finished."""
        blocks = ACTION_RE.findall(step.response)
        if len(blocks) != 1:
            step.kind = "format_error"
            self.add("user", "Format error: your reply must contain exactly ONE ```bash code block "
                             f"(found {len(blocks)}). Reply again with one THOUGHT and one command.")
            return False
        action = blocks[0].strip()
        step.action = action
        if SUBMIT_MARKER in action:
            reason = None
            if self.rejected_submits < self.config.max_rejected_submits:
                for h in self.hooks:
                    reason = h.on_submit(self)
                    if reason:
                        break
            if reason:
                self.rejected_submits += 1
                step.kind = "rejected_submit"
                self.add("user", reason)
                return False
            step.kind = "submit"
            return True
        for h in self.hooks:
            veto = h.before_action(self, action)
            if veto:
                step.kind = "vetoed"
                step.output = veto
                self.add("user", veto)
                return False
        res = self.env.execute(action, timeout=self.config.command_timeout)
        step.returncode = res.returncode
        step.output = res.output
        obs = None
        for h in self.hooks:
            obs = h.on_observation(self, action, res) or obs
        self.add("user", obs if obs is not None
                 else format_observation(res.returncode, res.output, self.config.max_obs_chars))
        return False
