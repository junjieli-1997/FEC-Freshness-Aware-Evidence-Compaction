"""FEC: Freshness-aware Evidence Compaction (see docs/DESIGN.md).

* Trigger: above ``threshold_tokens`` the turns before the last ``keep_recent`` are compacted.
  Before ``min_step`` steps, long old observations are only truncated to head + tail instead,
  unless the view exceeds ``hard_factor`` x threshold.
* Edits are detected from git: after every non read-only step the hook snapshots
  ``git status`` + blob hashes of changed files and diffs it with the previous snapshot, so edits
  made by python scripts, sed, tools ... are all seen and earlier reads of those files go STALE.
  Without git it falls back to command heuristics.
* The summary (at most ``summary_budget`` tokens) contains: modified files, the latest test run
  before the cut with its verbatim failure lines, task requirements not yet reflected in the git
  diff, current facts, stale evidence, the agent's own plan, and a command log.
* Verify after compaction: a submission with no test run since the last compaction is rejected once.
"""

from __future__ import annotations

import re

from agentkit.agent import ACTION_RE, estimate_tokens
from agentkit.verify_gate import TEST_CMD_RE

from .compaction import OBS_RE, ThresholdCompaction, _cmd, _thought
from .ledger import TEST_RE, Ledger, classify, edit_targets, paths_in, summarise_test

PLAN_RE = re.compile(r"\b(I'll|I will|Next|next,|need to|then I|remaining|still need|still have to|left to do|TODO|plan is)\b", re.I)
IDENT_RE = re.compile(r"`([^`]{2,80})`|\b([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+)\b|\b([a-z]+_[a-z_0-9]+)\b|\b([A-Z][a-z]+[A-Z]\w*)\b")
FILEISH = re.compile(r"/|\.(?:py|go|js|jsx|ts|tsx|java|rb|rs|c|h|cpp|yml|yaml|json|md|toml|cfg|ini|sh|html|css)$")
FAIL_LINE_RE = re.compile(
    r"^\s*(?:FAILED\b|ERROR\b|FAIL\b|--- FAIL|✕|×|●\s|\d+\)\s|panic:|E\s{2,}\w)"
    r"|\bAssertionError\b|\[build failed\]|\bundefined: |: error\b|\bError: ")

GIT_SNAPSHOT = (
    "git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo __NOGIT__; exit 0; }; "
    "git status --porcelain=v1 -uall 2>/dev/null | head -400 | while IFS= read -r l; do "
    "f=\"${l:3}\"; f=\"${f##* -> }\"; f=\"${f%\\\"}\"; f=\"${f#\\\"}\"; "
    "if [ -f \"$f\" ]; then echo \"$(git hash-object -- \"$f\" 2>/dev/null) $f\"; else echo \"deleted $f\"; fi; done")
GIT_DIFF = (
    "git diff --no-color HEAD 2>/dev/null | head -c 300000; "
    "git ls-files -o --exclude-standard 2>/dev/null | head -50 | while IFS= read -r f; do "
    "echo \"+++ b/$f\"; head -c 20000 \"$f\" 2>/dev/null | sed 's/^/+/'; done")

VERIFY_MESSAGE = ("Your context was compacted at step {step} and you have not run any tests since then, "
                  "so you have not verified the current state of the code. Re-run the tests that cover "
                  "your change now and read their output before submitting (if this task has no test "
                  "suite, run a command that exercises the changed behaviour).")

H_MODIFIED = "## Files you have modified (step of last edit)"
H_FACTS = "## Current facts (still valid)"
H_STALE = "## Superseded / stale evidence (re-read or re-run if you need it)"


# ---------------------------------------------------------------- helpers
def requirement_items(task: str, limit: int = 20) -> list[str]:
    """Bullet / numbered items of the task text (prefer a 'Requirements' section if present)."""
    m = re.search(r"(?im)^\s*#*\s*requirements?\s*:?\s*$", task)
    body = task[m.end():] if m else task
    items = []
    for line in body.splitlines():
        s = line.strip()
        mm = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)", s)
        if mm and len(mm.group(1)) >= 20:
            items.append(re.sub(r"\s+", " ", mm.group(1))[:180])
        if len(items) >= limit:
            break
    return items


def identifiers(text: str) -> list[str]:
    ids = []
    for m in IDENT_RE.finditer(text):
        tok = next(g for g in m.groups() if g)
        tok = tok.strip().split("(")[0].split(" ")[0]
        if len(tok) >= 3 and tok not in ids:
            ids.append(tok)
    return ids[:6]


def present(ident: str, diff: str, changed: set[str]) -> bool:
    """Is a requirement identifier reflected in the change? Paths/file names are matched against the
    changed files, dotted names by their last component, everything else by text in the diff hunks."""
    ident = ident.strip().strip("'\"")
    if FILEISH.search(ident) and " " not in ident:
        return any(c == ident or c.endswith("/" + ident) or ident.endswith(c) for c in changed)
    if "." in ident and re.fullmatch(r"[\w.]+", ident):
        return ident.split(".")[-1] in diff
    return ident in diff


def is_test(cmd: str) -> bool:
    return bool(TEST_RE.search(cmd) or TEST_CMD_RE.search(cmd))


def fail_lines(output: str, limit: int = 12) -> list[str]:
    out = []
    for line in output.splitlines():
        if FAIL_LINE_RE.search(line):
            s = re.sub(r"\s+", " ", line.strip())[:180]
            if s and s not in out:
                out.append(s)
        if len(out) >= limit:
            break
    return out


def split_sections(lines: list[str]) -> dict[str, list[str]]:
    """{'## heading': [body lines]} in order; lines before the first heading go under ''."""
    secs: dict[str, list[str]] = {"": []}
    cur = ""
    for line in lines:
        if line.startswith("## "):
            cur = line
            secs[cur] = []
        else:
            secs[cur].append(line)
    return secs


class FECCompaction(ThresholdCompaction):
    name = "fec"

    def __init__(self, threshold_tokens: int = 12000, keep_recent: int = 3, min_step: int = 6,
                 require_edit: bool = False, hard_factor: float = 2.0, summary_budget: int = 800,
                 early_truncate_chars: int = 1500, git: bool = True, verify_after_compaction: bool = True):
        super().__init__(threshold_tokens, keep_recent)
        self.ledger = Ledger()
        self.min_step, self.require_edit, self.hard_factor = min_step, require_edit, hard_factor
        self.budget = summary_budget
        self.early_truncate_chars = early_truncate_chars
        self.git = git
        self.git_ok: bool | None = None
        self.snapshot: dict[str, str] = {}
        self.changed_files: list[str] = []
        self.git_edit_steps = 0
        self.verify_after_compaction = verify_after_compaction
        self.edit_texts: list[str] = []
        self.tests: list[dict] = []   # every test run: step, cmd, rc, summary, failure lines
        self.n_gated = 0
        self.verify_reminders = 0

    # ------------------------------------------------------------ observing steps
    def on_start(self, agent):
        self._agent = agent
        self.ledger.workdir = (getattr(agent.env, "workdir", "") or "").rstrip("/")
        if self.git:
            self.snapshot = self._snap(agent)

    def _snap(self, agent) -> dict[str, str]:
        try:
            out = agent.env.execute(GIT_SNAPSHOT, timeout=60).output
        except Exception:  # noqa: BLE001 - a failing probe must not break the run
            self.git_ok = False
            return {}
        if "__NOGIT__" in out:
            self.git_ok = False
            return {}
        self.git_ok = True
        snap = {}
        for line in out.splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                snap[parts[1]] = parts[0]
        return snap

    def after_step(self, agent, step):
        if not (step.kind == "action" and step.action):
            return []
        self.ledger.observe(step.index, _thought(step.response), step.action, step.output, step.returncode)
        if step.returncode == 0 and (edit_targets(step.action) or re.search(r"sed\s+-i|tf-edit", step.action)):
            self.edit_texts.append(step.action)
        kind, by = classify(step.action)
        # git snapshot, skipped for pure read-only commands (nothing can have changed)
        if self.git and self.git_ok and not (kind in ("READ", "SEARCH") and set(by) <= {"READ", "SEARCH"}):
            snap = self._snap(agent)
            changed = [f for f, h in snap.items() if self.snapshot.get(f) != h]
            changed += [f for f in self.snapshot if f not in snap]  # reverted to HEAD
            self.snapshot = snap
            if changed:
                self.git_edit_steps += 1
                self.ledger.add_git_edits(step.index, changed)
                for f in changed:
                    if f not in self.changed_files:
                        self.changed_files.append(f)
        if is_test(step.action):
            summary, _ = summarise_test(step.output or "", step.returncode)
            self.tests.append({"step": step.index, "cmd": re.sub(r"\s+", " ", step.action.strip())[:120],
                               "rc": step.returncode, "summary": summary, "fails": fail_lines(step.output or "")})
        return []

    # ------------------------------------------------------------ trigger
    def _gated(self, agent) -> bool:
        return len(agent.steps) < self.min_step or (self.require_edit and not self.edit_texts)

    def _truncate_old(self, messages):
        a_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        keep_from = a_idx[-self.keep_recent] if len(a_idx) >= self.keep_recent else len(messages)
        out, n = [], self.early_truncate_chars
        for i, m in enumerate(messages):
            o = OBS_RE.search(m["content"]) if m["role"] == "user" and 0 < i < keep_from else None
            if o and len(o.group(2)) > 2 * n:
                body = o.group(2)
                text = (f"<returncode>{o.group(1)}</returncode>\n<output>\n{body[:n]}\n[... {len(body) - 2 * n} "
                        f"characters of this earlier output elided to save context ...]\n{body[-n:]}\n</output>")
                out.append({"role": m["role"], "content": text})
            else:
                out.append(m)
        return out

    def before_query(self, agent, messages):
        view = self._view(agent, messages)
        tokens = self._tokens(agent, view)
        if tokens <= self.threshold:
            return view
        if self.cut is None and self._gated(agent) and tokens <= self.hard_factor * self.threshold:
            self.n_gated += 1
            return self._truncate_old(view)
        return super().before_query(agent, messages)

    # ------------------------------------------------------------ verify after compaction
    def on_submit(self, agent):
        if not (self.verify_after_compaction and self.compaction_steps) or self.verify_reminders:
            return None
        last = self.compaction_steps[-1]   # index of the first step taken with the new summary
        if any(t["step"] >= last for t in self.tests):
            return None
        self.verify_reminders += 1
        return VERIFY_MESSAGE.format(step=last)

    # ------------------------------------------------------------ summary sections
    def _plan(self, msgs) -> list[str]:
        out = []
        for m in [x for x in msgs if x["role"] == "assistant"][-3:]:
            t = ACTION_RE.sub("", m["content"]).replace("THOUGHT:", "")
            for s in re.split(r"(?<=[.!?])\s+", t):
                if PLAN_RE.search(s) and len(s.strip()) > 15:
                    out.append(re.sub(r"\s+", " ", s.strip())[:220])
        return out[-4:]

    def _diff_text(self, agent) -> str:
        if self.git and self.git_ok:
            try:
                d = agent.env.execute(GIT_DIFF, timeout=60).output
                # changed lines, their +-3 context lines and hunk headers (which name the enclosing
                # function): editing a function body counts as touching that function
                return "\n".join(l for l in d.splitlines()
                                 if l[:1] in ("+", "-", " ", "@") and not l.startswith(("+++", "---")))
            except Exception:  # noqa: BLE001
                pass
        return "\n".join(self.edit_texts)

    def _requirements(self, agent) -> list[str]:
        """Requirement items of the task text with no identifier in the current change, and files
        named in the task that were not changed."""
        task = agent.vars.get("task", "")
        items = requirement_items(task)
        diff = self._diff_text(agent)
        source = "your current git diff" if (self.git and self.git_ok) else "your edit commands (no git here)"
        changed = set(self.changed_files) | {p for e in self.ledger.events if e.kind == "EDIT" for p in e.paths}
        checked = [(i, it) for i, it in enumerate(items, 1) if identifiers(it)]
        missing = [(i, it) for i, it in checked if not any(present(x, diff, changed) for x in identifiers(it))]
        lines = []
        if missing:
            lines.append("- Items with NO identifier in " + source + " yet:")
            lines += [f"  {i}. {it[:140]}" for i, it in missing[:10]]
            if len(missing) > 10:
                lines.append(f"  ... and {len(missing) - 10} more")
        named = [p for p in paths_in(task) if "/" in p]
        not_changed = [p for p in dict.fromkeys(named) if not any(c.endswith(p) or p.endswith(c) for c in changed)]
        if not_changed:
            lines.append("- Files named in the task that you have not changed: " + ", ".join(not_changed[:8]))
        return lines

    def _test_section(self, upto_step: int) -> list[str]:
        before = [t for t in self.tests if t["step"] < upto_step]
        if not before:
            return []
        t = before[-1]
        lines = [f"- `{t['cmd']}` -> {t['summary']} (exit {t['rc']})"]
        edited_after = [e.t for e in self.ledger.events if e.kind == "EDIT" and t["step"] < e.t < upto_step]
        if edited_after:
            lines.append(f"- code was edited after this run (step {edited_after[-1]}): re-run before relying on it")
        if t["fails"]:
            lines.append("- failure lines (verbatim):")
            lines += [f"  {f}" for f in t["fails"]]
        elif t["rc"] == 0:
            lines.append("- no failures reported")
        return lines

    # ------------------------------------------------------------ summary with a token budget
    def _summarise(self, messages, new_cut, upto_step):
        return self._fec_summary(messages[1:new_cut], upto_step)

    def _fec_summary(self, msgs, upto_step):
        led = self.ledger.render(upto_step).split("\n")
        header, secs = led[0], split_sections(led[1:])
        modified = secs.get(H_MODIFIED, [])
        facts = secs.get(H_FACTS, [])
        stale = secs.get(H_STALE, [])
        stale_note = False
        test = self._test_section(upto_step)
        req = self._requirements(self._agent)
        plan = self._plan(msgs)
        log = [f"- {_thought(m['content'])[:120]}  $ {_cmd(m['content'])[:100]}"
               for m in msgs if m["role"] == "assistant"][-15:]

        def render() -> str:
            out = [header]
            if modified:
                out += [H_MODIFIED] + modified
            if test:
                out += [f"## Latest test run before compaction (step {[t for t in self.tests if t['step'] < upto_step][-1]['step']})"] + test
            if req:
                out += ["## Task requirements not yet reflected in your changes"] + req
            out += [H_FACTS] + facts
            if stale:
                out += [H_STALE] + stale
            elif stale_note:
                out.append("## Stale evidence omitted: re-read files before relying on earlier outputs")
            if plan:
                out += ["## Where you were (your own plan just before compaction)"] + [f"- {p}" for p in plan]
            if log:
                out += ["## Command log of the compacted turns (latest)"] + log
            out.append("Continue from the plan above; re-read any file before editing it, and re-run the tests before you submit.")
            return "\n".join(out)

        def over() -> bool:
            return estimate_tokens(render()) > self.budget

        while over() and log:                                   # 1. command log
            log = log[1:]
        if over() and stale:                                    # 2. stale evidence
            stale, stale_note = [], True
        while over() and len(facts) > 6:                        # 3. oldest facts
            facts = facts[1:]
        req_head = [l for l in req if not l.startswith("  ")]   # 4. requirement items (keep headers)
        req_items = [l for l in req if l.startswith("  ")]
        while over() and len(req_items) > 3:
            req_items = req_items[:-1]
            req = req_head[:1] + req_items + req_head[1:]
        fl = [l for l in test if l.startswith("  ")]            # 5. failure lines
        while over() and len(fl) > 5:
            fl = fl[:-1]
            test = [l for l in test if not l.startswith("  ")] + fl
        while over() and len(facts) > 3:                        # 6. last resort
            facts = facts[1:]
        while over() and len(plan) > 2:
            plan = plan[1:]
        return render()

    def stats(self):
        st = super().stats()
        st.update({"n_gated_queries": self.n_gated, "budget_tokens": self.budget,
                   "git_available": self.git_ok, "git_edit_steps": self.git_edit_steps,
                   "changed_files": self.changed_files[:30], "verify_reminders": self.verify_reminders,
                   "n_test_runs": len(self.tests), "min_step": self.min_step, "require_edit": self.require_edit})
        return st
