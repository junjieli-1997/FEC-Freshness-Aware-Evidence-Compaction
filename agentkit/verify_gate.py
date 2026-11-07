"""Verification gate at submission time (part of FEC; see docs/DESIGN.md).

In 10 of the 14 SWE-bench Pro tasks that every configuration failed, the agent wrote that its
work was unverified / partial, or that the remaining test failures were "expected", "pre-existing",
"old expectations" or "environmental", and submitted anyway. Those tests were usually exactly the
hidden tests that graded the task.

When the agent submits and its last replies contain such a statement, the submission is rejected
once with a message asking it to name every failing / unverified test or requirement and, for
each, either fix it or give concrete evidence why it may stay as it is.
"""

from __future__ import annotations

import re

from .agent import ACTION_RE, Hook

TEST_CMD_RE = re.compile(r"\b(pytest|py\.test|runtests\.py|bin/test|tox\b|unittest|nosetests|go test|"
                         r"cargo test|npm (?:run )?test|yarn test|jest|mocha|make test|ctest|/tests/test\.sh)\b")

# statements that the work is unverified / incomplete, or that failures are acceptable
DOUBT_PATTERNS = {
    "unverified": r"\b(unverified|not verified|haven'?t (?:verified|run|tested|checked)|did not (?:run|verify|test)|"
                  r"didn'?t (?:run|verify|test)|could not (?:run|verify|test)|couldn'?t (?:run|verify|test)|"
                  r"cannot (?:run|verify)|can'?t (?:run|verify)|untested|without running)\b",
    "incomplete": r"\b(partly done|partially|only part|incomplete|not (?:fully )?(?:complete|done|met|implemented)|"
                  r"remaining gaps?|known gaps?|left (?:as is|for a follow-?up)|follow-?up|stub|placeholder)\b",
    "expected_failure": r"\b(expected (?:failures?|to fail|mismatch\w*)|failures? (?:are|is) expected|"
                        r"pre-?existing|predate|old (?:signature|expectation|wording|interface)s?|"
                        r"unrelated to (?:my|this|the) change|environmental|"
                        r"(?:leave|leaving) (?:the|those|that|these) tests?|not (?:editing|allowed to edit) (?:the )?test|"
                        r"conflicts? with the (?:spec|requirements?|pr))\b",
}

GATE_MESSAGE = """Before you submit: your last message says {why}.
The hidden tests that grade this task are usually exactly the tests you just called unverified,
expected to fail, outdated or environmental. Do this now, then submit again:
1. List every test or requirement that is failing or unverified, by exact name.
2. For each one, either fix the code, or state the concrete evidence that it may stay as it is
   (quote the sentence of the task that requires the new behaviour; "the test uses the old
   signature" is NOT enough - the graders use updated tests written for the required behaviour,
   so match the names and signatures given in the task exactly).
3. Run the relevant tests (or, if none can run here, a script that exercises the changed code)
   and show the result.{fails}"""


def doubts(text: str) -> list[str]:
    return [k for k, p in DOUBT_PATTERNS.items() if re.search(p, text, re.I)]


def last_failing_test_lines(agent, limit: int = 8) -> list[str]:
    """Failure lines of the most recent test command that exited non-zero (if it is the latest test run)."""
    for s in reversed(agent.steps):
        if s.kind == "action" and s.action and TEST_CMD_RE.search(s.action):
            if not s.returncode:
                return []
            lines = [l.strip() for l in (s.output or "").splitlines()
                     if re.match(r"^\s*(FAILED|ERROR|FAIL\b|--- FAIL|✕|×|●)", l)]
            return list(dict.fromkeys(lines))[:limit]
    return []


class VerifyGateHook(Hook):
    name = "verify_gate"

    def __init__(self, max_rejections: int = 1, lookback: int = 2):
        self.max_rejections = max_rejections
        self.lookback = lookback          # how many of the latest replies to inspect
        self.rejections = 0
        self.triggers: list[str] = []

    def on_submit(self, agent):
        if self.rejections >= self.max_rejections:
            return None
        recent = [s.response for s in agent.steps[-self.lookback:]]
        text = ACTION_RE.sub("", "\n".join(recent))
        why = doubts(text)
        if not why:
            return None
        self.rejections += 1
        self.triggers += why
        labels = {"unverified": "the work is unverified", "incomplete": "the work is incomplete or partial",
                  "expected_failure": "some failures are expected / pre-existing / not yours to fix"}
        fails = last_failing_test_lines(agent)
        extra = ("\nFailures in your latest test run:\n" + "\n".join(f"  {l[:200]}" for l in fails)) if fails else ""
        return GATE_MESSAGE.format(why=" and ".join(labels[w] for w in why), fails=extra)

    def stats(self):
        return {"rejections": self.rejections, "triggers": self.triggers}

