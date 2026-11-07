import os
import subprocess
import sys
from types import SimpleNamespace

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from agentkit.agent import Agent, AgentConfig, Step, estimate_tokens  # noqa: E402
from agentkit.env import LocalEnv  # noqa: E402
from agentkit.methods import load_method  # noqa: E402
from agentkit.models import FakeModel  # noqa: E402
from fec.compaction import CliffCompaction, redundant_rereads  # noqa: E402
from fec.fec import FECCompaction, fail_lines, identifiers, is_test, present, requirement_items, split_sections  # noqa: E402

PY = sys.executable
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TASK = """Change things.

## Requirements
- `alpha` in pkg/mod.py must return 2 instead of 1 for all callers.
- The `Gamma.reset_all` method must clear every cached entry.
- Document the change in docs/usage.md for end users.
"""


def b(cmd, thought="step"):
    return f"THOUGHT: {thought}\n\n```bash\n{cmd}\n```"


def big(i, n=1500):
    return b(f"python3 -c \"print('{i}' * {n})\"")


def git_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("from pkg.mod import alpha\n\ndef test_x():\n    assert alpha() == 3\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(c, cwd=tmp_path, check=True)


def run(tmp_path, hook, replies, task=TASK, submit=True):
    env = LocalEnv(str(tmp_path), prefix=f"python3() {{ {PY} \"$@\"; }}; export -f python3")
    model = FakeModel(list(replies) + ([b(SUBMIT)] * 3 if submit else []))
    agent = Agent(model, env, AgentConfig("sys", "task {task}"), hooks=[hook])
    res = agent.run(task)
    return res, model, agent


def test_registry():
    assert [h.name for h in load_method("cliff", "pro").hooks] == ["cliff"]
    assert [h.name for h in load_method("fec", "pro").hooks] == ["fec", "verify_gate"]
    assert load_method("baseline", "pro").hooks == []


# ------------------------------------------------------------------ shared threshold logic + Cliff
def test_compaction_triggers_and_is_append_only(tmp_path):
    for hook in (CliffCompaction(2000, 2), FECCompaction(2000, 2, git=False)):
        res, model, _ = run(tmp_path, hook, [big(i) for i in range(12)])
        assert hook.n_compactions >= 2 and res["exit_status"] == "submitted"
        sizes = [sum(len(m["content"]) for m in msgs) for _, msgs in model.requests]
        assert max(sizes) < 4 * 2000 + 4 * 1700
        # between two compactions the view only grows by appending
        for (_, a), (_, c) in zip(model.requests, model.requests[1:]):
            if len(a) > 1 and len(c) > len(a) and c[1]["content"] == a[1]["content"]:
                assert c[: len(a)] == a


def test_fec_ledger_accumulates_but_cliff_drops(tmp_path):
    fec, cliff = FECCompaction(2000, 2, git=False), CliffCompaction(2000, 2)
    _, m_fec, _ = run(tmp_path, fec, [big(i) for i in range(12)])
    _, m_cliff, _ = run(tmp_path, cliff, [big(i) for i in range(12)])
    assert "@step 0" in m_fec.requests[-1][1][1]["content"]       # earliest evidence still listed
    assert "print('0'" not in m_cliff.requests[-1][1][1]["content"]  # cliff re-compaction dropped it


def test_redundant_rereads():
    steps = [Step(0, "", "cat a.py", kind="action"), Step(1, "", "cd /app && cat a.py", kind="action"),
             Step(2, "", "sed -i 's/x/y/' a.py", kind="action"), Step(3, "", "cat a.py", kind="action")]
    assert redundant_rereads(steps, 0) == 1
    assert redundant_rereads(steps, None) == 0


# ------------------------------------------------------------------ FEC trigger
def test_gate_opens_at_step_6_without_edits():
    hook = FECCompaction()
    assert hook.min_step == 6 and hook.require_edit is False
    assert hook._gated(SimpleNamespace(steps=[0] * 5)) is True
    assert hook._gated(SimpleNamespace(steps=[0] * 6)) is False


def test_gate_truncates_instead_of_compacting(tmp_path):
    hook = FECCompaction(2000, 2, min_step=50, hard_factor=100, early_truncate_chars=300, git=False)
    _, model, _ = run(tmp_path, hook, [big(i) for i in range(8)])
    assert hook.n_compactions == 0 and hook.n_gated > 0
    last_view = model.requests[-1][1]
    assert any("characters of this earlier output elided" in m["content"] for m in last_view)
    assert "elided" not in last_view[-1]["content"]                  # recent turns stay verbatim


def test_hard_ceiling_forces_compaction(tmp_path):
    hook = FECCompaction(2000, 2, min_step=50, hard_factor=1.2, early_truncate_chars=5000, git=False)
    run(tmp_path, hook, [big(i) for i in range(8)])
    assert hook.n_compactions >= 1


# ------------------------------------------------------------------ git edit detection
def test_git_detects_script_edit_and_marks_read_stale(tmp_path):
    git_repo(tmp_path)
    script = ("python3 - <<'EOF'\nimport pathlib\np = pathlib.Path('pkg/' + 'mod.py')\n"
              "p.write_text(p.read_text().replace('return 1', 'return 2'))\nEOF")
    hook = FECCompaction(100000, 2)
    run(tmp_path, hook, [b("cat pkg/mod.py"), b(script), b("ls")])
    st = hook.stats()
    assert st["git_available"] is True and st["git_edit_steps"] == 1 and st["changed_files"] == ["pkg/mod.py"]
    assert [e for e in hook.ledger.events if e.kind == "READ"][0].status == "STALE"
    git_edit = [e for e in hook.ledger.events if e.kind == "EDIT"][0]
    assert git_edit.cmd == "(detected by git)" and git_edit.t == 1


def test_git_detects_revert_and_skips_read_only_steps(tmp_path):
    git_repo(tmp_path)
    hook = FECCompaction(100000, 2)
    _, _, agent = run(tmp_path, hook, [b("sed -i 's/return 1/return 3/' pkg/mod.py"), b("cat pkg/mod.py"),
                                       b("git checkout -q -- pkg/mod.py")])
    assert hook.git_edit_steps == 2        # the edit and the revert both change the snapshot
    calls = []
    hook2 = FECCompaction()
    hook2._snap = lambda ag: calls.append(1) or {}
    hook2.git, hook2.git_ok = True, True
    hook2.after_step(agent, Step(0, "x", "cat pkg/mod.py", "", 0, "action"))
    hook2.after_step(agent, Step(1, "x", "cd /app && grep -rn foo .", "", 0, "action"))
    assert calls == []                     # read-only steps never run the git probe


def test_no_git_falls_back(tmp_path):
    hook = FECCompaction(100000, 2)
    run(tmp_path, hook, [b("echo hi > notes.txt")])
    assert hook.stats()["git_available"] is False and hook.git_edit_steps == 0
    assert any(e.kind == "EDIT" and e.paths == ["notes.txt"] for e in hook.ledger.events)


# ------------------------------------------------------------------ requirements vs. real diff
def test_requirement_helpers():
    items = requirement_items(TASK)
    assert len(items) == 3 and items[0].startswith("`alpha`")
    assert "alpha" in identifiers(items[0]) and "Gamma.reset_all" in identifiers(items[1])


def test_requirements_use_real_diff(tmp_path):
    git_repo(tmp_path)
    # the command text mentions Gamma.reset_all, but the diff never contains it
    hook = FECCompaction(100000, 2)
    _, _, agent = run(tmp_path, hook, [b("sed -i 's/return 1/return 2/' pkg/mod.py # also Gamma.reset_all")])
    text = "\n".join(hook._requirements(agent))
    assert "your current git diff" in text and "checkable requirement items" not in text
    assert "2. The `Gamma.reset_all`" in text            # not in the diff -> listed as missing
    assert "1. `alpha`" not in text                       # in the diff -> not listed
    assert "Files named in the task that you have not changed: docs/usage.md" in text


def test_present_matches_paths_files_and_dotted_names():
    changed = {"lib/ai/model/tokencount.go", "qutebrowser/keyinput/keyutils.py"}
    diff = "@@ -10,3 +10,4 @@ class KeySequence:\n     def __init__(self, *keys):\n+        self._check(keys)"
    assert present("lib/ai/model/tokencount.go", "", changed)
    assert present("tokencount.go", "", changed)
    assert not present("lib/ai/client.go", "", changed)
    assert present("KeySequence.__init__", diff, changed)
    assert present("_check", diff, changed) and not present("reset_all", diff, changed)


# ------------------------------------------------------------------ verify after compaction
def test_verify_reminder_once_when_no_test_after_compaction():
    hook = FECCompaction()
    hook.compaction_steps = [8]
    hook.tests = [{"step": 5, "cmd": "pytest", "rc": 1, "summary": "1 failed", "fails": []}]
    msg = hook.on_submit(None)
    assert msg and "compacted at step 8" in msg
    assert hook.on_submit(None) is None                              # only once


def test_no_reminder_when_tested_after_compaction_or_never_compacted():
    hook = FECCompaction()
    hook.compaction_steps = [8]
    hook.tests = [{"step": 9, "cmd": "go test ./...", "rc": 0, "summary": "ok", "fails": []}]
    assert hook.on_submit(None) is None
    assert FECCompaction().on_submit(None) is None


# ------------------------------------------------------------------ summary content and budget
def test_is_test_and_fail_lines():
    assert is_test("cd /app && go test ./internal/config/ -run TestLoad")
    assert is_test("cd /app/applications/mail && npx jest src/app/logic")
    assert not is_test("cd /app && cat tests/test_x.py")
    out = "\n".join([
        "=== RUN   TestLoad", "--- FAIL: TestLoad (0.01s)", "    --- FAIL: TestLoad/advanced (0.00s)",
        "FAIL\tgo.flipt.io/flipt/internal/config\t0.2s", "ok  \tgo.flipt.io/flipt/internal/tracing\t0.1s",
        "  ✕ should retry when API call fails (12 ms)", "  ✓ loads elements (3 ms)",
        "FAILED tests/test_a.py::test_x - assert 1 == 2", "E       assert 1 == 2",
        "gost/ubuntu_test.go:20:5: undefined: isKernelSourcePackage",
    ])
    fl = fail_lines(out)
    assert "--- FAIL: TestLoad (0.01s)" in fl and "FAIL go.flipt.io/flipt/internal/config 0.2s" in fl
    assert "✕ should retry when API call fails (12 ms)" in fl
    assert "FAILED tests/test_a.py::test_x - assert 1 == 2" in fl and "E assert 1 == 2" in fl
    assert "gost/ubuntu_test.go:20:5: undefined: isKernelSourcePackage" in fl
    assert not any(l.startswith(("ok", "=== RUN", "✓")) for l in fl)


def test_split_sections():
    assert split_sections(["x", "## A", "a1", "a2", "## B", "b1"]) == {"": ["x"], "## A": ["a1", "a2"], "## B": ["b1"]}


def test_summary_keeps_test_failures_plan_and_fits_budget(tmp_path):
    git_repo(tmp_path)
    replies = [b("sed -i 's/return 1/return 2/' pkg/mod.py"), b("python3 -m pytest -q tests/test_a.py")]
    replies += [b(f"cat pkg/mod.py && python3 -c \"print('{i}' * 1500)\"",
                  "I'll check the output next.") for i in range(12)]
    hook = FECCompaction(2000, 2)
    res, _, agent = run(tmp_path, hook, replies)
    assert hook.n_compactions >= 1 and hook.compaction_steps[0] >= 6
    s = hook.summary
    assert "## Latest test run before compaction (step 1)" in s
    assert "FAILED tests/test_a.py::test_x" in s                     # concrete failure kept
    assert "## Task requirements not yet reflected in your changes" in s
    assert "Gamma.reset_all" in s and "docs/usage.md" in s
    assert "## Where you were" in s and "I'll check the output next." in s
    assert "pkg/mod.py @step 0" in s                                 # modified files listed
    assert all(t <= 800 for t in hook.summary_tokens) and estimate_tokens(s) <= 800
    # no test after the last compaction -> the first submission was rejected with a reminder
    assert hook.verify_reminders == 1 and res["rejected_submits"] == 1 and res["exit_status"] == "submitted"
    assert any(st.kind == "rejected_submit" for st in agent.steps)


def test_budget_drops_command_log_first(tmp_path):
    git_repo(tmp_path)
    replies = [b("sed -i 's/return 1/return 2/' pkg/mod.py")]
    replies += [b(f"cat pkg/mod.py && python3 -c \"print('{i}' * 1500)\"") for i in range(12)]
    hook = FECCompaction(2000, 2, summary_budget=300)
    run(tmp_path, hook, replies)
    assert "## Command log" not in hook.summary
    assert "## Files you have modified" in hook.summary and "pkg/mod.py" in hook.summary
