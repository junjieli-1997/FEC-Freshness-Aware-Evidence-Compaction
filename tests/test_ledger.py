import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from fec.ledger import Ledger, classify, edit_targets, segments, summarise_test, symbols_in  # noqa: E402


# ------------------------------------------------------------------ classification
def test_segments_skip_cd_and_heredoc_body():
    assert segments("cd /app && cat a.py && echo ---- && cat b.py") == ["cat a.py", "echo ----", "cat b.py"]
    assert segments("cd /app && python3 - <<'EOF'\nimport os\nprint(1)\nEOF") == ["python3 - <<'EOF'"]


def test_classify():
    assert classify("python -m pytest tests/test_a.py -x")[0] == "TEST"
    assert classify("cd tests && ./runtests.py forms_tests")[0] == "TEST"
    assert classify("cd /app && python -m pytest -q tests/test_x.py")[0] == "TEST"
    assert classify("sed -i 's/a/b/' src/mod.py")[0] == "EDIT"
    assert classify("cat <<'EOF' > src/new.py\nx=1\nEOF")[0] == "EDIT"
    assert classify("cd /app && python3 - <<'EOF'\nopen('lib/x.py','w').write('1')\nEOF")[0] == "EDIT"
    assert classify("cd /app && cat lib/ai/chat.go")[0] == "READ"
    assert classify("cd /app && sed -n '120,260p' lib/ai/model/agent.go")[0] == "READ"
    assert classify("cd /app && grep -n 'Complete(' lib/ai/chat.go")[0] == "READ"
    assert classify("cd /app && grep -rn 'Complete(' lib")[0] == "SEARCH"
    assert classify("cd /app && ls lib/ai && find . -name '*.go' | head")[0] == "SEARCH"
    assert classify("cd /app && python3 reproduce.py 2>&1 | tail -5")[0] == "RUN"
    kind, by = classify("cd /app && sed -i 's/a/b/' x.py && cat x.py")
    assert kind == "EDIT" and "READ" in by


def test_edit_targets():
    assert edit_targets("sed -i 's/a/b/' src/mod.py") == ["src/mod.py"]
    assert edit_targets("python - <<'EOF'\nopen('pkg/x.py','w').write('1')\nEOF") == ["pkg/x.py"]
    assert edit_targets("python r.py > /dev/null 2>&1") == []


def test_summarise_test_outputs():
    s, f = summarise_test("FAILED tests/t.py::test_x - AssertionError\n==== 1 failed, 3 passed in 0.2s ====", 1)
    assert "1 failed, 3 passed" in s and f == ["tests/t.py::test_x"]
    s, f = summarise_test("FAIL: test_y (forms.tests.T)\nRan 5 tests in 0.1s\nFAILED (failures=1)", 1)
    assert f == ["forms.tests.T.test_y"] and "Ran 5 tests" in s


def test_symbols_in():
    out = "  12\tdef foo(x):\nclass Bar:\nfunc (s *S) Run() {\nexport function baz() {}\n"
    assert symbols_in(out) == ["foo", "Bar", "Run", "baz"]


# ------------------------------------------------------------------ ledger
def test_reads_get_symbols_and_relative_paths():
    L = Ledger(workdir="/app")
    L.observe(0, "look", "cd /app && cat /app/lib/mod.py", "def alpha():\n    pass\nclass Beta:\n    pass\n", 0)
    ev = [e for e in L.events if e.kind == "READ"][0]
    assert ev.paths == ["lib/mod.py"] and "defines alpha, Beta" in ev.summary


def test_freshness_rules():
    L = Ledger()
    L.observe(0, "look", "cat src/mod.py", "a\nb\n", 0)
    L.observe(1, "test", "python -m pytest tests/t.py", "FAILED tests/t.py::test_x\n= 1 failed in 1s =", 1)
    L.observe(2, "The root cause is in src/mod.py.", "sed -i 's/a/b/' src/mod.py", "", 0)
    L.observe(3, "rerun", "python -m pytest tests/t.py", "= 1 passed in 1s =", 0)
    ev = {(e.t, e.kind): e for e in L.events}
    assert ev[(0, "READ")].status == "STALE"
    assert ev[(1, "TEST")].status == "SUPERSEDED"
    assert ev[(3, "TEST")].status == "FRESH"
    assert ev[(2, "NOTE")].status == "FRESH"  # its file was edited in the same step, not later
    text = L.render(10)
    assert "Current facts" in text and "SUPERSEDED" in text and "src/mod.py @step 2" in text
    # the superseded failure must not appear among the current facts
    current = text.split("## Current facts")[1].split("## Superseded")[0]
    assert "failing: tests/t.py::test_x" not in current


def test_git_edits_mark_reads_stale():
    L = Ledger()
    L.observe(0, "look", "cat pkg/mod.py", "x\n", 0)
    L.observe(1, "fix", "python3 fix.py", "", 0)        # the command text names no file
    assert L.events[0].status == "FRESH"
    L.add_git_edits(1, ["pkg/mod.py"])
    assert L.events[0].status == "STALE"
    assert L.events[-1].cmd == "(detected by git)" and L.events[-1].paths == ["pkg/mod.py"]
