import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentkit.agent import Agent, AgentConfig, Hook  # noqa: E402
from agentkit.bench import logparse  # noqa: E402
from agentkit.env import LocalEnv  # noqa: E402
from agentkit.models import FakeModel, merge_roles  # noqa: E402


def bash(cmd: str, thought: str = "t") -> str:
    return f"THOUGHT: {thought}\n\n```bash\n{cmd}\n```"


SUBMIT = bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
CFG = dict(system_template="sys {tool_cards}", instance_template="do: {task}")


def make(tmp_path, replies, hooks=(), **kw):
    model = FakeModel(list(replies))
    cfg = AgentConfig(**CFG, **kw)
    return Agent(model, LocalEnv(str(tmp_path)), cfg, hooks=list(hooks)), model


def test_runs_commands_and_submits(tmp_path):
    agent, model = make(tmp_path, [bash("echo hi > a.txt"), bash("cat a.txt"), SUBMIT])
    res = agent.run("x", tool_cards="")
    assert res["exit_status"] == "submitted"
    assert res["n_actions"] == 2
    assert "hi" in agent.history[-2]["content"]  # observation of `cat a.txt`
    assert (tmp_path / "a.txt").read_text().strip() == "hi"


def test_format_error_then_recovery(tmp_path):
    agent, _ = make(tmp_path, ["no code here", SUBMIT])
    res = agent.run("x", tool_cards="")
    assert agent.steps[0].kind == "format_error"
    assert res["exit_status"] == "submitted"


def test_step_limit(tmp_path):
    agent, _ = make(tmp_path, [bash("true")] * 10, step_limit=3)
    assert agent.run("x", tool_cards="")["exit_status"] == "step_limit"


def test_hooks_order_and_effects(tmp_path):
    calls = []

    class H(Hook):
        name = "h"

        def before_query(self, agent, messages):
            calls.append("q")
            return messages + [{"role": "user", "content": "INJECTED"}]

        def before_action(self, agent, action):
            return "VETO" if "rm" in action else None

        def after_step(self, agent, step):
            return ["NOTE"] if step.kind == "action" else []

        def on_submit(self, agent):
            return "NOT YET" if agent.rejected_submits == 0 else None

        def stats(self):
            return {"calls": len(calls)}

    agent, model = make(tmp_path, [bash("rm -rf x"), bash("echo ok"), SUBMIT, SUBMIT], hooks=[H()])
    res = agent.run("x", tool_cards="")
    assert agent.steps[0].kind == "vetoed"
    assert agent.steps[2].kind == "rejected_submit"
    assert res["exit_status"] == "submitted" and res["rejected_submits"] == 1
    assert any(m["content"] == "NOTE" for m in agent.history)
    # the injected message reaches the model but never the stored history
    assert model.requests[0][1][-1]["content"] == "INJECTED"
    assert all(m["content"] != "INJECTED" for m in agent.history)
    assert res["hooks"]["h"]["calls"] == 4


def test_merge_roles():
    m = merge_roles([{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                     {"role": "assistant", "content": "c"}])
    assert len(m) == 2 and m[0]["content"] == "a\n\nb"


def test_long_observation_truncated(tmp_path):
    agent, _ = make(tmp_path, [bash("python3 -c \"print('x'*30000)\""), SUBMIT], max_obs_chars=1000)
    agent.run("x", tool_cards="")
    assert "characters elided" in agent.history[2]["content"]


LOG = """+ : '>>>>> Start Test Output'
test_a (mod.T) ... ok
test_b (mod.T) ... FAIL
PASSED tests/test_x.py::test_one
FAILED tests/test_x.py::test_two - AssertionError: boom
test_sym ok
+ : '>>>>> End Test Output'"""


def test_parsers_and_grade():
    sec = logparse.test_section(LOG)
    assert logparse.parse_django(sec) == {"test_a (mod.T)": "PASSED", "test_b (mod.T)": "FAILED"}
    assert logparse.parse_pytest(sec)["tests/test_x.py::test_two"] == "FAILED"
    assert logparse.parse_pytest_v2(sec)["tests/test_x.py::test_one"] == "PASSED"
    assert logparse.parse_sympy(sec)["test_sym"] == "PASSED"
    rep = logparse.grade(LOG, "parse_log_django", ["test_a (mod.T)"], ["test_b (mod.T)"])
    assert rep["f2p_success"] == ["test_a (mod.T)"] and rep["p2p_failure"] == ["test_b (mod.T)"]
    assert rep["resolved"] is False
    assert logparse.grade("no markers", "parse_log_pytest", ["a"], [])["resolved"] is False
