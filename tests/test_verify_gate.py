import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentkit.agent import Agent, AgentConfig  # noqa: E402
from agentkit.env import LocalEnv  # noqa: E402
from agentkit.models import FakeModel  # noqa: E402
from agentkit.verify_gate import VerifyGateHook, doubts  # noqa: E402


def bash(cmd: str, thought: str = "t") -> str:
    return f"THOUGHT: {thought}\n\n```bash\n{cmd}\n```"


SUBMIT = "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"


def test_doubts_patterns():
    assert doubts("The remaining TestLoad failures are old expectations that predate the new defaults.") == ["expected_failure"]
    assert "unverified" in doubts("I haven't run the test suite, so treat the work as unverified.")
    assert "incomplete" in doubts("The work is only partly done.")
    assert "expected_failure" in doubts("The redis failures are environmental.")
    assert "expected_failure" in doubts("I'm not editing test files, so I'll leave those tests as they are.")
    assert doubts("All 76 tests pass and every requirement is implemented. Submitting.") == []


def run(tmp_path, replies):
    hook = VerifyGateHook()
    agent = Agent(FakeModel(list(replies)), LocalEnv(str(tmp_path)), AgentConfig("sys", "do: {task}"), hooks=[hook])
    return hook, agent, agent.run("x")


def test_gate_rejects_once_and_lists_latest_failures(tmp_path):
    failing = "printf 'FAILED tests/test_a.py::test_x - assert 1 == 2\\n'; exit 1  # pytest"
    replies = [bash(failing),
               "THOUGHT: The remaining failure is expected; the test uses the old signature.\n\n" + SUBMIT,
               bash("echo fixed"),
               "THOUGHT: Fixed and verified.\n\n" + SUBMIT]
    hook, agent, res = run(tmp_path, replies)
    assert res["exit_status"] == "submitted" and hook.rejections == 1 and res["rejected_submits"] == 1
    msg = agent.history[agent.history.index({"role": "assistant", "content": replies[1]}) + 1]["content"]
    assert "some failures are expected" in msg and "FAILED tests/test_a.py::test_x" in msg
    assert hook.stats()["triggers"] == ["expected_failure"]


def test_gate_passes_confident_submission_and_only_fires_once(tmp_path):
    hook, _, res = run(tmp_path, ["THOUGHT: All tests pass.\n\n" + SUBMIT])
    assert res["exit_status"] == "submitted" and hook.rejections == 0
    unsure = "THOUGHT: This is unverified.\n\n" + SUBMIT
    hook, _, res = run(tmp_path, [unsure, unsure])
    assert res["exit_status"] == "submitted" and hook.rejections == 1   # second unsure submit goes through
