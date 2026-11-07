"""Terminal-Bench 2.1 runner: agent and verifier share the task container (as in Harbor)."""

from __future__ import annotations

import os
import tomllib

from ..env import ContainerEnv

# Terminal-Bench 2.1 task folders (one directory per task with task.toml); see README.md
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS_DIR = os.environ.get("TB_TASKS_DIR", os.path.join(_ROOT, "benchmarks", "terminal-bench-2-1", "tasks"))


def load_task(name: str) -> dict:
    d = os.path.join(TASKS_DIR, name)
    with open(os.path.join(d, "task.toml"), "rb") as f:
        cfg = tomllib.load(f)
    with open(os.path.join(d, "instruction.md")) as f:
        instruction = f.read()
    return {"name": name, "dir": d, "config": cfg, "instruction": instruction}


def make_env(task: dict) -> ContainerEnv:
    e = task["config"]["environment"]
    return ContainerEnv(e["docker_image"], workdir=None, memory_mb=e.get("memory_mb"),
                        cpus=e.get("cpus"))


def task_text(task: dict) -> str:
    return task["instruction"]


def agent_timeout(task: dict) -> float:
    return float(task["config"].get("agent", {}).get("timeout_sec", 900))


def evaluate(env, task: dict) -> dict:
    timeout = float(task["config"].get("verifier", {}).get("timeout_sec", 900))
    env.execute("rm -rf /tests && mkdir -p /tests /logs/verifier", timeout=60)
    cp = env.copy_in(os.path.join(task["dir"], "tests") + "/.", "/tests")
    if cp.returncode != 0:
        return {"reward": 0.0, "resolved": False, "error": "copy tests failed: " + cp.output[-500:]}
    run = env.execute("bash /tests/test.sh", timeout=timeout)
    rew = env.execute("cat /logs/verifier/reward.txt", timeout=30).output.strip()
    try:
        reward = float(rew.split()[0])
    except (ValueError, IndexError):
        reward = 0.0
    return {"reward": reward, "resolved": reward >= 1.0, "verifier_rc": run.returncode,
            "verifier_tail": run.output[-4000:]}
