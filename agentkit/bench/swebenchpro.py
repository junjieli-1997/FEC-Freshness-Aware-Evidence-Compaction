"""SWE-bench Pro (V2, HARD-51) runner following the official locked protocol in
scaleapi/SWE-bench_Pro-os/v2:

* agent: instance image, repo at /app (or /testbed), **no network**, sees instruction.md;
* grade: the agent's diff is replayed on a pristine image (the README's authoritative "patch_replay"
  re-grade): apply diff -> copy tests/ to /tests -> bash /tests/test.sh -> /logs/verifier/reward.txt.
test.sh / run_script.sh / parser.py run only inside the container.
"""

from __future__ import annotations

import os
import re
import tomllib

from ..env import ContainerEnv

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# SWE-bench Pro V2 harness checkout (scaleapi/SWE-bench_Pro-os, folder v2); see README.md
SWEPRO_V2 = os.environ.get("SWEPRO_V2_DIR", os.path.join(_ROOT, "benchmarks", "swe-bench-pro", "v2"))
TASKS = os.path.join(SWEPRO_V2, "tasks")
HARD51 = os.path.join(SWEPRO_V2, "hard51_ids.txt")


def hard51() -> list[str]:
    return [l.strip() for l in open(HARD51) if l.strip()]


def load_instance(iid: str) -> dict:
    d = os.path.join(TASKS, iid)
    with open(os.path.join(d, "task.toml"), "rb") as f:
        cfg = tomllib.load(f)
    image = re.search(r"^FROM\s+(\S+)", open(os.path.join(d, "environment", "Dockerfile")).read(), re.M).group(1)
    return {"instance_id": iid, "dir": d, "config": cfg, "image": image,
            "instruction": open(os.path.join(d, "instruction.md")).read(),
            "gold_patch": open(os.path.join(d, "solution", "gold_patch.diff")).read()}


def _repo_dir(env) -> str:
    return "/app" if env.execute("test -d /app/.git", timeout=30).returncode == 0 else "/testbed"


def make_env(inst: dict) -> ContainerEnv:
    env = ContainerEnv(inst["image"], workdir=None, network="none")
    env.workdir = _repo_dir(env)
    env.execute(f"git config --global --add safe.directory {env.workdir}", timeout=30)
    env.start_head = env.execute("git rev-parse HEAD", timeout=30).output.strip().split()[-1]
    return env


def task_text(inst: dict) -> str:
    return inst["instruction"]


def extract_patch(env, inst: dict) -> str:
    r = env.execute(f"cd {env.workdir} && git add -A && git diff --cached --binary {env.start_head}", timeout=180)
    return r.output if r.returncode == 0 else ""


def evaluate(inst: dict, patch: str) -> dict:
    """Replay ``patch`` on a pristine image and run the official verifier."""
    if not patch.strip():
        return {"resolved": False, "reward": 0.0, "error": "empty patch", "applied": False}
    env = ContainerEnv(inst["image"], workdir=None)
    try:
        repo = _repo_dir(env)
        env.write_file("/tmp/agent.patch", patch)
        ap = env.execute(f"cd {repo} && git apply --binary -v /tmp/agent.patch", timeout=300)
        if ap.returncode != 0:
            return {"resolved": False, "reward": 0.0, "error": "patch did not apply", "applied": False,
                    "apply_log": ap.output[-2000:]}
        env.execute("rm -rf /tests && mkdir -p /tests /logs/verifier", timeout=60)
        env.copy_in(os.path.join(inst["dir"], "tests") + "/.", "/tests")
        timeout = float(inst["config"].get("verifier", {}).get("timeout_sec", 3000))
        run = env.execute(f"cd {repo} && bash /tests/test.sh", timeout=timeout)
        rew = env.execute("cat /logs/verifier/reward.txt", timeout=30).output.strip()
        try:
            reward = float(rew.split()[0])
        except (ValueError, IndexError):
            reward = 0.0
        return {"resolved": reward >= 1.0, "reward": reward, "applied": True, "verifier_tail": run.output[-3000:]}
    finally:
        env.close()
