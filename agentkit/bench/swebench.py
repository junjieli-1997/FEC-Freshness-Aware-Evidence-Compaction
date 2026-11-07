"""SWE-bench Verified runner: agent in the official instance image, grading in a fresh container."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from ..env import ContainerEnv
from . import logparse

# SWE-bench Verified test split (parquet from the princeton-nlp/SWE-bench_Verified dataset); see README.md
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.environ.get("SWEBENCH_VERIFIED_PARQUET", os.path.join(_ROOT, "benchmarks", "swe-bench-verified", "test-00000-of-00001.parquet"))
PREFIX = "source /opt/miniconda3/bin/activate >/dev/null 2>&1 && conda activate testbed >/dev/null 2>&1"
WORKDIR = "/testbed"


@lru_cache(maxsize=1)
def _df():
    import pandas as pd

    return pd.read_parquet(DATASET)


def load_instance(instance_id: str) -> dict:
    df = _df()
    r = df[df.instance_id == instance_id].iloc[0]
    d = r.to_dict()
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        v = d[k]
        d[k] = list(v) if not isinstance(v, str) else json.loads(v)
    return d


def make_env(inst: dict) -> ContainerEnv:
    env = ContainerEnv(inst["image"], workdir=WORKDIR, prefix=PREFIX)
    # The images carry an extra environment-setup commit on top of base_commit (file modes,
    # and for sphinx pinned deps in setup.py / tox.ini). Diff against the image's own HEAD so
    # the patch holds only the agent's changes and applies cleanly in a fresh container.
    env.execute("git config --global --add safe.directory /testbed", timeout=30)
    env.start_head = env.execute("git rev-parse HEAD", timeout=30).output.strip().split()[-1]
    return env


def task_text(inst: dict) -> str:
    return inst["problem_statement"]


def extract_patch(env, inst: dict) -> str:
    ref = getattr(env, "start_head", None) or inst["base_commit"]
    r = env.execute(f"cd {WORKDIR} && git add -A && git diff --cached {ref}", timeout=120)
    return r.output if r.returncode == 0 else ""


def agent_only_patch(inst: dict, patch_vs_base: str) -> str:
    """Strip the image's setup commit from a patch that was diffed against base_commit."""
    if not patch_vs_base.strip():
        return ""
    env = ContainerEnv(inst["image"], workdir=WORKDIR)
    try:
        env.execute("git config --global --add safe.directory /testbed", timeout=30)
        head = env.execute("git rev-parse HEAD", timeout=30).output.strip().split()[-1]
        env.write_file("/tmp/old.patch", patch_vs_base)
        r = env.execute(f"git checkout -q {inst['base_commit']} -- . && git apply /tmp/old.patch && "
                        f"git add -A && git diff --cached {head}", timeout=300)
        if r.returncode == 0:
            return r.output
        # not a base_commit-relative patch: it was already diffed against the image HEAD
        env.execute(f"git reset -q --hard {head} && git clean -fdq", timeout=120)
        if env.execute("git apply --check /tmp/old.patch", timeout=120).returncode == 0:
            return patch_vs_base
        raise RuntimeError("could not rebuild agent patch: " + r.output[-500:])
    finally:
        env.close()


def evaluate(inst: dict, patch: str) -> dict:
    """Apply ``patch`` in a fresh container and run the official eval script."""
    if not patch.strip():
        return {"resolved": False, "error": "empty patch", "applied": False}
    env = ContainerEnv(inst["image"], workdir=WORKDIR)
    try:
        env.write_file("/tmp/agent.patch", patch)
        ap = env.execute("git apply -v /tmp/agent.patch || patch --batch --fuzz=5 -p1 -i /tmp/agent.patch",
                         timeout=120)
        if ap.returncode != 0:
            return {"resolved": False, "error": "patch did not apply", "applied": False,
                    "apply_log": ap.output[-3000:]}
        env.write_file("/eval.sh", inst["eval_script"])
        run = env.execute("bash /eval.sh", timeout=1800)
        rep = logparse.grade(run.output, inst["log_parser"], inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"])
        rep["applied"] = True
        rep["eval_log_tail"] = run.output[-4000:]
        return rep
    finally:
        env.close()
