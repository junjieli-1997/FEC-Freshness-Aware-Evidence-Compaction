"""Run one method on a task list:  python -m agentkit.runner --bench swe --method fec --out DIR

Results land in DIR/<task>/{result.json,trajectory.json,patch.diff}; finished tasks are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .agent import Agent, AgentConfig
from .methods import load_method
from .models import make_model
from .prompts import PRO_INSTANCE, SWE_INSTANCE, SYSTEM, TB_INSTANCE

TASK_LISTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")


def run_one(bench: str, task_id: str, method: str, model_name: str, outdir: str,
            overrides: dict | None = None) -> dict:
    tdir = os.path.join(outdir, task_id)
    rpath = os.path.join(tdir, "result.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            return json.load(f)
    os.makedirs(tdir, exist_ok=True)
    spec = load_method(method, "swe" if bench == "pro" else bench)
    model = make_model(model_name)
    res: dict = {"task": task_id, "bench": bench, "method": method, "model": model_name}
    t_start = time.time()
    timing: dict = {}
    env = agent = None
    try:
        if bench == "swe":
            from .bench import swebench as B

            item = B.load_instance(task_id)
            cfg = AgentConfig(SYSTEM, SWE_INSTANCE, step_limit=60, cost_limit=1.0, wall_limit=2400,
                              command_timeout=120)
        elif bench == "pro":
            from .bench import swebenchpro as B

            item = B.load_instance(task_id)
            cfg = AgentConfig(SYSTEM, PRO_INSTANCE, step_limit=100, cost_limit=1.5, wall_limit=3600,
                              command_timeout=180)
        else:
            from .bench import tbench as B

            item = B.load_task(task_id)
            cfg = AgentConfig(SYSTEM, TB_INSTANCE, step_limit=60, cost_limit=1.0,
                              wall_limit=B.agent_timeout(item), command_timeout=180)
        for k, v in {**spec.config_overrides, **(overrides or {})}.items():
            setattr(cfg, k, v)
        if spec.instance_suffix:
            cfg.instance_template = cfg.instance_template + "\n\n" + spec.instance_suffix
        t0 = time.time()
        env = B.make_env(item)
        timing["env_start_s"] = round(time.time() - t0, 1)
        if spec.setup:
            spec.setup(env, bench)
        agent = Agent(model, env, cfg, hooks=spec.hooks)
        t0 = time.time()
        res["agent"] = agent.run(B.task_text(item), workdir=env.workdir, tool_cards=spec.tool_cards,
                                 command_timeout=int(cfg.command_timeout))
        timing["agent_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        if bench in ("swe", "pro"):
            patch = B.extract_patch(env, item)
            res["patch_ref"] = "image_head" if getattr(env, "start_head", None) else "base_commit"
            with open(os.path.join(tdir, "patch.diff"), "w") as f:
                f.write(patch)
            env.close()
            env = None
            res["eval"] = B.evaluate(item, patch)
        else:
            res["eval"] = B.evaluate(env, item)
        timing["eval_s"] = round(time.time() - t0, 1)
        res["resolved"] = bool(res["eval"].get("resolved"))
    except Exception as e:  # noqa: BLE001 - record infra failures per task
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        res["resolved"] = False
    finally:
        if env is not None:
            env.close()
    timing["total_s"] = round(time.time() - t_start, 1)
    res["timing"] = timing
    if agent is not None:
        with open(os.path.join(tdir, "trajectory.json"), "w") as f:
            json.dump({"system": agent.system, "history": agent.history,
                       "steps": [s.__dict__ for s in agent.steps]}, f, indent=1)
    with open(rpath, "w") as f:
        json.dump(res, f, indent=1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["swe", "tb", "pro"], required=True)
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--tasks", default=None, help="JSON list; defaults to tasks/{pro_hard51,tb_hard50,swe10}.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    tasks_file = a.tasks or os.path.join(TASK_LISTS, {"pro": "pro_hard51.json", "tb": "tb_hard50.json"}.get(a.bench, "swe10.json"))
    with open(tasks_file) as f:
        tasks = json.load(f)
    if a.only:
        tasks = [t for t in tasks if t in a.only]
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, a.bench, t, a.method, a.model, a.out) for t in tasks]
        for fu in futs:
            r = fu.result()
            ag = r.get("agent", {})
            print(f"{r['task']:<40} resolved={r['resolved']!s:<5} steps={ag.get('n_steps')} "
                  f"cost=${ag.get('total_cost', 0):.4f} {r.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
