"""Summarise result dirs:  python -m agentkit.report name=DIR [name=DIR ...]  (markdown to stdout)."""

from __future__ import annotations

import json
import os
import sys


def load(d: str) -> list[dict]:
    out = []
    if not os.path.isdir(d):
        return out
    for t in sorted(os.listdir(d)):
        p = os.path.join(d, t, "result.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if "error" not in r]
    ag = [r["agent"] for r in ok if "agent" in r]

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    p2p = [len(r["eval"].get("p2p_failure", [])) for r in ok if r.get("bench") == "swe"
           and r.get("eval", {}).get("applied")]
    return {
        "n": n,
        "resolved": sum(r["resolved"] for r in rows),
        "infra_errors": n - len(ok),
        "mean_cost": mean(a["total_cost"] for a in ag),
        "total_cost": sum(a["total_cost"] for a in ag),
        "mean_steps": mean(a["n_steps"] for a in ag),
        "mean_peak_ctx": mean(a["peak_context_tokens"] for a in ag),
        "p2p_fail_total": sum(p2p) if p2p else None,
        "submitted": sum(a["exit_status"] == "submitted" for a in ag),
    }


def table(named: dict[str, list[dict]]) -> str:
    lines = ["| method | resolved | submitted | mean steps | mean peak ctx (tok) | mean cost ($) | total cost ($) | P2P failures | infra errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, rows in named.items():
        s = summary(rows)
        lines.append(f"| {name} | {s['resolved']}/{s['n']} | {s['submitted']} | {s['mean_steps']:.1f} | "
                     f"{s['mean_peak_ctx']:.0f} | {s['mean_cost']:.4f} | {s['total_cost']:.3f} | "
                     f"{'' if s['p2p_fail_total'] is None else s['p2p_fail_total']} | {s['infra_errors']} |")
    return "\n".join(lines)


def per_task(named: dict[str, list[dict]]) -> str:
    names = list(named)
    tasks = sorted({r["task"] for rows in named.values() for r in rows})
    idx = {n: {r["task"]: r for r in rows} for n, rows in named.items()}
    lines = ["| task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for t in tasks:
        cells = []
        for n in names:
            r = idx[n].get(t)
            cells.append("—" if r is None else ("ERR" if "error" in r else ("✅" if r["resolved"] else "❌")))
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    named = {}
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        named[k] = load(v)
    print(table(named))
    print()
    print(per_task(named))


if __name__ == "__main__":
    main()
