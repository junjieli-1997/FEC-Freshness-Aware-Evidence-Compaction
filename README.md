# FEC: Freshness-aware Evidence Compaction for Coding Agents

FEC is a **context-compaction method for LLM coding agents**. When the agent's working context grows
past a threshold, FEC replaces the older turns with an **evidence ledger**: a compact record of what the
agent has read, searched, edited and tested, where each entry is marked **FRESH**, **STALE** (e.g. a file
was read and later edited) or **SUPERSEDED** (e.g. a test was re-run). The agent therefore keeps the facts
it still needs, and is told which earlier observations it must not trust anymore.

This repository contains the agent harness, FEC, the CliffCompaction baseline, unit tests and the task
lists used in our experiments. It contains **no result files**.

---

## What FEC does

| Component | What it does |
|---|---|
| **Evidence ledger** | every command becomes typed events (READ / SEARCH / EDIT / TEST / RUN / NOTE) with FRESH / STALE / SUPERSEDED status |
| **Segment-wise classification** | commands are split on `&&`, `;`, `\|\|` and `cd` segments are skipped, so `cd /app && cat x.py` is a READ |
| **Git-based edit detection** | a `git status` + blob-hash snapshot after every non-read-only step, so edits made by Python scripts or `sed` are seen and earlier reads go STALE (falls back to command heuristics without git) |
| **Compaction trigger** | above 12K tokens, all but the last 3 turns are compacted; before step 6 only long old outputs are truncated to head + tail (unless the view exceeds 2× the threshold) |
| **Summary content** | modified files, the latest test run before the cut with its **verbatim failure lines**, task requirements **not yet reflected in the git diff**, current facts, stale evidence, the agent's own plan, a command log |
| **800-token summary budget** | fixed drop order: command log → stale evidence → oldest facts → requirement items → failure lines |
| **Verify after compaction** | a submission with no test run since the last compaction is rejected once with a reminder to re-run the tests |
| **Submission verify gate** | if the agent's last replies say the work is *unverified*, *incomplete*, or that failures are *expected / pre-existing / environmental*, the submission is rejected once and the agent must name each failing test and fix it or justify it from the task text |

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design.

## Repository layout

```
agentkit/                 minimal bash-only agent harness (mini-swe-agent style, one ```bash block per reply)
  agent.py                agent loop + hook API (before_query / after_step / on_submit / ...)
  env.py                  Docker container environment (docker CLI)
  models.py               Anthropic + OpenAI clients with token/cost accounting
  prompts.py              system / instance prompts
  runner.py               run one method over a task list, write result.json / trajectory.json / patch.diff
  report.py               summarise result directories as markdown
  methods.py              method registry
  verify_gate.py          submission verify gate (part of FEC)
  bench/                  SWE-bench Pro (V2 locked protocol), Terminal-Bench 2.1, SWE-bench Verified
fec/
  ledger.py               evidence ledger: events, classification, FRESH/STALE/SUPERSEDED
  compaction.py           threshold / cut logic shared by both methods + CliffCompaction baseline
  fec.py                  FECCompaction: trigger, git edit detection, summary, verify after compaction
  method.py               method factories (cliff, fec)
tasks/                    task lists (pro_hard51, tb_hard50, swe10, swe_hard10)
tests/                    unit tests (37, no API calls, no Docker needed)
scripts/                  example experiment scripts
docs/DESIGN.md            design notes and experiment summary
```

## Installation

Requirements: Linux, Python ≥ 3.11, Docker (the agent only ever runs commands inside containers).

```bash
git clone <this repo> fec && cd fec
python3 -m pip install -r requirements.txt
python3 -m pytest -q tests          # 37 tests, ~2 s
```

API keys are read from the environment:

```bash
export ANTHROPIC_API_KEY=...        # or CLAUDE_API_KEY
export OPENAI_API_KEY=...           # only for gpt-* models
```

## Benchmarks

The benchmark data is **not** included. Download it and either place it under `benchmarks/` or point the
environment variables below to it.

| Benchmark | Source | Expected location (default) | Override |
|---|---|---|---|
| SWE-bench Pro V2 (HARD-51) | `v2/` folder of [scaleapi/SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) | `benchmarks/swe-bench-pro/v2/` (with `tasks/` and `hard51_ids.txt`) | `SWEPRO_V2_DIR` |
| Terminal-Bench 2.1 | [harbor-framework/terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1) | `benchmarks/terminal-bench-2-1/tasks/` | `TB_TASKS_DIR` |
| SWE-bench Verified (optional) | test split parquet of `princeton-nlp/SWE-bench_Verified` | `benchmarks/swe-bench-verified/test-00000-of-00001.parquet` | `SWEBENCH_VERIFIED_PARQUET` |

Container images are pulled on demand (`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` for SWE-bench Pro,
the per-task images in `task.toml` for Terminal-Bench). SWE-bench Pro is run with the official **locked
protocol**: the agent runs with `--network none`, and its diff is replayed on a pristine image before the
official `tests/test.sh` produces the reward.

## Running

```bash
# FEC with Claude Haiku on SWE-bench Pro HARD-51 (2 tasks in parallel)
python3 -m agentkit.runner --bench pro --model claude-haiku-5-5 --method fec \
    --tasks tasks/pro_hard51.json --out results/claude_haiku_5_5/pro51/fec --workers 2

# baselines:               --method baseline | cliff
# Terminal-Bench 2.1:      --bench tb --tasks tasks/tb_hard50.json
# a few tasks only:        --only <task_id> [<task_id> ...]

python3 -m agentkit.report fec=results/claude_haiku_5_5/pro51/fec cliff=results/claude_haiku_5_5/pro51/cliff
```

Each task writes `result.json` (resolved, timing, token usage, cost, hook statistics), `trajectory.json`
(every step: reply, command, output, tokens, time) and `patch.diff`. Finished tasks are skipped, so an
interrupted run can simply be restarted. `scripts/run_pro51.sh` and `scripts/run_tb50.sh` run several
methods in sequence.

### Methods

| `--method` | Description |
|---|---|
| `baseline` | the bare agent ("pre-trained": model + harness, no method) |
| `cliff` | CliffCompaction: above the threshold, older turns become a mechanical summary |
| `fec` | FEC compaction + submission verify gate |

Supported models: `claude-haiku-5-5` (default) and other Claude models via the Anthropic SDK; `gpt-5-mini`,
`gpt-5-nano` and other OpenAI chat models via the OpenAI SDK. Prices used for cost accounting are in
`agentkit/models.py` (`PRICES`); unknown models are reported with cost 0.

### Main parameters

| Parameter | Default | Where |
|---|---|---|
| compaction threshold / recent turns kept verbatim | 12,000 tokens / 3 turns | `fec/method.py` |
| earliest compaction step / require an edit first | 6 / no | `FECCompaction(min_step, require_edit)` |
| hard ceiling (compaction forced above) | 2 × threshold | `FECCompaction(hard_factor)` |
| summary budget | 800 tokens | `FECCompaction(summary_budget)` |
| verify reminder after compaction | on, once per run | `FECCompaction(verify_after_compaction)` |
| verify gate | rejects once per run | `VerifyGateHook(max_rejections)` |
| step / wall limits | Pro: 100 steps, 1 h; TB: 60 steps + task timeout | `agentkit/bench/*.py` |

## Results in brief

One run per task, `claude-haiku-5-5`, SWE-bench Pro HARD-51 (51 tasks):

| Method | Resolved | Mean tokens / task | First compaction (step) | Mean summary (tokens) |
|---|---|---|---|---|
| baseline | 30 | 250 K | – | – |
| CliffCompaction | 33 | 177 K | 9.1 | 611 |
| FEC | 28 | 286 K | 10.5 | 750 (max 799) |

FEC keeps its summaries within the 800-token budget, but **no configuration is separable from single-run noise in resolve rate**: with one run per task,
differences of a few tasks on a 51-task benchmark are within run-to-run variation. The verify gate fired on
35 of 51 tasks and its extra steps raised token usage. Use several runs per configuration before drawing
conclusions. See `docs/DESIGN.md` for details.

## Tests

```bash
python3 -m pytest -q tests
```

The tests use a scripted fake model and a local temporary git repository; they need neither API keys nor
Docker.

## License

No license file is included yet; add one before publishing.
