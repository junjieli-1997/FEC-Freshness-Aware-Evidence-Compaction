# FEC design notes

## 1. Problem

A coding agent working on a real repository accumulates long tool outputs: file listings, file
contents, test logs. Once the context passes a few tens of thousands of tokens, older outputs must be
dropped or summarised. Two things go wrong with naive summarisation:

1. **Staleness.** A summary that says "`foo()` returns `None` on line 42" is wrong once the agent has edited
   that file. The agent keeps reasoning from facts that are no longer true.
2. **Loss of what matters for finishing.** The concrete failing test names, the requirement items that are
   still missing, and the agent's own plan are exactly what it needs at the end. A generic summary drops
   them.

CliffCompaction, a common baseline, replaces the older turns with a fixed-format summary of thoughts and
commands. It does not track whether a fact is still valid.

## 2. Evidence ledger

Every executed command becomes one or more **events**:

| Event | Created from | Becomes STALE / SUPERSEDED when |
|---|---|---|
| READ (paths, symbols defined) | `cat`, `sed -n`, `head`, `nl`, `grep FILE` … | a later EDIT touches one of its paths |
| SEARCH (matched files) | `grep -r`, `rg`, `find`, `ls` … | a matched file is edited |
| EDIT (paths) | `sed -i`, redirections, heredocs, Python `open(...,'w')`, **git-detected changes** | – |
| TEST (summary, failing tests) | pytest, go test, jest, npm/yarn test, tests/test.sh … | re-run (SUPERSEDED) or code edited after it (STALE) |
| RUN | anything else | re-run |
| NOTE | sentences in the agent's thought such as "the root cause is …" | a file it mentions is edited (hint may be outdated) |

At compaction time, the turns older than the last `keep_recent` turns are replaced by:

```
[CONTEXT COMPACTED - evidence ledger of the earlier steps; recent turns follow verbatim]
## Files you have modified (step of last edit)
## Latest test run before compaction (step N)
## Task requirements not yet reflected in your changes
## Current facts (still valid)
## Superseded / stale evidence (re-read or re-run if you need it)
## Where you were (your own plan just before compaction)
## Command log of the compacted turns (latest)
```

The full, unmodified conversation is always kept in `agent.history`; only the view sent to the model is
rewritten (`before_query` hook), so compaction never loses data on disk.

## 3. FEC details

Code: `fec/ledger.py` (events and freshness), `fec/fec.py` (`FECCompaction`), `fec/compaction.py`
(threshold / cut logic shared with the `CliffCompaction` baseline), `agentkit/verify_gate.py`.

### 3.1 Command classification and edit detection

* A command is split into segments (`&&`, `||`, `;`, newlines before a heredoc); `cd` segments and heredoc
  bodies are skipped and each segment is classified. `grep` on explicit files is a READ, recursive grep /
  `find` / `ls` are SEARCH. A command that both edits and reads yields an EDIT and a READ event.
* After every step that is not purely read-only, FEC snapshots `git status` + blob hashes of the changed
  files and diffs it with the previous snapshot. Edits made by Python scripts, `sed`, tools … are therefore
  all recorded, and earlier READs of those files become STALE. Without a git work tree it falls back to
  command heuristics.
* The requirement section compares the bullet items of the task text against the real `git diff`
  (changed lines and hunk headers) and lists only items whose identifiers do not appear yet, plus files
  named in the task that were not changed.

### 3.2 Compaction trigger

```
tokens(view) ≤ threshold                        → no change
steps < min_step (6) and tokens ≤ 2×threshold   → truncate old outputs > 3,000 chars to head+tail
otherwise                                       → compact everything before the last 3 turns
```

### 3.3 Summary budget (800 tokens)

Sections are rendered in priority order, then reduced in this order until the summary fits:
1. command log (entry by entry, down to none);
2. stale evidence (replaced by one line);
3. oldest facts (down to 6);
4. requirement items (down to 3);
5. failure lines (down to 5);
6. last resort: facts to 3, plan to 2.

### 3.4 Verify after compaction

`after_step` records every test command (step, command, exit code, summary, failure lines). In
`on_submit`, if a compaction has happened and no test ran at or after the step following the last
compaction, the submission is rejected once with a reminder to re-run the tests.

### 3.5 Submission verify gate

`agentkit/verify_gate.py`, the second hook of the `fec` method. In `on_submit`, the last two replies (code blocks
removed) are matched against three groups of phrases:

| Group | Examples |
|---|---|
| unverified | unverified, haven't run / verified, could not run, untested |
| incomplete | partly done, partially, incomplete, not met, known gaps, follow-up, stub, placeholder |
| expected failure | failures are expected, pre-existing, predate, old signature / expectation / wording, unrelated to my change, environmental, leave those tests, not editing test files, conflicts with the spec |

On a match the submission is rejected once; the message asks for every failing or unverified test by name,
a fix or a justification quoted from the task, and a test run. The failure lines of the latest failing test
run are appended. The harness allows at most 2 rejected submissions per run in total.

## 4. What the experiments showed (SWE-bench Pro HARD-51, Claude Haiku, 1 run per task)

| Method | Resolved | Mean tokens / task | First compaction (step) | Mean summary (tokens) |
|---|---|---|---|---|
| baseline | 30 / 51 | 250 K | – | – |
| CliffCompaction | 33 / 51 | 177 K | 9.1 | 611 |
| FEC | 28 / 51 | 286 K | 10.5 | 750 (max 799) |

* FEC never exceeded its summary budget (max 799 tokens).
* With one run per task, resolve-rate differences of this size are within single-run noise; no method is
  separable from the others.
* The verify gate fired on 35/51 tasks (mostly "unverified"); it added steps and tokens without a
  measurable gain in resolve rate.
* Lessons:
  * a gate keyed on self-reported doubt is too broad for a model that routinely and honestly says
    "I haven't run X";
  * compaction quality matters less than *when* compaction happens;
  * method comparisons on 50-task benchmarks need several runs per configuration.

## 5. Known limitations

* The phrase lists of the verify gate are English-only and tuned on Claude Haiku transcripts.
* Git-based edit detection needs a git work tree; on most Terminal-Bench tasks FEC falls back to command
  heuristics.
* The requirement checklist detects *missing* work (identifiers absent from the diff), not *wrong* work.
* Token estimates use `len(text) / 4`.
