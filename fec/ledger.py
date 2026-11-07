"""Evidence ledger: typed events extracted from agent steps, with order-based freshness.

Every executed command is split into segments (``&&``, ``||``, ``;``, newlines before a heredoc;
``cd`` segments are skipped) and each segment is classified, so ``cd /app && cat x.py`` is a READ
and a command that both edits and reads yields an EDIT and a READ event. Edits that the command
text does not reveal (python scripts, tools, ...) are added from git by ``FECCompaction``.

Freshness rules:
  READ(p)@t    -> STALE       if EDIT(p)@t' with t' > t
  SEARCH@t     -> STALE       if any hit file was edited later
  TEST(c)@t    -> SUPERSEDED  if the same normalised test command ran again later
               -> STALE       elif any file was edited after t
  RUN(c)@t     -> SUPERSEDED  if the same command ran again later
  NOTE@t       -> STALE_HINT  if a path it mentions was edited later
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PATH_RE = re.compile(r"(?<![\w$])((?:\.{0,2}/)?[\w.\-/]*\w\.(?:py|pyx|pyi|rst|txt|cfg|toml|ini|json|"
                     r"ya?ml|md|c|cc|cpp|h|hpp|js|ts|go|rs|sh|java|rb|html|css|sql|lean|v|cob|tex|r|R))\b")
TEST_RE = re.compile(r"\b(pytest|py\.test|runtests\.py|bin/test|tox\b|unittest|nosetests|go test|"
                     r"cargo test|npm (?:run )?test|make test|ctest|/tests/test\.sh)\b")
NOTE_RE = re.compile(r"(root cause|the (?:bug|issue|problem) is|i found|confirmed|this means|"
                     r"the fix|is caused by|works now|now passes|fails because)", re.I)

# ---------------------------------------------------------------- command segments
SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\n)\s*")
READ_SEG = re.compile(r"^(?:cat|head|tail|nl|less|more|wc|awk|bat|tf-view|sed\s+-n)\b")
SEARCH_SEG = re.compile(r"^(?:grep\s+(?:-\w*[rR]\w*|--recursive)|rg|ag|ack|find|fd|tree|ls|git\s+(?:grep|ls-files|log|show)|tf-search|tf-find-tests)\b")
EDIT_SEG = re.compile(r"(?:^|\s)(?:sed\s+-i|perl\s+-pi|tee\s|tf-edit|patch\b|git\s+apply|mv\s|cp\s|rm\s|touch\s|mkdir\s)")
REDIRECT = re.compile(r"(?<![0-9&>])>>?\s*['\"]?([^\s'\";&|)]+)")

# definitions kept on READ events (Python, Go, JS/TS)
SYMBOL_RE = re.compile(
    r"^\s*(?:\d+[:\s|]+)?\s*(?:"
    r"(?:async\s+)?def\s+(\w+)|class\s+(\w+)|func\s+(?:\([^)]*\)\s*)?(\w+)|type\s+(\w+)\s+(?:struct|interface)"
    r"|(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)|(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\("
    r")", re.M)


def norm_cmd(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd.strip())[:200]


def paths_in(text: str) -> list[str]:
    seen = []
    for p in PATH_RE.findall(text):
        p = p.lstrip("./") if p.startswith("./") else p
        if p not in seen:
            seen.append(p)
    return seen


def edit_targets(cmd: str) -> list[str]:
    """Files a command writes (heuristic)."""
    out = []
    for m in re.finditer(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*['\"]?([\w./-]+)", cmd):
        if m.group(1) not in ("/dev/null", "&1", "&2"):
            out.append(m.group(1))
    m = re.search(r"sed\s+-i[^\n]*?\s([\w./-]+\.\w+)\s*$", cmd.split("&&")[-1].strip()) or \
        re.search(r"sed\s+-i\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+([\w./-]+)", cmd)
    if m:
        out.append(m.group(1))
    for m in re.finditer(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]", cmd):
        out.append(m.group(1))
    for m in re.finditer(r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_text", cmd):
        out.append(m.group(1))
    if re.search(r"\b(patch|git apply)\b", cmd):
        out += paths_in(cmd)
    clean = []
    for p in out:
        p = p[2:] if p.startswith("./") else p
        if p not in clean and not p.startswith("/tmp/"):
            clean.append(p)
    return clean


def symbols_in(output: str, limit: int = 15) -> list[str]:
    out: list[str] = []
    for m in SYMBOL_RE.finditer(output):
        name = next((g for g in m.groups() if g), None)
        if name and name not in out:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def summarise_test(output: str, rc: int | None) -> tuple[str, list[str]]:
    """(one-line summary, failing test names) for common runners."""
    fails = []
    for line in output.splitlines():
        s = line.strip()
        m = re.match(r"^(FAILED|ERROR)\s+([^\s(]\S*)", s)
        if m:
            fails.append(m.group(2))
            continue
        m = re.match(r"^(FAIL|ERROR):\s+(\S+)\s+\(([\w.]+)\)", s)  # django / unittest
        if m:
            fails.append(f"{m.group(3)}.{m.group(2)}")
            continue
        m = re.match(r"^(test_\w+)\s+(F|E)$", s)  # sympy
        if m:
            fails.append(m.group(1))
    summary = None
    for pat in (r"=+ (.*?(?:passed|failed|error).*?) (?:in [\d.]+s )?=+", r"(Ran \d+ tests? in [\d.]+s)",
                r"(tests finished: .*)", r"^(OK(?: \(.*\))?)$", r"^(FAILED \(.*\))$", r"(\d+ passed.*)"):
        found = re.findall(pat, output, re.M)
        if found:
            summary = (summary + "; " if summary else "") + found[-1].strip()
            if len(summary) > 120:
                break
    if summary is None:
        summary = "exit code %s" % rc
    fails = list(dict.fromkeys(fails))
    return summary[:160], fails[:12]


def segments(cmd: str) -> list[str]:
    """Command segments, excluding heredoc bodies and `cd` segments."""
    lines = cmd.strip().split("\n")
    head = []
    for line in lines:
        head.append(line)
        if "<<" in line:
            break
    segs = [s.strip() for s in SPLIT_RE.split("\n".join(head)) if s.strip()]
    return [s for s in segs if not re.match(r"^(?:cd|pushd|popd)(?:\s|$)", s) and not s.startswith("#")]


def seg_kind(seg: str) -> str:
    if TEST_RE.search(seg):
        return "TEST"
    if EDIT_SEG.search(seg):
        return "EDIT"
    m = REDIRECT.search(seg)
    if m and not m.group(1).startswith(("/dev/", "/tmp/", "&")):
        return "EDIT"
    if READ_SEG.match(seg):
        return "READ"
    if re.match(r"^grep\b", seg):
        return "READ" if paths_in(seg) else "SEARCH"
    if SEARCH_SEG.match(seg):
        return "SEARCH"
    return "RUN"


def classify(cmd: str) -> tuple[str, dict[str, list[str]]]:
    """(primary kind, {kind: [segments]}). Priority: EDIT > TEST > READ > SEARCH > RUN."""
    by: dict[str, list[str]] = {}
    for s in segments(cmd) or [cmd.strip()]:
        by.setdefault(seg_kind(s), []).append(s)
    if "EDIT" not in by and edit_targets(cmd):  # python heredoc with open(..., 'w') etc.
        by.setdefault("EDIT", []).append(cmd)
    for k in ("EDIT", "TEST", "READ", "SEARCH", "RUN"):
        if k in by:
            return k, by
    return "RUN", by


@dataclass
class Event:
    t: int
    kind: str                 # READ | EDIT | TEST | RUN | SEARCH | NOTE
    cmd: str = ""
    paths: list[str] = field(default_factory=list)
    summary: str = ""
    fails: list[str] = field(default_factory=list)
    rc: int | None = None
    status: str = "FRESH"     # FRESH | STALE | SUPERSEDED | STALE_HINT
    reason: str = ""


class Ledger:
    def __init__(self, workdir: str = ""):
        self.events: list[Event] = []
        self.workdir = workdir.rstrip("/")

    def _rel(self, p: str) -> str:
        if self.workdir and p.startswith(self.workdir + "/"):
            return p[len(self.workdir) + 1:]
        return p[2:] if p.startswith("./") else p

    def observe(self, t: int, thought: str, cmd: str, output: str, rc: int | None) -> None:
        kind, by = classify(cmd)
        n = norm_cmd(cmd)
        if kind == "TEST":
            s, fails = summarise_test(output, rc)
            self.events.append(Event(t, "TEST", n, paths_in(cmd), s, fails, rc))
        elif kind == "EDIT":
            tg = [self._rel(p) for p in (edit_targets(cmd) or paths_in(" ".join(by.get("EDIT", []))))]
            ok = "ok" if rc == 0 else f"exit {rc}: {output.strip()[-120:]}"
            self.events.append(Event(t, "EDIT", n[:150], tg, ok, rc=rc))
        elif kind == "SEARCH":
            hits = [self._rel(p) for p in paths_in(output)][:10]
            self.events.append(Event(t, "SEARCH", n[:150], hits,
                                     f"{output.count(chr(10))} lines; files: {', '.join(hits[:6])}", rc=rc))
        elif kind == "RUN":
            tail = " | ".join(output.strip().splitlines()[-3:])[:200] if rc else output.strip()[:160]
            self.events.append(Event(t, "RUN", n[:150], paths_in(cmd), tail, rc=rc))
        # a READ event whenever any segment reads files (also alongside an EDIT / SEARCH)
        if "READ" in by:
            rp = []
            for s in by["READ"]:
                rp += [self._rel(p) for p in paths_in(s)]
            rp = list(dict.fromkeys(rp))
            syms = symbols_in(output)
            summ = f"{output.count(chr(10))} lines" + ("; defines " + ", ".join(syms) if syms else "")
            self.events.append(Event(t, "READ", " && ".join(by["READ"])[:150], rp, summ, rc=rc))
        for sent in re.split(r"(?<=[.!?])\s+", thought):
            if NOTE_RE.search(sent):
                self.events.append(Event(t, "NOTE", "", paths_in(sent), sent.strip()[:220]))
                break
        self.refresh()

    def add_git_edits(self, t: int, files: list[str]) -> None:
        """Record files git saw change in step t (unless an EDIT event already names them)."""
        named = {p for e in self.events if e.t == t and e.kind == "EDIT" for p in e.paths}
        new = [f for f in files if f not in named]
        if new:
            self.events.append(Event(t, "EDIT", "(detected by git)", new, "changed on disk", rc=0))
            self.refresh()

    def refresh(self) -> None:
        edits = [e for e in self.events if e.kind == "EDIT" and e.rc in (0, None)]
        for e in self.events:
            e.status, e.reason = "FRESH", ""
            later_edits = [x for x in edits if x.t > e.t]
            if e.kind == "READ":
                hit = [x for x in later_edits if set(x.paths) & set(e.paths)]
                if hit:
                    e.status, e.reason = "STALE", f"edited @step {hit[0].t}"
            elif e.kind == "SEARCH":
                hit = [x for x in later_edits if set(x.paths) & set(e.paths)]
                if hit:
                    e.status, e.reason = "STALE", f"a matched file was edited @step {hit[0].t}"
            elif e.kind in ("TEST", "RUN"):
                again = [x for x in self.events if x.kind == e.kind and x.t > e.t and x.cmd == e.cmd]
                if again:
                    e.status, e.reason = "SUPERSEDED", f"re-run @step {again[-1].t}"
                elif e.kind == "TEST" and later_edits:
                    e.status, e.reason = "STALE", f"code edited @step {later_edits[0].t}"
            elif e.kind == "NOTE":
                hit = [x for x in later_edits if set(x.paths) & set(e.paths)]
                if hit:
                    e.status, e.reason = "STALE_HINT", f"{hit[0].paths[0]} edited @step {hit[0].t}"

    def render(self, upto_step: int, max_fresh: int = 40, max_stale: int = 15) -> str:
        evs = [e for e in self.events if e.t < upto_step]
        fresh = [e for e in evs if e.status in ("FRESH", "STALE_HINT")]
        stale = [e for e in evs if e.status in ("STALE", "SUPERSEDED")]
        edited = {}
        for e in evs:
            if e.kind == "EDIT":
                for p in e.paths:
                    edited[p] = e.t
        lines = ["[CONTEXT COMPACTED - evidence ledger of the earlier steps; recent turns follow verbatim]"]
        if edited:
            lines.append("## Files you have modified (step of last edit)")
            lines += [f"- {p} @step {t}" for p, t in edited.items()]
        lines.append("## Current facts (still valid)")
        for e in fresh[-max_fresh:]:
            lines.append("- " + _fmt(e))
        if stale:
            lines.append("## Superseded / stale evidence (re-read or re-run if you need it)")
            for e in stale[-max_stale:]:
                lines.append(f"- {e.kind} @step {e.t}: {e.cmd[:90]} -> {e.status} ({e.reason})")
        return "\n".join(lines)


def _fmt(e: Event) -> str:
    if e.kind == "TEST":
        f = f"; failing: {', '.join(e.fails)}" if e.fails else ""
        return f"TEST @step {e.t}: `{e.cmd[:110]}` -> {e.summary}{f}"
    if e.kind == "EDIT":
        return f"EDIT @step {e.t}: {', '.join(e.paths) or e.cmd[:80]} ({e.summary})"
    if e.kind == "NOTE":
        tag = " [hint may be outdated: " + e.reason + "]" if e.status == "STALE_HINT" else ""
        return f"NOTE @step {e.t}: \"{e.summary}\"{tag}"
    return f"{e.kind} @step {e.t}: `{e.cmd[:110]}` -> {e.summary}"
