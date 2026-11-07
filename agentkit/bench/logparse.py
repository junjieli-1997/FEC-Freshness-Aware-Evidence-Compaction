"""SWE-bench test-log parsing and grading, re-implemented after swebench/harness (log_parsers, grading).

Only the parsers needed by the repositories in our task list are included.
"""

from __future__ import annotations

import re

STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL")
START, END = ">>>>> Start Test Output", ">>>>> End Test Output"
_SKIP_SUMMARY = re.compile(r"^\[\d+\]$")


def _skip_summary(status: str, name: str) -> bool:
    return status == "SKIPPED" and bool(_SKIP_SUMMARY.match(name))


def parse_pytest(log: str) -> dict[str, str]:
    m = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            tc = line.split()
            if len(tc) <= 1 or _skip_summary(tc[0], tc[1]):
                continue
            m[tc[1]] = tc[0]
    return m


def parse_pytest_options(log: str) -> dict[str, str]:
    opt = re.compile(r"(.*?)\[(.*)\]")
    m = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            tc = line.split()
            if len(tc) <= 1 or _skip_summary(tc[0], tc[1]):
                continue
            g = opt.search(tc[1])
            if g:
                main, option = g.groups()
                if option.startswith("/") and not option.startswith("//") and "*" not in option:
                    option = "/" + option.split("/")[-1]
                name = f"{main}[{option}]"
            else:
                name = tc[1]
            m[name] = tc[0]
    return m


def parse_pytest_v2(log: str) -> dict[str, str]:
    m = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    tr = str.maketrans("", "", escapes)
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(tr)
        if any(line.startswith(s) for s in STATUSES):
            if line.startswith("FAILED"):
                line = line.split(" - ", 1)[0]
            tc = line.split()
            if len(tc) >= 2 and not _skip_summary(tc[0], tc[1]):
                m[" ".join(tc[1:])] = tc[0]
        elif any(line.endswith(s) for s in STATUSES):
            tc = line.split()
            if len(tc) >= 2:
                m[" ".join(tc[:-1])] = tc[-1]
    return m


def parse_django(log: str) -> dict[str, str]:
    m = {}
    prev = None
    for line in log.split("\n"):
        line = line.strip()
        if "--version is equivalent to version" in line:
            m["--version is equivalent to version"] = "PASSED"
        if " ... " in line:
            prev = line.split(" ... ")[0]
        for suf in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suf):
                if line.startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                m[line.rsplit(suf, 1)[0]] = "PASSED"
                break
        if " ... skipped" in line:
            m[line.split(" ... skipped")[0]] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            m[line.split(" ... FAIL")[0]] = "FAILED"
        if line.startswith("FAIL:"):
            m[line.split()[1].strip()] = "FAILED"
        if line.endswith(" ... ERROR"):
            m[line.split(" ... ERROR")[0]] = "ERROR"
        if line.startswith("ERROR:"):
            m[line.split()[1].strip()] = "ERROR"
        if line.lstrip().startswith("ok") and prev is not None:
            m[prev] = "PASSED"
    for pat in (r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
                r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
                r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$"):
        for g in re.finditer(pat, log, re.MULTILINE):
            m[g.group(1)] = "PASSED"
    return m


def parse_sympy(log: str) -> dict[str, str]:
    m = {}
    for g in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        m[f"{g[1]}.py:{g[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                m[line.split()[0]] = "ERROR"
            if line.endswith(" F"):
                m[line.split()[0]] = "FAILED"
            if line.endswith(" ok"):
                m[line.split()[0]] = "PASSED"
    return m


PARSERS = {
    "parse_log_pytest": parse_pytest,
    "parse_log_xarray": parse_pytest,
    "parse_log_requests": parse_pytest_options,
    "parse_log_pylint": parse_pytest_options,
    "parse_log_scikit": parse_pytest_v2,
    "parse_log_sphinx": parse_pytest_v2,
    "parse_log_django": parse_django,
    "parse_log_sympy": parse_sympy,
}


def test_section(log: str) -> str | None:
    if START not in log or END not in log:
        return None
    return log.split(START, 1)[1].split(END, 1)[0]


def grade(log: str, parser_name: str, fail_to_pass: list[str], pass_to_pass: list[str]) -> dict:
    """Return a SWE-bench style report: resolved + per-list success/failure."""
    section = test_section(log)
    if section is None:
        return {"resolved": False, "error": "test markers not found", "f2p_success": [],
                "f2p_failure": list(fail_to_pass), "p2p_success": [], "p2p_failure": list(pass_to_pass)}
    status = PARSERS[parser_name](section)

    def ok(t: str) -> bool:
        return status.get(t) in ("PASSED", "XFAIL")

    rep = {
        "f2p_success": [t for t in fail_to_pass if ok(t)],
        "f2p_failure": [t for t in fail_to_pass if not ok(t)],
        "p2p_success": [t for t in pass_to_pass if ok(t)],
        "p2p_failure": [t for t in pass_to_pass if not ok(t)],
    }
    rep["resolved"] = not rep["f2p_failure"] and not rep["p2p_failure"]
    return rep
