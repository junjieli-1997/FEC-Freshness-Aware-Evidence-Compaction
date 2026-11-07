"""Execution environments.

``ContainerEnv`` drives a long-lived container through the docker (or podman) CLI.  Each
command runs in a fresh ``bash -lc`` (no persistent shell state, like mini-swe-agent).
``LocalEnv`` runs in a host directory and exists only for unit tests with scripted models.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass


@dataclass
class ExecResult:
    returncode: int
    output: str
    timed_out: bool = False


CLI = os.environ.get("CONTAINER_CLI", "docker")


def _run(args: list[str], timeout: float | None = None, input: str | None = None) -> ExecResult:
    try:
        # merge stderr into stdout at the source so interleaving is preserved (SWE-bench eval
        # scripts print their test-output markers via `set -x` on stderr)
        p = subprocess.run(args, input=input, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout, errors="replace")
        return ExecResult(p.returncode, p.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return ExecResult(124, out + f"\n[command timed out after {timeout}s]", True)


class ContainerEnv:
    def __init__(self, image: str, workdir: str | None = None, prefix: str = "",
                 env: dict[str, str] | None = None, memory_mb: int | None = None,
                 cpus: float | None = None, name: str | None = None, network: str | None = None):
        self.image = image
        self.workdir = workdir
        self.prefix = prefix
        self.env = env or {}
        self.name = name or f"ak-{uuid.uuid4().hex[:10]}"
        args = [CLI, "run", "-d", "--name", self.name]
        if workdir:
            args += ["-w", workdir]
        if memory_mb:
            args += ["--memory", f"{memory_mb}m"]
        if cpus:
            args += ["--cpus", str(cpus)]
        if network:
            args += ["--network", network]
        for k, v in self.env.items():
            args += ["-e", f"{k}={v}"]
        args += ["--entrypoint", "", image, "sleep", "infinity"]
        r = _run(args, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"container start failed: {r.output[-2000:]}")
        if not workdir:
            self.workdir = self.execute("pwd", timeout=30).output.strip() or "/"

    def execute(self, command: str, timeout: float = 120, cwd: str | None = None) -> ExecResult:
        full = f"{self.prefix}\n{command}" if self.prefix else command
        wd = cwd or self.workdir
        # no -w when the workdir is unknown: docker exec then uses the image's WORKDIR
        args = [CLI, "exec"] + (["-w", wd] if wd else []) + [self.name, "bash", "-c", full]
        r = _run(args, timeout=timeout)
        if r.timed_out:
            # the docker exec client was killed; make sure the process inside dies too
            _run([CLI, "exec", self.name, "bash", "-c",
                  "pkill -9 -f " + shlex.quote(command[:40]) + " || true"], timeout=30)
        return r

    def write_file(self, path: str, content: str) -> ExecResult:
        return _run([CLI, "exec", "-i", self.name, "bash", "-c",
                     f"mkdir -p $(dirname {shlex.quote(path)}) && cat > {shlex.quote(path)}"],
                    timeout=120, input=content)

    def copy_in(self, src: str, dst: str) -> ExecResult:
        return _run([CLI, "cp", src, f"{self.name}:{dst}"], timeout=600)

    def close(self) -> None:
        _run([CLI, "rm", "-f", self.name], timeout=300)


class LocalEnv:
    """Host-directory environment for tests only (commands come from scripted fake models)."""

    def __init__(self, workdir: str, prefix: str = ""):
        self.workdir = workdir
        self.prefix = prefix

    def execute(self, command: str, timeout: float = 120, cwd: str | None = None) -> ExecResult:
        full = f"{self.prefix}\n{command}" if self.prefix else command
        try:
            p = subprocess.run(["bash", "-c", full], cwd=cwd or self.workdir, capture_output=True,
                               text=True, timeout=timeout, errors="replace")
            return ExecResult(p.returncode, (p.stdout or "") + (p.stderr or ""))
        except subprocess.TimeoutExpired:
            return ExecResult(124, f"[command timed out after {timeout}s]", True)

    def write_file(self, path: str, content: str) -> ExecResult:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return ExecResult(0, "")

    def close(self) -> None:
        pass


def cli_available() -> bool:
    try:
        return subprocess.run([CLI, "info"], capture_output=True, timeout=30).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
