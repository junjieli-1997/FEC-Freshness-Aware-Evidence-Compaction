"""Method registry: maps a method name to a factory that returns a MethodSpec (hooks + options)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MethodSpec:
    name: str
    hooks: list = field(default_factory=list)
    tool_cards: str = ""
    setup: Callable | None = None          # setup(env, bench) before the agent starts
    config_overrides: dict = field(default_factory=dict)
    instance_suffix: str = ""              # appended to the instance prompt


# method name -> (module, factory)
REGISTRY: dict[str, tuple[str, str]] = {
    "cliff": ("fec.method", "make_cliff"),   # CliffCompaction baseline
    "fec": ("fec.method", "make_fec"),       # FEC compaction + submission verify gate
}


def load_method(name: str, bench: str) -> MethodSpec:
    if name == "baseline":
        return MethodSpec("baseline")
    module, factory = REGISTRY[name]
    return getattr(importlib.import_module(module), factory)(bench)
