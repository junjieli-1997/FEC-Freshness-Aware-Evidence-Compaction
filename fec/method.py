from agentkit.methods import MethodSpec
from agentkit.verify_gate import VerifyGateHook

from .compaction import CliffCompaction
from .fec import FECCompaction

THRESHOLD = 12000
KEEP_RECENT = 3


def make_cliff(bench: str) -> MethodSpec:
    """CliffCompaction baseline: mechanical summary of the older turns above the threshold."""
    return MethodSpec("cliff", hooks=[CliffCompaction(THRESHOLD, KEEP_RECENT)])


def make_fec(bench: str) -> MethodSpec:
    """FEC = evidence-ledger compaction + the submission verify gate."""
    return MethodSpec("fec", hooks=[FECCompaction(THRESHOLD, KEEP_RECENT), VerifyGateHook()])
