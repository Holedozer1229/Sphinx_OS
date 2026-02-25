"""
Artificial Intelligence subsystem for SphinxOS

Provides:
- ASI SphinxOS Advanced IIT v5.0: Integrated Information Theory engine
  combining Φ^max (minimum information partition search), GWT broadcast
  measure, and the composite Φ_total consciousness score used in block
  consensus.
- ASI SphinxOS Advanced IIT v6.0: Extends v5.0 with temporal-depth Φ (τ),
  per-concept Intrinsic Causal Power (ICP), exclusion-principle CES pruning,
  and a tripartite composite score (α·Φ_τ + β·GWT_S + γ·ICP_avg).
"""

from .iit_v5 import (
    ASISphinxOSIITv5,
    IITv5Engine,
    PhiStructure,
    CauseEffectRepertoire,
    Partition,
)
from .iit_v6 import (
    ASISphinxOSIITv6,
    IITv6Engine,
)

__version__ = "6.0.0"
__all__ = [
    "ASISphinxOSIITv5",
    "IITv5Engine",
    "ASISphinxOSIITv6",
    "IITv6Engine",
    "PhiStructure",
    "CauseEffectRepertoire",
    "Partition",
]
