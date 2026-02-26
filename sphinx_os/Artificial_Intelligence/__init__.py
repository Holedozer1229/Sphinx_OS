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
- ASI SphinxOS Advanced IIT v7.0: Extends v6.0 with Octonionic Fano plane
  mechanics (Φ_fano) and non-abelian physics (Φ_nab), yielding a 5-term
  composite score (α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab).
- ASI SphinxOS Advanced IIT v8.0: Extends v7.0 with Quantum Gravity curvature
  (Φ_qg) and Holographic Ryu-Takayanagi entanglement entropy (Φ_holo), yielding
  a 7-term composite score and a QG-augmented consciousness-consensus condition.
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
from .iit_v7 import (
    ASISphinxOSIITv7,
    IITv7Engine,
    PhiStructureV7,
    ScoreDiagnostic,
    RiemannZeroEvidence,
    RiemannZeroProbe,
    FANO_LINES,
    FANO_POINTS,
    CLASSIFICATION_EXACT_ZERO,
    CLASSIFICATION_NEAR_ZERO,
    CLASSIFICATION_NONZERO,
    NEAR_ZERO_THRESHOLD_DEFAULT,
)
from .iit_v8 import (
    ASISphinxOSIITv8,
    IITv8Engine,
    PhiStructureV8,
)
from .riemann_proof import (
    RiemannHypothesisVerifier,
    VerificationReport,
    VERDICT_CONSISTENT,
    VERDICT_COUNTEREXAMPLE,
)

__version__ = "8.0.0"
__all__ = [
    "ASISphinxOSIITv5",
    "IITv5Engine",
    "ASISphinxOSIITv6",
    "IITv6Engine",
    "ASISphinxOSIITv7",
    "IITv7Engine",
    "ASISphinxOSIITv8",
    "IITv8Engine",
    "PhiStructure",
    "PhiStructureV7",
    "PhiStructureV8",
    "ScoreDiagnostic",
    "RiemannZeroEvidence",
    "RiemannZeroProbe",
    "RiemannHypothesisVerifier",
    "VerificationReport",
    "CauseEffectRepertoire",
    "Partition",
    "FANO_LINES",
    "FANO_POINTS",
    "CLASSIFICATION_EXACT_ZERO",
    "CLASSIFICATION_NEAR_ZERO",
    "CLASSIFICATION_NONZERO",
    "NEAR_ZERO_THRESHOLD_DEFAULT",
    "VERDICT_CONSISTENT",
    "VERDICT_COUNTEREXAMPLE",
]
