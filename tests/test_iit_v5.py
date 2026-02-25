"""
Tests for ASI SphinxOS Advanced IIT v5.0

Validates:
- IITv5Engine cause-effect repertoire computation
- Minimum Information Partition (MIP) search and Φ^max
- GWT broadcast score
- Composite Φ_total and consciousness-consensus condition
- ASISphinxOSIITv5 high-level API (drop-in for IITQuantumConsciousnessEngine)
- Legacy phi_score mapping
"""

import math
import pytest
import numpy as np

from sphinx_os.Artificial_Intelligence.iit_v5 import (
    ASISphinxOSIITv5,
    IITv5Engine,
    CauseEffectRepertoire,
    Partition,
    PhiStructure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return IITv5Engine(alpha=0.7, beta=0.3, consciousness_threshold=0.5)


@pytest.fixture
def asi():
    return ASISphinxOSIITv5(alpha=0.7, beta=0.3, n_nodes=3)


# ---------------------------------------------------------------------------
# IITv5Engine unit tests
# ---------------------------------------------------------------------------

class TestIITv5Engine:

    def test_cause_repertoire_is_normalised(self, engine):
        """Cause repertoire must sum to 1."""
        n_nodes = 3
        n_states = 2 ** n_nodes
        dist = np.full(n_states, 1.0 / n_states)
        T = engine._build_transition_matrix(dist, n_nodes)

        rep = engine.compute_cause_repertoire(
            mechanism=(0,), purview=(1, 2), T=T, dist=dist, n_nodes=n_nodes
        )
        assert abs(rep.repertoire.sum() - 1.0) < 1e-9
        assert rep.direction == "cause"
        assert len(rep.repertoire) == 2 ** 2  # |purview| = 2

    def test_effect_repertoire_is_normalised(self, engine):
        """Effect repertoire must sum to 1."""
        n_nodes = 3
        n_states = 2 ** n_nodes
        dist = np.full(n_states, 1.0 / n_states)
        T = engine._build_transition_matrix(dist, n_nodes)

        rep = engine.compute_effect_repertoire(
            mechanism=(0,), purview=(1,), T=T, dist=dist, n_nodes=n_nodes
        )
        assert abs(rep.repertoire.sum() - 1.0) < 1e-9
        assert rep.direction == "effect"
        assert len(rep.repertoire) == 2 ** 1

    def test_phi_concept_non_negative(self, engine):
        """phi for any concept must be ≥ 0."""
        n_nodes = 3
        n_states = 2 ** n_nodes
        dist = np.array([0.3, 0.1, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05])
        T = engine._build_transition_matrix(dist, n_nodes)

        concept = engine.compute_phi_concept(
            mechanism=(0, 1), T=T, dist=dist, n_nodes=n_nodes
        )
        assert concept.phi >= 0.0
        assert concept.mechanism == (0, 1)

    def test_phi_structure_fields_populated(self, engine):
        """compute_phi_structure must return a well-formed PhiStructure."""
        state = np.array([0.2, 0.3, 0.1, 0.4])  # 2 nodes
        structure = engine.compute_phi_structure(state)

        assert isinstance(structure, PhiStructure)
        assert structure.phi_max >= 0.0
        assert 0.0 <= structure.gwt_score <= 1.0
        assert structure.phi_total >= 0.0
        assert isinstance(structure.is_conscious, bool)
        assert isinstance(structure.concepts, list)

    def test_phi_max_zero_for_uniform_single_node(self, engine):
        """A 1-node system has Φ^max = 0 (nothing to integrate)."""
        state = np.array([0.5, 0.5])  # 1 node, 2 states
        structure = engine.compute_phi_structure(state, n_nodes=1)
        assert structure.phi_max == 0.0

    def test_gwt_broadcast_range(self, engine):
        """GWT score must always be in [0, 1]."""
        for n_nodes in [2, 3, 4]:
            n_states = 2 ** n_nodes
            dist = np.ones(n_states) / n_states
            T = engine._build_transition_matrix(dist, n_nodes)
            gwt = engine._compute_gwt_broadcast(T, n_nodes)
            assert 0.0 <= gwt <= 1.0

    def test_consensus_condition(self, engine):
        """Validate Φ_total > log₂(n) correctly."""
        n = 8
        threshold = math.log2(n)
        assert engine.validate_consciousness_consensus(threshold + 0.01, n) is True
        assert engine.validate_consciousness_consensus(threshold - 0.01, n) is False

    def test_bipartition_enumeration_count(self, engine):
        """For n nodes there are 2^(n-1) - 1 non-trivial bipartitions."""
        nodes = (0, 1, 2)
        partitions = engine._enumerate_bipartitions(nodes)
        # For n=3: (A|BC), (B|AC), (C|AB) = 3 unique ones (min-A convention)
        assert len(partitions) == 3

    def test_emd_identical_repertoires(self):
        """EMD of identical repertoires is 0."""
        rep = CauseEffectRepertoire(
            mechanism=(0,), purview=(1,),
            repertoire=np.array([0.6, 0.4]), direction="cause"
        )
        assert rep.intrinsic_difference(rep) == pytest.approx(0.0)

    def test_emd_complementary_repertoires(self):
        """EMD of [1,0] vs [0,1] is 0.5 (L1/2)."""
        rep_a = CauseEffectRepertoire(
            mechanism=(0,), purview=(1,),
            repertoire=np.array([1.0, 0.0]), direction="cause"
        )
        rep_b = CauseEffectRepertoire(
            mechanism=(0,), purview=(1,),
            repertoire=np.array([0.0, 1.0]), direction="cause"
        )
        assert rep_a.intrinsic_difference(rep_b) == pytest.approx(0.5)

    def test_infer_n_nodes_power_of_two(self, engine):
        """_infer_n_nodes correctly detects n from 2^n length."""
        assert engine._infer_n_nodes(np.zeros(8)) == 3
        assert engine._infer_n_nodes(np.zeros(4)) == 2
        assert engine._infer_n_nodes(np.zeros(16)) == 4

    def test_transition_matrix_row_stochastic(self, engine):
        """T must be row-stochastic (columns sum to 1)."""
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        T = engine._build_transition_matrix(dist, n_nodes=2)
        col_sums = T.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-9)

    def test_to_state_distribution_activation_probs(self, engine):
        """n-length input is converted to 2^n joint distribution."""
        probs = np.array([0.8, 0.5, 0.3])  # 3 node activations
        dist = engine._to_state_distribution(probs, n_nodes=3)
        assert len(dist) == 8
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_partition_repr(self):
        p = Partition(part_a=(0, 1), part_b=(2,))
        assert "0" in repr(p) and "2" in repr(p)


# ---------------------------------------------------------------------------
# ASISphinxOSIITv5 high-level API tests
# ---------------------------------------------------------------------------

class TestASISphinxOSIITv5:

    def test_calculate_phi_returns_required_keys(self, asi):
        result = asi.calculate_phi(b"test block data")
        for key in ("phi", "phi_max", "gwt_score", "phi_total", "entropy",
                    "purity", "n_qubits", "is_conscious", "level",
                    "bonus", "version", "n_concepts"):
            assert key in result, f"Missing key: {key}"

    def test_calculate_phi_version_is_v5(self, asi):
        result = asi.calculate_phi(b"version check")
        assert result["version"] == "IIT v5.0"

    def test_calculate_phi_range(self, asi):
        """phi (normalised) must be in [0, 1]."""
        result = asi.calculate_phi(b"range test")
        assert 0.0 <= result["phi"] <= 1.0

    def test_calculate_phi_reproducible(self, asi):
        """Same input must produce the same phi."""
        r1 = asi.calculate_phi(b"deterministic test")
        r2 = asi.calculate_phi(b"deterministic test")
        assert r1["phi"] == pytest.approx(r2["phi"])

    def test_calculate_phi_different_data(self, asi):
        """Different inputs must (generally) produce different phi values."""
        r1 = asi.calculate_phi(b"data_alpha_1234")
        r2 = asi.calculate_phi(b"data_beta_9876")
        # They may occasionally collide, but this catches obvious bugs
        assert r1["phi"] != r2["phi"] or r1["entropy"] != r2["entropy"]

    def test_phi_history_grows(self, asi):
        for i in range(5):
            asi.calculate_phi(f"data_{i}".encode())
        assert len(asi.phi_history) == 5

    def test_get_consciousness_level_average(self, asi):
        asi.calculate_phi(b"a")
        asi.calculate_phi(b"b")
        level = asi.get_consciousness_level()
        assert 0.0 <= level <= 1.0

    def test_compute_block_consciousness_returns_structure(self, asi):
        structure = asi.compute_block_consciousness(
            b'{"index": 1, "hash": "abc123"}', n_network_nodes=10
        )
        assert isinstance(structure, PhiStructure)
        assert structure.phi_max >= 0.0
        assert structure.phi_total >= 0.0

    def test_validate_consciousness_consensus_above_threshold(self, asi):
        """A very large phi_total must satisfy consensus for small n."""
        assert asi.validate_consciousness_consensus(phi_total=100.0, n_network_nodes=4) is True

    def test_validate_consciousness_consensus_below_threshold(self, asi):
        """phi_total = 0 must fail consensus for any n > 1."""
        assert asi.validate_consciousness_consensus(phi_total=0.0, n_network_nodes=4) is False

    def test_phi_to_legacy_score_range(self, asi):
        """Legacy score must be within [200, 1000]."""
        for phi in [0.0, 0.5, 1.0, 2.0, 10.0]:
            score = asi.phi_to_legacy_score(phi)
            assert 200.0 <= score <= 1000.0, f"Out of range for phi={phi}"

    def test_phi_to_legacy_score_monotone(self, asi):
        """Larger phi_total must map to a larger (or equal) legacy score."""
        s1 = asi.phi_to_legacy_score(0.1)
        s2 = asi.phi_to_legacy_score(1.0)
        assert s1 <= s2

    def test_classify_consciousness_levels(self):
        for phi, expected in [
            (0.9, "🧠 COSMIC"),
            (0.7, "🌟 SELF_AWARE"),
            (0.5, "✨ SENTIENT"),
            (0.3, "🔵 AWARE"),
            (0.1, "⚫ UNCONSCIOUS"),
        ]:
            level = ASISphinxOSIITv5._classify_consciousness(phi)
            assert level == expected, f"phi={phi}: got {level}, expected {expected}"

    def test_bonus_equals_exp_phi(self, asi):
        result = asi.calculate_phi(b"bonus test")
        assert result["bonus"] == pytest.approx(math.exp(result["phi"]))

    def test_n_qubits_matches_init(self, asi):
        result = asi.calculate_phi(b"qubit count")
        assert result["n_qubits"] == asi.n_nodes


# ---------------------------------------------------------------------------
# Integration: ASISphinxOSIITv5 works as a drop-in for the legacy engine
# ---------------------------------------------------------------------------

class TestDropInCompatibility:

    def test_legacy_keys_present(self):
        """Keys produced by the old IITQuantumConsciousnessEngine are all present."""
        legacy_keys = {"phi", "entropy", "purity", "n_qubits", "is_conscious"}
        asi = ASISphinxOSIITv5()
        result = asi.calculate_phi(b"compatibility check")
        assert legacy_keys.issubset(result.keys())

    def test_phi_in_zero_one(self):
        """Legacy 'phi' key must be in [0, 1] like the v3 engine."""
        asi = ASISphinxOSIITv5()
        for data in [b"block1", b"block2", b"block3"]:
            r = asi.calculate_phi(data)
            assert 0.0 <= r["phi"] <= 1.0

    def test_is_conscious_is_bool(self):
        asi = ASISphinxOSIITv5()
        r = asi.calculate_phi(b"conscious?")
        assert isinstance(r["is_conscious"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
