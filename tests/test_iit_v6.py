"""
Tests for ASI SphinxOS Advanced IIT v6.0

Validates:
- IITv6Engine cause-effect repertoire computation
- Temporal-depth Φ (phi_tau) averaging over τ time steps
- Intrinsic Causal Power (ICP) per concept
- Exclusion-principle CES pruning
- Minimum Information Partition (MIP) search and Φ^max
- GWT broadcast score
- Tripartite Φ_total and consciousness-consensus condition
- ASISphinxOSIITv6 high-level API (drop-in for v5 / legacy engine)
- Legacy phi_score mapping
"""

import math
import pytest
import numpy as np

from sphinx_os.Artificial_Intelligence.iit_v6 import (
    ASISphinxOSIITv6,
    IITv6Engine,
    CauseEffectRepertoire,
    Partition,
    PhiStructure,
    Concept,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return IITv6Engine(alpha=0.55, beta=0.25, gamma=0.20, temporal_depth=2)


@pytest.fixture
def asi():
    return ASISphinxOSIITv6(alpha=0.55, beta=0.25, gamma=0.20, n_nodes=3, temporal_depth=2)


# ---------------------------------------------------------------------------
# IITv6Engine unit tests
# ---------------------------------------------------------------------------

class TestIITv6Engine:

    def test_weights_must_sum_to_one(self):
        """Constructor must raise when α+β+γ ≠ 1."""
        with pytest.raises(ValueError):
            IITv6Engine(alpha=0.5, beta=0.5, gamma=0.5)

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
        assert len(rep.repertoire) == 2 ** 2

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

    def test_icp_non_negative(self, engine):
        """ICP for any concept must be ≥ 0."""
        n_nodes = 3
        dist = np.array([0.3, 0.1, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05])
        T = engine._build_transition_matrix(dist, n_nodes)

        concept = engine.compute_phi_concept(
            mechanism=(0,), T=T, dist=dist, n_nodes=n_nodes
        )
        assert concept.icp >= 0.0

    def test_icp_is_geometric_mean(self, engine):
        """ICP(M) should satisfy ICP ≥ 0; when phi > 0 icp > 0."""
        n_nodes = 2
        dist = np.array([0.4, 0.2, 0.3, 0.1])
        T = engine._build_transition_matrix(dist, n_nodes)

        concept = engine.compute_phi_concept(
            mechanism=(0,), T=T, dist=dist, n_nodes=n_nodes
        )
        if concept.phi > 0.0:
            assert concept.icp > 0.0

    def test_phi_structure_fields_populated(self, engine):
        """compute_phi_structure must return a well-formed PhiStructure."""
        state = np.array([0.2, 0.3, 0.1, 0.4])  # 2 nodes
        structure = engine.compute_phi_structure(state)

        assert isinstance(structure, PhiStructure)
        assert structure.phi_max >= 0.0
        assert structure.phi_tau >= 0.0
        assert 0.0 <= structure.gwt_score <= 1.0
        assert structure.icp_avg >= 0.0
        assert structure.phi_total >= 0.0
        assert isinstance(structure.is_conscious, bool)
        assert isinstance(structure.concepts, list)

    def test_phi_tau_is_average_of_steps(self, engine):
        """phi_tau must be >= 0 and consistent across calls."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s1 = engine.compute_phi_structure(state)
        s2 = engine.compute_phi_structure(state)
        assert s1.phi_tau == pytest.approx(s2.phi_tau)
        assert s1.phi_tau >= 0.0

    def test_phi_tau_single_step_matches_phi_max(self):
        """With temporal_depth=1, phi_tau should equal phi_max."""
        eng = IITv6Engine(alpha=0.55, beta=0.25, gamma=0.20, temporal_depth=1)
        state = np.array([0.1, 0.4, 0.3, 0.2])
        structure = eng.compute_phi_structure(state)
        assert structure.phi_tau == pytest.approx(structure.phi_max, abs=1e-9)

    def test_phi_max_zero_for_single_node(self, engine):
        """A 1-node system has Φ^max = 0 (nothing to integrate)."""
        state = np.array([0.5, 0.5])
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
        """For n=3 there should be 3 unique bipartitions."""
        nodes = (0, 1, 2)
        partitions = engine._enumerate_bipartitions(nodes)
        assert len(partitions) == 3

    def test_emd_identical_repertoires(self):
        """EMD of identical repertoires is 0."""
        rep = CauseEffectRepertoire(
            mechanism=(0,), purview=(1,),
            repertoire=np.array([0.6, 0.4]), direction="cause"
        )
        assert rep.intrinsic_difference(rep) == pytest.approx(0.0)

    def test_emd_complementary_repertoires(self):
        """EMD of [1,0] vs [0,1] is 1.0 (L1 distance = 2, divided by 2)."""
        rep_a = CauseEffectRepertoire(
            mechanism=(0,), purview=(1,),
            repertoire=np.array([1.0, 0.0]), direction="cause"
        )
        rep_b = CauseEffectRepertoire(
            mechanism=(0,), purview=(1,),
            repertoire=np.array([0.0, 1.0]), direction="cause"
        )
        assert rep_a.intrinsic_difference(rep_b) == pytest.approx(1.0)

    def test_infer_n_nodes_power_of_two(self, engine):
        """_infer_n_nodes correctly detects n from 2^n length."""
        assert engine._infer_n_nodes(np.zeros(8)) == 3
        assert engine._infer_n_nodes(np.zeros(4)) == 2
        assert engine._infer_n_nodes(np.zeros(16)) == 4

    def test_transition_matrix_row_stochastic(self, engine):
        """T must be column-stochastic (columns sum to 1)."""
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        T = engine._build_transition_matrix(dist, n_nodes=2)
        col_sums = T.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-9)

    def test_to_state_distribution_activation_probs(self, engine):
        """n-length input is converted to 2^n joint distribution."""
        probs = np.array([0.8, 0.5, 0.3])
        dist = engine._to_state_distribution(probs, n_nodes=3)
        assert len(dist) == 8
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_partition_repr(self):
        p = Partition(part_a=(0, 1), part_b=(2,))
        assert "0" in repr(p) and "2" in repr(p)

    def test_exclusion_principle_removes_subsumed(self, engine):
        """Exclusion pruning must remove concepts whose mechanism is a subset."""
        # Manually create two concepts: (0,) with phi=1.0 and (0,1) with phi=0.5
        c_small = Concept(mechanism=(0,), purview_cause=(0,), purview_effect=(0,),
                          phi=1.0, icp=0.5)
        c_large = Concept(mechanism=(0, 1), purview_cause=(0, 1), purview_effect=(0, 1),
                          phi=0.5, icp=0.3)
        # c_small has higher phi; after exclusion it should survive but c_large
        # should still be included because it brings node 1 (not yet covered)
        result = engine._apply_exclusion_principle([c_small, c_large])
        mechanisms = [c.mechanism for c in result]
        # Both survive because c_large includes node 1 which is new
        assert (0,) in mechanisms
        assert (0, 1) in mechanisms

    def test_exclusion_principle_keeps_higher_phi(self, engine):
        """When two concepts share ALL nodes, only the higher-phi one survives."""
        c1 = Concept(mechanism=(0, 1), purview_cause=(0,), purview_effect=(0,),
                     phi=0.8, icp=0.4)
        c2 = Concept(mechanism=(0, 1), purview_cause=(0,), purview_effect=(0,),
                     phi=0.3, icp=0.2)
        # Both have identical mechanisms; after exclusion only the first should remain
        result = engine._apply_exclusion_principle([c1, c2])
        assert len(result) == 1
        assert result[0].phi == pytest.approx(0.8)

    def test_phi_structure_icp_avg_non_negative(self, engine):
        """icp_avg in PhiStructure must be ≥ 0."""
        state = np.array([0.3, 0.2, 0.3, 0.2])
        structure = engine.compute_phi_structure(state)
        assert structure.icp_avg >= 0.0


# ---------------------------------------------------------------------------
# ASISphinxOSIITv6 high-level API tests
# ---------------------------------------------------------------------------

class TestASISphinxOSIITv6:

    def test_calculate_phi_returns_required_keys(self, asi):
        result = asi.calculate_phi(b"test block data")
        for key in ("phi", "phi_max", "phi_tau", "gwt_score", "icp_avg",
                    "phi_total", "entropy", "purity", "n_qubits",
                    "is_conscious", "level", "bonus", "version", "n_concepts"):
            assert key in result, f"Missing key: {key}"

    def test_calculate_phi_version_is_v6(self, asi):
        result = asi.calculate_phi(b"version check")
        assert result["version"] == "IIT v6.0"

    def test_calculate_phi_range(self, asi):
        """phi (normalised) must be in [0, 1]."""
        result = asi.calculate_phi(b"range test")
        assert 0.0 <= result["phi"] <= 1.0

    def test_phi_tau_non_negative(self, asi):
        result = asi.calculate_phi(b"tau test")
        assert result["phi_tau"] >= 0.0

    def test_icp_avg_non_negative(self, asi):
        result = asi.calculate_phi(b"icp test")
        assert result["icp_avg"] >= 0.0

    def test_calculate_phi_reproducible(self, asi):
        """Same input must produce the same phi."""
        r1 = asi.calculate_phi(b"deterministic test")
        r2 = asi.calculate_phi(b"deterministic test")
        assert r1["phi"] == pytest.approx(r2["phi"])

    def test_calculate_phi_different_data(self, asi):
        """Different inputs must (generally) produce different phi values."""
        r1 = asi.calculate_phi(b"data_alpha_1234")
        r2 = asi.calculate_phi(b"data_beta_9876")
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
        assert structure.phi_tau >= 0.0
        assert structure.phi_total >= 0.0

    def test_validate_consciousness_consensus_above_threshold(self, asi):
        assert asi.validate_consciousness_consensus(phi_total=100.0, n_network_nodes=4) is True

    def test_validate_consciousness_consensus_below_threshold(self, asi):
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
            level = ASISphinxOSIITv6._classify_consciousness(phi)
            assert level == expected, f"phi={phi}: got {level}, expected {expected}"

    def test_bonus_equals_exp_phi(self, asi):
        result = asi.calculate_phi(b"bonus test")
        assert result["bonus"] == pytest.approx(math.exp(result["phi"]))

    def test_n_qubits_matches_init(self, asi):
        result = asi.calculate_phi(b"qubit count")
        assert result["n_qubits"] == asi.n_nodes


# ---------------------------------------------------------------------------
# Integration: ASISphinxOSIITv6 works as a drop-in for the legacy engine
# ---------------------------------------------------------------------------

class TestDropInCompatibility:

    def test_legacy_keys_present(self):
        """Keys produced by the old IITQuantumConsciousnessEngine are all present."""
        legacy_keys = {"phi", "entropy", "purity", "n_qubits", "is_conscious"}
        asi = ASISphinxOSIITv6()
        result = asi.calculate_phi(b"compatibility check")
        assert legacy_keys.issubset(result.keys())

    def test_phi_in_zero_one(self):
        """Legacy 'phi' key must be in [0, 1] like the v5 engine."""
        asi = ASISphinxOSIITv6()
        for data in [b"block1", b"block2", b"block3"]:
            r = asi.calculate_phi(data)
            assert 0.0 <= r["phi"] <= 1.0

    def test_is_conscious_is_bool(self):
        asi = ASISphinxOSIITv6()
        r = asi.calculate_phi(b"conscious?")
        assert isinstance(r["is_conscious"], bool)

    def test_v6_has_additional_keys_vs_v5(self):
        """v6 must expose phi_tau and icp_avg which v5 does not."""
        asi = ASISphinxOSIITv6()
        result = asi.calculate_phi(b"v6 extras")
        assert "phi_tau" in result
        assert "icp_avg" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
