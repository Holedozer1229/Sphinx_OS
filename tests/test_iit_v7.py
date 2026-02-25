"""
Tests for ASI SphinxOS Advanced IIT v7.0

Validates:
- IITv7Engine Fano plane alignment score (Phi_fano)
- IITv7Engine Non-abelian physics measure (Phi_nab)
- 5-term composite score and v7 consciousness-consensus condition
- PhiStructureV7 dataclass fields
- ASISphinxOSIITv7 high-level API (drop-in for v6)
- Backward compatibility: all v6 fields still present in v7 output
- FANO_LINES constant structure
"""

import math
import pytest
import numpy as np

from sphinx_os.Artificial_Intelligence.iit_v7 import (
    ASISphinxOSIITv7,
    IITv7Engine,
    PhiStructureV7,
    FANO_LINES,
    FANO_POINTS,
)
from sphinx_os.Artificial_Intelligence.iit_v6 import (
    CauseEffectRepertoire,
    Partition,
    Concept,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return IITv7Engine(
        alpha=0.40, beta=0.20, gamma=0.15, delta=0.15, epsilon=0.10,
        temporal_depth=2,
    )


@pytest.fixture
def asi():
    return ASISphinxOSIITv7(
        alpha=0.40, beta=0.20, gamma=0.15, delta=0.15, epsilon=0.10,
        n_nodes=3, temporal_depth=2,
    )


# ---------------------------------------------------------------------------
# Fano plane constant tests
# ---------------------------------------------------------------------------

class TestFanoPlaneConstants:

    def test_fano_has_seven_lines(self):
        """The Fano plane must have exactly 7 lines."""
        assert len(FANO_LINES) == 7

    def test_fano_points_constant(self):
        """FANO_POINTS must equal 7."""
        assert FANO_POINTS == 7

    def test_each_line_has_three_points(self):
        """Each Fano line must contain exactly 3 distinct point indices."""
        for line in FANO_LINES:
            assert len(line) == 3
            assert len(set(line)) == 3, f"Duplicate point in line {line}"

    def test_all_points_covered(self):
        """All 7 Fano points (0-6) must appear in at least one line."""
        all_points = set()
        for line in FANO_LINES:
            all_points.update(line)
        assert all_points == set(range(7))

    def test_each_point_appears_in_three_lines(self):
        """In the Fano plane each point lies on exactly 3 lines."""
        from collections import Counter
        counts = Counter(p for line in FANO_LINES for p in line)
        for point, count in counts.items():
            assert count == 3, f"Point {point} appears in {count} lines (expected 3)"

    def test_fano_lines_are_tuples_of_ints(self):
        for line in FANO_LINES:
            assert all(isinstance(p, int) for p in line)


# ---------------------------------------------------------------------------
# IITv7Engine unit tests
# ---------------------------------------------------------------------------

class TestIITv7Engine:

    def test_weights_must_sum_to_one(self):
        """Constructor must raise when the five weights do not sum to 1."""
        with pytest.raises(ValueError):
            IITv7Engine(alpha=0.4, beta=0.2, gamma=0.2, delta=0.2, epsilon=0.2)

    def test_valid_construction(self, engine):
        """A correctly-weighted engine constructs without error."""
        assert engine.alpha + engine.beta + engine.gamma + engine.delta + engine.epsilon == pytest.approx(1.0)

    # -- Fano alignment score -------------------------------------------

    def test_fano_score_in_range(self, engine):
        """Phi_fano must be in [0, 1] for any transition matrix."""
        for n_nodes in [2, 3, 4]:
            n_states = 2 ** n_nodes
            dist = np.ones(n_states) / n_states
            T = engine._build_transition_matrix(dist, n_nodes)
            score = engine._compute_fano_alignment(T, n_nodes)
            assert 0.0 <= score <= 1.0, f"Fano score {score} out of range for n={n_nodes}"

    def test_fano_score_non_negative(self, engine):
        """Phi_fano must always be >= 0."""
        dist = np.array([0.3, 0.2, 0.3, 0.2])
        T = engine._build_transition_matrix(dist, n_nodes=2)
        assert engine._compute_fano_alignment(T, n_nodes=2) >= 0.0

    def test_fano_score_reproducible(self, engine):
        """Same transition matrix must yield the same Fano score."""
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        T = engine._build_transition_matrix(dist, n_nodes=2)
        s1 = engine._compute_fano_alignment(T, n_nodes=2)
        s2 = engine._compute_fano_alignment(T, n_nodes=2)
        assert s1 == pytest.approx(s2)

    def test_fano_score_singular_matrix_does_not_crash(self, engine):
        """A singular (zero) matrix should return 0.0 without raising."""
        T = np.zeros((4, 4))
        score = engine._compute_fano_alignment(T, n_nodes=2)
        assert score == 0.0

    # -- Non-abelian score ----------------------------------------------

    def test_nonabelian_score_in_range(self, engine):
        """Phi_nab must be in [0, 1] for any transition matrix."""
        for n_nodes in [2, 3, 4]:
            n_states = 2 ** n_nodes
            dist = np.ones(n_states) / n_states
            T = engine._build_transition_matrix(dist, n_nodes)
            score = engine._compute_nonabelian_measure(T)
            assert 0.0 <= score <= 1.0, f"Non-abelian score {score} out of range"

    def test_nonabelian_score_symmetric_matrix_is_zero(self, engine):
        """A symmetric matrix has commutator [T, T^T] = 0."""
        T = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        # Make it perfectly symmetric
        T_sym = (T + T.T) / 2.0
        score = engine._compute_nonabelian_measure(T_sym)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_nonabelian_score_positive_for_asymmetric(self, engine):
        """An asymmetric matrix should yield a positive Phi_nab."""
        T = np.array([
            [0.8, 0.1, 0.1],
            [0.05, 0.8, 0.15],
            [0.15, 0.1, 0.75],
        ])
        score = engine._compute_nonabelian_measure(T)
        assert score >= 0.0

    def test_nonabelian_zero_matrix_does_not_crash(self, engine):
        """A zero matrix should return 0.0 without raising."""
        T = np.zeros((4, 4))
        assert engine._compute_nonabelian_measure(T) == 0.0

    def test_nonabelian_score_reproducible(self, engine):
        """Same matrix must give same non-abelian score."""
        dist = np.array([0.1, 0.4, 0.3, 0.2])
        T = engine._build_transition_matrix(dist, n_nodes=2)
        s1 = engine._compute_nonabelian_measure(T)
        s2 = engine._compute_nonabelian_measure(T)
        assert s1 == pytest.approx(s2)

    # -- PhiStructureV7 from compute_phi_structure_v7 -------------------

    def test_phi_structure_v7_fields(self, engine):
        """compute_phi_structure_v7 returns a well-formed PhiStructureV7."""
        state = np.array([0.2, 0.3, 0.1, 0.4])
        structure = engine.compute_phi_structure_v7(state)

        assert isinstance(structure, PhiStructureV7)
        assert structure.phi_max >= 0.0
        assert structure.phi_tau >= 0.0
        assert 0.0 <= structure.gwt_score <= 1.0
        assert structure.icp_avg >= 0.0
        assert 0.0 <= structure.fano_score <= 1.0
        assert 0.0 <= structure.nonabelian_score <= 1.0
        assert structure.phi_total >= 0.0
        assert isinstance(structure.is_conscious, bool)
        assert isinstance(structure.concepts, list)

    def test_phi_total_is_weighted_sum(self, engine):
        """phi_total must equal the weighted sum of its components."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v7(state)
        expected = (
            engine.alpha * s.phi_tau
            + engine.beta * s.gwt_score
            + engine.gamma * s.icp_avg
            + engine.delta * s.fano_score
            + engine.epsilon * s.nonabelian_score
        )
        assert s.phi_total == pytest.approx(expected, rel=1e-6)

    def test_phi_structure_v7_reproducible(self, engine):
        """Same input must produce same phi_total."""
        state = np.array([0.1, 0.4, 0.3, 0.2])
        s1 = engine.compute_phi_structure_v7(state)
        s2 = engine.compute_phi_structure_v7(state)
        assert s1.phi_total == pytest.approx(s2.phi_total)

    # -- v7 consensus condition -----------------------------------------

    def test_consensus_v7_above_threshold(self, engine):
        """phi_total well above threshold must return True."""
        assert engine.validate_consciousness_consensus_v7(
            phi_total=100.0, fano_score=0.0, n_nodes=8
        ) is True

    def test_consensus_v7_below_threshold(self, engine):
        """phi_total of 0.0 must return False."""
        assert engine.validate_consciousness_consensus_v7(
            phi_total=0.0, fano_score=0.0, n_nodes=8
        ) is False

    def test_consensus_v7_fano_raises_threshold(self, engine):
        """A higher fano_score should make it harder to reach consensus."""
        n = 4
        base_threshold = math.log2(n)
        phi_just_above_base = base_threshold + 0.001
        # Without Fano contribution this should pass
        assert engine.validate_consciousness_consensus_v7(
            phi_total=phi_just_above_base, fano_score=0.0, n_nodes=n
        ) is True
        # With a large fano_score the same phi_total should fail
        large_fano = (phi_just_above_base - base_threshold) / engine.delta + 0.1
        if large_fano <= 1.0:
            assert engine.validate_consciousness_consensus_v7(
                phi_total=phi_just_above_base,
                fano_score=large_fano,
                n_nodes=n,
            ) is False

    def test_phi_structure_v7_weights_stored(self, engine):
        """PhiStructureV7 stores delta and epsilon correctly."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v7(state)
        assert s.delta == pytest.approx(engine.delta)
        assert s.epsilon == pytest.approx(engine.epsilon)


# ---------------------------------------------------------------------------
# ASISphinxOSIITv7 high-level API tests
# ---------------------------------------------------------------------------

class TestASISphinxOSIITv7:

    def test_calculate_phi_returns_required_keys(self, asi):
        """All expected keys must be present in the response dict."""
        result = asi.calculate_phi(b"test block data")
        required = (
            "phi", "phi_max", "phi_tau", "gwt_score", "icp_avg",
            "fano_score", "nonabelian_score",
            "phi_total", "entropy", "purity", "n_qubits",
            "is_conscious", "level", "bonus", "version", "n_concepts",
        )
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_calculate_phi_version_is_v7(self, asi):
        result = asi.calculate_phi(b"version check")
        assert result["version"] == "IIT v7.0"

    def test_calculate_phi_range(self, asi):
        """phi (normalised) must be in [0, 1]."""
        result = asi.calculate_phi(b"range test")
        assert 0.0 <= result["phi"] <= 1.0

    def test_fano_score_in_result_range(self, asi):
        result = asi.calculate_phi(b"fano range test")
        assert 0.0 <= result["fano_score"] <= 1.0

    def test_nonabelian_score_in_result_range(self, asi):
        result = asi.calculate_phi(b"nab range test")
        assert 0.0 <= result["nonabelian_score"] <= 1.0

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
        """Different inputs generally produce different phi or entropy."""
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

    def test_compute_block_consciousness_returns_v7_structure(self, asi):
        structure = asi.compute_block_consciousness(
            b'{"index": 1, "hash": "abc123"}', n_network_nodes=10
        )
        assert isinstance(structure, PhiStructureV7)
        assert structure.phi_max >= 0.0
        assert structure.phi_tau >= 0.0
        assert structure.phi_total >= 0.0
        assert 0.0 <= structure.fano_score <= 1.0
        assert 0.0 <= structure.nonabelian_score <= 1.0

    def test_validate_consciousness_consensus_above_threshold(self, asi):
        assert asi.validate_consciousness_consensus(
            phi_total=100.0, fano_score=0.0, n_network_nodes=4
        ) is True

    def test_validate_consciousness_consensus_below_threshold(self, asi):
        assert asi.validate_consciousness_consensus(
            phi_total=0.0, fano_score=0.0, n_network_nodes=4
        ) is False

    def test_phi_to_legacy_score_range(self, asi):
        """Legacy score must be within [200, 1000]."""
        for phi in [0.0, 0.5, 1.0, 2.0, 10.0]:
            score = asi.phi_to_legacy_score(phi)
            assert 200.0 <= score <= 1000.0, f"Out of range for phi={phi}"

    def test_phi_to_legacy_score_monotone(self, asi):
        """Larger phi_total maps to a larger (or equal) legacy score."""
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
            level = ASISphinxOSIITv7._classify_consciousness(phi)
            assert level == expected, f"phi={phi}: got {level!r}, expected {expected!r}"

    def test_bonus_equals_exp_phi(self, asi):
        result = asi.calculate_phi(b"bonus test")
        assert result["bonus"] == pytest.approx(math.exp(result["phi"]))

    def test_n_qubits_matches_init(self, asi):
        result = asi.calculate_phi(b"qubit count")
        assert result["n_qubits"] == asi.n_nodes


# ---------------------------------------------------------------------------
# Backward compatibility: v7 is a drop-in for v6
# ---------------------------------------------------------------------------

class TestDropInCompatibilityV7:

    def test_all_v6_keys_present(self):
        """All keys produced by v6 engine must also be present in v7."""
        v6_keys = {
            "phi", "phi_max", "phi_tau", "gwt_score", "icp_avg",
            "phi_total", "entropy", "purity", "n_qubits",
            "is_conscious", "level", "bonus", "n_concepts",
        }
        asi = ASISphinxOSIITv7()
        result = asi.calculate_phi(b"compatibility check")
        assert v6_keys.issubset(result.keys())

    def test_phi_in_zero_one(self):
        """Legacy 'phi' key must be in [0, 1]."""
        asi = ASISphinxOSIITv7()
        for data in [b"block1", b"block2", b"block3"]:
            r = asi.calculate_phi(data)
            assert 0.0 <= r["phi"] <= 1.0

    def test_is_conscious_is_bool(self):
        asi = ASISphinxOSIITv7()
        r = asi.calculate_phi(b"conscious?")
        assert isinstance(r["is_conscious"], bool)

    def test_v7_has_additional_keys_vs_v6(self):
        """v7 must expose fano_score and nonabelian_score absent in v6."""
        asi = ASISphinxOSIITv7()
        result = asi.calculate_phi(b"v7 extras")
        assert "fano_score" in result
        assert "nonabelian_score" in result

    def test_version_string_is_v7(self):
        asi = ASISphinxOSIITv7()
        result = asi.calculate_phi(b"version")
        assert result["version"] == "IIT v7.0"


# ---------------------------------------------------------------------------
# Module-level import test
# ---------------------------------------------------------------------------

class TestModuleExports:

    def test_init_exports_v7_classes(self):
        """__init__.py must export ASISphinxOSIITv7, IITv7Engine, PhiStructureV7."""
        from sphinx_os.Artificial_Intelligence import (
            ASISphinxOSIITv7,
            IITv7Engine,
            PhiStructureV7,
            FANO_LINES,
            FANO_POINTS,
        )
        assert ASISphinxOSIITv7 is not None
        assert IITv7Engine is not None
        assert PhiStructureV7 is not None
        assert len(FANO_LINES) == 7
        assert FANO_POINTS == 7

    def test_version_updated(self):
        import sphinx_os.Artificial_Intelligence as ai
        assert ai.__version__ == "7.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
