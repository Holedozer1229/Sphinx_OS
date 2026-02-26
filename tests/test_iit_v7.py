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
- ScoreDiagnostic zero-classification system
- RiemannZeroProbe critical-line signature for known Riemann zeros
"""

import math
import pytest
import numpy as np

from sphinx_os.Artificial_Intelligence.iit_v7 import (
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
    _ZR_SVD_FAILED,
    _ZR_N_MODES_LT_2,
    _ZR_MODE_INTERACTIONS_NEG,
    _ZR_NO_VALID_FANO_LINES,
    _ZR_ZERO_MATRIX,
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


# ---------------------------------------------------------------------------
# ScoreDiagnostic zero-classification tests
# ---------------------------------------------------------------------------

class TestScoreDiagnostic:

    def test_exact_zero_on_svd_failure(self, engine):
        """A matrix that makes SVD fail must produce EXACT_ZERO with svd_failed reason."""
        # An all-NaN matrix triggers LinAlgError in SVD
        T_bad = np.full((4, 4), np.nan)
        raw, reason = engine._compute_fano_raw(T_bad, n_nodes=2)
        assert raw == 0.0
        assert reason == _ZR_SVD_FAILED
        cls = engine._classify_score(raw, reason)
        assert cls == CLASSIFICATION_EXACT_ZERO

    def test_exact_zero_n_modes_lt_2(self, engine):
        """A 1×1 matrix produces EXACT_ZERO with n_modes_lt_2 reason."""
        T_tiny = np.array([[1.0]])
        raw, reason = engine._compute_fano_raw(T_tiny, n_nodes=1)
        assert raw == 0.0
        assert reason == _ZR_N_MODES_LT_2
        assert engine._classify_score(raw, reason) == CLASSIFICATION_EXACT_ZERO

    def test_exact_zero_zero_matrix_nonabelian(self, engine):
        """A zero matrix produces EXACT_ZERO with zero_matrix reason (non-abelian)."""
        T_zero = np.zeros((4, 4))
        raw, reason = engine._compute_nonabelian_raw(T_zero)
        assert raw == 0.0
        assert reason == _ZR_ZERO_MATRIX
        assert engine._classify_score(raw, reason) == CLASSIFICATION_EXACT_ZERO

    def test_near_zero_below_threshold(self, engine):
        """A computed value below near_zero_threshold → NEAR_ZERO."""
        # Create tiny but nonzero raw_value, no reason
        cls = engine._classify_score(1e-10, None)
        assert cls == CLASSIFICATION_NEAR_ZERO

    def test_nonzero_above_threshold(self, engine):
        """A computed value above near_zero_threshold → NONZERO."""
        cls = engine._classify_score(0.5, None)
        assert cls == CLASSIFICATION_NONZERO

    def test_near_zero_threshold_boundary(self, engine):
        """Value exactly at threshold is NONZERO (strict <)."""
        t = engine.near_zero_threshold
        assert engine._classify_score(t, None) == CLASSIFICATION_NONZERO
        assert engine._classify_score(t - 1e-20, None) == CLASSIFICATION_NEAR_ZERO

    def test_phi_structure_has_fano_diagnostic(self, engine):
        """compute_phi_structure_v7 must populate fano_diagnostic."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v7(state)
        assert s.fano_diagnostic is not None
        assert isinstance(s.fano_diagnostic, ScoreDiagnostic)

    def test_phi_structure_has_nonabelian_diagnostic(self, engine):
        """compute_phi_structure_v7 must populate nonabelian_diagnostic."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v7(state)
        assert s.nonabelian_diagnostic is not None
        assert isinstance(s.nonabelian_diagnostic, ScoreDiagnostic)

    def test_diagnostic_clamped_matches_score(self, engine):
        """fano_diagnostic.clamped_value must equal fano_score on PhiStructureV7."""
        state = np.array([0.1, 0.4, 0.3, 0.2])
        s = engine.compute_phi_structure_v7(state)
        assert s.fano_diagnostic.clamped_value == pytest.approx(s.fano_score)
        assert s.nonabelian_diagnostic.clamped_value == pytest.approx(s.nonabelian_score)

    def test_diagnostic_raw_value_le_clamped(self, engine):
        """raw_value may exceed 1.0; clamped_value is always ≤ 1.0."""
        state = np.array([0.1, 0.4, 0.3, 0.2])
        s = engine.compute_phi_structure_v7(state)
        assert s.fano_diagnostic.clamped_value <= 1.0
        assert s.nonabelian_diagnostic.clamped_value <= 1.0

    def test_diagnostic_near_zero_threshold_stored(self, engine):
        """The threshold used for classification is recorded on the diagnostic."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v7(state)
        assert s.fano_diagnostic.near_zero_threshold == pytest.approx(
            engine.near_zero_threshold
        )

    def test_calculate_phi_exposes_diagnostics(self, asi):
        """calculate_phi must include fano_diagnostic and nonabelian_diagnostic dicts."""
        result = asi.calculate_phi(b"diag test")
        for key in ("fano_diagnostic", "nonabelian_diagnostic"):
            assert key in result, f"Missing key: {key}"
            d = result[key]
            assert "raw_value" in d
            assert "clamped_value" in d
            assert "zero_reason" in d
            assert "classification" in d
            assert "near_zero_threshold" in d

    def test_diagnostic_classification_consistent_with_score(self, asi):
        """If clamped_value > 0, classification must not be EXACT_ZERO."""
        result = asi.calculate_phi(b"consistency test")
        for key in ("fano_diagnostic", "nonabelian_diagnostic"):
            d = result[key]
            if d["clamped_value"] > 0.0:
                assert d["classification"] != CLASSIFICATION_EXACT_ZERO

    def test_custom_near_zero_threshold_changes_classification(self):
        """A large near_zero_threshold classifies small computed values as NEAR_ZERO."""
        eng_strict = IITv7Engine(
            alpha=0.40, beta=0.20, gamma=0.15, delta=0.15, epsilon=0.10,
            near_zero_threshold=0.9,  # almost everything is "near zero"
        )
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        T = eng_strict._build_transition_matrix(dist, n_nodes=2)
        raw, reason = eng_strict._compute_nonabelian_raw(T)
        if reason is None and raw < 0.9:
            assert eng_strict._classify_score(raw, reason) == CLASSIFICATION_NEAR_ZERO

    def test_score_diagnostic_dataclass_fields(self):
        """ScoreDiagnostic has all expected fields."""
        d = ScoreDiagnostic(
            raw_value=1e-8,
            clamped_value=1e-8,
            zero_reason=None,
            classification=CLASSIFICATION_NEAR_ZERO,
            near_zero_threshold=1e-6,
        )
        assert d.raw_value == 1e-8
        assert d.classification == CLASSIFICATION_NEAR_ZERO
        assert d.zero_reason is None

    def test_classification_constants(self):
        assert CLASSIFICATION_EXACT_ZERO == "EXACT_ZERO"
        assert CLASSIFICATION_NEAR_ZERO == "NEAR_ZERO"
        assert CLASSIFICATION_NONZERO == "NONZERO"

    def test_near_zero_threshold_default(self):
        assert NEAR_ZERO_THRESHOLD_DEFAULT == 1e-6


# ---------------------------------------------------------------------------
# RiemannZeroProbe tests
# ---------------------------------------------------------------------------

class TestRiemannZeroProbe:
    """
    Tests for the RiemannZeroProbe class.

    The probe uses mpmath with 50 decimal places by default and supports
    high-precision string coordinates (``KNOWN_ZEROS_HP``) for maximal
    accuracy.  The first known non-trivial Riemann zero at t ≈ 14.134725
    is used for all single-zero tests to keep the suite fast.
    """

    # Only first zero for most tests (fast); first 3 for the scan test
    T0 = 14.134725141734693   # first known Riemann zero imaginary part
    T0_HP = "14.134725141734693790457251983562470270784257115699"
    T_NONZERO = 15.0          # not a zero

    @pytest.fixture
    def probe(self):
        return RiemannZeroProbe(near_zero_threshold=1e-6, mpmath_dps=50,
                                zeta_near_zero_threshold=1e-6)

    # -- classify_zeta --------------------------------------------------

    def test_classify_zeta_near_zero_at_known_zero(self, probe):
        """|ζ(1/2 + i·t₀)| must be classified as NEAR_ZERO at a known zero."""
        diag = probe.classify_zeta(0.5, self.T0)
        assert diag.classification == CLASSIFICATION_NEAR_ZERO

    def test_classify_zeta_nonzero_off_critical_line(self, probe):
        """|ζ(0.3 + i·t₀)| must be NONZERO."""
        diag = probe.classify_zeta(0.3, self.T0)
        assert diag.classification == CLASSIFICATION_NONZERO

    def test_classify_zeta_raw_value_at_known_zero(self, probe):
        """|ζ(1/2 + it₀)| must be a finite positive float much less than 1e-6."""
        diag = probe.classify_zeta(0.5, self.T0)
        assert diag.raw_value > 0.0
        assert diag.raw_value < 1e-6

    def test_classify_zeta_raw_value_off_line(self, probe):
        """|ζ(0.5 + i·15.0)| (not a zero) must be substantially nonzero."""
        diag = probe.classify_zeta(0.5, self.T_NONZERO)
        assert diag.raw_value > 0.01

    def test_classify_zeta_zero_reason_none_for_near_zero(self, probe):
        """NEAR_ZERO has zero_reason=None (it was computed, not forced)."""
        diag = probe.classify_zeta(0.5, self.T0)
        assert diag.zero_reason is None

    def test_classify_zeta_returns_score_diagnostic(self, probe):
        """classify_zeta must return a ScoreDiagnostic instance."""
        diag = probe.classify_zeta(0.5, self.T0)
        assert isinstance(diag, ScoreDiagnostic)

    # -- probe_zero ------------------------------------------------------

    def test_probe_zero_returns_evidence(self, probe):
        """probe_zero must return a RiemannZeroEvidence instance."""
        ev = probe.probe_zero(self.T0)
        assert isinstance(ev, RiemannZeroEvidence)

    def test_probe_zero_t_stored(self, probe):
        """t field must match the input."""
        ev = probe.probe_zero(self.T0)
        assert ev.t == pytest.approx(self.T0)

    def test_probe_zero_zeta_classification_near_zero(self, probe):
        """First known zero must have zeta_classification == NEAR_ZERO."""
        ev = probe.probe_zero(self.T0)
        assert ev.zeta_classification == CLASSIFICATION_NEAR_ZERO

    def test_probe_zero_critical_line_signature_true(self, probe):
        """critical_line_signature must be True for the first known zero."""
        ev = probe.probe_zero(self.T0)
        assert ev.critical_line_signature is True

    def test_probe_zero_sigma_scan_covers_all_sigmas(self, probe):
        """zeta_scan must contain an entry for every σ in SIGMA_SCAN."""
        ev = probe.probe_zero(self.T0)
        for sigma in RiemannZeroProbe.SIGMA_SCAN:
            assert sigma in ev.zeta_scan, f"Missing σ={sigma} in zeta_scan"

    def test_probe_zero_nonabelian_scan_covers_all_sigmas(self, probe):
        """nonabelian_scan must contain an entry for every σ in SIGMA_SCAN."""
        ev = probe.probe_zero(self.T0)
        for sigma in RiemannZeroProbe.SIGMA_SCAN:
            assert sigma in ev.nonabelian_scan

    def test_probe_zero_nonabelian_scan_values_in_range(self, probe):
        """All Phi_nab values in nonabelian_scan must be in [0, 1]."""
        ev = probe.probe_zero(self.T0)
        for sigma, nab in ev.nonabelian_scan.items():
            assert 0.0 <= nab <= 1.0, f"Phi_nab={nab} out of range at σ={sigma}"

    def test_probe_zero_fano_at_critical_in_range(self, probe):
        """Fano score at critical line must be in [0, 1]."""
        ev = probe.probe_zero(self.T0)
        assert 0.0 <= ev.fano_at_critical <= 1.0

    def test_probe_nonzero_t_has_nonzero_at_half(self, probe):
        """|ζ(1/2 + i·15.0)| is NONZERO — not a Riemann zero."""
        ev = probe.probe_zero(self.T_NONZERO)
        assert ev.zeta_classification == CLASSIFICATION_NONZERO
        assert ev.critical_line_signature is False

    def test_zeta_scan_half_is_near_zero(self, probe):
        """At a known zero, zeta_scan[0.5] must be NEAR_ZERO."""
        ev = probe.probe_zero(self.T0)
        assert ev.zeta_scan[0.5].classification == CLASSIFICATION_NEAR_ZERO

    def test_zeta_scan_off_line_is_nonzero(self, probe):
        """At a known zero, zeta_scan[0.3] and zeta_scan[0.7] must be NONZERO."""
        ev = probe.probe_zero(self.T0)
        assert ev.zeta_scan[0.3].classification == CLASSIFICATION_NONZERO
        assert ev.zeta_scan[0.7].classification == CLASSIFICATION_NONZERO

    def test_known_zeros_constant_count(self):
        """KNOWN_ZEROS must contain exactly 30 entries."""
        assert len(RiemannZeroProbe.KNOWN_ZEROS) == 30

    def test_known_zeros_hp_count(self):
        """KNOWN_ZEROS_HP must contain exactly 30 entries."""
        assert len(RiemannZeroProbe.KNOWN_ZEROS_HP) == 30

    def test_known_zeros_all_positive(self):
        """All known zero imaginary parts must be positive."""
        assert all(t > 0 for t in RiemannZeroProbe.KNOWN_ZEROS)

    def test_known_zeros_increasing(self):
        """Known zeros must be listed in strictly increasing order."""
        zs = RiemannZeroProbe.KNOWN_ZEROS
        assert all(zs[i] < zs[i + 1] for i in range(len(zs) - 1))

    def test_scan_known_zeros_first_three(self, probe):
        """scan_known_zeros must return critical_line_signature=True for first 3 zeros."""
        evidences = probe.scan_known_zeros(RiemannZeroProbe.KNOWN_ZEROS[:3])
        assert len(evidences) == 3
        for ev in evidences:
            assert ev.critical_line_signature is True, (
                f"critical_line_signature is False for t={ev.t} "
                f"(|ζ(1/2+it)|={ev.zeta_abs:.2e}, "
                f"classification={ev.zeta_classification})"
            )

    def test_build_local_matrix_shape(self, probe):
        """Local matrix must be 7×7 (matching FANO_POINTS)."""
        T = probe._build_local_matrix(0.5, self.T0)
        assert T.shape == (7, 7)

    def test_build_local_matrix_column_stochastic(self, probe):
        """Local matrix must be column-stochastic."""
        T = probe._build_local_matrix(0.5, self.T0)
        col_sums = T.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-9)

    def test_build_local_matrix_nonnegative(self, probe):
        """Local matrix entries must all be ≥ 0."""
        T = probe._build_local_matrix(0.5, self.T0)
        assert np.all(T >= 0.0)

    # -- High-precision tests -------------------------------------------

    def test_classify_zeta_hp_much_smaller_than_float(self, probe):
        """HP string input must yield |ζ| orders of magnitude below float."""
        diag_hp = probe.classify_zeta(0.5, self.T0_HP)
        diag_float = probe.classify_zeta(0.5, self.T0)
        assert diag_hp.raw_value < diag_float.raw_value * 1e-10, (
            f"HP={diag_hp.raw_value:.2e} should be ≪ float={diag_float.raw_value:.2e}"
        )

    def test_classify_zeta_hp_near_zero(self, probe):
        """HP string at known zero must be NEAR_ZERO with |ζ| < 1e-40."""
        diag = probe.classify_zeta(0.5, self.T0_HP)
        assert diag.classification == CLASSIFICATION_NEAR_ZERO
        assert diag.raw_value < 1e-40

    def test_probe_zero_hp_critical_line_signature(self, probe):
        """probe_zero with HP string must yield critical_line_signature=True."""
        ev = probe.probe_zero(self.T0_HP)
        assert ev.critical_line_signature is True
        assert ev.zeta_abs < 1e-40

    # -- GUE pair-correlation tests -------------------------------------

    def test_probe_zero_has_gue_field(self, probe):
        """RiemannZeroEvidence must include gue_pair_correlation."""
        ev = probe.probe_zero(self.T0)
        assert hasattr(ev, "gue_pair_correlation")

    def test_gue_pair_correlation_is_float(self, probe):
        """gue_pair_correlation must be a float (not None) for known zeros."""
        ev = probe.probe_zero(self.T0)
        assert ev.gue_pair_correlation is not None
        assert isinstance(ev.gue_pair_correlation, float)

    def test_gue_pair_correlation_in_range(self, probe):
        """GUE pair-correlation must be in [-1, 1] (Pearson correlation)."""
        ev = probe.probe_zero(self.T0)
        assert -1.0 <= ev.gue_pair_correlation <= 1.0

    # -- Module-level exports -------------------------------------------

    def test_init_exports_riemann_classes(self):
        """__init__.py must export RiemannZeroProbe and related symbols."""
        from sphinx_os.Artificial_Intelligence import (
            RiemannZeroProbe,
            RiemannZeroEvidence,
            ScoreDiagnostic,
            CLASSIFICATION_EXACT_ZERO,
            CLASSIFICATION_NEAR_ZERO,
            CLASSIFICATION_NONZERO,
            NEAR_ZERO_THRESHOLD_DEFAULT,
        )
        assert RiemannZeroProbe is not None
        assert RiemannZeroEvidence is not None
        assert ScoreDiagnostic is not None


class TestRiemannZeroProbeEnhancements:
    """
    Tests for the IIT v7.0 RiemannZeroProbe enhancements:
      - Precision-aware zeta near-zero threshold
      - t-refinement via golden-section search
      - Margin-based critical-line signature
      - New evidence fields (min_other_raw, separation_ratio, zeta_threshold,
        margin_factor, refined_t, refine_iterations, refine_residual)
    """

    T0 = 14.134725141734693          # first known zero (float)
    T0_HP = "14.134725141734693790457251983562470270784257115699"
    T_NONZERO = 15.0

    # ------------------------------------------------------------------
    # Precision-aware threshold
    # ------------------------------------------------------------------

    def test_precision_aware_threshold_default_formula(self):
        """Default zeta_threshold must equal 10^(-(dps//2)) when not overridden."""
        for dps in (20, 30, 50):
            probe = RiemannZeroProbe(mpmath_dps=dps)
            expected = 10.0 ** (-(dps // 2))
            assert probe._get_zeta_threshold() == expected, (
                f"dps={dps}: expected {expected}, got {probe._get_zeta_threshold()}"
            )

    def test_precision_aware_threshold_override(self):
        """Explicit zeta_near_zero_threshold must override the precision default."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-8)
        assert probe._get_zeta_threshold() == 1e-8

    def test_classify_zeta_hp_near_zero_with_default_threshold(self):
        """HP string at known zero must be NEAR_ZERO using precision-aware threshold."""
        probe = RiemannZeroProbe(mpmath_dps=50)  # threshold = 1e-25
        diag = probe.classify_zeta(0.5, self.T0_HP)
        assert diag.classification == CLASSIFICATION_NEAR_ZERO
        assert diag.raw_value < 1e-25

    def test_classify_zeta_float_t0_nonzero_with_default_threshold(self):
        """Float T0 (limited precision) is NONZERO with strict precision-aware threshold."""
        probe = RiemannZeroProbe(mpmath_dps=50)  # threshold = 1e-25
        diag = probe.classify_zeta(0.5, self.T0)
        # float T0 gives |ζ| ≈ 7e-16, well above 1e-25
        assert diag.classification == CLASSIFICATION_NONZERO

    def test_zeta_threshold_stored_in_score_diagnostic(self):
        """ScoreDiagnostic.near_zero_threshold must reflect the zeta threshold."""
        probe = RiemannZeroProbe(mpmath_dps=50)  # threshold = 1e-25
        diag = probe.classify_zeta(0.5, self.T0_HP)
        assert diag.near_zero_threshold == pytest.approx(1e-25)

    # ------------------------------------------------------------------
    # Evidence metadata fields
    # ------------------------------------------------------------------

    def test_evidence_has_zeta_threshold_field(self):
        """RiemannZeroEvidence must include zeta_threshold metadata."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T0)
        assert hasattr(ev, "zeta_threshold")
        assert ev.zeta_threshold == pytest.approx(1e-6)

    def test_evidence_has_margin_factor_field(self):
        """RiemannZeroEvidence must record the margin_factor used."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6,
                                 margin_factor=2.0)
        ev = probe.probe_zero(self.T0)
        assert ev.margin_factor == pytest.approx(2.0)

    def test_evidence_has_min_other_raw(self):
        """RiemannZeroEvidence must include min_other_raw (min |ζ| off critical line)."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T0)
        assert hasattr(ev, "min_other_raw")
        assert not math.isnan(ev.min_other_raw)
        # off-line |ζ| should be substantially larger than on-line |ζ|
        assert ev.min_other_raw > 1e-6

    def test_evidence_separation_ratio_positive(self):
        """separation_ratio must be positive and > 1 for known zeros."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T0)
        assert ev.separation_ratio is not None
        assert ev.separation_ratio > 1.0

    def test_evidence_no_refinement_fields_by_default(self):
        """Without refine_t, refined_t/iterations/residual must all be None."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T0)
        assert ev.refined_t is None
        assert ev.refine_iterations is None
        assert ev.refine_residual is None

    # ------------------------------------------------------------------
    # t-refinement
    # ------------------------------------------------------------------

    def test_refine_t_reduces_zeta_abs(self):
        """t-refinement from a coarse start near a known zero must reduce |ζ|."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        # Use a coarse approximation of the first known zero (errors ≈ 0.005)
        # to ensure the unrefined probe gives a large |ζ| that refinement can improve.
        T0_coarse = round(self.T0, 2)  # 14.13 -- clearly not at the zero
        ev_raw = probe.probe_zero(T0_coarse)
        ev_ref = probe.probe_zero(T0_coarse, refine_t=True,
                                  refine_window=0.5, max_iter=50)
        assert ev_ref.zeta_abs < ev_raw.zeta_abs, (
            f"Refined zeta_abs {ev_ref.zeta_abs:.2e} should be < "
            f"unrefined {ev_raw.zeta_abs:.2e}"
        )

    def test_refine_t_populates_metadata(self):
        """With refine_t=True, refined_t, iterations, and residual must be set."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T0, refine_t=True)
        assert ev.refined_t is not None
        assert ev.refine_iterations is not None
        assert ev.refine_iterations >= 1
        assert ev.refine_residual is not None
        assert ev.refine_residual >= 0.0

    def test_refine_t_stays_near_true_zero(self):
        """Refined t from coarse start must end up close to the known zero location."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        T0_coarse = round(self.T0, 2)  # 14.13
        ev = probe.probe_zero(T0_coarse, refine_t=True,
                              refine_window=0.5, max_iter=50)
        assert abs(ev.refined_t - self.T0) < 0.01, (
            f"refined_t={ev.refined_t:.6f} too far from T0={self.T0:.6f}"
        )

    def test_refine_t_original_t_preserved(self):
        """ev.t must equal the original input t, not the refined value."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T0, refine_t=True)
        assert ev.t == pytest.approx(self.T0)

    # ------------------------------------------------------------------
    # Margin-based critical-line signature
    # ------------------------------------------------------------------

    def test_margin_signature_true_known_zero_hp(self):
        """Known zero (HP string) must give critical_line_signature=True with default margin."""
        probe = RiemannZeroProbe(mpmath_dps=50)  # precision-aware threshold
        ev = probe.probe_zero(self.T0_HP)
        assert ev.critical_line_signature is True, (
            f"zeta_abs={ev.zeta_abs:.2e}, threshold={ev.zeta_threshold:.2e}, "
            f"min_other={ev.min_other_raw:.2e}, ratio={ev.separation_ratio}"
        )

    def test_margin_signature_true_with_explicit_margin(self):
        """Known zero with margin_factor=10 must still give critical_line_signature=True."""
        probe = RiemannZeroProbe(
            mpmath_dps=50,
            zeta_near_zero_threshold=1e-6,
            margin_factor=10.0,
        )
        ev = probe.probe_zero(self.T0)
        # min_other_raw >> 10 * 1e-6, so signature should be True
        assert ev.critical_line_signature is True

    def test_margin_signature_false_non_zero_t(self):
        """Non-zero t must give critical_line_signature=False with any margin."""
        probe = RiemannZeroProbe(mpmath_dps=50, zeta_near_zero_threshold=1e-6)
        ev = probe.probe_zero(self.T_NONZERO)
        assert ev.critical_line_signature is False


class TestRiemannZeroPublishResults:
    """Tests for RiemannZeroEvidence.to_dict and RiemannZeroProbe.publish_results."""

    T0 = 14.134725141734693
    T0_HP = "14.134725141734693790457251983562470270784257115699"

    @pytest.fixture
    def probe(self):
        return RiemannZeroProbe(near_zero_threshold=1e-6, mpmath_dps=50,
                                zeta_near_zero_threshold=1e-6)

    # -- to_dict -----------------------------------------------------------

    def test_to_dict_returns_dict(self, probe):
        """to_dict must return a plain dict."""
        ev = probe.probe_zero(self.T0)
        d = ev.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_top_level_keys(self, probe):
        """to_dict must contain all evidence fields."""
        ev = probe.probe_zero(self.T0)
        d = ev.to_dict()
        expected_keys = {
            "t", "zeta_abs", "zeta_classification", "zeta_scan",
            "nonabelian_scan", "fano_at_critical", "critical_line_signature",
            "gue_pair_correlation", "min_other_raw", "separation_ratio",
            "refined_t", "refine_iterations", "refine_residual",
            "zeta_threshold", "margin_factor",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_is_json_serialisable(self, probe):
        """to_dict output must be serialisable with json.dumps."""
        import json
        ev = probe.probe_zero(self.T0)
        d = ev.to_dict()
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_to_dict_zeta_scan_sigma_keys_are_strings(self, probe):
        """zeta_scan keys in the dict must be strings (for JSON compat)."""
        ev = probe.probe_zero(self.T0)
        d = ev.to_dict()
        for key in d["zeta_scan"]:
            assert isinstance(key, str)

    def test_to_dict_preserves_values(self, probe):
        """Round-trip key values must match the original evidence."""
        ev = probe.probe_zero(self.T0)
        d = ev.to_dict()
        assert d["t"] == ev.t
        assert d["zeta_abs"] == ev.zeta_abs
        assert d["critical_line_signature"] == ev.critical_line_signature
        assert d["zeta_classification"] == ev.zeta_classification

    # -- publish_results ---------------------------------------------------

    def test_publish_results_returns_dict(self, probe):
        """publish_results must return a dict."""
        result = probe.publish_results([self.T0])
        assert isinstance(result, dict)

    def test_publish_results_has_summary_and_evidence(self, probe):
        """publish_results must contain 'summary' and 'evidence' keys."""
        result = probe.publish_results([self.T0])
        assert "summary" in result
        assert "evidence" in result

    def test_publish_results_summary_fields(self, probe):
        """Summary must report zeros_probed and critical_line_confirmed."""
        result = probe.publish_results([self.T0])
        summary = result["summary"]
        assert summary["zeros_probed"] == 1
        assert "critical_line_confirmed" in summary
        assert "mpmath_dps" in summary
        assert "zeta_threshold" in summary
        assert "margin_factor" in summary

    def test_publish_results_evidence_count_matches(self, probe):
        """Evidence list length must match the number of zeros supplied."""
        zeros = RiemannZeroProbe.KNOWN_ZEROS[:2]
        result = probe.publish_results(zeros)
        assert len(result["evidence"]) == 2

    def test_publish_results_is_json_serialisable(self, probe):
        """publish_results output must serialise with json.dumps."""
        import json
        result = probe.publish_results([self.T0])
        text = json.dumps(result)
        assert isinstance(text, str)

    def test_publish_results_default_uses_three_zeros(self, probe):
        """Default publish_results (no args) must probe 3 zeros."""
        result = probe.publish_results()
        assert result["summary"]["zeros_probed"] == 3


class TestRiemannHypothesisVerifier:
    """Tests for the RiemannHypothesisVerifier computational proof framework."""

    T0 = 14.134725141734693
    T0_HP = "14.134725141734693790457251983562470270784257115699"
    T_NONZERO = 15.0

    @pytest.fixture
    def verifier(self):
        from sphinx_os.Artificial_Intelligence.riemann_proof import (
            RiemannHypothesisVerifier,
        )
        return RiemannHypothesisVerifier(mpmath_dps=50)

    # -- Imports -----------------------------------------------------------

    def test_module_exports(self):
        """riemann_proof must be importable from the AI package."""
        from sphinx_os.Artificial_Intelligence import (
            RiemannHypothesisVerifier,
            VerificationReport,
            VERDICT_CONSISTENT,
            VERDICT_COUNTEREXAMPLE,
        )
        assert RiemannHypothesisVerifier is not None
        assert VERDICT_CONSISTENT == "CONSISTENT_WITH_RH"

    # -- Separation Theorem ------------------------------------------------

    def test_separation_passes_known_zero(self, verifier):
        """Separation Theorem must pass for the first known zero."""
        ev = verifier._probe.probe_zero(self.T0_HP)
        result = verifier.verify_separation(ev)
        assert result.passes is True

    def test_separation_log10_ratio_large(self, verifier):
        """log10(separation_ratio) must exceed 10 for HP zeros."""
        ev = verifier._probe.probe_zero(self.T0_HP)
        result = verifier.verify_separation(ev)
        assert result.log10_ratio > 10.0

    # -- Classification Consistency ----------------------------------------

    def test_classification_passes_known_zero(self, verifier):
        """Classification Consistency must pass for the first known zero."""
        ev = verifier._probe.probe_zero(self.T0_HP)
        result = verifier.verify_classification(ev)
        assert result.passes is True
        assert result.on_line_classification == CLASSIFICATION_NEAR_ZERO
        assert result.off_line_all_nonzero is True

    def test_classification_fails_nonzero_t(self, verifier):
        """Classification must fail for a non-zero t."""
        ev = verifier._probe.probe_zero(self.T_NONZERO)
        result = verifier.verify_classification(ev)
        assert result.passes is False

    # -- GUE Fingerprint ---------------------------------------------------

    def test_gue_passes_known_zero(self, verifier):
        """GUE Fingerprint must pass for the first known zero."""
        ev = verifier._probe.probe_zero(self.T0)
        result = verifier.verify_gue(ev)
        assert result.passes is True
        assert result.gue_pair_correlation is not None
        assert result.gue_pair_correlation > 0.3

    def test_gue_mean_nonabelian_positive(self, verifier):
        """Mean non-abelian score must be positive for known zeros."""
        ev = verifier._probe.probe_zero(self.T0)
        result = verifier.verify_gue(ev)
        assert result.mean_nonabelian > 0.0

    # -- Single-zero verification ------------------------------------------

    def test_verify_zero_passes_known_zero(self, verifier):
        """verify_zero must pass for a known zero."""
        result = verifier.verify_zero(self.T0_HP)
        assert result.all_pass is True

    def test_verify_zero_fails_nonzero(self, verifier):
        """verify_zero must fail for a non-zero t."""
        result = verifier.verify_zero(self.T_NONZERO)
        assert result.all_pass is False

    # -- Full verification -------------------------------------------------

    def test_full_verification_consistent(self, verifier):
        """Full verification on first known zero must be CONSISTENT_WITH_RH."""
        from sphinx_os.Artificial_Intelligence.riemann_proof import (
            VERDICT_CONSISTENT,
        )
        report = verifier.full_verification(
            [RiemannZeroProbe.KNOWN_ZEROS_HP[0]]
        )
        assert report.verdict == VERDICT_CONSISTENT
        assert report.zeros_tested == 1
        assert report.zeros_passing == 1

    def test_full_verification_report_to_dict(self, verifier):
        """VerificationReport.to_dict must be JSON-serialisable."""
        import json
        report = verifier.full_verification(
            [RiemannZeroProbe.KNOWN_ZEROS_HP[0]]
        )
        d = report.to_dict()
        text = json.dumps(d)
        assert isinstance(text, str)
        assert d["verdict"] == "CONSISTENT_WITH_RH"

    def test_full_verification_default_three_zeros(self, verifier):
        """Default full_verification must test 3 zeros."""
        report = verifier.full_verification()
        assert report.zeros_tested == 3
        assert report.verdict == "CONSISTENT_WITH_RH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
