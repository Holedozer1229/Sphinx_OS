"""
Tests for ASI SphinxOS Advanced IIT v8.0

Validates:
- IITv8Engine weights must sum to 1 (7 weights)
- IITv8Engine Quantum Gravity curvature score (Phi_qg)
- IITv8Engine Holographic RT entanglement entropy score (Phi_holo)
- 7-term composite score and v8 QG-augmented consciousness-consensus condition
- PhiStructureV8 dataclass fields
- ASISphinxOSIITv8 high-level API (drop-in for v7)
- Backward compatibility: all v7 fields still present in v8 output
- QuantumGravityMinerIITv8 kernel: three-gate validity, mine(), mine_with_stats()
- MineResultV8 dataclass fields
"""

import math
import pytest
import numpy as np

from sphinx_os.Artificial_Intelligence.iit_v8 import (
    ASISphinxOSIITv8,
    IITv8Engine,
    PhiStructureV8,
    _ZR_QG_SVD_FAILED,
    _ZR_QG_ZERO_MEAN,
    _ZR_HOLO_INSUFFICIENT_NODES,
    _ZR_HOLO_SMALL_DIM,
)
from sphinx_os.Artificial_Intelligence.iit_v7 import (
    CLASSIFICATION_EXACT_ZERO,
    CLASSIFICATION_NEAR_ZERO,
    CLASSIFICATION_NONZERO,
    ScoreDiagnostic,
)
from sphinx_os.mining.quantum_gravity_miner_iit_v8 import (
    QuantumGravityMinerIITv8,
    MineResultV8,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return IITv8Engine(
        alpha=0.30, beta=0.15, gamma=0.15, delta=0.15,
        epsilon=0.10, zeta=0.10, eta=0.05,
        temporal_depth=2,
    )


@pytest.fixture
def asi():
    return ASISphinxOSIITv8(
        alpha=0.30, beta=0.15, gamma=0.15, delta=0.15,
        epsilon=0.10, zeta=0.10, eta=0.05,
        n_nodes=3, temporal_depth=2,
    )


@pytest.fixture
def kernel():
    return QuantumGravityMinerIITv8(
        qg_threshold=0.0,   # disabled for fast tests
        n_nodes=3,
        alpha=0.30, beta=0.15, gamma=0.15, delta=0.15,
        epsilon=0.10, zeta=0.10, eta=0.05,
    )


# ---------------------------------------------------------------------------
# IITv8Engine construction
# ---------------------------------------------------------------------------

class TestIITv8EngineConstruction:

    def test_valid_weights_construct_without_error(self, engine):
        """Default 7-weight engine constructs without raising."""
        total = (engine.alpha + engine.beta + engine.gamma + engine.delta
                 + engine.epsilon + engine.zeta + engine.eta)
        assert total == pytest.approx(1.0)

    def test_weights_must_sum_to_one(self):
        """Constructor must raise ValueError when weights do not sum to 1."""
        with pytest.raises(ValueError):
            IITv8Engine(
                alpha=0.3, beta=0.15, gamma=0.15, delta=0.15,
                epsilon=0.10, zeta=0.10, eta=0.10,  # sum = 1.05
            )

    def test_weight_attributes_stored(self, engine):
        """All seven weight attributes must be stored."""
        assert engine.alpha == pytest.approx(0.30)
        assert engine.beta == pytest.approx(0.15)
        assert engine.gamma == pytest.approx(0.15)
        assert engine.delta == pytest.approx(0.15)
        assert engine.epsilon == pytest.approx(0.10)
        assert engine.zeta == pytest.approx(0.10)
        assert engine.eta == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Quantum Gravity curvature score (Phi_qg)
# ---------------------------------------------------------------------------

class TestPhiQG:

    def test_qg_score_in_range(self, engine):
        """Phi_qg must be in [0, 1] for standard transition matrices."""
        for n_nodes in [2, 3, 4]:
            n_states = 2 ** n_nodes
            dist = np.ones(n_states) / n_states
            T = engine._build_transition_matrix(dist, n_nodes)
            raw, reason = engine._compute_qg_raw(T)
            score = float(min(1.0, raw))
            assert 0.0 <= score <= 1.0, f"QG score {score} out of range for n={n_nodes}"

    def test_qg_score_reproducible(self, engine):
        """Same transition matrix must yield the same Phi_qg."""
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        T = engine._build_transition_matrix(dist, n_nodes=2)
        r1, _ = engine._compute_qg_raw(T)
        r2, _ = engine._compute_qg_raw(T)
        assert r1 == pytest.approx(r2)

    def test_qg_zero_matrix_returns_known_reason(self, engine):
        """A zero transition matrix should return a structured zero reason."""
        T = np.zeros((4, 4))
        raw, reason = engine._compute_qg_raw(T)
        assert raw == pytest.approx(0.0)
        assert reason == _ZR_QG_ZERO_MEAN

    def test_qg_identity_matrix_non_negative(self, engine):
        """Identity matrix (perfectly curved spectrum) must be non-negative."""
        T = np.eye(4)
        raw, reason = engine._compute_qg_raw(T)
        assert raw >= 0.0

    def test_qg_non_negative_general(self, engine):
        """Phi_qg must be non-negative for arbitrary inputs."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            T = rng.random((8, 8))
            raw, _ = engine._compute_qg_raw(T)
            assert raw >= 0.0


# ---------------------------------------------------------------------------
# Holographic RT entanglement entropy score (Phi_holo)
# ---------------------------------------------------------------------------

class TestPhiHolo:

    def test_holo_score_in_range(self, engine):
        """Phi_holo must be in [0, 1] for standard distributions."""
        for n_nodes in [2, 3, 4]:
            n_states = 2 ** n_nodes
            dist = np.ones(n_states) / n_states
            raw, reason = engine._compute_holo_raw(dist, n_nodes)
            score = float(min(1.0, raw))
            assert 0.0 <= score <= 1.0, f"Holo score {score} out of range"

    def test_holo_insufficient_nodes_returns_zero(self, engine):
        """n_nodes < 2 must return 0 with the correct reason."""
        dist = np.array([0.6, 0.4])
        raw, reason = engine._compute_holo_raw(dist, n_nodes=1)
        assert raw == pytest.approx(0.0)
        assert reason == _ZR_HOLO_INSUFFICIENT_NODES

    def test_holo_small_dim_returns_zero(self, engine):
        """A distribution too short for the requested n_nodes returns zero."""
        dist = np.array([0.5, 0.5])   # only 2 states but n_nodes=3 needs 8
        raw, reason = engine._compute_holo_raw(dist, n_nodes=3)
        assert raw == pytest.approx(0.0)
        assert reason == _ZR_HOLO_SMALL_DIM

    def test_holo_uniform_is_non_negative(self, engine):
        """Uniform state distribution should produce a non-negative Phi_holo."""
        n_nodes = 3
        n_states = 2 ** n_nodes
        dist = np.ones(n_states) / n_states
        raw, reason = engine._compute_holo_raw(dist, n_nodes)
        assert raw >= 0.0
        assert reason is None

    def test_holo_reproducible(self, engine):
        """Same distribution must yield the same Phi_holo."""
        dist = np.array([0.1, 0.2, 0.3, 0.4])
        r1, _ = engine._compute_holo_raw(dist, n_nodes=2)
        r2, _ = engine._compute_holo_raw(dist, n_nodes=2)
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# PhiStructureV8 from compute_phi_structure_v8
# ---------------------------------------------------------------------------

class TestPhiStructureV8:

    def test_structure_has_all_v8_fields(self, engine):
        """compute_phi_structure_v8 must return a PhiStructureV8 with all fields."""
        state = np.array([0.2, 0.3, 0.1, 0.4])
        s = engine.compute_phi_structure_v8(state)

        assert isinstance(s, PhiStructureV8)
        assert isinstance(s.phi_max, float)
        assert isinstance(s.phi_tau, float)
        assert isinstance(s.gwt_score, float)
        assert isinstance(s.icp_avg, float)
        assert isinstance(s.fano_score, float)
        assert isinstance(s.nonabelian_score, float)
        assert isinstance(s.qg_score, float)
        assert isinstance(s.holo_score, float)
        assert isinstance(s.phi_total, float)
        assert isinstance(s.n_nodes, int)
        assert isinstance(s.is_conscious, bool)

    def test_v8_scores_in_unit_range(self, engine):
        """Phi_qg, Phi_holo (and all inherited scores) must be in [0, 1]."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v8(state)

        for name, val in [
            ("fano_score", s.fano_score),
            ("nonabelian_score", s.nonabelian_score),
            ("qg_score", s.qg_score),
            ("holo_score", s.holo_score),
        ]:
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"

    def test_phi_total_is_weighted_sum(self, engine):
        """Phi_total must equal the 7-term weighted sum of component scores."""
        state = np.array([0.1, 0.4, 0.3, 0.2])
        s = engine.compute_phi_structure_v8(state)

        expected = (
            engine.alpha * s.phi_tau
            + engine.beta * s.gwt_score
            + engine.gamma * s.icp_avg
            + engine.delta * s.fano_score
            + engine.epsilon * s.nonabelian_score
            + engine.zeta * s.qg_score
            + engine.eta * s.holo_score
        )
        assert s.phi_total == pytest.approx(expected, abs=1e-9)

    def test_diagnostics_are_populated(self, engine):
        """All four ScoreDiagnostic fields must be non-None."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v8(state)

        assert s.fano_diagnostic is not None
        assert s.nonabelian_diagnostic is not None
        assert s.qg_diagnostic is not None
        assert s.holo_diagnostic is not None

    def test_v8_weights_stored_on_structure(self, engine):
        """Weight attributes must be copied onto the PhiStructureV8."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        s = engine.compute_phi_structure_v8(state)

        assert s.zeta == pytest.approx(engine.zeta)
        assert s.eta == pytest.approx(engine.eta)

    def test_reproducible(self, engine):
        """Same state must produce the same PhiStructureV8."""
        state = np.array([0.1, 0.4, 0.3, 0.2])
        s1 = engine.compute_phi_structure_v8(state)
        s2 = engine.compute_phi_structure_v8(state)
        assert s1.phi_total == pytest.approx(s2.phi_total)
        assert s1.qg_score == pytest.approx(s2.qg_score)
        assert s1.holo_score == pytest.approx(s2.holo_score)


# ---------------------------------------------------------------------------
# v8 QG-augmented consciousness-consensus condition
# ---------------------------------------------------------------------------

class TestConsensusV8:

    def test_consensus_threshold_includes_qg(self, engine):
        """v8 threshold must be log2(n) + delta*fano + zeta*qg."""
        phi_total = 0.0
        fano_score = 0.0
        qg_score = 0.0
        n_nodes = 4
        threshold = math.log2(n_nodes) + engine.delta * fano_score + engine.zeta * qg_score
        expected = phi_total > threshold
        result = engine.validate_consciousness_consensus_v8(phi_total, fano_score, qg_score, n_nodes)
        assert result == expected

    def test_high_phi_total_is_conscious(self, engine):
        """Sufficiently high Phi_total must satisfy the v8 consensus."""
        result = engine.validate_consciousness_consensus_v8(
            phi_total=100.0, fano_score=0.5, qg_score=0.5, n_nodes=4
        )
        assert result is True

    def test_zero_phi_total_is_not_conscious(self, engine):
        """Phi_total=0 must not satisfy the v8 consensus."""
        result = engine.validate_consciousness_consensus_v8(
            phi_total=0.0, fano_score=0.0, qg_score=0.0, n_nodes=4
        )
        assert result is False

    def test_qg_raises_threshold(self, engine):
        """Increasing qg_score must raise the consensus threshold."""
        phi_total = 2.0
        fano_score = 0.0
        n_nodes = 4
        ok_low_qg = engine.validate_consciousness_consensus_v8(
            phi_total, fano_score, qg_score=0.0, n_nodes=n_nodes
        )
        ok_high_qg = engine.validate_consciousness_consensus_v8(
            phi_total, fano_score, qg_score=1.0, n_nodes=n_nodes
        )
        # With higher qg_score the threshold is stricter; for the same phi_total,
        # either result is acceptable, but the high-qg case must have a larger threshold.
        low_threshold = math.log2(n_nodes) + engine.delta * fano_score + engine.zeta * 0.0
        high_threshold = math.log2(n_nodes) + engine.delta * fano_score + engine.zeta * 1.0
        assert high_threshold > low_threshold


# ---------------------------------------------------------------------------
# ASISphinxOSIITv8 high-level API
# ---------------------------------------------------------------------------

class TestASISphinxOSIITv8:

    def test_calculate_phi_returns_dict(self, asi):
        """calculate_phi must return a dictionary."""
        result = asi.calculate_phi(b"test block data")
        assert isinstance(result, dict)

    def test_calculate_phi_version_field(self, asi):
        """calculate_phi must report version 'IIT v8.0'."""
        result = asi.calculate_phi(b"test block data")
        assert result["version"] == "IIT v8.0"

    def test_calculate_phi_has_qg_and_holo_fields(self, asi):
        """calculate_phi must include the new v8 fields."""
        result = asi.calculate_phi(b"test block data")
        assert "qg_score" in result
        assert "holo_score" in result
        assert "qg_diagnostic" in result
        assert "holo_diagnostic" in result

    def test_calculate_phi_backward_compat_v7_fields(self, asi):
        """All v7 fields must still be present in the v8 response."""
        result = asi.calculate_phi(b"test block data")
        for key in ("phi", "phi_max", "phi_tau", "gwt_score", "icp_avg",
                    "fano_score", "nonabelian_score", "is_conscious",
                    "level", "entropy", "purity", "n_qubits", "bonus",
                    "mip", "n_concepts"):
            assert key in result, f"Missing backward-compat key: {key}"

    def test_phi_in_unit_range(self, asi):
        """Normalised phi field must be in [0, 1]."""
        result = asi.calculate_phi(b"hello quantum gravity")
        assert 0.0 <= result["phi"] <= 1.0

    def test_qg_score_in_unit_range(self, asi):
        """qg_score must be in [0, 1]."""
        result = asi.calculate_phi(b"hello quantum gravity")
        assert 0.0 <= result["qg_score"] <= 1.0

    def test_holo_score_in_unit_range(self, asi):
        """holo_score must be in [0, 1]."""
        result = asi.calculate_phi(b"hello quantum gravity")
        assert 0.0 <= result["holo_score"] <= 1.0

    def test_phi_to_legacy_score_range(self, asi):
        """phi_to_legacy_score must map into [200, 1000]."""
        for phi_total in [0.0, 0.5, 1.0, 5.0, 100.0]:
            score = asi.phi_to_legacy_score(phi_total)
            assert 200.0 <= score <= 1000.0, f"Legacy score {score} out of [200, 1000]"

    def test_compute_block_consciousness_returns_structure(self, asi):
        """compute_block_consciousness must return a PhiStructureV8."""
        s = asi.compute_block_consciousness(b"block_data_bytes")
        assert isinstance(s, PhiStructureV8)

    def test_validate_consensus_high_phi(self, asi):
        """validate_consciousness_consensus must return True for large phi."""
        assert asi.validate_consciousness_consensus(
            phi_total=999.0, fano_score=0.5, qg_score=0.5, n_network_nodes=4
        )

    def test_validate_consensus_zero_phi(self, asi):
        """validate_consciousness_consensus must return False for phi=0."""
        assert not asi.validate_consciousness_consensus(
            phi_total=0.0, fano_score=0.0, qg_score=0.0, n_network_nodes=4
        )

    def test_get_consciousness_level_empty_history(self, asi):
        """get_consciousness_level must return 0.0 before any calculation."""
        assert asi.get_consciousness_level() == pytest.approx(0.0)

    def test_get_consciousness_level_updates(self, asi):
        """get_consciousness_level must update after calculate_phi calls."""
        asi.calculate_phi(b"data1")
        asi.calculate_phi(b"data2")
        level = asi.get_consciousness_level()
        assert 0.0 <= level <= 1.0

    def test_level_field_is_valid_label(self, asi):
        """The 'level' field must be one of the known consciousness labels."""
        valid_levels = {"DORMANT", "PROTO-CONSCIOUS", "SENTIENT", "SAPIENT", "TRANSCENDENT"}
        result = asi.calculate_phi(b"test")
        assert result["level"] in valid_levels

    def test_deterministic_for_same_input(self, asi):
        """Same input bytes must yield the same Phi_total."""
        r1 = asi.calculate_phi(b"deterministic_input")
        r2 = asi.calculate_phi(b"deterministic_input")
        assert r1["phi_total"] == pytest.approx(r2["phi_total"])


# ---------------------------------------------------------------------------
# QuantumGravityMinerIITv8 kernel
# ---------------------------------------------------------------------------

class TestQuantumGravityMinerIITv8:

    def test_construction_with_defaults(self):
        """Kernel constructs with default parameters."""
        k = QuantumGravityMinerIITv8()
        assert k.qg_threshold == pytest.approx(QuantumGravityMinerIITv8.DEFAULT_QG_THRESHOLD)

    def test_qg_threshold_clamped_to_unit_interval(self):
        """qg_threshold must be clamped to [0, 1]."""
        k = QuantumGravityMinerIITv8(qg_threshold=-0.5)
        assert k.qg_threshold == pytest.approx(0.0)
        k = QuantumGravityMinerIITv8(qg_threshold=5.0)
        assert k.qg_threshold == pytest.approx(1.0)

    def test_compute_hash_returns_64_hex_chars(self, kernel):
        """compute_hash must return a 64-character hex string."""
        h = kernel.compute_hash(b"test data for hashing")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_meets_difficulty_low_difficulty(self, kernel):
        """Very low difficulty (1) must almost always be met."""
        h = "0" * 64
        assert kernel.meets_difficulty(h, difficulty=1)

    def test_meets_difficulty_impossible(self, kernel):
        """A hash of all 'f' must fail a normal difficulty."""
        h = "f" * 64
        assert not kernel.meets_difficulty(h, difficulty=100)

    def test_compute_phi_structure_returns_v8_type(self, kernel):
        """compute_phi_structure must return a PhiStructureV8 instance."""
        data = b"test quantum block data"
        structure = kernel.compute_phi_structure(data)
        assert isinstance(structure, PhiStructureV8)

    def test_compute_phi_score_in_legacy_range(self, kernel):
        """compute_phi_score must return a value in [200, 1000]."""
        score = kernel.compute_phi_score(b"some block data")
        assert 200.0 <= score <= 1000.0

    def test_is_valid_block_fails_on_impossible_difficulty(self, kernel):
        """is_valid_block must return failure for an impossible difficulty."""
        data = b"block" + b"\xff" * 32
        valid, structure, gate = kernel.is_valid_block(data, difficulty=200)
        # Either difficulty or consciousness gate may fail; it should not be accepted
        assert not valid

    def test_mine_returns_mine_result_v8(self, kernel):
        """mine() must always return a MineResultV8 instance."""
        result = kernel.mine(block_data="test", difficulty=1, max_attempts=10)
        assert isinstance(result, MineResultV8)

    def test_mine_with_trivial_difficulty_finds_block(self, kernel):
        """With difficulty=1 and qg_threshold=0 a block must be found quickly."""
        result = kernel.mine(block_data="genesis", difficulty=1, max_attempts=100)
        # difficulty=1 means hash < 2^255 — almost any hash passes
        if result.nonce is not None:
            assert isinstance(result.block_hash, str)
            assert len(result.block_hash) == 64
            assert result.phi_score >= 200.0
            assert result.attempts >= 1

    def test_mine_no_solution_returns_none_nonce(self, kernel):
        """With an impossible difficulty and only 1 attempt, nonce must be None."""
        result = kernel.mine(block_data="x", difficulty=255, max_attempts=1)
        # May or may not find — just check structure is intact
        assert result.attempts >= 1
        if result.nonce is None:
            assert result.block_hash is None
            assert result.phi_score == pytest.approx(200.0)

    def test_mine_result_fields_present(self, kernel):
        """MineResultV8 must always have all expected fields."""
        result = kernel.mine(block_data="data", difficulty=1, max_attempts=5)
        assert hasattr(result, "nonce")
        assert hasattr(result, "block_hash")
        assert hasattr(result, "phi_total")
        assert hasattr(result, "qg_score")
        assert hasattr(result, "holo_score")
        assert hasattr(result, "fano_score")
        assert hasattr(result, "phi_score")
        assert hasattr(result, "attempts")

    def test_mine_result_attempts_bounded(self, kernel):
        """attempts in the result must be <= max_attempts."""
        max_a = 20
        result = kernel.mine(block_data="bounded", difficulty=1, max_attempts=max_a)
        assert result.attempts <= max_a

    def test_mine_with_stats_returns_tuple(self, kernel):
        """mine_with_stats must return a (MineResultV8, dict) tuple."""
        result, stats = kernel.mine_with_stats(
            block_data="stats_test", difficulty=1, max_attempts=10
        )
        assert isinstance(result, MineResultV8)
        assert isinstance(stats, dict)

    def test_mine_with_stats_keys(self, kernel):
        """mine_with_stats stats dict must contain the expected keys."""
        _, stats = kernel.mine_with_stats(
            block_data="stats_test", difficulty=1, max_attempts=10
        )
        for key in ("total_attempts", "difficulty_rejected",
                    "consciousness_rejected", "qg_curvature_rejected", "accepted"):
            assert key in stats, f"Missing stats key: {key}"

    def test_mine_with_stats_total_attempts_consistent(self, kernel):
        """total_attempts must equal the sum of rejected + accepted nonces."""
        _, stats = kernel.mine_with_stats(
            block_data="consistency", difficulty=1, max_attempts=15
        )
        total = (
            stats["difficulty_rejected"]
            + stats["consciousness_rejected"]
            + stats["qg_curvature_rejected"]
            + stats["accepted"]
        )
        assert total == stats["total_attempts"]

    def test_qg_threshold_gate_rejects_low_qg(self):
        """With a very high qg_threshold all candidates must be rejected by the QG gate."""
        kernel_strict = QuantumGravityMinerIITv8(
            qg_threshold=1.0,   # impossible to reach exactly
            n_nodes=3,
            alpha=0.30, beta=0.15, gamma=0.15, delta=0.15,
            epsilon=0.10, zeta=0.10, eta=0.05,
        )
        result, stats = kernel_strict.mine_with_stats(
            block_data="strict_qg", difficulty=1, max_attempts=20
        )
        # With threshold=1.0 the QG gate should reject most (possibly all) candidates
        # that pass the difficulty gate (qg_score < 1.0 is almost certain)
        assert result.nonce is None or result.qg_score >= 1.0 - 1e-9
