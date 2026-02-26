"""
Tests for the BTC Wormhole bridge protocol.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.bridge.btc_wormhole import (
    BTCWormhole,
    WormholeStatus,
    WormholeTransfer,
    SpectralAttestation,
    WORMHOLE_VERSION,
    WORMHOLE_FEE_RATE,
    WORMHOLE_ROUTES,
    PHI_GATE_THRESHOLD,
    GUARDIAN_FEE_SHARE,
    MAX_PHI_DISCOUNT,
)


# ---------------------------------------------------------------------------
# BTCWormhole initialisation
# ---------------------------------------------------------------------------

class TestBTCWormholeInit:
    """Verify constructor defaults and basic attributes."""

    def test_default_guardian_count(self):
        wh = BTCWormhole()
        assert len(wh.guardians) == 9

    def test_default_required_signatures(self):
        wh = BTCWormhole()
        assert wh.required_signatures == 5

    def test_default_phi_threshold(self):
        wh = BTCWormhole()
        assert wh.phi_threshold == PHI_GATE_THRESHOLD

    def test_custom_guardian_count(self):
        wh = BTCWormhole(guardian_count=5, required_signatures=3)
        assert len(wh.guardians) == 5
        assert wh.required_signatures == 3

    def test_initial_stats_zeroed(self):
        wh = BTCWormhole()
        stats = wh.get_stats()
        assert stats["total_volume"] == 0.0
        assert stats["transfers_initiated"] == 0
        assert stats["version"] == WORMHOLE_VERSION


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------

class TestRouteValidation:
    """Verify supported and unsupported wormhole routes."""

    @pytest.mark.parametrize("src,dst", WORMHOLE_ROUTES)
    def test_valid_routes(self, src, dst):
        assert BTCWormhole.validate_route(src, dst)

    def test_invalid_route_unknown_chain(self):
        assert not BTCWormhole.validate_route("dogecoin", "sphinx")

    def test_invalid_route_same_chain(self):
        assert not BTCWormhole.validate_route("btc", "btc")

    def test_supported_routes_list(self):
        routes = BTCWormhole.supported_routes()
        assert len(routes) == len(WORMHOLE_ROUTES)
        assert all("source" in r and "destination" in r for r in routes)


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

class TestFeeCalculation:
    """Verify fee computation with and without Φ discount."""

    def test_base_fee_no_phi(self):
        fee, net = BTCWormhole.calculate_fee(1000.0, phi_score=0.0)
        assert fee == pytest.approx(1000.0 * WORMHOLE_FEE_RATE)
        assert net == pytest.approx(1000.0 - fee)

    def test_max_phi_discount(self):
        fee_no, _ = BTCWormhole.calculate_fee(1000.0, phi_score=0.0)
        fee_max, _ = BTCWormhole.calculate_fee(1000.0, phi_score=1.0)
        # With full Φ → 50 % discount
        assert fee_max == pytest.approx(fee_no * (1.0 - MAX_PHI_DISCOUNT))

    def test_fee_positive_for_positive_amount(self):
        fee, net = BTCWormhole.calculate_fee(42.0, phi_score=0.5)
        assert fee > 0
        assert net > 0
        assert net < 42.0

    def test_net_plus_fee_equals_amount(self):
        amount = 123.456
        fee, net = BTCWormhole.calculate_fee(amount, phi_score=0.3)
        assert fee + net == pytest.approx(amount)


# ---------------------------------------------------------------------------
# Spectral attestation
# ---------------------------------------------------------------------------

class TestSpectralAttestation:
    """Verify spectral hash attestation generation."""

    def test_attestation_fields(self):
        att = BTCWormhole.create_spectral_attestation("test_data", 0.7)
        assert isinstance(att.block_hash, str)
        assert len(att.block_hash) == 64
        assert isinstance(att.spectral_hash, str)
        assert len(att.spectral_hash) == 64
        assert att.phi_score == 0.7

    def test_attestation_deterministic(self):
        a1 = BTCWormhole.create_spectral_attestation("same", 0.5)
        a2 = BTCWormhole.create_spectral_attestation("same", 0.5)
        assert a1.block_hash == a2.block_hash
        assert a1.spectral_hash == a2.spectral_hash

    def test_attestation_different_for_different_data(self):
        a1 = BTCWormhole.create_spectral_attestation("alpha", 0.5)
        a2 = BTCWormhole.create_spectral_attestation("beta", 0.5)
        assert a1.spectral_hash != a2.spectral_hash

    def test_attestation_to_dict(self):
        att = BTCWormhole.create_spectral_attestation("dict_test", 0.6)
        d = att.to_dict()
        assert "block_hash" in d
        assert "spectral_hash" in d
        assert "zeta_weight" in d
        assert "phi_score" in d


# ---------------------------------------------------------------------------
# Transfer lifecycle — initiate
# ---------------------------------------------------------------------------

class TestInitiateTransfer:
    """Verify transfer initiation."""

    def test_initiate_returns_id(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "SENDER", "RECIP")
        assert tid is not None
        assert len(tid) == 64

    def test_initiate_invalid_route(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("doge", "sphinx", 1.0, "S", "R")
        assert tid is None

    def test_initiate_zero_amount(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 0.0, "S", "R")
        assert tid is None

    def test_initiate_negative_amount(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", -5.0, "S", "R")
        assert tid is None

    def test_initiate_locks_funds(self):
        wh = BTCWormhole()
        wh.initiate_transfer("btc", "sphinx", 10.0, "ALICE", "BOB")
        assert wh.get_locked_balance("ALICE") > 0

    def test_initiate_updates_stats(self):
        wh = BTCWormhole()
        wh.initiate_transfer("btc", "sphinx", 5.0, "S", "R")
        assert wh.stats["transfers_initiated"] == 1
        assert wh.stats["total_volume"] == 5.0

    def test_transfer_status_attested(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        t = wh.get_transfer(tid)
        assert t["status"] == WormholeStatus.ATTESTED.value


# ---------------------------------------------------------------------------
# Guardian signatures + Φ gate
# ---------------------------------------------------------------------------

class TestGuardianSignatures:
    """Verify multi-sig and Φ-gate consensus."""

    def _setup(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 10.0, "S", "R")
        return wh, tid

    def test_valid_signatures_pass(self):
        wh, tid = self._setup()
        sigs = wh.guardians[:5]
        assert wh.submit_guardian_signatures(tid, sigs, 0.6)

    def test_insufficient_signatures_fail(self):
        wh, tid = self._setup()
        sigs = wh.guardians[:3]
        assert not wh.submit_guardian_signatures(tid, sigs, 0.6)

    def test_low_phi_fails(self):
        wh, tid = self._setup()
        sigs = wh.guardians[:5]
        assert not wh.submit_guardian_signatures(tid, sigs, 0.3)  # below 0.5

    def test_unknown_transfer_id_fails(self):
        wh, _ = self._setup()
        sigs = wh.guardians[:5]
        assert not wh.submit_guardian_signatures("bad_id", sigs, 0.6)

    def test_double_submission_fails(self):
        wh, tid = self._setup()
        sigs = wh.guardians[:5]
        wh.submit_guardian_signatures(tid, sigs, 0.6)
        # Second submission should fail (already PHI_GATED)
        assert not wh.submit_guardian_signatures(tid, sigs, 0.6)

    def test_guardian_reward_tracked(self):
        wh, tid = self._setup()
        sigs = wh.guardians[:5]
        wh.submit_guardian_signatures(tid, sigs, 0.6)
        assert wh.stats["guardian_rewards"] > 0


# ---------------------------------------------------------------------------
# ZK proof
# ---------------------------------------------------------------------------

class TestZKProof:
    """Verify ZK proof generation and validation."""

    def test_proof_deterministic(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        t = wh.transfers[tid]
        p1 = BTCWormhole.generate_zk_proof(t)
        p2 = BTCWormhole.generate_zk_proof(t)
        assert p1 == p2

    def test_proof_verifies(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        t = wh.transfers[tid]
        proof = BTCWormhole.generate_zk_proof(t)
        assert BTCWormhole.verify_zk_proof(proof, t)

    def test_invalid_proof_rejected(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        t = wh.transfers[tid]
        assert not BTCWormhole.verify_zk_proof("bad_proof", t)


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

class TestFinalisation:
    """Verify transfer finalisation."""

    def _setup_gated(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 10.0, "S", "R", 0.6)
        sigs = wh.guardians[:5]
        wh.submit_guardian_signatures(tid, sigs, 0.6)
        return wh, tid

    def test_finalise_success(self):
        wh, tid = self._setup_gated()
        assert wh.finalise_transfer(tid)
        t = wh.get_transfer(tid)
        assert t["status"] == WormholeStatus.FINALISED.value

    def test_finalise_mints_wrapped_btc(self):
        wh, tid = self._setup_gated()
        wh.finalise_transfer(tid)
        assert wh.get_wrapped_balance("R") > 0

    def test_finalise_updates_stats(self):
        wh, tid = self._setup_gated()
        wh.finalise_transfer(tid)
        assert wh.stats["transfers_finalised"] == 1

    def test_finalise_without_gate_fails(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        assert not wh.finalise_transfer(tid)

    def test_finalise_nonexistent_fails(self):
        wh = BTCWormhole()
        assert not wh.finalise_transfer("nonexistent")

    def test_zk_proof_present_after_finalise(self):
        wh, tid = self._setup_gated()
        wh.finalise_transfer(tid)
        t = wh.get_transfer(tid)
        assert t["zk_proof"] is not None

    def test_finalised_at_timestamp(self):
        wh, tid = self._setup_gated()
        wh.finalise_transfer(tid)
        t = wh.get_transfer(tid)
        assert t["finalised_at"] is not None


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

class TestFailure:
    """Verify transfer failure handling."""

    def test_fail_unlocks_funds(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 5.0, "ALICE", "BOB")
        locked_before = wh.get_locked_balance("ALICE")
        assert locked_before > 0
        wh.fail_transfer(tid)
        assert wh.get_locked_balance("ALICE") == 0.0

    def test_fail_sets_status(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        wh.fail_transfer(tid)
        assert wh.get_transfer(tid)["status"] == WormholeStatus.FAILED.value

    def test_fail_finalised_rejected(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R", 0.6)
        wh.submit_guardian_signatures(tid, wh.guardians[:5], 0.6)
        wh.finalise_transfer(tid)
        assert not wh.fail_transfer(tid)

    def test_fail_nonexistent_rejected(self):
        wh = BTCWormhole()
        assert not wh.fail_transfer("bad_id")

    def test_fail_updates_stats(self):
        wh = BTCWormhole()
        tid = wh.initiate_transfer("btc", "sphinx", 1.0, "S", "R")
        wh.fail_transfer(tid)
        assert wh.stats["transfers_failed"] == 1


# ---------------------------------------------------------------------------
# End-to-end convenience method
# ---------------------------------------------------------------------------

class TestExecuteTransfer:
    """Verify the one-shot execute_transfer helper."""

    def test_e2e_success(self):
        wh = BTCWormhole()
        t = wh.execute_transfer("btc", "sphinx", 10.0, "ALICE", "BOB", 0.7)
        assert t is not None
        assert t.status == WormholeStatus.FINALISED

    def test_e2e_invalid_route(self):
        wh = BTCWormhole()
        t = wh.execute_transfer("doge", "sphinx", 1.0, "S", "R")
        assert t is None

    def test_e2e_low_phi_fails(self):
        wh = BTCWormhole()
        t = wh.execute_transfer("btc", "sphinx", 1.0, "S", "R", phi_score=0.3)
        assert t is None

    @pytest.mark.parametrize("src,dst", WORMHOLE_ROUTES)
    def test_e2e_all_routes(self, src, dst):
        wh = BTCWormhole()
        t = wh.execute_transfer(src, dst, 1.0, "S", "R", 0.6)
        assert t is not None
        assert t.status == WormholeStatus.FINALISED


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    """Verify to_dict output."""

    def test_transfer_to_dict_keys(self):
        wh = BTCWormhole()
        t = wh.execute_transfer("btc", "sphinx", 1.0, "SENDER", "RECIPIENT", 0.6)
        d = t.to_dict()
        expected_keys = {
            "transfer_id", "source_chain", "destination_chain", "amount",
            "sender", "recipient", "fee", "net_amount", "status",
            "attestation", "guardian_signatures", "collective_phi",
            "zk_proof", "failure_reason", "created_at", "finalised_at",
        }
        assert expected_keys == set(d.keys())

    def test_stats_keys(self):
        wh = BTCWormhole()
        stats = wh.get_stats()
        assert "version" in stats
        assert "fee_rate_bps" in stats
        assert "supported_routes" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
