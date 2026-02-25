"""
Tests for SKYNT-BTC hard fork and Spectral IIT PoW.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.blockchain.btc_hard_fork import (
    BTC_GENESIS_HASH,
    BTC_GENESIS_TIMESTAMP,
    SKYNTBTCParams,
    SKYNTBTCGenesis,
    SKYNTBTCChain,
)
from sphinx_os.mining.spectral_iit_pow import SpectralIITPow, _compute_iit_phi
from sphinx_os.mining.merge_miner import MergeMiningCoordinator
from sphinx_os.mining.miner import SphinxMiner
from sphinx_os.blockchain.core import SphinxSkynetBlockchain


# ---------------------------------------------------------------------------
# SpectralIITPow unit tests
# ---------------------------------------------------------------------------

class TestSpectralIITPow:
    """Tests for the Spectral IIT PoW engine."""

    def test_compute_hash_returns_64_char_hex(self):
        engine = SpectralIITPow()
        h = engine.compute_hash(b"test_block_data_42")
        assert isinstance(h, str)
        assert len(h) == 64
        # must be valid hex
        int(h, 16)

    def test_compute_hash_deterministic(self):
        engine = SpectralIITPow()
        data = b"deterministic_test"
        assert engine.compute_hash(data) == engine.compute_hash(data)

    def test_compute_phi_range(self):
        engine = SpectralIITPow()
        for seed in [b"alpha", b"beta", b"gamma_delta_epsilon"]:
            phi = engine.compute_phi(seed)
            assert 0.0 <= phi <= 1.0, f"phi out of range: {phi}"

    def test_compute_phi_deterministic(self):
        engine = SpectralIITPow()
        data = b"phi_determinism"
        assert engine.compute_phi(data) == engine.compute_phi(data)

    def test_compute_phi_score_range(self):
        """Legacy [200, 1000] scale."""
        engine = SpectralIITPow()
        score = engine.compute_phi_score(b"score_test")
        assert 200.0 <= score <= 1000.0

    def test_meets_difficulty_easy(self):
        """A hash of all zeros trivially meets any difficulty."""
        easy_hash = "0" * 64
        assert SpectralIITPow.meets_difficulty(easy_hash, 1)

    def test_meets_difficulty_impossible(self):
        """A hash of all 'f' should not meet any realistic difficulty."""
        hard_hash = "f" * 64
        # difficulty=1 target is 2^255, 0xfff...f > 2^255
        assert not SpectralIITPow.meets_difficulty(hard_hash, 2 ** 30)

    def test_iit_phi_independent_helper(self):
        phi = _compute_iit_phi(b"standalone_test")
        assert 0.0 <= phi <= 1.0

    def test_mine_low_difficulty(self):
        """With a very low difficulty, mining should succeed quickly."""
        engine = SpectralIITPow(phi_threshold=0.0)  # no Φ gate
        nonce, h, phi = engine.mine("block_data", difficulty=1, max_attempts=200)
        assert nonce is not None
        assert h is not None
        assert phi is not None and 0.0 <= phi <= 1.0

    def test_mine_with_phi_score_low_difficulty(self):
        """mine_with_phi_score should return phi in [200, 1000]."""
        engine = SpectralIITPow(phi_threshold=0.0)
        nonce, h, phi_score = engine.mine_with_phi_score(
            "block_data_score", difficulty=1, max_attempts=200
        )
        assert nonce is not None
        assert 200.0 <= phi_score <= 1000.0


# ---------------------------------------------------------------------------
# SKYNTBTCParams tests
# ---------------------------------------------------------------------------

class TestSKYNTBTCParams:
    """Tests for SKYNT-BTC chain parameters."""

    def test_ticker(self):
        assert SKYNTBTCParams.TICKER == "SKYNT"

    def test_pow_algorithm(self):
        assert SKYNTBTCParams.POW_ALGORITHM == "spectral"

    def test_block_reward_genesis(self):
        assert SKYNTBTCParams.block_reward(0) == 50.0

    def test_block_reward_first_halving(self):
        assert SKYNTBTCParams.block_reward(840_000) == 25.0

    def test_block_reward_second_halving(self):
        assert SKYNTBTCParams.block_reward(1_680_000) == 12.5

    def test_block_reward_very_late(self):
        # After 64 halvings reward is 0
        assert SKYNTBTCParams.block_reward(840_000 * 65) == 0.0

    def test_max_supply(self):
        assert SKYNTBTCParams.MAX_SUPPLY == 42_000_000.0

    def test_iit_phi_threshold(self):
        assert 0.0 < SKYNTBTCParams.IIT_PHI_THRESHOLD <= 1.0


# ---------------------------------------------------------------------------
# SKYNTBTCGenesis tests
# ---------------------------------------------------------------------------

class TestSKYNTBTCGenesis:
    """Tests for the SKYNT-BTC genesis block."""

    @pytest.fixture(scope="class")
    def genesis(self):
        return SKYNTBTCGenesis.create()

    def test_genesis_index_is_zero(self, genesis):
        assert genesis.index == 0

    def test_genesis_previous_hash_is_btc_genesis(self, genesis):
        """Hard-fork anchor: must point to Bitcoin's genesis hash."""
        assert genesis.previous_hash == BTC_GENESIS_HASH

    def test_btc_genesis_hash_format(self):
        """Bitcoin genesis hash is a 64-char lowercase hex string."""
        assert len(BTC_GENESIS_HASH) == 64
        int(BTC_GENESIS_HASH, 16)  # must be valid hex

    def test_genesis_timestamp_matches_btc(self, genesis):
        assert genesis.timestamp == BTC_GENESIS_TIMESTAMP

    def test_genesis_pow_algorithm(self, genesis):
        assert genesis.pow_algorithm == "spectral"

    def test_genesis_phi_score_maximum(self, genesis):
        assert genesis.phi_score == 1000.0

    def test_genesis_has_coinbase(self, genesis):
        assert len(genesis.transactions) >= 1
        assert genesis.transactions[0].is_coinbase()

    def test_genesis_coinbase_address(self, genesis):
        coinbase = genesis.transactions[0]
        assert any(
            out.address == "SKYNT_GENESIS_ADDRESS" for out in coinbase.outputs
        )

    def test_genesis_fork_record_output(self, genesis):
        """The zero-value fork-record output must be present."""
        coinbase = genesis.transactions[0]
        fork_outs = [
            out for out in coinbase.outputs if out.address == "SKYNT_FORK_RECORD"
        ]
        assert len(fork_outs) == 1
        assert fork_outs[0].amount == 0.0

    def test_genesis_hash_is_set(self, genesis):
        assert genesis.hash != ""
        assert len(genesis.hash) == 64


# ---------------------------------------------------------------------------
# SKYNTBTCChain tests
# ---------------------------------------------------------------------------

class TestSKYNTBTCChain:
    """Tests for the SKYNT-BTC chain class."""

    @pytest.fixture
    def chain(self):
        return SKYNTBTCChain()

    def test_chain_initialises_with_genesis(self, chain):
        assert len(chain.chain) == 1
        assert chain.chain[0].index == 0

    def test_genesis_hard_forks_from_btc(self, chain):
        assert chain.chain[0].previous_hash == BTC_GENESIS_HASH

    def test_genesis_pow_is_spectral(self, chain):
        assert chain.chain[0].pow_algorithm == "spectral"

    def test_create_block_uses_spectral(self, chain):
        block = chain.create_block(miner_address="MINER_1", phi_score=600.0)
        assert block.pow_algorithm == "spectral"

    def test_create_block_index(self, chain):
        block = chain.create_block(miner_address="MINER_1")
        assert block.index == 1

    def test_add_block_rejects_wrong_algorithm(self, chain):
        block = chain.create_block(miner_address="MINER_1")
        block.pow_algorithm = "sha256"       # wrong algorithm
        block.hash = block.calculate_hash()
        assert chain.add_block(block) is False

    def test_add_block_accepts_spectral(self, chain):
        block = chain.create_block(miner_address="MINER_1")
        # block.pow_algorithm is already "spectral"
        block.hash = block.calculate_hash()
        result = chain.add_block(block)
        assert result is True
        assert len(chain.chain) == 2

    def test_get_chain_info_keys(self, chain):
        info = chain.get_chain_info()
        required_keys = {
            "chain", "ticker", "chain_length", "btc_fork_point",
            "pow_algorithm", "iit_phi_threshold", "deployed",
        }
        assert required_keys.issubset(info.keys())

    def test_get_chain_info_pow_algorithm(self, chain):
        info = chain.get_chain_info()
        assert info["pow_algorithm"] == "spectral"

    def test_get_chain_info_fork_point(self, chain):
        info = chain.get_chain_info()
        assert info["btc_fork_point"] == BTC_GENESIS_HASH

    def test_deploy_idempotent(self, chain):
        receipt1 = chain.deploy()
        receipt2 = chain.deploy()
        assert receipt1["status"] == "deployed"
        assert receipt1["deployed_at"] == receipt2["deployed_at"]

    def test_deploy_receipt_keys(self, chain):
        receipt = chain.deploy()
        required = {
            "status", "chain", "ticker", "genesis_hash",
            "btc_fork_point", "pow_algorithm", "iit_phi_threshold",
            "initial_block_reward", "halving_interval", "max_supply",
        }
        assert required.issubset(receipt.keys())

    def test_deploy_receipt_genesis_hash(self, chain):
        receipt = chain.deploy()
        assert receipt["genesis_hash"] == chain.chain[0].hash

    def test_genesis_balance(self, chain):
        """Genesis miner should hold the genesis coinbase reward."""
        balance = chain.get_balance("SKYNT_GENESIS_ADDRESS")
        # 50 SKYNT × 2.0 Φ-boost = 100 SKYNT at genesis
        assert balance == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Merge miner integration: 'skynt' is now a supported chain
# ---------------------------------------------------------------------------

class TestMergeMinerSkynt:
    """Tests that 'skynt' is accepted as an auxiliary chain."""

    def test_enable_skynt_chain(self):
        bc = SphinxSkynetBlockchain()
        primary = SphinxMiner(blockchain=bc, miner_address="TEST")
        coord = MergeMiningCoordinator(primary)
        coord.enable_chain("skynt")
        assert "skynt" in coord.enabled_chains

    def test_skynt_blocks_stat_initialised(self):
        bc = SphinxSkynetBlockchain()
        primary = SphinxMiner(blockchain=bc, miner_address="TEST")
        coord = MergeMiningCoordinator(primary)
        assert "skynt_blocks" in coord.get_stats()

    def test_unsupported_chain_raises(self):
        bc = SphinxSkynetBlockchain()
        primary = SphinxMiner(blockchain=bc, miner_address="TEST")
        coord = MergeMiningCoordinator(primary)
        with pytest.raises(ValueError):
            coord.enable_chain("doge")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
