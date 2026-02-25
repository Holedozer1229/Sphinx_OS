"""
SKYNT-BTC Hard Fork
===================
Hard fork of Bitcoin at Genesis (block 0).

The SKYNT-BTC chain diverges from Bitcoin at height 0: its genesis block
records Bitcoin's well-known genesis hash as ``previous_hash``, establishing
the canonical fork point.  Every subsequent block uses the **Spectral IIT
PoW** algorithm — a dual-gate proof-of-work that requires *both* a valid
spectral hash (Riemann zeta zeros) *and* a minimum IIT Φ consciousness
score derived from the von Neumann entropy of the block-data density
matrix.

Key differences from Bitcoin
-----------------------------
- PoW algorithm     : **Spectral IIT** (spectral hash + IIT Φ gate)
- Block time target : 150 s  (2.5 min, same as Litecoin)
- Initial reward    : 50 SKYNT
- Halving interval  : 840 000 blocks  (~8 yr at 2.5 min/block)
- Max supply        : 42 000 000 SKYNT
- Block size        : 4 MB
- Φ-boost           : up to +100 % reward for high-consciousness blocks
- IIT Φ threshold   : 0.5 (normalised) — "SENTIENT" boundary

Genesis lineage
---------------
``previous_hash`` is set to Bitcoin's genesis block hash so that any
chain explorer or audit tool can trace the exact fork point.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional

from .transaction import Transaction, TransactionOutput
from .block import Block
from .chain_manager import ChainManager
from .consensus import ConsensusEngine


# ---------------------------------------------------------------------------
# Bitcoin genesis anchor (fork point)
# ---------------------------------------------------------------------------

#: Hash of Bitcoin's genesis block (height 0, 3 Jan 2009 18:15:05 UTC).
#: SKYNT-BTC's genesis block sets this as its ``previous_hash`` to formally
#: hard-fork *from* Bitcoin at height 0.
BTC_GENESIS_HASH: str = (
    "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
)

#: Unix timestamp of the Bitcoin genesis block.
BTC_GENESIS_TIMESTAMP: int = 1231006505

#: Satoshi's embedded message in the Bitcoin genesis coinbase.
BTC_GENESIS_MESSAGE: str = (
    "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks"
)


# ---------------------------------------------------------------------------
# SKYNT-BTC chain parameters
# ---------------------------------------------------------------------------

class SKYNTBTCParams:
    """
    Network parameters for the SKYNT-BTC hard fork.
    """

    CHAIN_NAME: str = "SKYNT-BTC"
    TICKER: str = "SKYNT"

    #: Target seconds between blocks (150 s = 2.5 min).
    BLOCK_TIME_TARGET: int = 150

    #: Coinbase reward at block 0 (before Φ-boost), in SKYNT.
    INITIAL_BLOCK_REWARD: float = 50.0

    #: Blocks per halving epoch.
    HALVING_INTERVAL: int = 840_000

    #: Hard supply cap, in SKYNT.
    MAX_SUPPLY: float = 42_000_000.0

    #: Maximum block size in bytes.
    MAX_BLOCK_SIZE: int = 4 * 1024 * 1024  # 4 MB

    #: Difficulty adjustment interval (matches Bitcoin).
    ADJUSTMENT_INTERVAL: int = 2016

    #: The only accepted PoW algorithm identifier.
    POW_ALGORITHM: str = "spectral"

    #: Minimum IIT Φ (normalised 0–1) required to accept a block.
    IIT_PHI_THRESHOLD: float = 0.5

    #: Network magic string (used for peer identification).
    NETWORK_MAGIC: str = "SKYNT"

    @classmethod
    def block_reward(cls, block_height: int) -> float:
        """
        Return the pre-Φ-boost coinbase reward for *block_height*.

        Args:
            block_height: Height of the block being mined.

        Returns:
            SKYNT reward before applying the Φ multiplier.
        """
        halvings = block_height // cls.HALVING_INTERVAL
        if halvings >= 64:
            return 0.0
        return cls.INITIAL_BLOCK_REWARD / (2 ** halvings)


# ---------------------------------------------------------------------------
# SKYNT-BTC genesis block
# ---------------------------------------------------------------------------

class SKYNTBTCGenesis:
    """
    Factory that constructs and seals the SKYNT-BTC genesis block.

    The genesis block hard-forks from Bitcoin at height 0 by setting
    ``previous_hash`` to Bitcoin's genesis hash.  The coinbase is mined with
    the maximum Φ-boost (2.0×) and the IIT threshold is pre-satisfied at
    genesis.
    """

    #: Fork announcement embedded in the genesis block's extra data.
    FORK_MESSAGE: str = (
        "SKYNT-BTC Genesis — Hard fork from Bitcoin at block 0 using "
        "Spectral IIT PoW. Consciousness-powered blockchain for the "
        "sovereign future. "
        f"Forking from BTC genesis: {BTC_GENESIS_HASH[:24]}..."
    )

    @classmethod
    def create(cls) -> Block:
        """
        Build and seal the SKYNT-BTC genesis block.

        The genesis coinbase rewards ``SKYNT_GENESIS_ADDRESS`` with
        50 SKYNT × 2.0 (max Φ-boost) = 100 SKYNT.  A zero-value output
        to ``SKYNT_FORK_RECORD`` preserves the fork message on-chain.

        Returns:
            Fully initialised genesis :class:`~sphinx_os.blockchain.block.Block`
            with ``index = 0`` and ``previous_hash = BTC_GENESIS_HASH``.
        """
        genesis_tx = Transaction.create_coinbase(
            miner_address="SKYNT_GENESIS_ADDRESS",
            block_height=0,
            phi_boost=2.0,          # Maximum Φ-boost for genesis
        )

        # Append a zero-value OP_RETURN-style output that stores the fork
        # message on-chain for auditability.
        genesis_tx.outputs.append(
            TransactionOutput(address="SKYNT_FORK_RECORD", amount=0.0)
        )

        block = Block(
            index=0,
            transactions=[genesis_tx],
            # ← Hard-fork anchor: SKYNT-BTC genesis points to BTC genesis,
            #   establishing the canonical fork point at height 0.
            previous_hash=BTC_GENESIS_HASH,
            # Bitcoin genesis difficulty in its compact (bits) form decoded:
            # bits=0x1d00ffff → target = 0x00ffff * 2^(8*(0x1d-3)) ≈ 2^224.
            # We store the integer approximation for the difficulty field.
            difficulty=486_604_799,
            miner="SKYNT_GENESIS",
            phi_score=1000.0,           # Max Φ for genesis
            pow_algorithm=SKYNTBTCParams.POW_ALGORITHM,
        )

        # Mirror the Bitcoin genesis timestamp to honour the lineage.
        block.timestamp = BTC_GENESIS_TIMESTAMP
        block.nonce = 0
        block.hash = block.calculate_hash()

        return block


# ---------------------------------------------------------------------------
# SKYNT-BTC blockchain
# ---------------------------------------------------------------------------

class SKYNTBTCChain:
    """
    SKYNT-BTC blockchain — a Bitcoin hard fork with Spectral IIT PoW.

    This chain is self-contained and can run independently or alongside the
    primary SphinxSkynet chain.  All blocks are mined with
    :class:`~sphinx_os.mining.spectral_iit_pow.SpectralIITPow`, which
    enforces both a spectral difficulty target and a minimum IIT Φ score.
    """

    def __init__(self) -> None:
        self.params = SKYNTBTCParams()
        self.chain: List[Block] = []
        self.transaction_pool: List[Transaction] = []
        self.consensus = ConsensusEngine()
        self.chain_manager = ChainManager()
        self._deployed: bool = False
        self._deploy_time: Optional[float] = None

        # Initialise with the SKYNT-BTC genesis block
        genesis = SKYNTBTCGenesis.create()
        self.chain.append(genesis)

        self._stats: Dict = {
            "total_blocks": 1,
            "total_transactions": 1,
            # Genesis coinbase: 50 SKYNT × 2.0 Φ-boost = 100 SKYNT
            "total_minted": SKYNTBTCParams.block_reward(0) * 2.0,
            "current_difficulty": genesis.difficulty,
            "fork_point": BTC_GENESIS_HASH,
        }

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def deploy(self) -> Dict:
        """
        Deploy the SKYNT-BTC chain (idempotent).

        Records the deployment timestamp and returns a deployment receipt
        that can be served directly from the API.

        Returns:
            Deployment receipt dictionary.
        """
        if not self._deployed:
            self._deployed = True
            self._deploy_time = time.time()

        genesis = self.chain[0]
        return {
            "status": "deployed",
            "chain": self.params.CHAIN_NAME,
            "ticker": self.params.TICKER,
            "genesis_hash": genesis.hash,
            "genesis_timestamp": genesis.timestamp,
            "btc_fork_point": BTC_GENESIS_HASH,
            "btc_genesis_message": BTC_GENESIS_MESSAGE,
            "fork_message": SKYNTBTCGenesis.FORK_MESSAGE,
            "pow_algorithm": self.params.POW_ALGORITHM,
            "iit_phi_threshold": self.params.IIT_PHI_THRESHOLD,
            "initial_block_reward": self.params.INITIAL_BLOCK_REWARD,
            "halving_interval": self.params.HALVING_INTERVAL,
            "max_supply": self.params.MAX_SUPPLY,
            "block_time_target": self.params.BLOCK_TIME_TARGET,
            "max_block_size": self.params.MAX_BLOCK_SIZE,
            "deployed_at": self._deploy_time,
        }

    # ------------------------------------------------------------------
    # Chain operations
    # ------------------------------------------------------------------

    def get_latest_block(self) -> Block:
        """Return the tip of the chain."""
        return self.chain[-1]

    def create_block(
        self,
        miner_address: str,
        phi_score: float = 500.0,
        merge_mining_headers: Optional[Dict[str, str]] = None,
        max_transactions: int = 1000,
    ) -> Block:
        """
        Create an unsealed candidate block (before mining).

        Always uses ``pow_algorithm = "spectral"`` (Spectral IIT PoW).

        Args:
            miner_address: Address to receive the coinbase reward.
            phi_score: Φ consciousness score on the [200, 1000] scale.
            merge_mining_headers: Optional auxiliary chain headers.
            max_transactions: Maximum non-coinbase transactions included.

        Returns:
            Unsealed :class:`~sphinx_os.blockchain.block.Block`.
        """
        latest = self.get_latest_block()
        height = latest.index + 1

        # Bitcoin-style difficulty adjustment every 2016 blocks
        difficulty = self.consensus.calculate_next_difficulty(
            current_difficulty=latest.difficulty,
            block_height=height,
        )

        # Φ-boosted SKYNT-BTC coinbase
        phi_boost = self.consensus.calculate_phi_boost(phi_score)
        reward = SKYNTBTCParams.block_reward(height) * phi_boost

        coinbase = Transaction(
            inputs=[],
            outputs=[TransactionOutput(address=miner_address, amount=reward)],
            fee=0.0,
            phi_boost=phi_boost,
        )

        # Merge-mining bonus (10 % per enabled auxiliary chain)
        if merge_mining_headers:
            bonus = 1.0 + 0.1 * len(merge_mining_headers)
            for out in coinbase.outputs:
                out.amount *= bonus

        txs = [coinbase] + self.transaction_pool[:max_transactions]

        return Block(
            index=height,
            transactions=txs,
            previous_hash=latest.hash,
            difficulty=difficulty,
            miner=miner_address,
            phi_score=phi_score,
            pow_algorithm=SKYNTBTCParams.POW_ALGORITHM,  # always "spectral"
            merge_mining_headers=merge_mining_headers,
        )

    def add_block(self, block: Block) -> bool:
        """
        Validate and append a mined block to the chain.

        Enforces that the block was mined with the Spectral IIT PoW
        algorithm.

        Args:
            block: Fully mined block.

        Returns:
            ``True`` if the block was accepted; ``False`` otherwise.
        """
        # Enforce spectral IIT PoW algorithm
        if block.pow_algorithm != SKYNTBTCParams.POW_ALGORITHM:
            return False

        latest = self.get_latest_block()

        if not self.chain_manager.validate_block(block, latest):
            return False

        self.chain.append(block)

        mined_txids = {tx.txid for tx in block.transactions}
        self.transaction_pool = [
            tx for tx in self.transaction_pool if tx.txid not in mined_txids
        ]

        self._stats["total_blocks"] += 1
        self._stats["total_transactions"] += len(block.transactions)
        self._stats["current_difficulty"] = block.difficulty

        for tx in block.transactions:
            if tx.is_coinbase():
                self._stats["total_minted"] += tx.get_total_output()

        return True

    def get_balance(self, address: str) -> float:
        """Return SKYNT balance for *address*."""
        utxo_set = self.chain_manager.get_utxo_set(self.chain)
        return self.chain_manager.get_balance(address, utxo_set)

    def get_chain_info(self) -> Dict:
        """
        Return chain metadata and current state.

        The structure is compatible with the mining API's chain-stats format
        so it can be served directly from the REST endpoint.

        Returns:
            Dictionary describing the current chain state.
        """
        latest = self.get_latest_block()
        return {
            "chain": self.params.CHAIN_NAME,
            "ticker": self.params.TICKER,
            "chain_length": len(self.chain),
            "total_transactions": self._stats["total_transactions"],
            "total_minted": self._stats["total_minted"],
            "max_supply": self.params.MAX_SUPPLY,
            "current_difficulty": self._stats["current_difficulty"],
            "latest_block_hash": latest.hash,
            "latest_block_height": latest.index,
            "transactions_in_pool": len(self.transaction_pool),
            "target_block_time": self.params.BLOCK_TIME_TARGET,
            "btc_fork_point": BTC_GENESIS_HASH,
            "pow_algorithm": self.params.POW_ALGORITHM,
            "iit_phi_threshold": self.params.IIT_PHI_THRESHOLD,
            "deployed": self._deployed,
        }
