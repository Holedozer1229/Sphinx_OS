"""
SphinxSkynet Cross-Chain Bridge Module

Trustless cross-chain bridge with:
- Lock & mint mechanism
- Burn & release mechanism
- Multi-signature validation (5-of-9)
- Support for BTC, ETH, ETC, MATIC, AVAX, BNB, STX
- ZK-proof verification
- BTC Wormhole: quantum-secured BTC bridging with spectral attestations
"""

from .bridge import CrossChainBridge, BridgeStatus, BridgeTransaction
from .relayer import BridgeRelayer
from .validator import BridgeValidator, ZKProofVerifier
from .btc_wormhole import (
    BTCWormhole,
    WormholeStatus,
    WormholeTransfer,
    SpectralAttestation,
)

__all__ = [
    'CrossChainBridge',
    'BridgeStatus',
    'BridgeTransaction',
    'BridgeRelayer',
    'BridgeValidator',
    'ZKProofVerifier',
    'BTCWormhole',
    'WormholeStatus',
    'WormholeTransfer',
    'SpectralAttestation',
]
