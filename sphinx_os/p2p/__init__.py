"""
SphinxSkynet Peer-to-Peer Network Module

Provides the complete P2P infrastructure for block, transaction, and
bridge-attestation propagation across the SphinxSkynet network:

- **Peer**           — identity and connection state
- **PeerDiscovery**  — bootstrap + peer-exchange discovery
- **GossipProtocol** — epidemic message dissemination with dedup
- **P2PNetwork**     — high-level façade tying everything together
- **P2PMessage / MessageType** — message envelope and type tags
"""

from .peer import (
    Peer,
    PeerState,
    DEFAULT_PORT,
    PEER_STALE_TIMEOUT,
    BAN_THRESHOLD,
    BAN_DURATION,
)
from .message import (
    P2PMessage,
    MessageType,
    PROTOCOL_VERSION,
    MAX_PAYLOAD_SIZE,
    MESSAGE_TTL,
    create_block_message,
    create_transaction_message,
    create_peer_exchange_message,
    create_bridge_attestation_message,
    create_ping,
    create_pong,
    create_version_message,
)
from .discovery import (
    PeerDiscovery,
    MAX_KNOWN_PEERS,
    EXCHANGE_BATCH_SIZE,
    MIN_CONNECTED_PEERS,
)
from .gossip import (
    GossipProtocol,
    GOSSIP_FANOUT,
    SEEN_CACHE_SIZE,
)
from .network import (
    P2PNetwork,
    MAX_OUTBOUND,
    MAX_INBOUND,
)

__all__ = [
    # Peer
    "Peer",
    "PeerState",
    "DEFAULT_PORT",
    "PEER_STALE_TIMEOUT",
    "BAN_THRESHOLD",
    "BAN_DURATION",
    # Message
    "P2PMessage",
    "MessageType",
    "PROTOCOL_VERSION",
    "MAX_PAYLOAD_SIZE",
    "MESSAGE_TTL",
    "create_block_message",
    "create_transaction_message",
    "create_peer_exchange_message",
    "create_bridge_attestation_message",
    "create_ping",
    "create_pong",
    "create_version_message",
    # Discovery
    "PeerDiscovery",
    "MAX_KNOWN_PEERS",
    "EXCHANGE_BATCH_SIZE",
    "MIN_CONNECTED_PEERS",
    # Gossip
    "GossipProtocol",
    "GOSSIP_FANOUT",
    "SEEN_CACHE_SIZE",
    # Network
    "P2PNetwork",
    "MAX_OUTBOUND",
    "MAX_INBOUND",
]
