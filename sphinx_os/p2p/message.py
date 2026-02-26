"""
P2P Message Protocol for SphinxSkynet

Defines the message types and serialisation used across the SphinxSkynet
peer-to-peer network.  Every message is a compact envelope containing a
type tag, originator, payload, and cryptographic digest.

Supported message types
-----------------------
- BLOCK          — new block announcement
- TRANSACTION    — new transaction broadcast
- PEER_EXCHANGE  — peer-list gossip
- BRIDGE_ATTESTATION — BTC Wormhole spectral attestation
- PING / PONG    — liveness probe
- VERSION        — protocol handshake
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Current P2P protocol version.
PROTOCOL_VERSION: str = "1.0.0"

#: Maximum payload size in bytes (1 MiB).
MAX_PAYLOAD_SIZE: int = 1_048_576

#: Message time-to-live (seconds).  Messages older than this are discarded.
MESSAGE_TTL: int = 300


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageType(Enum):
    """Enumeration of all P2P message types."""
    BLOCK = "block"
    TRANSACTION = "transaction"
    PEER_EXCHANGE = "peer_exchange"
    BRIDGE_ATTESTATION = "bridge_attestation"
    PING = "ping"
    PONG = "pong"
    VERSION = "version"


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class P2PMessage:
    """
    Envelope for a single P2P message.

    Attributes
    ----------
    msg_type : MessageType
        The type of message.
    sender_id : str
        Hex identifier of the originating peer.
    payload : dict
        Arbitrary JSON-serialisable data.
    timestamp : float
        Unix epoch when the message was created.
    msg_id : str
        SHA-256 digest of (type + sender + payload + timestamp).
    ttl : int
        Remaining hops / seconds before discard.
    """
    msg_type: MessageType
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    msg_id: str = ""
    ttl: int = MESSAGE_TTL

    def __post_init__(self) -> None:
        if not self.msg_id:
            self.msg_id = self._compute_id()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def _compute_id(self) -> str:
        """Deterministic message ID from content."""
        raw = (
            f"{self.msg_type.value}"
            f"{self.sender_id}"
            f"{json.dumps(self.payload, sort_keys=True)}"
            f"{self.timestamp}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Return ``True`` if the message passes basic sanity checks."""
        if not self.msg_id:
            return False
        if self.ttl <= 0:
            return False
        age = time.time() - self.timestamp
        if age > MESSAGE_TTL or age < -60:
            return False
        payload_bytes = json.dumps(self.payload, sort_keys=True).encode()
        if len(payload_bytes) > MAX_PAYLOAD_SIZE:
            return False
        return True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "msg_id": self.msg_id,
            "ttl": self.ttl,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> P2PMessage:
        """Deserialise from a dictionary."""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            sender_id=data["sender_id"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            msg_id=data.get("msg_id", ""),
            ttl=data.get("ttl", MESSAGE_TTL),
        )

    @classmethod
    def from_json(cls, raw: str) -> P2PMessage:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_block_message(sender_id: str, block_data: Dict) -> P2PMessage:
    """Create a BLOCK announcement message."""
    return P2PMessage(
        msg_type=MessageType.BLOCK,
        sender_id=sender_id,
        payload=block_data,
    )


def create_transaction_message(sender_id: str, tx_data: Dict) -> P2PMessage:
    """Create a TRANSACTION broadcast message."""
    return P2PMessage(
        msg_type=MessageType.TRANSACTION,
        sender_id=sender_id,
        payload=tx_data,
    )


def create_peer_exchange_message(
    sender_id: str, peers: List[Dict[str, Any]]
) -> P2PMessage:
    """Create a PEER_EXCHANGE gossip message."""
    return P2PMessage(
        msg_type=MessageType.PEER_EXCHANGE,
        sender_id=sender_id,
        payload={"peers": peers},
    )


def create_bridge_attestation_message(
    sender_id: str, attestation: Dict
) -> P2PMessage:
    """Create a BRIDGE_ATTESTATION message for the BTC Wormhole."""
    return P2PMessage(
        msg_type=MessageType.BRIDGE_ATTESTATION,
        sender_id=sender_id,
        payload=attestation,
    )


def create_ping(sender_id: str) -> P2PMessage:
    """Create a PING liveness probe."""
    return P2PMessage(
        msg_type=MessageType.PING,
        sender_id=sender_id,
        payload={"nonce": hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]},
    )


def create_pong(sender_id: str, ping_nonce: str) -> P2PMessage:
    """Create a PONG reply."""
    return P2PMessage(
        msg_type=MessageType.PONG,
        sender_id=sender_id,
        payload={"nonce": ping_nonce},
    )


def create_version_message(
    sender_id: str,
    version: str = PROTOCOL_VERSION,
    chain_height: int = 0,
) -> P2PMessage:
    """Create a VERSION handshake message."""
    return P2PMessage(
        msg_type=MessageType.VERSION,
        sender_id=sender_id,
        payload={"version": version, "chain_height": chain_height},
    )
