"""
P2P Peer Identity and Connection Management

Provides the :class:`Peer` data class that encapsulates everything the
network layer needs to know about a remote node:

- Identity (public-key derived ``peer_id``)
- Network address (``host``, ``port``)
- Connection state (``CONNECTED``, ``DISCONNECTED``, ``BANNED``)
- Metrics (latency, last-seen, message counts)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default P2P listen port.
DEFAULT_PORT: int = 9333

#: Time (seconds) after which a peer is considered stale.
PEER_STALE_TIMEOUT: int = 600

#: Maximum allowed latency (seconds) before a peer is deprioritised.
MAX_LATENCY: float = 10.0

#: Number of consecutive failures before a peer is banned.
BAN_THRESHOLD: int = 5

#: Duration (seconds) of a ban.
BAN_DURATION: int = 3600


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PeerState(Enum):
    """Connection state of a peer."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    BANNED = "banned"


# ---------------------------------------------------------------------------
# Peer dataclass
# ---------------------------------------------------------------------------

@dataclass
class Peer:
    """
    Represents a remote SphinxSkynet node.

    Parameters
    ----------
    host : str
        IP address or hostname.
    port : int
        TCP port (default ``DEFAULT_PORT``).
    peer_id : str
        Hex identifier derived from the peer's public key.  If omitted,
        one is generated from ``host:port``.
    """
    host: str
    port: int = DEFAULT_PORT
    peer_id: str = ""
    state: PeerState = PeerState.DISCONNECTED
    last_seen: float = 0.0
    latency: float = 0.0
    chain_height: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    failures: int = 0
    banned_until: float = 0.0
    version: str = ""

    def __post_init__(self) -> None:
        if not self.peer_id:
            self.peer_id = hashlib.sha256(
                f"{self.host}:{self.port}".encode()
            ).hexdigest()

    # ------------------------------------------------------------------
    # Address helpers
    # ------------------------------------------------------------------

    @property
    def address(self) -> str:
        """Return ``host:port`` string."""
        return f"{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Mark this peer as connected."""
        self.state = PeerState.CONNECTED
        self.last_seen = time.time()

    def disconnect(self) -> None:
        """Mark this peer as disconnected."""
        self.state = PeerState.DISCONNECTED

    def record_message_sent(self) -> None:
        """Increment sent counter and refresh last-seen."""
        self.messages_sent += 1
        self.last_seen = time.time()

    def record_message_received(self) -> None:
        """Increment received counter and refresh last-seen."""
        self.messages_received += 1
        self.last_seen = time.time()

    def record_failure(self) -> None:
        """Record a communication failure.  Auto-bans after threshold."""
        self.failures += 1
        if self.failures >= BAN_THRESHOLD:
            self.ban()

    def ban(self, duration: int = BAN_DURATION) -> None:
        """Ban the peer for *duration* seconds."""
        self.state = PeerState.BANNED
        self.banned_until = time.time() + duration

    def is_banned(self) -> bool:
        """Return ``True`` if the peer is currently banned."""
        if self.state != PeerState.BANNED:
            return False
        if time.time() > self.banned_until:
            # Ban expired
            self.state = PeerState.DISCONNECTED
            self.failures = 0
            return False
        return True

    def is_stale(self) -> bool:
        """Return ``True`` if the peer has not been seen recently."""
        if self.last_seen == 0.0:
            return True
        return (time.time() - self.last_seen) > PEER_STALE_TIMEOUT

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "peer_id": self.peer_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.value,
            "last_seen": self.last_seen,
            "latency": self.latency,
            "chain_height": self.chain_height,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "failures": self.failures,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Peer:
        """Deserialise from a dictionary."""
        p = cls(
            host=data["host"],
            port=data.get("port", DEFAULT_PORT),
            peer_id=data.get("peer_id", ""),
        )
        p.state = PeerState(data.get("state", "disconnected"))
        p.last_seen = data.get("last_seen", 0.0)
        p.latency = data.get("latency", 0.0)
        p.chain_height = data.get("chain_height", 0)
        p.messages_sent = data.get("messages_sent", 0)
        p.messages_received = data.get("messages_received", 0)
        p.failures = data.get("failures", 0)
        p.version = data.get("version", "")
        return p

    def __hash__(self) -> int:
        return hash(self.peer_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Peer):
            return self.peer_id == other.peer_id
        return NotImplemented
