"""
Peer Discovery for SphinxSkynet P2P Network

Manages bootstrap nodes and dynamic peer exchange so that every node
can find and maintain connections to a healthy set of peers.

Discovery strategies
--------------------
1. **Bootstrap** — hard-coded seed nodes contacted on first launch.
2. **Peer exchange** — connected peers share their known-peer lists.
3. **Periodic refresh** — stale peers are pruned and new ones requested.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from .peer import (
    Peer,
    PeerState,
    DEFAULT_PORT,
    PEER_STALE_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of peers in the address book.
MAX_KNOWN_PEERS: int = 256

#: Number of peers returned in a single exchange response.
EXCHANGE_BATCH_SIZE: int = 8

#: Minimum peers before requesting more via exchange.
MIN_CONNECTED_PEERS: int = 3

#: Default bootstrap seed nodes (testnet).
DEFAULT_BOOTSTRAP_NODES: List[Dict[str, object]] = [
    {"host": "seed1.sphinxskynet.io", "port": DEFAULT_PORT},
    {"host": "seed2.sphinxskynet.io", "port": DEFAULT_PORT},
    {"host": "seed3.sphinxskynet.io", "port": DEFAULT_PORT},
]


# ---------------------------------------------------------------------------
# PeerDiscovery
# ---------------------------------------------------------------------------

class PeerDiscovery:
    """
    Manages the peer address book and discovery lifecycle.

    Parameters
    ----------
    bootstrap_nodes : list of dict, optional
        Seed nodes to contact on first launch.
    max_peers : int
        Maximum number of peers to track.
    """

    def __init__(
        self,
        bootstrap_nodes: Optional[List[Dict[str, object]]] = None,
        max_peers: int = MAX_KNOWN_PEERS,
    ) -> None:
        self.max_peers = max_peers
        self._peers: Dict[str, Peer] = {}  # peer_id → Peer

        seeds = bootstrap_nodes if bootstrap_nodes is not None else DEFAULT_BOOTSTRAP_NODES
        for node in seeds:
            p = Peer(
                host=str(node["host"]),
                port=int(node.get("port", DEFAULT_PORT)),
            )
            self.add_peer(p)

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, peer: Peer) -> bool:
        """
        Add a peer to the address book.

        Returns ``True`` if the peer was added (or updated).
        Returns ``False`` if the book is full or the peer is banned.
        """
        if peer.is_banned():
            return False
        if peer.peer_id in self._peers:
            # Update existing entry
            existing = self._peers[peer.peer_id]
            existing.last_seen = max(existing.last_seen, peer.last_seen)
            existing.chain_height = max(existing.chain_height, peer.chain_height)
            if peer.version:
                existing.version = peer.version
            return True
        if len(self._peers) >= self.max_peers:
            # Evict stale peer if possible
            if not self._evict_stale():
                return False
        self._peers[peer.peer_id] = peer
        return True

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer by ID.  Returns ``True`` if it existed."""
        return self._peers.pop(peer_id, None) is not None

    def get_peer(self, peer_id: str) -> Optional[Peer]:
        """Look up a peer by ID."""
        return self._peers.get(peer_id)

    @property
    def known_peers(self) -> List[Peer]:
        """Return all known peers (including disconnected, excluding banned)."""
        return [p for p in self._peers.values() if not p.is_banned()]

    @property
    def connected_peers(self) -> List[Peer]:
        """Return only currently connected peers."""
        return [
            p for p in self._peers.values()
            if p.state == PeerState.CONNECTED and not p.is_banned()
        ]

    @property
    def peer_count(self) -> int:
        """Number of known (non-banned) peers."""
        return len(self.known_peers)

    # ------------------------------------------------------------------
    # Peer exchange
    # ------------------------------------------------------------------

    def get_peers_for_exchange(self) -> List[Dict]:
        """
        Return a batch of peers suitable for sharing with a remote node.

        Connected, non-stale peers are preferred.
        """
        candidates = sorted(
            [p for p in self.known_peers if not p.is_stale()],
            key=lambda p: p.last_seen,
            reverse=True,
        )
        return [p.to_dict() for p in candidates[:EXCHANGE_BATCH_SIZE]]

    def process_peer_exchange(self, peer_dicts: List[Dict]) -> int:
        """
        Ingest a list of peer descriptors received from a remote node.

        Returns the number of *new* peers added.
        """
        added = 0
        for pd in peer_dicts:
            try:
                p = Peer.from_dict(pd)
                if p.peer_id not in self._peers:
                    if self.add_peer(p):
                        added += 1
            except (KeyError, ValueError):
                continue
        return added

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def needs_more_peers(self) -> bool:
        """Return ``True`` if below the minimum connected-peer threshold."""
        return len(self.connected_peers) < MIN_CONNECTED_PEERS

    def prune_stale(self) -> int:
        """
        Remove peers that have not been seen within the stale timeout.

        Returns the number of peers removed.
        """
        stale_ids = [
            pid for pid, p in self._peers.items()
            if p.is_stale() and p.state != PeerState.CONNECTED
        ]
        for pid in stale_ids:
            del self._peers[pid]
        return len(stale_ids)

    def prune_banned(self) -> int:
        """Remove peers whose ban has expired (they will re-enter as fresh)."""
        expired = [
            pid for pid, p in self._peers.items()
            if p.state == PeerState.BANNED and not p.is_banned()
        ]
        for pid in expired:
            del self._peers[pid]
        return len(expired)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_stale(self) -> bool:
        """Evict one stale peer to make room.  Returns ``True`` on success."""
        for pid, p in list(self._peers.items()):
            if p.is_stale() and p.state != PeerState.CONNECTED:
                del self._peers[pid]
                return True
        return False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Snapshot of the discovery state."""
        return {
            "peer_count": self.peer_count,
            "connected": len(self.connected_peers),
            "max_peers": self.max_peers,
            "peers": [p.to_dict() for p in self.known_peers],
        }
