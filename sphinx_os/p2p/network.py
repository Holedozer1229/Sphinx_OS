"""
SphinxSkynet P2P Network Layer

Ties together peer discovery, gossip propagation, and message handling
into a single :class:`P2PNetwork` façade that can be used by the
blockchain node.

Responsibilities
----------------
- Maintain a set of connected peers via :class:`PeerDiscovery`.
- Broadcast blocks, transactions, and bridge attestations via
  :class:`GossipProtocol`.
- Respond to ping / pong liveness probes.
- Provide a clean API for the node to send and receive data.
"""

from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Callable, Dict, List, Optional

from .peer import Peer, PeerState, DEFAULT_PORT
from .discovery import PeerDiscovery, MAX_KNOWN_PEERS
from .gossip import GossipProtocol, GOSSIP_FANOUT
from .message import (
    P2PMessage,
    MessageType,
    PROTOCOL_VERSION,
    create_block_message,
    create_transaction_message,
    create_peer_exchange_message,
    create_bridge_attestation_message,
    create_ping,
    create_pong,
    create_version_message,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum outbound connections a node should maintain.
MAX_OUTBOUND: int = 8

#: Maximum inbound connections a node should accept.
MAX_INBOUND: int = 32


# ---------------------------------------------------------------------------
# P2PNetwork
# ---------------------------------------------------------------------------

class P2PNetwork:
    """
    High-level P2P network manager for a SphinxSkynet node.

    Parameters
    ----------
    host : str
        Local listen address (default ``"0.0.0.0"``).
    port : int
        Local listen port (default ``DEFAULT_PORT``).
    node_id : str, optional
        Hex identifier for this node.  Auto-generated if omitted.
    bootstrap_nodes : list, optional
        Seed nodes for initial discovery.
    max_peers : int
        Maximum peer-book size.
    fanout : int
        Gossip fan-out factor.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        node_id: Optional[str] = None,
        bootstrap_nodes: Optional[List[Dict]] = None,
        max_peers: int = MAX_KNOWN_PEERS,
        fanout: int = GOSSIP_FANOUT,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id or hashlib.sha256(
            f"{host}:{port}:{time.time()}".encode()
        ).hexdigest()

        self.discovery = PeerDiscovery(
            bootstrap_nodes=bootstrap_nodes,
            max_peers=max_peers,
        )
        self.gossip = GossipProtocol(
            node_id=self.node_id,
            fanout=fanout,
        )

        # Track chain height for version handshake
        self.chain_height: int = 0

        self._running: bool = False

        # Wire up internal gossip handlers
        self.gossip.subscribe(MessageType.PING, self._handle_ping)
        self.gossip.subscribe(MessageType.PEER_EXCHANGE, self._handle_peer_exchange)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for connections (placeholder for transport)."""
        self._running = True

    def stop(self) -> None:
        """Stop the network layer."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect_to_peer(self, host: str, port: int = DEFAULT_PORT) -> Optional[Peer]:
        """
        Establish an outbound connection to a peer.

        In a real implementation this would open a TCP / QUIC socket.
        Here we model the state transitions.
        """
        peer = Peer(host=host, port=port)
        if peer.is_banned():
            return None
        peer.connect()
        peer.version = PROTOCOL_VERSION
        self.discovery.add_peer(peer)
        return peer

    def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect and remove a peer."""
        peer = self.discovery.get_peer(peer_id)
        if peer is None:
            return False
        peer.disconnect()
        return True

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast_block(self, block_data: Dict) -> P2PMessage:
        """Broadcast a new block to connected peers."""
        msg = create_block_message(self.node_id, block_data)
        self._broadcast(msg)
        return msg

    def broadcast_transaction(self, tx_data: Dict) -> P2PMessage:
        """Broadcast a new transaction to connected peers."""
        msg = create_transaction_message(self.node_id, tx_data)
        self._broadcast(msg)
        return msg

    def broadcast_bridge_attestation(self, attestation: Dict) -> P2PMessage:
        """Broadcast a BTC Wormhole bridge attestation."""
        msg = create_bridge_attestation_message(self.node_id, attestation)
        self._broadcast(msg)
        return msg

    def request_peer_exchange(self) -> P2PMessage:
        """Send a peer-exchange request to connected peers."""
        peers_to_share = self.discovery.get_peers_for_exchange()
        msg = create_peer_exchange_message(self.node_id, peers_to_share)
        self._broadcast(msg)
        return msg

    def send_ping(self, peer_id: str) -> Optional[P2PMessage]:
        """Send a PING to a specific peer."""
        peer = self.discovery.get_peer(peer_id)
        if peer is None or peer.state != PeerState.CONNECTED:
            return None
        msg = create_ping(self.node_id)
        self.gossip.originate(msg)
        peer.record_message_sent()
        return msg

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def receive_message(self, message: P2PMessage) -> None:
        """
        Process an incoming message from the transport layer.

        Delegates to the gossip protocol for dedup and handler dispatch,
        then relays to a random subset of connected peers.
        """
        # Update sender metrics
        sender = self.discovery.get_peer(message.sender_id)
        if sender:
            sender.record_message_received()

        # Gossip processing (handlers, dedup, TTL)
        self.gossip.receive(message)

        # Relay to fanout peers (excluding sender)
        self._relay(message, exclude=message.sender_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_connected_peers(self) -> List[Dict]:
        """Return serialised list of connected peers."""
        return [p.to_dict() for p in self.discovery.connected_peers]

    def get_network_stats(self) -> Dict[str, Any]:
        """Return network-layer statistics."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "running": self._running,
            "protocol_version": PROTOCOL_VERSION,
            "chain_height": self.chain_height,
            "known_peers": self.discovery.peer_count,
            "connected_peers": len(self.discovery.connected_peers),
            "gossip": self.gossip.get_stats(),
        }

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_ping(self, message: P2PMessage) -> None:
        """Respond to a PING with a PONG."""
        nonce = message.payload.get("nonce", "")
        pong = create_pong(self.node_id, nonce)
        self.gossip.originate(pong)
        # In a real network the pong would be sent back to the sender.

    def _handle_peer_exchange(self, message: P2PMessage) -> None:
        """Ingest peers from a PEER_EXCHANGE message."""
        peer_list = message.payload.get("peers", [])
        self.discovery.process_peer_exchange(peer_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _broadcast(self, message: P2PMessage) -> None:
        """Mark as originated and relay to all connected peers."""
        self.gossip.originate(message)
        self._relay(message)

    def _relay(self, message: P2PMessage, exclude: str = "") -> None:
        """Forward *message* to up to *fanout* connected peers."""
        candidates = [
            p for p in self.discovery.connected_peers
            if p.peer_id != exclude and p.peer_id != self.node_id
        ]
        targets = candidates[:self.gossip.fanout]
        for peer in targets:
            peer.record_message_sent()
