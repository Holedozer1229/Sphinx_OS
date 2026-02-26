"""
Tests for the SphinxSkynet P2P infrastructure.

Covers:
- Peer identity and state management
- Message creation, serialisation, and validation
- Peer discovery (bootstrap, exchange, pruning)
- Gossip protocol (dedup, TTL, subscriptions)
- P2PNetwork façade (broadcast, connect, stats)
"""

import json
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.p2p import (
    Peer,
    PeerState,
    DEFAULT_PORT,
    PEER_STALE_TIMEOUT,
    BAN_THRESHOLD,
    BAN_DURATION,
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
    PeerDiscovery,
    MAX_KNOWN_PEERS,
    EXCHANGE_BATCH_SIZE,
    MIN_CONNECTED_PEERS,
    GossipProtocol,
    GOSSIP_FANOUT,
    SEEN_CACHE_SIZE,
    P2PNetwork,
    MAX_OUTBOUND,
    MAX_INBOUND,
)


# ============================================================================
# Peer
# ============================================================================


class TestPeer:
    """Verify Peer identity and state management."""

    def test_default_port(self):
        p = Peer(host="10.0.0.1")
        assert p.port == DEFAULT_PORT

    def test_auto_peer_id(self):
        p = Peer(host="10.0.0.1", port=9333)
        assert len(p.peer_id) == 64  # SHA-256 hex

    def test_custom_peer_id(self):
        p = Peer(host="10.0.0.1", peer_id="custom_id")
        assert p.peer_id == "custom_id"

    def test_address_property(self):
        p = Peer(host="192.168.1.1", port=8080)
        assert p.address == "192.168.1.1:8080"

    def test_initial_state_disconnected(self):
        p = Peer(host="10.0.0.1")
        assert p.state == PeerState.DISCONNECTED

    def test_connect(self):
        p = Peer(host="10.0.0.1")
        p.connect()
        assert p.state == PeerState.CONNECTED
        assert p.last_seen > 0

    def test_disconnect(self):
        p = Peer(host="10.0.0.1")
        p.connect()
        p.disconnect()
        assert p.state == PeerState.DISCONNECTED

    def test_record_message_sent(self):
        p = Peer(host="10.0.0.1")
        p.record_message_sent()
        assert p.messages_sent == 1
        assert p.last_seen > 0

    def test_record_message_received(self):
        p = Peer(host="10.0.0.1")
        p.record_message_received()
        assert p.messages_received == 1

    def test_record_failure_increments(self):
        p = Peer(host="10.0.0.1")
        p.record_failure()
        assert p.failures == 1

    def test_auto_ban_after_threshold(self):
        p = Peer(host="10.0.0.1")
        for _ in range(BAN_THRESHOLD):
            p.record_failure()
        assert p.is_banned()
        assert p.state == PeerState.BANNED

    def test_ban_duration(self):
        p = Peer(host="10.0.0.1")
        p.ban(duration=1)  # 1 second
        assert p.is_banned()

    def test_stale_detection(self):
        p = Peer(host="10.0.0.1")
        # Never seen → stale
        assert p.is_stale()
        p.last_seen = time.time()
        assert not p.is_stale()

    def test_to_dict_keys(self):
        p = Peer(host="10.0.0.1", port=9333)
        d = p.to_dict()
        expected = {
            "peer_id", "host", "port", "state", "last_seen",
            "latency", "chain_height", "messages_sent",
            "messages_received", "failures", "version",
        }
        assert set(d.keys()) == expected

    def test_from_dict_roundtrip(self):
        p = Peer(host="10.0.0.1", port=9333)
        p.connect()
        p.chain_height = 42
        d = p.to_dict()
        p2 = Peer.from_dict(d)
        assert p2.host == p.host
        assert p2.port == p.port
        assert p2.peer_id == p.peer_id
        assert p2.chain_height == 42

    def test_equality(self):
        p1 = Peer(host="10.0.0.1", port=9333)
        p2 = Peer(host="10.0.0.1", port=9333)
        assert p1 == p2

    def test_hash(self):
        p1 = Peer(host="10.0.0.1", port=9333)
        p2 = Peer(host="10.0.0.1", port=9333)
        assert hash(p1) == hash(p2)


# ============================================================================
# P2PMessage
# ============================================================================


class TestP2PMessage:
    """Verify message creation, serialisation, and validation."""

    def test_auto_msg_id(self):
        msg = P2PMessage(
            msg_type=MessageType.BLOCK,
            sender_id="node_1",
            payload={"hash": "abc"},
        )
        assert len(msg.msg_id) == 64

    def test_deterministic_id(self):
        ts = time.time()
        m1 = P2PMessage(MessageType.BLOCK, "node_1", {"x": 1}, timestamp=ts)
        m2 = P2PMessage(MessageType.BLOCK, "node_1", {"x": 1}, timestamp=ts)
        assert m1.msg_id == m2.msg_id

    def test_different_payload_different_id(self):
        ts = time.time()
        m1 = P2PMessage(MessageType.BLOCK, "node_1", {"x": 1}, timestamp=ts)
        m2 = P2PMessage(MessageType.BLOCK, "node_1", {"x": 2}, timestamp=ts)
        assert m1.msg_id != m2.msg_id

    def test_is_valid_fresh_message(self):
        msg = P2PMessage(MessageType.PING, "node_1", {"nonce": "abc"})
        assert msg.is_valid()

    def test_invalid_zero_ttl(self):
        msg = P2PMessage(MessageType.PING, "node_1", {"nonce": "abc"}, ttl=0)
        assert not msg.is_valid()

    def test_to_dict_roundtrip(self):
        msg = P2PMessage(MessageType.TRANSACTION, "node_1", {"txid": "abc"})
        d = msg.to_dict()
        msg2 = P2PMessage.from_dict(d)
        assert msg2.msg_id == msg.msg_id
        assert msg2.msg_type == msg.msg_type
        assert msg2.payload == msg.payload

    def test_to_json_roundtrip(self):
        msg = P2PMessage(MessageType.BLOCK, "node_1", {"h": 100})
        j = msg.to_json()
        msg2 = P2PMessage.from_json(j)
        assert msg2.msg_id == msg.msg_id


class TestMessageFactories:
    """Verify factory functions produce correct message types."""

    def test_create_block_message(self):
        msg = create_block_message("node_1", {"hash": "0xabc", "height": 42})
        assert msg.msg_type == MessageType.BLOCK
        assert msg.sender_id == "node_1"

    def test_create_transaction_message(self):
        msg = create_transaction_message("node_1", {"txid": "0x123"})
        assert msg.msg_type == MessageType.TRANSACTION

    def test_create_peer_exchange_message(self):
        peers = [{"host": "10.0.0.1", "port": 9333}]
        msg = create_peer_exchange_message("node_1", peers)
        assert msg.msg_type == MessageType.PEER_EXCHANGE
        assert msg.payload["peers"] == peers

    def test_create_bridge_attestation_message(self):
        att = {"spectral_hash": "abc", "phi_score": 0.919}
        msg = create_bridge_attestation_message("node_1", att)
        assert msg.msg_type == MessageType.BRIDGE_ATTESTATION

    def test_create_ping_pong(self):
        ping = create_ping("node_1")
        assert ping.msg_type == MessageType.PING
        nonce = ping.payload["nonce"]
        pong = create_pong("node_2", nonce)
        assert pong.msg_type == MessageType.PONG
        assert pong.payload["nonce"] == nonce

    def test_create_version_message(self):
        msg = create_version_message("node_1", chain_height=1000)
        assert msg.msg_type == MessageType.VERSION
        assert msg.payload["version"] == PROTOCOL_VERSION
        assert msg.payload["chain_height"] == 1000


# ============================================================================
# PeerDiscovery
# ============================================================================


class TestPeerDiscovery:
    """Verify bootstrap, peer exchange, and pruning."""

    def test_bootstrap_nodes_loaded(self):
        d = PeerDiscovery()
        assert d.peer_count == 3  # default seeds

    def test_custom_bootstrap(self):
        seeds = [{"host": "10.0.0.1", "port": 9333}]
        d = PeerDiscovery(bootstrap_nodes=seeds)
        assert d.peer_count == 1

    def test_add_peer(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        assert d.add_peer(p)
        assert d.peer_count == 1

    def test_add_duplicate_updates(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        d.add_peer(p)
        p2 = Peer(host="10.0.0.1")
        p2.chain_height = 100
        d.add_peer(p2)
        assert d.peer_count == 1
        assert d.get_peer(p.peer_id).chain_height == 100

    def test_remove_peer(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        d.add_peer(p)
        assert d.remove_peer(p.peer_id)
        assert d.peer_count == 0

    def test_max_peers_enforced(self):
        d = PeerDiscovery(bootstrap_nodes=[], max_peers=2)
        d.add_peer(Peer(host="10.0.0.1"))
        d.add_peer(Peer(host="10.0.0.2"))
        # Third peer should fail (no stale to evict since they're fresh)
        p3 = Peer(host="10.0.0.3")
        p3.last_seen = time.time()  # not stale
        result = d.add_peer(p3)
        # If both existing peers are stale (last_seen=0), one gets evicted
        # so the result depends on their state.
        assert d.peer_count <= 2

    def test_connected_peers_filter(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p1 = Peer(host="10.0.0.1")
        p1.connect()
        p2 = Peer(host="10.0.0.2")
        d.add_peer(p1)
        d.add_peer(p2)
        assert len(d.connected_peers) == 1

    def test_get_peers_for_exchange(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        for i in range(10):
            p = Peer(host=f"10.0.0.{i}")
            p.last_seen = time.time()
            d.add_peer(p)
        exchange = d.get_peers_for_exchange()
        assert len(exchange) <= EXCHANGE_BATCH_SIZE

    def test_process_peer_exchange(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        peer_dicts = [
            {"host": "10.0.0.1", "port": 9333, "peer_id": "aaa"},
            {"host": "10.0.0.2", "port": 9333, "peer_id": "bbb"},
        ]
        added = d.process_peer_exchange(peer_dicts)
        assert added == 2
        assert d.peer_count == 2

    def test_process_peer_exchange_ignores_duplicates(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        d.add_peer(p)
        peer_dicts = [{"host": "10.0.0.1", "port": DEFAULT_PORT}]
        added = d.process_peer_exchange(peer_dicts)
        assert added == 0

    def test_needs_more_peers(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        assert d.needs_more_peers()
        for i in range(MIN_CONNECTED_PEERS):
            p = Peer(host=f"10.0.0.{i}")
            p.connect()
            d.add_peer(p)
        assert not d.needs_more_peers()

    def test_prune_stale(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        p.last_seen = 0  # ancient
        d.add_peer(p)
        pruned = d.prune_stale()
        assert pruned == 1
        assert d.peer_count == 0

    def test_prune_stale_keeps_connected(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        p.connect()
        p.last_seen = 0  # stale but connected
        d.add_peer(p)
        pruned = d.prune_stale()
        assert pruned == 0
        assert d.peer_count == 1

    def test_banned_peer_not_added(self):
        d = PeerDiscovery(bootstrap_nodes=[])
        p = Peer(host="10.0.0.1")
        p.ban()
        assert not d.add_peer(p)

    def test_to_dict(self):
        d = PeerDiscovery(bootstrap_nodes=[{"host": "10.0.0.1"}])
        state = d.to_dict()
        assert "peer_count" in state
        assert "connected" in state
        assert "peers" in state


# ============================================================================
# GossipProtocol
# ============================================================================


class TestGossipProtocol:
    """Verify deduplication, TTL, and subscriptions."""

    def test_receive_valid_message(self):
        g = GossipProtocol("node_1")
        msg = create_block_message("node_2", {"hash": "abc"})
        g.receive(msg)
        assert g.stats["messages_received"] == 1

    def test_dedup_blocks_duplicate(self):
        g = GossipProtocol("node_1")
        msg = create_block_message("node_2", {"hash": "abc"})
        g.receive(msg)
        g.receive(msg)  # same msg_id
        assert g.stats["messages_dropped_dup"] == 1

    def test_ttl_zero_dropped(self):
        g = GossipProtocol("node_1")
        msg = P2PMessage(MessageType.BLOCK, "node_2", {"h": 1}, ttl=0)
        g.receive(msg)
        assert g.stats["messages_dropped_invalid"] == 1

    def test_subscribe_handler_called(self):
        g = GossipProtocol("node_1")
        received = []
        g.subscribe(MessageType.BLOCK, lambda m: received.append(m))
        msg = create_block_message("node_2", {"hash": "abc"})
        g.receive(msg)
        assert len(received) == 1
        assert received[0].msg_id == msg.msg_id

    def test_unsubscribe(self):
        g = GossipProtocol("node_1")
        received = []
        handler = lambda m: received.append(m)
        g.subscribe(MessageType.BLOCK, handler)
        g.unsubscribe(MessageType.BLOCK, handler)
        msg = create_block_message("node_2", {"hash": "abc"})
        g.receive(msg)
        assert len(received) == 0

    def test_originate_marks_seen(self):
        g = GossipProtocol("node_1")
        msg = create_block_message("node_1", {"hash": "my_block"})
        g.originate(msg)
        assert g.has_seen(msg.msg_id)

    def test_has_seen_false_for_unknown(self):
        g = GossipProtocol("node_1")
        assert not g.has_seen("unknown_id")

    def test_get_stats(self):
        g = GossipProtocol("node_1")
        stats = g.get_stats()
        assert "messages_received" in stats
        assert "seen_cache_size" in stats

    def test_seen_cache_eviction(self):
        g = GossipProtocol("node_1")
        # Manually fill the cache beyond limit
        for i in range(SEEN_CACHE_SIZE + 100):
            g._mark_seen(f"msg_{i}")
        assert len(g._seen) <= SEEN_CACHE_SIZE


# ============================================================================
# P2PNetwork
# ============================================================================


class TestP2PNetwork:
    """Verify the high-level P2P network façade."""

    def _make_network(self, **kwargs):
        return P2PNetwork(
            host="127.0.0.1",
            port=9333,
            bootstrap_nodes=[],
            **kwargs,
        )

    def test_init_defaults(self):
        net = self._make_network()
        assert len(net.node_id) == 64
        assert not net.is_running

    def test_start_stop(self):
        net = self._make_network()
        net.start()
        assert net.is_running
        net.stop()
        assert not net.is_running

    def test_connect_to_peer(self):
        net = self._make_network()
        peer = net.connect_to_peer("10.0.0.1", 9333)
        assert peer is not None
        assert peer.state == PeerState.CONNECTED
        assert net.discovery.peer_count >= 1

    def test_disconnect_peer(self):
        net = self._make_network()
        peer = net.connect_to_peer("10.0.0.1")
        assert net.disconnect_peer(peer.peer_id)
        assert peer.state == PeerState.DISCONNECTED

    def test_disconnect_unknown_peer(self):
        net = self._make_network()
        assert not net.disconnect_peer("unknown_id")

    def test_broadcast_block(self):
        net = self._make_network()
        net.connect_to_peer("10.0.0.1")
        msg = net.broadcast_block({"hash": "abc", "height": 42})
        assert msg.msg_type == MessageType.BLOCK
        assert net.gossip.has_seen(msg.msg_id)

    def test_broadcast_transaction(self):
        net = self._make_network()
        msg = net.broadcast_transaction({"txid": "tx_1"})
        assert msg.msg_type == MessageType.TRANSACTION

    def test_broadcast_bridge_attestation(self):
        net = self._make_network()
        att = {"spectral_hash": "abc", "phi": 0.919}
        msg = net.broadcast_bridge_attestation(att)
        assert msg.msg_type == MessageType.BRIDGE_ATTESTATION

    def test_request_peer_exchange(self):
        net = self._make_network()
        msg = net.request_peer_exchange()
        assert msg.msg_type == MessageType.PEER_EXCHANGE

    def test_send_ping(self):
        net = self._make_network()
        peer = net.connect_to_peer("10.0.0.1")
        msg = net.send_ping(peer.peer_id)
        assert msg is not None
        assert msg.msg_type == MessageType.PING

    def test_send_ping_disconnected_returns_none(self):
        net = self._make_network()
        assert net.send_ping("nonexistent") is None

    def test_receive_message(self):
        net = self._make_network()
        received = []
        net.gossip.subscribe(MessageType.BLOCK, lambda m: received.append(m))
        msg = create_block_message("remote_node", {"hash": "abc"})
        net.receive_message(msg)
        assert len(received) == 1

    def test_receive_deduplicates(self):
        net = self._make_network()
        received = []
        net.gossip.subscribe(MessageType.BLOCK, lambda m: received.append(m))
        msg = create_block_message("remote_node", {"hash": "abc"})
        net.receive_message(msg)
        net.receive_message(msg)
        assert len(received) == 1

    def test_get_connected_peers(self):
        net = self._make_network()
        net.connect_to_peer("10.0.0.1")
        net.connect_to_peer("10.0.0.2")
        peers = net.get_connected_peers()
        assert len(peers) == 2

    def test_get_network_stats(self):
        net = self._make_network()
        stats = net.get_network_stats()
        assert "node_id" in stats
        assert "protocol_version" in stats
        assert "known_peers" in stats
        assert "gossip" in stats

    def test_custom_node_id(self):
        net = P2PNetwork(
            host="127.0.0.1",
            port=9333,
            node_id="custom_node",
            bootstrap_nodes=[],
        )
        assert net.node_id == "custom_node"


# ============================================================================
# Integration — multi-node gossip
# ============================================================================


class TestMultiNodeGossip:
    """Verify message propagation across a simulated multi-node network."""

    def test_two_node_block_propagation(self):
        """Node A broadcasts a block; Node B receives it."""
        net_a = P2PNetwork(host="127.0.0.1", port=9001, node_id="A", bootstrap_nodes=[])
        net_b = P2PNetwork(host="127.0.0.1", port=9002, node_id="B", bootstrap_nodes=[])

        # Wire up connections
        net_a.connect_to_peer("127.0.0.1", 9002)
        net_b.connect_to_peer("127.0.0.1", 9001)

        # A broadcasts a block
        block = {"hash": "0xblock42", "height": 42}
        msg = net_a.broadcast_block(block)

        # Simulate delivery to B
        received_by_b = []
        net_b.gossip.subscribe(MessageType.BLOCK, lambda m: received_by_b.append(m))
        net_b.receive_message(msg)

        assert len(received_by_b) == 1
        assert received_by_b[0].payload["hash"] == "0xblock42"

    def test_three_node_gossip_dedup(self):
        """A → B → C; C should not echo back to B."""
        net_a = P2PNetwork(host="127.0.0.1", port=9001, node_id="A", bootstrap_nodes=[])
        net_b = P2PNetwork(host="127.0.0.1", port=9002, node_id="B", bootstrap_nodes=[])
        net_c = P2PNetwork(host="127.0.0.1", port=9003, node_id="C", bootstrap_nodes=[])

        msg = net_a.broadcast_transaction({"txid": "tx_1"})

        # B receives from A
        net_b.receive_message(msg)
        # C receives from B (same msg_id)
        net_c.receive_message(msg)
        # B tries to receive again (from C) — should be deduped
        net_b.receive_message(msg)

        assert net_b.gossip.stats["messages_dropped_dup"] == 1

    def test_bridge_attestation_propagation(self):
        """Verify BTC Wormhole attestations propagate through the network."""
        net_a = P2PNetwork(host="127.0.0.1", port=9001, node_id="A", bootstrap_nodes=[])
        net_b = P2PNetwork(host="127.0.0.1", port=9002, node_id="B", bootstrap_nodes=[])

        att = {
            "spectral_hash": "zeta(1/2+it) * 87T",
            "phi_score": 0.919,
            "bridge_invariant": "PRESERVED",
        }
        msg = net_a.broadcast_bridge_attestation(att)

        received = []
        net_b.gossip.subscribe(
            MessageType.BRIDGE_ATTESTATION, lambda m: received.append(m)
        )
        net_b.receive_message(msg)

        assert len(received) == 1
        assert received[0].payload["phi_score"] == 0.919


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
