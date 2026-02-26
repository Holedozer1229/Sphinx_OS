"""
Gossip Protocol for SphinxSkynet P2P Network

Implements epidemic-style message dissemination with deduplication.
Every node that receives a message forwards it to a random subset of
its connected peers (fanout) unless the message has already been seen.

Key properties
--------------
- **Deduplication** — each message ID is tracked; duplicates are dropped.
- **Fanout** — configurable number of peers per relay hop.
- **TTL decay** — messages are decremented on each hop and dropped at 0.
- **Subscription** — handlers can register for specific message types.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set

from .message import MessageType, P2PMessage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default gossip fanout — number of peers each message is relayed to.
GOSSIP_FANOUT: int = 4

#: Maximum number of seen-message IDs to remember.
SEEN_CACHE_SIZE: int = 10_000

#: TTL decrement per hop.
TTL_DECREMENT: int = 1


# ---------------------------------------------------------------------------
# GossipProtocol
# ---------------------------------------------------------------------------

class GossipProtocol:
    """
    Epidemic gossip engine.

    Parameters
    ----------
    node_id : str
        Identifier of the local node.
    fanout : int
        Number of peers to forward each message to (default ``GOSSIP_FANOUT``).
    """

    def __init__(
        self,
        node_id: str,
        fanout: int = GOSSIP_FANOUT,
    ) -> None:
        self.node_id = node_id
        self.fanout = fanout

        # Deduplication cache (OrderedDict used as an LRU set)
        self._seen: OrderedDict[str, float] = OrderedDict()

        # Message-type → list of handler callbacks
        self._handlers: Dict[MessageType, List[Callable]] = {}

        # Metrics
        self.stats: Dict[str, int] = {
            "messages_received": 0,
            "messages_relayed": 0,
            "messages_dropped_dup": 0,
            "messages_dropped_ttl": 0,
            "messages_dropped_invalid": 0,
        }

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(
        self, msg_type: MessageType, handler: Callable[[P2PMessage], None]
    ) -> None:
        """Register a handler for *msg_type*."""
        self._handlers.setdefault(msg_type, []).append(handler)

    def unsubscribe(
        self, msg_type: MessageType, handler: Callable[[P2PMessage], None]
    ) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers.get(msg_type, [])
        if handler in handlers:
            handlers.remove(handler)

    # ------------------------------------------------------------------
    # Receive path
    # ------------------------------------------------------------------

    def receive(self, message: P2PMessage) -> List[str]:
        """
        Process an incoming gossip message.

        1. Validate the message.
        2. Check for duplicates.
        3. Invoke registered handlers.
        4. Return a list of peer IDs that the message should be relayed to
           (the caller is responsible for actually sending it).

        Parameters
        ----------
        message : P2PMessage
            The incoming message.

        Returns
        -------
        list of str
            Peer IDs that should receive the relayed message.
        """
        self.stats["messages_received"] += 1

        # 1. Basic validation
        if not message.is_valid():
            self.stats["messages_dropped_invalid"] += 1
            return []

        # 2. Deduplication
        if message.msg_id in self._seen:
            self.stats["messages_dropped_dup"] += 1
            return []
        self._mark_seen(message.msg_id)

        # 3. TTL check
        if message.ttl <= 0:
            self.stats["messages_dropped_ttl"] += 1
            return []

        # 4. Invoke handlers
        for handler in self._handlers.get(message.msg_type, []):
            handler(message)

        # 5. Prepare relay (decrement TTL)
        message.ttl -= TTL_DECREMENT
        if message.ttl <= 0:
            return []

        self.stats["messages_relayed"] += 1
        # The caller picks up to `fanout` connected peers to forward to.
        # Returning an empty list here because we don't own the peer list;
        # instead the Network layer will handle actual fan-out.
        return []

    # ------------------------------------------------------------------
    # Originate
    # ------------------------------------------------------------------

    def originate(self, message: P2PMessage) -> None:
        """
        Mark a locally originated message as seen so it is not echoed back.
        """
        self._mark_seen(message.msg_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_seen(self, msg_id: str) -> bool:
        """Return ``True`` if this message ID has already been processed."""
        return msg_id in self._seen

    def get_stats(self) -> Dict[str, int]:
        """Return gossip metrics."""
        return {**self.stats, "seen_cache_size": len(self._seen)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mark_seen(self, msg_id: str) -> None:
        """Add a message ID to the seen cache with LRU eviction."""
        self._seen[msg_id] = time.time()
        while len(self._seen) > SEEN_CACHE_SIZE:
            self._seen.popitem(last=False)
