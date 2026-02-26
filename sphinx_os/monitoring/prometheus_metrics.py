# sphinx_os/monitoring/prometheus_metrics.py
"""
Prometheus Metrics for SphinxOS
================================

Exposes operational metrics via the ``prometheus_client`` library so that any
Prometheus-compatible scraper (Prometheus, Grafana, VictoriaMetrics, …) can
collect them from the standard ``/metrics`` HTTP endpoint.

Metric groups
-------------
- **Blockchain** – chain height, best block timestamp, total transactions,
  total SPHINX mined.
- **Mining** – hashrate, blocks found, average Φ score, total rewards,
  active miners in pool.
- **Transaction pool** – number of pending transactions.
- **Bridge** – locked volume, total bridge transactions, total bridge fees.
- **Stratum** – connected workers, accepted/rejected shares.

Usage
-----
::

    from sphinx_os.monitoring import SphinxMetrics

    metrics = SphinxMetrics()              # singleton – safe to call many times
    metrics.on_block_mined(block)          # call from the miner loop
    metrics.on_transaction_added(tx)       # call from the mempool
    metrics.on_share_submitted(accepted=True)  # call from the stratum server
"""

from __future__ import annotations

import time
import logging
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        REGISTRY,
        make_asgi_app,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False

logger = logging.getLogger("SphinxOS.Metrics")

# --------------------------------------------------------------------------- #
# Singleton guard                                                              #
# --------------------------------------------------------------------------- #

_instance: Optional["SphinxMetrics"] = None


class SphinxMetrics:
    """
    Centralised Prometheus metric registry for SphinxOS.

    Only one instance is created per process (singleton pattern).  Call
    ``SphinxMetrics()`` anywhere in the codebase to obtain the same object.
    """

    def __new__(cls) -> "SphinxMetrics":
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialised = False
        return _instance

    # ---------------------------------------------------------------------- #
    # Initialisation                                                           #
    # ---------------------------------------------------------------------- #

    def __init__(self) -> None:
        if self._initialised:
            return
        self._initialised = True

        if not _PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not installed – metrics are disabled. "
                "Install with: pip install prometheus_client"
            )
            self._enabled = False
            return

        self._enabled = True
        self._registry = REGISTRY

        # ------------------------------------------------------------------ #
        # Blockchain metrics                                                   #
        # ------------------------------------------------------------------ #

        self.blockchain_height = Gauge(
            "sphinxos_blockchain_height",
            "Current chain height (number of blocks minus genesis).",
        )
        self.blockchain_best_block_timestamp = Gauge(
            "sphinxos_blockchain_best_block_timestamp_seconds",
            "Unix timestamp of the most recently mined block.",
        )
        self.blockchain_transactions_total = Counter(
            "sphinxos_blockchain_transactions_total",
            "Total number of transactions ever confirmed on-chain.",
        )
        self.blockchain_mined_supply = Gauge(
            "sphinxos_blockchain_mined_supply_sphinx",
            "Total SPHINX tokens mined so far.",
        )

        # ------------------------------------------------------------------ #
        # Mining metrics                                                       #
        # ------------------------------------------------------------------  #

        self.mining_hashrate = Gauge(
            "sphinxos_mining_hashrate_hps",
            "Current mining hashrate in hashes per second.",
        )
        self.mining_blocks_total = Counter(
            "sphinxos_mining_blocks_total",
            "Total number of blocks mined by this node.",
        )
        self.mining_rewards_total = Counter(
            "sphinxos_mining_rewards_total_sphinx",
            "Total SPHINX rewards earned by this node.",
        )
        self.mining_phi_score = Gauge(
            "sphinxos_mining_phi_score",
            "Average Φ (IIT consciousness) score of recently mined blocks.",
        )
        self.mining_active_workers = Gauge(
            "sphinxos_mining_active_workers",
            "Number of workers currently active in the mining pool.",
        )
        self.mining_block_time = Histogram(
            "sphinxos_mining_block_time_seconds",
            "Time taken to mine each block.",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        # ------------------------------------------------------------------ #
        # Transaction pool metrics                                             #
        # ------------------------------------------------------------------ #

        self.mempool_size = Gauge(
            "sphinxos_mempool_pending_transactions",
            "Number of transactions waiting in the memory pool.",
        )

        # ------------------------------------------------------------------ #
        # Bridge metrics                                                       #
        # ------------------------------------------------------------------ #

        self.bridge_volume_total = Counter(
            "sphinxos_bridge_volume_total",
            "Cumulative value (in source-chain units) bridged through the cross-chain bridge.",
        )
        self.bridge_transactions_total = Counter(
            "sphinxos_bridge_transactions_total",
            "Total number of cross-chain bridge transactions.",
        )
        self.bridge_fees_total = Counter(
            "sphinxos_bridge_fees_total",
            "Total fees collected by the cross-chain bridge.",
        )

        # ------------------------------------------------------------------ #
        # Stratum metrics                                                      #
        # ------------------------------------------------------------------ #

        self.stratum_connected_workers = Gauge(
            "sphinxos_stratum_connected_workers",
            "Number of Stratum workers currently connected.",
        )
        self.stratum_shares_accepted_total = Counter(
            "sphinxos_stratum_shares_accepted_total",
            "Total accepted Stratum shares.",
        )
        self.stratum_shares_rejected_total = Counter(
            "sphinxos_stratum_shares_rejected_total",
            "Total rejected Stratum shares.",
        )

        logger.info("Prometheus metrics initialised.")

    # ---------------------------------------------------------------------- #
    # Convenience update methods                                               #
    # ---------------------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """``True`` when prometheus_client is installed and metrics are live."""
        return self._enabled

    def on_block_mined(self, block, rewards: float = 0.0, block_time: float = 0.0) -> None:
        """
        Update metrics after a new block is appended to the chain.

        Parameters
        ----------
        block :
            The newly mined :class:`~sphinx_os.blockchain.block.Block`.
        rewards :
            SPHINX tokens earned for this block.
        block_time :
            Elapsed seconds from work-start to block acceptance.
        """
        if not self._enabled:
            return
        self.blockchain_height.set(block.index)
        self.blockchain_best_block_timestamp.set(getattr(block, "timestamp", time.time()))
        tx_count = len(getattr(block, "transactions", []))
        self.blockchain_transactions_total.inc(tx_count)
        if rewards > 0:
            self.mining_rewards_total.inc(rewards)
        self.mining_blocks_total.inc()
        phi = getattr(block, "phi_score", 0.0)
        if phi:
            self.mining_phi_score.set(phi)
        if block_time > 0:
            self.mining_block_time.observe(block_time)

    def on_transaction_added(self, pool_size: int) -> None:
        """Update mempool size gauge after a transaction is added."""
        if not self._enabled:
            return
        self.mempool_size.set(pool_size)

    def set_hashrate(self, hps: float) -> None:
        """Update the hashrate gauge."""
        if not self._enabled:
            return
        self.mining_hashrate.set(hps)

    def set_active_workers(self, count: int) -> None:
        """Update the active-workers gauge."""
        if not self._enabled:
            return
        self.mining_active_workers.set(count)

    def set_mined_supply(self, total_sphinx: float) -> None:
        """Update the total mined supply gauge."""
        if not self._enabled:
            return
        self.blockchain_mined_supply.set(total_sphinx)

    def on_bridge_transaction(self, volume: float, fee: float) -> None:
        """Update bridge counters after a cross-chain transaction is locked."""
        if not self._enabled:
            return
        self.bridge_volume_total.inc(volume)
        self.bridge_transactions_total.inc()
        if fee > 0:
            self.bridge_fees_total.inc(fee)

    def on_share_submitted(self, accepted: bool) -> None:
        """Update Stratum share counters."""
        if not self._enabled:
            return
        if accepted:
            self.stratum_shares_accepted_total.inc()
        else:
            self.stratum_shares_rejected_total.inc()

    def on_worker_connected(self) -> None:
        """Increment connected Stratum worker count."""
        if not self._enabled:
            return
        self.stratum_connected_workers.inc()

    def on_worker_disconnected(self) -> None:
        """Decrement connected Stratum worker count."""
        if not self._enabled:
            return
        self.stratum_connected_workers.dec()

    # ---------------------------------------------------------------------- #
    # ASGI / WSGI integration helpers                                          #
    # ---------------------------------------------------------------------- #

    def make_asgi_app(self):
        """
        Return a bare ASGI application that serves the ``/metrics`` endpoint.

        Useful for mounting under FastAPI::

            from sphinx_os.monitoring import SphinxMetrics
            from fastapi import FastAPI

            app = FastAPI()
            metrics_app = SphinxMetrics().make_asgi_app()
            app.mount("/metrics", metrics_app)
        """
        if not self._enabled:
            raise RuntimeError("prometheus_client is not installed.")
        return make_asgi_app(registry=self._registry)

    def generate_latest(self) -> bytes:
        """Return the current metric snapshot as UTF-8 encoded text."""
        if not self._enabled:
            return b""
        return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        """MIME type for the Prometheus text exposition format."""
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST
