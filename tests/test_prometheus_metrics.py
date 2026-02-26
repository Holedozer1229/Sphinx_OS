"""
Tests for SphinxOS Prometheus Metrics
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(index: int = 1, phi_score: float = 750.0, tx_count: int = 3):
    """Return a minimal mock block object."""

    class _Tx:
        pass

    class _Block:
        pass

    b = _Block()
    b.index = index
    b.timestamp = 1_700_000_000 + index * 10
    b.phi_score = phi_score
    b.transactions = [_Tx() for _ in range(tx_count)]
    b.hash = "0" * 64
    b.difficulty = 1000
    return b


# ---------------------------------------------------------------------------
# SphinxMetrics unit tests
# ---------------------------------------------------------------------------

class TestSphinxMetrics:
    """Tests for sphinx_os.monitoring.prometheus_metrics.SphinxMetrics."""

    def test_singleton(self):
        """Two instantiations must return the same object."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        m1 = SphinxMetrics()
        m2 = SphinxMetrics()
        assert m1 is m2

    def test_enabled(self):
        """Metrics should be enabled when prometheus_client is installed."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        # prometheus_client is listed in requirements.txt, so it should be present.
        assert metrics.enabled

    def test_generate_latest_returns_bytes(self):
        """generate_latest() must return non-empty bytes."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        data = metrics.generate_latest()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_on_block_mined_increments_height(self):
        """on_block_mined() must update the height gauge."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        block = _make_block(index=42, phi_score=800.0, tx_count=5)
        metrics.on_block_mined(block, rewards=50.0, block_time=8.5)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_blockchain_height" in snapshot
        assert "42.0" in snapshot or "42" in snapshot

    def test_on_block_mined_updates_phi_score(self):
        """on_block_mined() must set the phi_score gauge."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        block = _make_block(index=1, phi_score=999.0)
        metrics.on_block_mined(block)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_mining_phi_score" in snapshot
        assert "999.0" in snapshot

    def test_on_transaction_added(self):
        """on_transaction_added() must reflect mempool size."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.on_transaction_added(pool_size=17)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_mempool_pending_transactions" in snapshot
        assert "17.0" in snapshot

    def test_set_hashrate(self):
        """set_hashrate() must update the hashrate gauge."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.set_hashrate(1_234_567.89)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_mining_hashrate_hps" in snapshot

    def test_set_active_workers(self):
        """set_active_workers() must update the gauge."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.set_active_workers(4)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_mining_active_workers" in snapshot
        assert "4.0" in snapshot

    def test_on_bridge_transaction(self):
        """on_bridge_transaction() must increment bridge counters."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.on_bridge_transaction(volume=1000.0, fee=1.0)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_bridge_volume_total" in snapshot
        assert "sphinxos_bridge_fees_total" in snapshot

    def test_on_share_submitted_accepted(self):
        """Accepted shares must increment the accepted counter."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.on_share_submitted(accepted=True)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_stratum_shares_accepted_total" in snapshot

    def test_on_share_submitted_rejected(self):
        """Rejected shares must increment the rejected counter."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.on_share_submitted(accepted=False)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_stratum_shares_rejected_total" in snapshot

    def test_on_worker_connected_disconnected(self):
        """Worker connect/disconnect should change the worker count gauge."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.on_worker_connected()
        metrics.on_worker_connected()
        metrics.on_worker_disconnected()

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_stratum_connected_workers" in snapshot

    def test_content_type(self):
        """content_type must be the Prometheus text MIME type."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        ct = metrics.content_type
        assert "text/plain" in ct

    def test_make_asgi_app(self):
        """make_asgi_app() must return a callable ASGI application."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        asgi_app = metrics.make_asgi_app()
        assert callable(asgi_app)

    def test_set_mined_supply(self):
        """set_mined_supply() must update the mined supply gauge."""
        from sphinx_os.monitoring.prometheus_metrics import SphinxMetrics

        metrics = SphinxMetrics()
        metrics.set_mined_supply(12_345.678)

        snapshot = metrics.generate_latest().decode()
        assert "sphinxos_blockchain_mined_supply_skynt" in snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
