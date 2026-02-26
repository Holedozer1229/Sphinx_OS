"""
Tests for SphinxOS Stratum v1 Server
"""

import asyncio
import json
import sys
import os
import time

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeBlock:
    def __init__(self, index: int = 0):
        self.index = index
        self.hash = "0" * 64
        self.difficulty = 1
        self.timestamp = int(time.time())
        self.transactions = []


class _FakeBlockchain:
    """Minimal blockchain stub used by the Stratum server tests."""

    def __init__(self):
        self._block = _FakeBlock(index=100)

    def get_latest_block(self):
        return self._block

    def add_block(self, block):
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blockchain():
    return _FakeBlockchain()


@pytest.fixture
def stratum_server(blockchain):
    from sphinx_os.mining.stratum_server import StratumServer
    return StratumServer(blockchain, host="127.0.0.1", port=0, difficulty=1)


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

class TestStratumServerUnit:
    """Non-network unit tests for StratumServer internals."""

    def test_initial_worker_count(self, stratum_server):
        assert stratum_server.worker_count == 0

    def test_extranonce1_uniqueness(self, stratum_server):
        e1 = stratum_server._next_extranonce1()
        e2 = stratum_server._next_extranonce1()
        e3 = stratum_server._next_extranonce1()
        assert e1 != e2 != e3
        assert len(e1) == 8  # EXTRANONCE1_BYTES * 2 hex chars

    def test_make_job_structure(self, stratum_server):
        job = stratum_server._make_job()
        required_keys = {
            "job_id", "prevhash", "coinbase1", "coinbase2",
            "merkle_branch", "version", "nbits", "ntime", "clean_jobs",
        }
        assert required_keys.issubset(job.keys())
        assert len(job["job_id"]) == 8
        assert job["prevhash"] == "0" * 64  # fake block hash
        assert job["clean_jobs"] is True

    def test_make_job_sets_current_job(self, stratum_server):
        stratum_server._make_job()
        assert stratum_server._current_job is not None
        assert stratum_server._current_job_id != ""

    def test_validate_share_stale_job(self, stratum_server):
        stratum_server._make_job()
        from sphinx_os.mining.stratum_server import StratumWorker

        # Build a fake worker (no real socket needed for this test)
        worker = object.__new__(StratumWorker)
        worker.extranonce1 = "aabbccdd"
        result = stratum_server._validate_share(
            worker, "stale_job_id", "00000000", "deadbeef", "cafebabe"
        )
        assert result is False

    def test_validate_share_correct_job(self, stratum_server):
        stratum_server._make_job()
        from sphinx_os.mining.stratum_server import StratumWorker

        worker = object.__new__(StratumWorker)
        worker.extranonce1 = "aabbccdd"
        # With difficulty=1 almost every nonce should be accepted
        result = stratum_server._validate_share(
            worker,
            stratum_server._current_job_id,
            "00000000",
            stratum_server._current_job["ntime"],
            "00000001",
        )
        # With difficulty=1 the target is just below 2^256, so this must accept.
        assert result is True


# ---------------------------------------------------------------------------
# Integration tests (real asyncio TCP)
# ---------------------------------------------------------------------------

class TestStratumServerIntegration:
    """Integration tests that spin up a real asyncio TCP server."""

    @pytest.mark.asyncio
    async def test_subscribe_and_authorize(self, blockchain):
        """A worker can subscribe and authorize over a real TCP connection."""
        from sphinx_os.mining.stratum_server import StratumServer

        server = StratumServer(blockchain, host="127.0.0.1", port=0, difficulty=1)

        # Start server on a random port
        tcp_server = await asyncio.start_server(
            server._handle_client, "127.0.0.1", 0
        )
        port = tcp_server.sockets[0].getsockname()[1]

        async with tcp_server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # --- mining.subscribe ---
            subscribe_msg = json.dumps({
                "id": 1,
                "method": "mining.subscribe",
                "params": ["TestMiner/1.0"],
            }) + "\n"
            writer.write(subscribe_msg.encode())
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=5)
            resp = json.loads(line)
            assert resp["id"] == 1
            assert resp["error"] is None
            result = resp["result"]
            assert isinstance(result, list) and len(result) == 3
            extranonce1 = result[1]
            extranonce2_size = result[2]
            assert len(extranonce1) == 8  # 4 bytes hex
            assert extranonce2_size == 4

            # Consume set_difficulty and notify notifications
            for _ in range(2):
                await asyncio.wait_for(reader.readline(), timeout=5)

            # --- mining.authorize ---
            auth_msg = json.dumps({
                "id": 2,
                "method": "mining.authorize",
                "params": ["SPHINX_MINER_ADDRESS", "x"],
            }) + "\n"
            writer.write(auth_msg.encode())
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=5)
            resp = json.loads(line)
            assert resp["id"] == 2
            assert resp["result"] is True
            assert resp["error"] is None

            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_submit_share(self, blockchain):
        """An authorised worker can submit a share and get a valid response."""
        from sphinx_os.mining.stratum_server import StratumServer

        server = StratumServer(blockchain, host="127.0.0.1", port=0, difficulty=1)

        tcp_server = await asyncio.start_server(
            server._handle_client, "127.0.0.1", 0
        )
        port = tcp_server.sockets[0].getsockname()[1]

        async with tcp_server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Subscribe
            writer.write((json.dumps({"id": 1, "method": "mining.subscribe", "params": []}) + "\n").encode())
            await writer.drain()
            await asyncio.wait_for(reader.readline(), timeout=5)  # result
            for _ in range(2):
                await asyncio.wait_for(reader.readline(), timeout=5)  # notifications

            # Authorize
            writer.write((json.dumps({"id": 2, "method": "mining.authorize", "params": ["ADDR", "x"]}) + "\n").encode())
            await writer.drain()
            await asyncio.wait_for(reader.readline(), timeout=5)  # result

            # Submit (difficulty=1 so almost any nonce is accepted)
            job_id = server._current_job_id
            ntime = server._current_job["ntime"]
            submit_msg = json.dumps({
                "id": 3,
                "method": "mining.submit",
                "params": ["ADDR", job_id, "00000000", ntime, "00000001"],
            }) + "\n"
            writer.write(submit_msg.encode())
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=5)
            resp = json.loads(line)
            assert resp["id"] == 3
            # With difficulty=1 the share must be accepted
            assert resp["result"] is True

            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_unauthorized_submit_rejected(self, blockchain):
        """A worker that hasn't authorised gets a rejection on submit."""
        from sphinx_os.mining.stratum_server import StratumServer

        server = StratumServer(blockchain, host="127.0.0.1", port=0, difficulty=1)
        tcp_server = await asyncio.start_server(
            server._handle_client, "127.0.0.1", 0
        )
        port = tcp_server.sockets[0].getsockname()[1]

        async with tcp_server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            # Subscribe but do NOT authorize
            writer.write((json.dumps({"id": 1, "method": "mining.subscribe", "params": []}) + "\n").encode())
            await writer.drain()
            await asyncio.wait_for(reader.readline(), timeout=5)
            for _ in range(2):
                await asyncio.wait_for(reader.readline(), timeout=5)

            # Submit without authorizing
            job_id = server._current_job_id
            submit_msg = json.dumps({
                "id": 4,
                "method": "mining.submit",
                "params": ["ADDR", job_id, "00000000", "deadbeef", "cafebabe"],
            }) + "\n"
            writer.write(submit_msg.encode())
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=5)
            resp = json.loads(line)
            assert resp["result"] is False

            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, blockchain):
        """Unknown Stratum methods must return an error response."""
        from sphinx_os.mining.stratum_server import StratumServer

        server = StratumServer(blockchain, host="127.0.0.1", port=0, difficulty=1)
        tcp_server = await asyncio.start_server(
            server._handle_client, "127.0.0.1", 0
        )
        port = tcp_server.sockets[0].getsockname()[1]

        async with tcp_server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            writer.write((json.dumps({"id": 5, "method": "mining.ping", "params": []}) + "\n").encode())
            await writer.drain()

            line = await asyncio.wait_for(reader.readline(), timeout=5)
            resp = json.loads(line)
            assert resp["id"] == 5
            assert resp["result"] is None
            assert resp["error"] is not None

            writer.close()
            await writer.wait_closed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
