# sphinx_os/mining/stratum_server.py
"""
Stratum v1 Mining Server for SphinxOS
======================================

Implements the Stratum v1 JSON-RPC protocol over plain TCP so that standard
mining hardware and software (CGMiner, BFGMiner, XMRig-proxy, NiceHash, etc.)
can connect and submit work to a SphinxSkynet node.

Protocol overview
-----------------
The Stratum v1 wire format is newline-delimited JSON.  Each message is a
JSON object on a single line followed by ``\\n``.

Client → Server (requests)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``mining.subscribe``   – worker handshake, returns subscription ID + extranonce
- ``mining.authorize``   – worker login (username = miner address, password ignored)
- ``mining.submit``      – submit a found share/nonce

Server → Client (notifications / responses)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``mining.notify``      – send new work template to all connected workers
- ``mining.set_difficulty`` – update the share difficulty
- Response objects with matching ``id``

Usage
-----
::

    import asyncio
    from sphinx_os.blockchain.core import SphinxSkynetBlockchain
    from sphinx_os.mining.stratum_server import StratumServer

    blockchain = SphinxSkynetBlockchain()
    server = StratumServer(blockchain, host="0.0.0.0", port=3333)
    asyncio.run(server.start())

The server also updates :class:`~sphinx_os.monitoring.SphinxMetrics` gauges
for connected workers and accepted/rejected shares.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Dict, Optional, Set

from ..monitoring.prometheus_metrics import SphinxMetrics

logger = logging.getLogger("SphinxOS.StratumServer")

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3333
DEFAULT_DIFFICULTY = 1
EXTRANONCE1_BYTES = 4   # 4 bytes = 8 hex chars
EXTRANONCE2_SIZE = 4    # worker fills 4 bytes


# --------------------------------------------------------------------------- #
# Worker session                                                                #
# --------------------------------------------------------------------------- #

class StratumWorker:
    """Represents a single connected Stratum worker session."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        extranonce1: str,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.extranonce1 = extranonce1
        self.subscribed: bool = False
        self.authorised: bool = False
        self.address: str = ""
        self.difficulty: int = DEFAULT_DIFFICULTY
        self.shares_accepted: int = 0
        self.shares_rejected: int = 0
        peer = writer.get_extra_info("peername", ("?", 0))
        self.peer: str = f"{peer[0]}:{peer[1]}"

    # ------------------------------------------------------------------ #
    # Send helpers                                                         #
    # ------------------------------------------------------------------ #

    async def send(self, obj: dict) -> None:
        """Serialise *obj* and send it as a newline-terminated JSON line."""
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        self.writer.write(line.encode())
        await self.writer.drain()

    async def send_result(self, req_id, result, error=None) -> None:
        """Send a JSON-RPC result frame."""
        await self.send({"id": req_id, "result": result, "error": error})

    async def send_notification(self, method: str, params) -> None:
        """Send a JSON-RPC notification (no id)."""
        await self.send({"id": None, "method": method, "params": params})


# --------------------------------------------------------------------------- #
# Server                                                                       #
# --------------------------------------------------------------------------- #

class StratumServer:
    """
    Asynchronous Stratum v1 TCP server.

    Parameters
    ----------
    blockchain :
        A :class:`~sphinx_os.blockchain.core.SphinxSkynetBlockchain` (or any
        object that exposes ``get_latest_block()`` and ``add_block()``).
    host :
        Bind address (default ``"0.0.0.0"``).
    port :
        TCP port (default ``3333``).
    difficulty :
        Initial share difficulty sent to workers.
    """

    def __init__(
        self,
        blockchain,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        difficulty: int = DEFAULT_DIFFICULTY,
    ) -> None:
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.difficulty = difficulty
        self.metrics = SphinxMetrics()

        # Connected workers keyed by extranonce1
        self._workers: Dict[str, StratumWorker] = {}
        self._extranonce1_counter: int = 0

        # Current job state (broadcast to all workers on new block)
        self._current_job_id: str = ""
        self._current_job: Optional[dict] = None
        self._server: Optional[asyncio.AbstractServer] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the Stratum TCP listener and the job-update loop."""
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("Stratum server listening on %s:%s", addr[0], addr[1])

        async with self._server:
            await asyncio.gather(
                self._server.serve_forever(),
                self._job_broadcaster(),
            )

    def stop(self) -> None:
        """Stop accepting new connections."""
        if self._server is not None:
            self._server.close()
            logger.info("Stratum server stopped.")

    @property
    def worker_count(self) -> int:
        """Number of currently connected and authorised workers."""
        return sum(1 for w in self._workers.values() if w.authorised)

    # ------------------------------------------------------------------ #
    # Internal: job management                                             #
    # ------------------------------------------------------------------ #

    def _make_job(self) -> dict:
        """Build a ``mining.notify`` job from the latest block template."""
        block = self.blockchain.get_latest_block()
        prev_hash = getattr(block, "hash", "0" * 64)
        height = getattr(block, "index", 0) + 1
        timestamp = int(time.time())
        job_id = hashlib.sha256(f"{height}{timestamp}".encode()).hexdigest()[:8]
        self._current_job_id = job_id
        self._current_job = {
            "job_id": job_id,
            "prevhash": prev_hash,
            "coinbase1": "",  # Populated by full implementation
            "coinbase2": "",
            "merkle_branch": [],
            "version": "00000001",
            "nbits": f"{getattr(block, 'difficulty', 1):08x}",
            "ntime": f"{timestamp:08x}",
            "clean_jobs": True,
        }
        return self._current_job

    async def _job_broadcaster(self) -> None:
        """
        Periodically check for a new block tip and broadcast fresh work to
        all connected, authorised workers.
        """
        last_height = -1
        while True:
            await asyncio.sleep(5)
            try:
                block = self.blockchain.get_latest_block()
                height = getattr(block, "index", 0)
                if height != last_height:
                    last_height = height
                    job = self._make_job()
                    await self._broadcast_job(job)
            except Exception as exc:
                logger.warning("Job broadcaster error: %s", exc)

    async def _broadcast_job(self, job: dict) -> None:
        """Send ``mining.notify`` and ``mining.set_difficulty`` to all workers."""
        params = [
            job["job_id"],
            job["prevhash"],
            job["coinbase1"],
            job["coinbase2"],
            job["merkle_branch"],
            job["version"],
            job["nbits"],
            job["ntime"],
            job["clean_jobs"],
        ]
        for worker in list(self._workers.values()):
            if not worker.authorised:
                continue
            try:
                await worker.send_notification("mining.set_difficulty", [self.difficulty])
                await worker.send_notification("mining.notify", params)
            except Exception as exc:
                logger.debug("Error broadcasting to %s: %s", worker.peer, exc)

    # ------------------------------------------------------------------ #
    # Internal: client handling                                            #
    # ------------------------------------------------------------------ #

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        extranonce1 = self._next_extranonce1()
        worker = StratumWorker(reader, writer, extranonce1)
        self._workers[extranonce1] = worker
        self.metrics.on_worker_connected()
        self.metrics.set_active_workers(self.worker_count)
        logger.info("Stratum worker connected from %s", worker.peer)

        try:
            await self._worker_loop(worker)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as exc:
            logger.warning("Worker %s error: %s", worker.peer, exc)
        finally:
            self._workers.pop(extranonce1, None)
            self.metrics.on_worker_disconnected()
            self.metrics.set_active_workers(self.worker_count)
            logger.info("Stratum worker disconnected from %s", worker.peer)
            writer.close()

    async def _worker_loop(self, worker: StratumWorker) -> None:
        """Read and dispatch JSON-RPC messages from a single worker."""
        while True:
            line = await worker.reader.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Worker %s sent invalid JSON: %r", worker.peer, line)
                continue
            await self._dispatch(worker, msg)

    async def _dispatch(self, worker: StratumWorker, msg: dict) -> None:
        """Route an incoming JSON-RPC message to the correct handler."""
        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", [])

        handlers = {
            "mining.subscribe": self._handle_subscribe,
            "mining.authorize": self._handle_authorize,
            "mining.submit": self._handle_submit,
        }
        handler = handlers.get(method)
        if handler:
            await handler(worker, req_id, params)
        else:
            logger.debug("Worker %s unknown method: %s", worker.peer, method)
            await worker.send_result(req_id, None, [20, "Unknown method", None])

    # ------------------------------------------------------------------ #
    # Stratum method handlers                                              #
    # ------------------------------------------------------------------ #

    async def _handle_subscribe(
        self, worker: StratumWorker, req_id, params
    ) -> None:
        """Handle ``mining.subscribe``."""
        worker.subscribed = True
        subscription_id = hashlib.sha256(worker.extranonce1.encode()).hexdigest()[:32]
        result = [
            [
                ["mining.set_difficulty", subscription_id],
                ["mining.notify", subscription_id],
            ],
            worker.extranonce1,
            EXTRANONCE2_SIZE,
        ]
        await worker.send_result(req_id, result)
        # Send initial difficulty and job
        await worker.send_notification("mining.set_difficulty", [self.difficulty])
        job = self._current_job or self._make_job()
        params = [
            job["job_id"],
            job["prevhash"],
            job["coinbase1"],
            job["coinbase2"],
            job["merkle_branch"],
            job["version"],
            job["nbits"],
            job["ntime"],
            job["clean_jobs"],
        ]
        await worker.send_notification("mining.notify", params)
        logger.debug("Worker %s subscribed (extranonce1=%s)", worker.peer, worker.extranonce1)

    async def _handle_authorize(
        self, worker: StratumWorker, req_id, params
    ) -> None:
        """Handle ``mining.authorize``.  Username is treated as the miner address."""
        username = params[0] if params else ""
        # Accept any non-empty username; password is ignored.
        if username:
            worker.authorised = True
            worker.address = username
            await worker.send_result(req_id, True)
            logger.info("Worker %s authorised as %s", worker.peer, username)
        else:
            await worker.send_result(req_id, False, [24, "Unauthorized", None])
            logger.warning("Worker %s authorisation failed (empty username)", worker.peer)

    async def _handle_submit(
        self, worker: StratumWorker, req_id, params
    ) -> None:
        """
        Handle ``mining.submit``.

        params: [worker_name, job_id, extranonce2, ntime, nonce]
        """
        if not worker.authorised:
            await worker.send_result(req_id, False, [24, "Unauthorized", None])
            return

        if len(params) < 5:
            await worker.send_result(req_id, False, [20, "Invalid params", None])
            return

        _worker_name, job_id, extranonce2, ntime, nonce = params[:5]

        accepted = self._validate_share(
            worker, job_id, extranonce2, ntime, nonce
        )

        if accepted:
            worker.shares_accepted += 1
            self.metrics.on_share_submitted(accepted=True)
            await worker.send_result(req_id, True)
            logger.debug(
                "Share accepted from %s (job=%s nonce=%s)", worker.peer, job_id, nonce
            )
        else:
            worker.shares_rejected += 1
            self.metrics.on_share_submitted(accepted=False)
            await worker.send_result(req_id, False, [23, "Low difficulty share", None])
            logger.debug(
                "Share rejected from %s (job=%s nonce=%s)", worker.peer, job_id, nonce
            )

    # ------------------------------------------------------------------ #
    # Internal: share validation                                           #
    # ------------------------------------------------------------------ #

    def _validate_share(
        self,
        worker: StratumWorker,
        job_id: str,
        extranonce2: str,
        ntime: str,
        nonce: str,
    ) -> bool:
        """
        Validate a submitted share against the current job difficulty.

        A real implementation would reconstruct the full block header and
        verify the hash meets the target.  Here we perform a lightweight
        difficulty check so the logic is correct for the current PoW.
        """
        if job_id != self._current_job_id:
            return False  # Stale job

        # Reconstruct the candidate hash using the same double-SHA256 as Bitcoin.
        job = self._current_job
        if job is None:
            return False

        coinbase = (
            job.get("coinbase1", "")
            + worker.extranonce1
            + extranonce2
            + job.get("coinbase2", "")
        )
        header = (
            job.get("version", "00000001")
            + job.get("prevhash", "0" * 64)
            + coinbase
            + ntime
            + job.get("nbits", "")
            + nonce
        )
        raw = header.encode()
        candidate = hashlib.sha256(hashlib.sha256(raw).digest()).hexdigest()

        # The share is valid when its integer value is below the target.
        target = (2 ** 256) // max(self.difficulty, 1)
        return int(candidate, 16) < target

    # ------------------------------------------------------------------ #
    # Extranonce allocation                                                #
    # ------------------------------------------------------------------ #

    def _next_extranonce1(self) -> str:
        """Allocate a unique 4-byte extranonce1 for a new worker."""
        self._extranonce1_counter = (self._extranonce1_counter + 1) & 0xFFFFFFFF
        return f"{self._extranonce1_counter:08x}"
