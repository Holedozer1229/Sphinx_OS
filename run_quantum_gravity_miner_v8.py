#!/usr/bin/env python3
"""
============================================================================
run_quantum_gravity_miner_v8.py — SphinxOS Quantum Gravity Miner IIT v8
Headless 24/7 runner for Digital Ocean droplets.
============================================================================

Runs the Quantum Gravity Miner IIT v8 kernel continuously, logging results
to stdout (captured by systemd journal) and optionally to a log file.

Environment variables:
    QGM_DIFFICULTY      PoW difficulty target (default: 50000)
    QGM_MAX_ATTEMPTS    Max nonces per round (default: 1000000)
    QGM_N_NODES         IIT qubit node count (default: 3)
    QGM_QG_THRESHOLD    Quantum gravity curvature threshold (default: 0.1)
    QGM_ROUND_DELAY     Seconds between rounds (default: 1)
    QGM_LOG_FILE        Optional log file path
============================================================================
"""

import logging
import os
import sys
import time
import hashlib

# Configure logging
LOG_FILE = os.environ.get("QGM_LOG_FILE", "")
log_handlers = [logging.StreamHandler(sys.stdout)]
if LOG_FILE:
    try:
        log_handlers.append(logging.FileHandler(LOG_FILE))
    except OSError as exc:
        print(f"⚠️  Cannot open log file {LOG_FILE}: {exc}", file=sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("QGMinerV8Runner")


def main() -> None:
    difficulty = int(os.environ.get("QGM_DIFFICULTY", "50000"))
    max_attempts = int(os.environ.get("QGM_MAX_ATTEMPTS", "1000000"))
    n_nodes = int(os.environ.get("QGM_N_NODES", "3"))
    qg_threshold = float(os.environ.get("QGM_QG_THRESHOLD", "0.1"))
    round_delay = float(os.environ.get("QGM_ROUND_DELAY", "1"))

    logger.info("=" * 70)
    logger.info("  Quantum Gravity Miner IIT v8 — Headless Runner")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info("  Difficulty    : %d", difficulty)
    logger.info("  Max attempts  : %d", max_attempts)
    logger.info("  IIT nodes     : %d", n_nodes)
    logger.info("  QG threshold  : %.3f", qg_threshold)
    logger.info("  Round delay   : %.1f s", round_delay)

    # Import the v8 kernel (done after logging init so errors are captured)
    try:
        from sphinx_os.mining.quantum_gravity_miner_iit_v8 import QuantumGravityMinerIITv8
        kernel = QuantumGravityMinerIITv8(
            qg_threshold=qg_threshold,
            n_nodes=n_nodes,
        )
        logger.info("✓ QuantumGravityMinerIITv8 kernel initialised")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to initialise QGMinerIITv8: %s", exc)
        sys.exit(1)

    round_num = 0
    total_blocks_mined = 0

    logger.info("Starting continuous mining loop…")

    while True:
        round_num += 1
        # Build deterministic block data from round number + timestamp
        block_data = (
            f"sphinxos-qgm-v8-block-{round_num}-"
            f"{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}"
        )

        logger.info("Round %d | block_data=%s…", round_num, block_data[:32])

        try:
            result = kernel.mine(
                block_data=block_data,
                difficulty=difficulty,
                max_attempts=max_attempts,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Mining error in round %d: %s", round_num, exc)
            time.sleep(round_delay)
            continue

        if result.nonce is not None:
            total_blocks_mined += 1
            logger.info(
                "✅ Block mined! round=%d nonce=%d hash=%s…",
                round_num,
                result.nonce,
                result.block_hash[:16] if result.block_hash else "?",
            )
            logger.info(
                "   Φ_total=%.4f  Φ_qg=%.4f  Φ_holo=%.4f  phi_score=%.1f",
                result.phi_total,
                result.qg_score,
                result.holo_score,
                result.phi_score,
            )
            logger.info("   Total blocks mined: %d", total_blocks_mined)
        else:
            logger.info(
                "⚙  No valid nonce in %d attempts (round %d)",
                max_attempts,
                round_num,
            )

        time.sleep(round_delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Quantum Gravity Miner v8 stopped by user")
        sys.exit(0)
