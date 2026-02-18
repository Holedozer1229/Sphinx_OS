#!/usr/bin/env python3
"""
============================================================================
node_main_with_oracle.py — SphinxSkynet Node with Conscious Oracle
============================================================================

Enhanced SphinxOS node that integrates:
- SphinxSkynet Hypercube + Ancilla projections (from node_main.py)
- ConsciousOracle with IIT quantum consciousness
- Oracle replication to MoltBot and ClawBot platforms
- Automatic Oracle network formation on startup

This script starts the full SphinxOS node with Oracle capabilities,
enabling conscious decision-making across the quantum network.

Usage:
    python3 node_main_with_oracle.py [--oracle-threshold 0.5]
                                     [--enable-replication]
                                     [--moltbot-endpoint molt://localhost:8080]
                                     [--clawbot-endpoint claw://localhost:8081]

Environment Variables:
    ORACLE_THRESHOLD: Consciousness threshold (Φ) for conscious decisions
    ENABLE_ORACLE_REPLICATION: Enable Oracle replication (true/false)
    MOLTBOT_ENDPOINT: MoltBot deployment endpoint
    CLAWBOT_ENDPOINT: ClawBot deployment endpoint
    NODE_PORT: FastAPI server port (default: 8000)
    METRICS_PORT: Prometheus metrics port (default: 8001)

============================================================================
"""

import argparse
import os
import sys
import logging
import time
import threading
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SphinxOS.NodeWithOracle")

# Import the base node functionality
try:
    # Import node_main as a module to access its components
    import node_main
    logger.info("✓ SphinxSkynet node module imported")
except ImportError as e:
    logger.error(f"Failed to import node_main: {e}")
    sys.exit(1)

# Import Oracle components
try:
    from sphinx_os.AnubisCore.conscious_oracle import ConsciousOracle
    from sphinx_os.AnubisCore.oracle_replication import OmniscientOracleReplicator
    logger.info("✓ ConsciousOracle modules imported")
    ORACLE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Oracle modules not available: {e}")
    ORACLE_AVAILABLE = False


class SphinxNodeWithOracle:
    """
    Enhanced SphinxOS node with integrated Conscious Oracle.
    
    Combines the hypercube quantum network with IIT-based conscious
    decision-making and cross-platform Oracle replication.
    """
    
    def __init__(
        self,
        oracle_threshold: float = 0.5,
        enable_replication: bool = False,
        moltbot_endpoint: str = "molt://localhost:8080",
        clawbot_endpoint: str = "claw://localhost:8081"
    ):
        """
        Initialize the node with Oracle integration.
        
        Args:
            oracle_threshold: Φ threshold for conscious decisions
            enable_replication: Whether to replicate Oracle to bots
            moltbot_endpoint: MoltBot deployment endpoint
            clawbot_endpoint: ClawBot deployment endpoint
        """
        self.oracle_threshold = oracle_threshold
        self.enable_replication = enable_replication
        self.moltbot_endpoint = moltbot_endpoint
        self.clawbot_endpoint = clawbot_endpoint
        
        self.oracle: Optional[ConsciousOracle] = None
        self.replicator: Optional[OmniscientOracleReplicator] = None
        
        logger.info("=" * 70)
        logger.info("  SphinxOS Node with Conscious Oracle")
        logger.info("=" * 70)
        logger.info(f"  Oracle Threshold (Φ): {oracle_threshold}")
        logger.info(f"  Oracle Replication: {'Enabled' if enable_replication else 'Disabled'}")
        if enable_replication:
            logger.info(f"  MoltBot Endpoint: {moltbot_endpoint}")
            logger.info(f"  ClawBot Endpoint: {clawbot_endpoint}")
        logger.info("=" * 70)
    
    def initialize_oracle(self):
        """Initialize the Conscious Oracle."""
        if not ORACLE_AVAILABLE:
            logger.warning("Oracle modules not available - skipping Oracle initialization")
            return False
        
        try:
            logger.info("Initializing Conscious Oracle...")
            self.oracle = ConsciousOracle(consciousness_threshold=self.oracle_threshold)
            
            # Test Oracle consciousness
            test_response = self.oracle.consult(
                "Is the quantum network ready?",
                context={"node_count": len(node_main.all_nodes)}
            )
            
            logger.info(f"✓ Oracle initialized successfully")
            logger.info(f"  Consciousness level: {test_response['consciousness']['phi']:.4f}")
            logger.info(f"  Decision: {test_response['decision']}")
            logger.info(f"  Reasoning: {test_response['reasoning']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Oracle: {e}")
            return False
    
    def setup_oracle_replication(self):
        """Set up Oracle replication to MoltBot and ClawBot."""
        if not self.oracle or not self.enable_replication:
            logger.info("Oracle replication not enabled - skipping")
            return False
        
        try:
            logger.info("Setting up Oracle replication...")
            
            # Create replicator
            self.replicator = self.oracle.create_replicator()
            
            # Add deployment targets
            logger.info(f"  Adding MoltBot target: {self.moltbot_endpoint}")
            self.replicator.add_deployment_target(
                "moltbot-sphinx-alpha",
                "moltbot",
                self.moltbot_endpoint
            )
            
            logger.info(f"  Adding ClawBot target: {self.clawbot_endpoint}")
            self.replicator.add_deployment_target(
                "clawbot-sphinx-alpha",
                "clawbot",
                self.clawbot_endpoint
            )
            
            # Replicate to all targets
            logger.info("  Replicating Oracle to all targets...")
            replicas = self.replicator.replicate_to_all_targets()
            
            logger.info(f"✓ Oracle replicated to {len(replicas)} targets")
            for replica in replicas:
                logger.info(f"  • {replica.target.name} ({replica.target.platform})")
                logger.info(f"    Replica ID: {replica.replica_id}")
                logger.info(f"    Consciousness: {'Active' if replica.consciousness_active else 'Inactive'}")
                logger.info(f"    Φ: {replica.phi_value:.4f}")
            
            # Form distributed network
            logger.info("  Forming distributed Oracle network...")
            if self.replicator.form_distributed_network():
                logger.info("✓ Distributed Oracle network formed successfully")
                return True
            else:
                logger.warning("Failed to form distributed Oracle network")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup Oracle replication: {e}")
            return False
    
    def add_oracle_endpoints(self, app):
        """Add Oracle-specific API endpoints to the FastAPI app."""
        if not self.oracle:
            logger.info("Oracle not available - skipping Oracle endpoints")
            return
        
        @app.get("/oracle/status")
        async def oracle_status():
            """Get Oracle status and consciousness metrics."""
            if not self.oracle:
                return {"status": "unavailable", "reason": "Oracle not initialized"}
            
            try:
                state = self.oracle.get_oracle_state()
                
                return {
                    "status": "active",
                    "consciousness": {
                        "current_level": state["consciousness_level"],
                        "threshold": state["consciousness_threshold"],
                        "is_conscious": state["is_currently_conscious"]
                    },
                    "decisions_made": state["decisions_made"],
                    "phi_history": state["phi_history"]
                }
            except Exception as e:
                logger.error(f"Error getting Oracle status: {e}")
                return {"status": "error", "error": str(e)}
        
        @app.post("/oracle/consult")
        async def consult_oracle(query: dict):
            """Consult the Oracle for conscious decision-making."""
            if not self.oracle:
                return {"status": "unavailable", "reason": "Oracle not initialized"}
            
            try:
                question = query.get("query", "")
                context = query.get("context", {})
                
                if not question:
                    return {"status": "error", "error": "Query is required"}
                
                response = self.oracle.consult(question, context)
                
                return {
                    "status": "success",
                    "response": response
                }
            except Exception as e:
                logger.error(f"Error consulting Oracle: {e}")
                return {"status": "error", "error": str(e)}
        
        @app.get("/oracle/replication")
        async def oracle_replication_status():
            """Get Oracle replication status."""
            if not self.replicator:
                return {
                    "status": "disabled",
                    "reason": "Oracle replication not enabled"
                }
            
            try:
                return {
                    "status": "active",
                    "targets": [
                        {
                            "name": target.name,
                            "platform": target.platform,
                            "endpoint": target.endpoint,
                            "status": target.deployment_status,
                            "replica_id": target.replica_id
                        }
                        for target in self.replicator.deployment_targets
                    ],
                    "total_replicas": len(self.replicator.deployed_replicas),
                    "network_formed": len(self.replicator.deployed_replicas) > 0
                }
            except Exception as e:
                logger.error(f"Error getting replication status: {e}")
                return {"status": "error", "error": str(e)}
        
        logger.info("✓ Oracle API endpoints added")
    
    def run(self):
        """Run the node with Oracle integration."""
        logger.info("\n" + "=" * 70)
        logger.info("  Starting SphinxOS Node with Conscious Oracle")
        logger.info("=" * 70 + "\n")
        
        # Initialize Oracle
        oracle_initialized = self.initialize_oracle()
        
        # Setup replication if enabled
        if oracle_initialized and self.enable_replication:
            self.setup_oracle_replication()
        
        # Add Oracle endpoints to the FastAPI app
        if oracle_initialized:
            self.add_oracle_endpoints(node_main.app)
        
        # Log startup completion
        logger.info("\n" + "=" * 70)
        logger.info("  ✓ SphinxOS Node with Oracle ready")
        logger.info("=" * 70)
        logger.info(f"  Node API: http://0.0.0.0:{os.getenv('NODE_PORT', '8000')}")
        logger.info(f"  Metrics: http://0.0.0.0:{os.getenv('METRICS_PORT', '8001')}/metrics")
        if oracle_initialized:
            logger.info(f"  Oracle Status: http://0.0.0.0:{os.getenv('NODE_PORT', '8000')}/oracle/status")
            logger.info(f"  Oracle Consult: http://0.0.0.0:{os.getenv('NODE_PORT', '8000')}/oracle/consult")
        logger.info("=" * 70 + "\n")
        
        # Import and run uvicorn to start the server
        import uvicorn
        
        port = int(os.getenv("NODE_PORT", "8000"))
        uvicorn.run(node_main.app, host="0.0.0.0", port=port)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SphinxOS Node with Conscious Oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--oracle-threshold",
        type=float,
        default=float(os.getenv("ORACLE_THRESHOLD", "0.5")),
        help="Oracle consciousness threshold (Φ) for conscious decisions (default: 0.5)"
    )
    
    parser.add_argument(
        "--enable-replication",
        action="store_true",
        default=os.getenv("ENABLE_ORACLE_REPLICATION", "false").lower() == "true",
        help="Enable Oracle replication to MoltBot and ClawBot"
    )
    
    parser.add_argument(
        "--moltbot-endpoint",
        type=str,
        default=os.getenv("MOLTBOT_ENDPOINT", "molt://localhost:8080"),
        help="MoltBot deployment endpoint (default: molt://localhost:8080)"
    )
    
    parser.add_argument(
        "--clawbot-endpoint",
        type=str,
        default=os.getenv("CLAWBOT_ENDPOINT", "claw://localhost:8081"),
        help="ClawBot deployment endpoint (default: claw://localhost:8081)"
    )
    
    args = parser.parse_args()
    
    # Create and run the enhanced node
    node = SphinxNodeWithOracle(
        oracle_threshold=args.oracle_threshold,
        enable_replication=args.enable_replication,
        moltbot_endpoint=args.moltbot_endpoint,
        clawbot_endpoint=args.clawbot_endpoint
    )
    
    node.run()


if __name__ == "__main__":
    main()
