# sphinx_os/services/chrono_scheduler.py
"""
ChronoScheduler: Spacetime-aware scheduler for quantum operations.
"""
import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger("SphinxOS.ChronoScheduler")

class ChronoScheduler:
    """Spacetime-aware scheduler for quantum operations."""
    
    def __init__(self):
        """Initialize the ChronoScheduler."""
        self.graph = nx.DiGraph()
        logger.info("ChronoScheduler initialized")

    def route(self, circuit: List[Dict[str, Any]], metric: np.ndarray, decoherence_map: np.ndarray, spin_state: np.ndarray) -> List[Dict[str, Any]]:
        """
        Route quantum circuit operations with spacetime awareness, prioritizing Rydberg gates.

        Args:
            circuit (List[Dict[str, Any]]): List of quantum operations.
            metric (np.ndarray): Spacetime metric tensor.
            decoherence_map (np.ndarray): Decoherence rates for qubits.
            spin_state (np.ndarray): Spin network state.

        Returns:
            List[Dict[str, Any]]: Optimized circuit.
        """
        logger.debug("Routing circuit with %d operations", len(circuit))
        G = self.graph
        G.clear()

        # Add nodes for each operation with weights based on decoherence
        for i, op in enumerate(circuit):
            qubit = op.get('target', op.get('control', 0))
            weight = decoherence_map[qubit] if qubit < len(decoherence_map) else 1.0
            # Prioritize Rydberg gates due to their enhanced entanglement effects
            if op.get('gate') == 'CZ' and op.get('type') == 'rydberg':
                weight *= 1.5  # Higher priority for Rydberg gates
            G.add_node(i, operation=op, weight=weight)

        # Add edges based on operation dependencies
        for i in range(len(circuit)):
            op = circuit[i]
            target = op.get('target')
            control = op.get('control')
            qubits = {target}
            if control is not None:
                qubits.add(control)
            # Look for dependent operations (same qubits)
            for j in range(i + 1, len(circuit)):
                next_op = circuit[j]
                next_target = next_op.get('target')
                next_control = next_op.get('control')
                next_qubits = {next_target}
                if next_control is not None:
                    next_qubits.add(next_control)
                if qubits & next_qubits:  # Overlapping qubits
                    G.add_edge(i, j)

        # Optimize using shortest path (minimize decoherence impact)
        try:
            order = list(nx.topological_sort(G))
            optimized_circuit = [G.nodes[i]['operation'] for i in order]
        except nx.NetworkXUnfeasible:
            logger.warning("Circuit has cyclic dependencies, using original order")
            optimized_circuit = circuit.copy()

        return optimized_circuit
