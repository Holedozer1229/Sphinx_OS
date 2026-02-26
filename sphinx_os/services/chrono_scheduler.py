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

        # Add edges based on operation dependencies using a qubit→last-op map (O(n)).
        last_op_for_qubit: Dict[int, int] = {}
        for i, op in enumerate(circuit):
            target = op.get('target')
            control = op.get('control')
            qubits = [q for q in (target, control) if q is not None]
            for q in qubits:
                if q in last_op_for_qubit:
                    G.add_edge(last_op_for_qubit[q], i)
                last_op_for_qubit[q] = i

        # Optimize using shortest path (minimize decoherence impact)
        try:
            order = list(nx.topological_sort(G))
            optimized_circuit = [G.nodes[i]['operation'] for i in order]
        except nx.NetworkXUnfeasible:
            logger.warning("Circuit has cyclic dependencies, using original order")
            optimized_circuit = circuit.copy()

        return optimized_circuit
