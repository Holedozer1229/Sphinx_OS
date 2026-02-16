"""
Circuit Builder for zk-EVM

Programmatic circuit construction for complex operations.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Signal:
    """Circuit signal"""
    name: str
    signal_type: str  # "input" or "output"
    size: Optional[int] = None  # For arrays
    

@dataclass
class Component:
    """Circuit component"""
    name: str
    template: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)


class CircuitBuilder:
    """
    Programmatic circuit builder for Circom.
    
    Allows building complex circuits through a Python API.
    """
    
    def __init__(self, circuit_name: str):
        self.circuit_name = circuit_name
        self.signals: List[Signal] = []
        self.components: List[Component] = []
        self.constraints: List[str] = []
        self.includes: List[str] = []
    
    def add_include(self, path: str):
        """Add include directive"""
        self.includes.append(path)
    
    def add_input(self, name: str, size: Optional[int] = None):
        """Add input signal"""
        self.signals.append(Signal(name, "input", size))
    
    def add_output(self, name: str, size: Optional[int] = None):
        """Add output signal"""
        self.signals.append(Signal(name, "output", size))
    
    def add_component(
        self,
        name: str,
        template: str,
        inputs: Dict[str, str] = None,
        outputs: List[str] = None
    ):
        """Add component"""
        self.components.append(Component(
            name=name,
            template=template,
            inputs=inputs or {},
            outputs=outputs or []
        ))
    
    def add_constraint(self, constraint: str):
        """Add constraint"""
        self.constraints.append(constraint)
    
    def build(self) -> str:
        """Build complete Circom circuit"""
        
        circuit = "pragma circom 2.0.0;\n\n"
        
        # Includes
        for include in self.includes:
            circuit += f'include "{include}";\n'
        
        if self.includes:
            circuit += "\n"
        
        # Template definition
        circuit += f"template {self.circuit_name}() {{\n"
        
        # Signals
        for signal in self.signals:
            if signal.size:
                circuit += f"    signal {signal.signal_type} {signal.name}[{signal.size}];\n"
            else:
                circuit += f"    signal {signal.signal_type} {signal.name};\n"
        
        if self.signals:
            circuit += "\n"
        
        # Components
        for component in self.components:
            circuit += f"    component {component.name} = {component.template};\n"
            
            for input_name, input_value in component.inputs.items():
                circuit += f"    {component.name}.{input_name} <== {input_value};\n"
        
        if self.components:
            circuit += "\n"
        
        # Constraints
        for constraint in self.constraints:
            circuit += f"    {constraint};\n"
        
        circuit += "}\n\n"
        
        # Main component
        circuit += f"component main = {self.circuit_name}();\n"
        
        return circuit
    
    def save(self, filepath: str):
        """Save circuit to file"""
        with open(filepath, 'w') as f:
            f.write(self.build())
    
    @staticmethod
    def create_balance_checker() -> 'CircuitBuilder':
        """Create a balance checker circuit"""
        builder = CircuitBuilder("BalanceChecker")
        builder.add_include("circomlib/circuits/comparators.circom")
        
        builder.add_input("balance")
        builder.add_input("amount")
        builder.add_output("sufficient")
        
        builder.add_component(
            "gte",
            "GreaterEqThan(64)",
            inputs={"in[0]": "balance", "in[1]": "amount"}
        )
        
        builder.add_constraint("sufficient <== gte.out")
        builder.add_constraint("sufficient === 1")
        
        return builder
    
    @staticmethod
    def create_yield_calculator() -> 'CircuitBuilder':
        """Create a yield calculator circuit"""
        builder = CircuitBuilder("YieldCalculator")
        
        builder.add_input("amount")
        builder.add_input("phi_score")
        builder.add_input("base_apr")
        builder.add_output("yield_amount")
        
        builder.add_constraint("signal phi_boost")
        builder.add_constraint("phi_boost <== 1000 + (phi_score - 500) / 2")
        builder.add_constraint("signal boosted_apr")
        builder.add_constraint("boosted_apr <== (base_apr * phi_boost) / 1000")
        builder.add_constraint("yield_amount <== (amount * boosted_apr) / 10000")
        
        return builder


if __name__ == "__main__":
    # Demo: Create balance checker
    checker = CircuitBuilder.create_balance_checker()
    print("Balance Checker Circuit:")
    print("="*50)
    print(checker.build())
    
    # Demo: Create yield calculator
    calc = CircuitBuilder.create_yield_calculator()
    print("\nYield Calculator Circuit:")
    print("="*50)
    print(calc.build())
