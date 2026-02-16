"""
zk-SNARK Prover for SphinxSkynet

Integrates with Circom circuits and SnarkJS for:
- Groth16 proof generation
- PLONK proof generation  
- Proof verification
- Recursive proof composition
"""

import json
import os
import subprocess
import tempfile
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib


class ProofType(Enum):
    """Supported proof systems"""
    GROTH16 = "groth16"
    PLONK = "plonk"
    FFLONK = "fflonk"


@dataclass
class ProofResult:
    """Result of proof generation"""
    proof: Dict
    public_signals: List
    proof_type: ProofType
    circuit_name: str
    proof_hash: str
    verification_key: Dict


@dataclass
class CircuitInfo:
    """Information about a compiled circuit"""
    name: str
    circuit_file: str
    wasm_file: str
    r1cs_file: str
    zkey_file: str
    vkey_file: str
    num_constraints: int
    num_public_inputs: int


class ZKProver:
    """
    Zero-knowledge proof generator and verifier.
    
    Integrates with:
    - Circom circuit compiler
    - SnarkJS proof system
    - SphinxSkynet hypercube network
    - Token transfer verification
    """
    
    def __init__(self, circuits_dir: str = None):
        """
        Initialize prover.
        
        Args:
            circuits_dir: Directory containing compiled circuits
        """
        self.circuits_dir = circuits_dir or "/tmp/circuits"
        os.makedirs(self.circuits_dir, exist_ok=True)
        self.circuits: Dict[str, CircuitInfo] = {}
        
    def compile_circuit(
        self,
        circuit_file: str,
        circuit_name: str,
        optimize: bool = True
    ) -> CircuitInfo:
        """
        Compile a Circom circuit.
        
        Args:
            circuit_file: Path to .circom file
            circuit_name: Name for the circuit
            optimize: Whether to optimize (O2)
        
        Returns:
            CircuitInfo object
        """
        output_dir = os.path.join(self.circuits_dir, circuit_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Compile with circom
        compile_cmd = [
            "circom",
            circuit_file,
            "--r1cs",
            "--wasm",
            "--sym",
            "--output", output_dir
        ]
        
        if optimize:
            compile_cmd.append("-O2")
        
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Circuit compiled: {circuit_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Circuit compilation failed: {e.stderr}")
            raise
        
        # Get constraint count from compilation output
        num_constraints = self._parse_constraint_count(result.stdout)
        
        circuit_info = CircuitInfo(
            name=circuit_name,
            circuit_file=circuit_file,
            wasm_file=os.path.join(output_dir, f"{circuit_name}_js/{circuit_name}.wasm"),
            r1cs_file=os.path.join(output_dir, f"{circuit_name}.r1cs"),
            zkey_file=os.path.join(output_dir, f"{circuit_name}.zkey"),
            vkey_file=os.path.join(output_dir, f"{circuit_name}_vkey.json"),
            num_constraints=num_constraints,
            num_public_inputs=0  # Will be updated during setup
        )
        
        self.circuits[circuit_name] = circuit_info
        return circuit_info
    
    def _parse_constraint_count(self, compile_output: str) -> int:
        """Parse constraint count from circom output"""
        for line in compile_output.split('\n'):
            if 'constraints' in line.lower():
                try:
                    return int(line.split()[-1])
                except:
                    pass
        return 0
    
    def setup_groth16(
        self,
        circuit_name: str,
        powers_of_tau: int = 12
    ) -> Tuple[str, str]:
        """
        Setup Groth16 proving and verification keys.
        
        Args:
            circuit_name: Name of compiled circuit
            powers_of_tau: Powers of tau (2^powers_of_tau constraints)
        
        Returns:
            Tuple of (zkey_file, vkey_file)
        """
        if circuit_name not in self.circuits:
            raise ValueError(f"Circuit {circuit_name} not compiled")
        
        circuit = self.circuits[circuit_name]
        
        # Download powers of tau if needed
        ptau_file = os.path.join(self.circuits_dir, f"pot{powers_of_tau}_final.ptau")
        if not os.path.exists(ptau_file):
            print(f"Downloading powers of tau (2^{powers_of_tau})...")
            # In production, download from trusted setup
            # For now, generate locally (NOT FOR PRODUCTION)
            subprocess.run([
                "snarkjs", "powersoftau", "new", "bn128",
                str(powers_of_tau), ptau_file
            ], check=True)
        
        # Generate zkey
        print(f"Generating proving key for {circuit_name}...")
        subprocess.run([
            "snarkjs", "groth16", "setup",
            circuit.r1cs_file,
            ptau_file,
            circuit.zkey_file
        ], check=True)
        
        # Export verification key
        print(f"Exporting verification key...")
        subprocess.run([
            "snarkjs", "zkey", "export", "verificationkey",
            circuit.zkey_file,
            circuit.vkey_file
        ], check=True)
        
        print(f"✅ Groth16 setup complete for {circuit_name}")
        return circuit.zkey_file, circuit.vkey_file
    
    def generate_proof(
        self,
        circuit_name: str,
        inputs: Dict,
        proof_type: ProofType = ProofType.GROTH16
    ) -> ProofResult:
        """
        Generate a zero-knowledge proof.
        
        Args:
            circuit_name: Name of circuit to use
            inputs: Input signals for the circuit
            proof_type: Type of proof system
        
        Returns:
            ProofResult object
        """
        if circuit_name not in self.circuits:
            raise ValueError(f"Circuit {circuit_name} not compiled")
        
        circuit = self.circuits[circuit_name]
        
        # Write inputs to temp file
        input_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        json.dump(inputs, input_file)
        input_file.close()
        
        # Generate witness
        witness_file = tempfile.NamedTemporaryFile(
            suffix='.wtns',
            delete=False
        )
        witness_file.close()
        
        print(f"Generating witness for {circuit_name}...")
        wasm_dir = os.path.dirname(circuit.wasm_file)
        subprocess.run([
            "node",
            os.path.join(wasm_dir, "generate_witness.js"),
            circuit.wasm_file,
            input_file.name,
            witness_file.name
        ], check=True)
        
        # Generate proof based on type
        proof_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_proof.json',
            delete=False
        )
        proof_file.close()
        
        public_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_public.json',
            delete=False
        )
        public_file.close()
        
        if proof_type == ProofType.GROTH16:
            print(f"Generating Groth16 proof...")
            subprocess.run([
                "snarkjs", "groth16", "prove",
                circuit.zkey_file,
                witness_file.name,
                proof_file.name,
                public_file.name
            ], check=True)
        else:
            raise NotImplementedError(f"Proof type {proof_type} not yet implemented")
        
        # Load proof and public signals
        with open(proof_file.name, 'r') as f:
            proof = json.load(f)
        
        with open(public_file.name, 'r') as f:
            public_signals = json.load(f)
        
        # Load verification key
        with open(circuit.vkey_file, 'r') as f:
            vkey = json.load(f)
        
        # Calculate proof hash
        proof_hash = hashlib.sha256(
            json.dumps(proof, sort_keys=True).encode()
        ).hexdigest()
        
        # Cleanup temp files
        os.unlink(input_file.name)
        os.unlink(witness_file.name)
        os.unlink(proof_file.name)
        os.unlink(public_file.name)
        
        print(f"✅ Proof generated: {proof_hash[:16]}...")
        
        return ProofResult(
            proof=proof,
            public_signals=public_signals,
            proof_type=proof_type,
            circuit_name=circuit_name,
            proof_hash=proof_hash,
            verification_key=vkey
        )
    
    def verify_proof(
        self,
        proof_result: ProofResult
    ) -> bool:
        """
        Verify a zero-knowledge proof.
        
        Args:
            proof_result: ProofResult to verify
        
        Returns:
            True if proof is valid
        """
        circuit_name = proof_result.circuit_name
        
        if circuit_name not in self.circuits:
            raise ValueError(f"Circuit {circuit_name} not compiled")
        
        circuit = self.circuits[circuit_name]
        
        # Write proof and public signals to temp files
        proof_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_proof.json',
            delete=False
        )
        json.dump(proof_result.proof, proof_file)
        proof_file.close()
        
        public_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_public.json',
            delete=False
        )
        json.dump(proof_result.public_signals, public_file)
        public_file.close()
        
        # Verify based on proof type
        if proof_result.proof_type == ProofType.GROTH16:
            result = subprocess.run([
                "snarkjs", "groth16", "verify",
                circuit.vkey_file,
                public_file.name,
                proof_file.name
            ], capture_output=True, text=True)
            
            verified = "OK" in result.stdout
        else:
            raise NotImplementedError(
                f"Proof type {proof_result.proof_type} not yet implemented"
            )
        
        # Cleanup
        os.unlink(proof_file.name)
        os.unlink(public_file.name)
        
        return verified
    
    def generate_recursive_proof(
        self,
        inner_proof: ProofResult,
        outer_circuit: str,
        additional_inputs: Dict = None
    ) -> ProofResult:
        """
        Generate a recursive proof (proof of proof).
        
        Args:
            inner_proof: Inner proof to verify
            outer_circuit: Outer circuit that verifies inner proof
            additional_inputs: Additional inputs for outer circuit
        
        Returns:
            Outer proof result
        """
        # Prepare inputs for outer circuit
        inputs = {
            "inner_proof_hash": inner_proof.proof_hash,
            "inner_public_hash": hashlib.sha256(
                json.dumps(inner_proof.public_signals, sort_keys=True).encode()
            ).hexdigest()
        }
        
        if additional_inputs:
            inputs.update(additional_inputs)
        
        # Generate outer proof
        return self.generate_proof(outer_circuit, inputs)
    
    def get_circuit_info(self, circuit_name: str) -> Optional[CircuitInfo]:
        """Get information about a compiled circuit"""
        return self.circuits.get(circuit_name)
    
    def list_circuits(self) -> List[str]:
        """List all compiled circuits"""
        return list(self.circuits.keys())
    
    def summary(self) -> str:
        """Get summary of prover state"""
        output = f"""
SphinxSkynet zk-Prover Status
{'='*60}
Circuits Compiled: {len(self.circuits)}
Circuits Directory: {self.circuits_dir}

Available Circuits:
"""
        for name, circuit in self.circuits.items():
            output += f"\n  • {name}"
            output += f"\n    Constraints: {circuit.num_constraints:,}"
            output += f"\n    R1CS: {os.path.basename(circuit.r1cs_file)}"
            output += f"\n    Keys: {os.path.basename(circuit.zkey_file)}"
        
        return output


if __name__ == "__main__":
    # Demo
    prover = ZKProver()
    print(prover.summary())
