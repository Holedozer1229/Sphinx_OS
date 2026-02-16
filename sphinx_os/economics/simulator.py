"""
Economic Simulator for SphinxOS

Models treasury revenue, user growth, and flywheel effects across different scenarios.
"""

import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random

from .yield_calculator import YieldCalculator


@dataclass
class SimulationScenario:
    """Scenario parameters for economic simulation"""
    name: str
    num_users: int
    avg_stx_per_user: float
    phi_mean: float
    phi_stddev: float
    btc_price_usd: float
    cycles_per_year: int = 26  # ~2 week cycles
    nft_holder_percentage: float = 0.15
    pool_efficiency: float = 0.95


@dataclass
class SimulationResult:
    """Results from economic simulation"""
    scenario: SimulationScenario
    annual_treasury_btc: float
    annual_treasury_usd: float
    annual_user_yield_btc: float
    annual_user_yield_usd: float
    avg_user_annual_btc: float
    avg_user_annual_usd: float
    treasury_percentage: float
    roi_percentage: float


class EconomicSimulator:
    """
    Simulate economic performance across different scenarios.
    
    Models:
    - User growth and retention
    - BTC price dynamics
    - Treasury revenue accumulation
    - Flywheel effects
    """
    
    def __init__(self):
        """Initialize economic simulator"""
        self.calculator = YieldCalculator()
    
    def generate_user_population(
        self,
        num_users: int,
        avg_stx: float,
        phi_mean: float,
        phi_stddev: float,
        nft_percentage: float
    ) -> Tuple[Dict[str, float], Dict[str, float], set]:
        """
        Generate synthetic user population.
        
        Args:
            num_users: Number of users to generate
            avg_stx: Average STX delegation per user
            phi_mean: Mean spectral integration score
            phi_stddev: Standard deviation of Φ scores
            nft_percentage: Percentage of users with NFTs
            
        Returns:
            Tuple of (delegations, phi_scores, nft_holders)
        """
        delegations = {}
        phi_scores = {}
        nft_holders = set()
        
        for i in range(num_users):
            user_id = f"user_{i}"
            
            # STX delegation (log-normal distribution)
            stx = random.lognormvariate(math.log(avg_stx), 0.5)
            delegations[user_id] = max(100, stx)  # Min 100 STX
            
            # Φ score (normal distribution, clipped)
            phi = random.normalvariate(phi_mean, phi_stddev)
            phi_scores[user_id] = max(200, min(1000, phi))  # Range [200, 1000]
            
            # NFT holder (random selection)
            if random.random() < nft_percentage:
                nft_holders.add(user_id)
        
        return delegations, phi_scores, nft_holders
    
    def simulate_cycle(
        self,
        delegations: Dict[str, float],
        phi_scores: Dict[str, float],
        nft_holders: set,
        btc_reward_per_cycle: float
    ) -> Tuple[float, float]:
        """
        Simulate a single PoX cycle.
        
        Args:
            delegations: User STX delegations
            phi_scores: User Φ scores
            nft_holders: Set of NFT holders
            btc_reward_per_cycle: Total BTC rewards for cycle
            
        Returns:
            Tuple of (treasury_btc, user_btc)
        """
        results = self.calculator.calculate_batch_yields(
            delegations, btc_reward_per_cycle, phi_scores, nft_holders
        )
        
        treasury_btc = self.calculator.get_treasury_total(results)
        user_btc = sum(r.effective_payout for r in results.values())
        
        return treasury_btc, user_btc
    
    def simulate_scenario(
        self,
        scenario: SimulationScenario,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run full simulation for a scenario.
        
        Args:
            scenario: Simulation parameters
            verbose: Print progress updates
            
        Returns:
            SimulationResult with annual projections
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Simulating: {scenario.name}")
            print(f"{'='*70}")
            print(f"Users: {scenario.num_users:,}")
            print(f"Avg STX: {scenario.avg_stx_per_user:,}")
            print(f"Φ mean: {scenario.phi_mean}")
            print(f"BTC price: ${scenario.btc_price_usd:,}")
        
        # Generate population
        delegations, phi_scores, nft_holders = self.generate_user_population(
            scenario.num_users,
            scenario.avg_stx_per_user,
            scenario.phi_mean,
            scenario.phi_stddev,
            scenario.nft_holder_percentage
        )
        
        total_stx = sum(delegations.values())
        
        # Estimate BTC reward per cycle based on total STX
        # Rough estimate: ~0.00001 BTC per 1000 STX per cycle
        btc_per_cycle = total_stx * 0.00001 / 1000
        
        # Simulate all cycles for the year
        annual_treasury_btc = 0.0
        annual_user_btc = 0.0
        
        for cycle in range(scenario.cycles_per_year):
            treasury_btc, user_btc = self.simulate_cycle(
                delegations, phi_scores, nft_holders, btc_per_cycle
            )
            annual_treasury_btc += treasury_btc
            annual_user_btc += user_btc
        
        # Convert to USD
        annual_treasury_usd = annual_treasury_btc * scenario.btc_price_usd
        annual_user_yield_usd = annual_user_btc * scenario.btc_price_usd
        
        # Calculate per-user averages
        avg_user_annual_btc = annual_user_btc / scenario.num_users
        avg_user_annual_usd = annual_user_yield_usd / scenario.num_users
        
        # Calculate metrics
        total_annual_btc = annual_treasury_btc + annual_user_btc
        treasury_percentage = (annual_treasury_btc / total_annual_btc * 100) if total_annual_btc > 0 else 0
        
        # ROI calculation (simplified - assumes STX value stable)
        avg_investment_usd = scenario.avg_stx_per_user * 0.5  # Assume $0.50 per STX
        roi_percentage = (avg_user_annual_usd / avg_investment_usd * 100) if avg_investment_usd > 0 else 0
        
        result = SimulationResult(
            scenario=scenario,
            annual_treasury_btc=annual_treasury_btc,
            annual_treasury_usd=annual_treasury_usd,
            annual_user_yield_btc=annual_user_btc,
            annual_user_yield_usd=annual_user_yield_usd,
            avg_user_annual_btc=avg_user_annual_btc,
            avg_user_annual_usd=avg_user_annual_usd,
            treasury_percentage=treasury_percentage,
            roi_percentage=roi_percentage
        )
        
        if verbose:
            self.print_result(result)
        
        return result
    
    def print_result(self, result: SimulationResult):
        """Print formatted simulation result"""
        print(f"\n📊 ANNUAL PROJECTIONS")
        print(f"{'─'*70}")
        print(f"Treasury Revenue:   {result.annual_treasury_btc:.4f} BTC (${result.annual_treasury_usd:,.2f})")
        print(f"User Yield:         {result.annual_user_yield_btc:.4f} BTC (${result.annual_user_yield_usd:,.2f})")
        print(f"Avg User Yield:     {result.avg_user_annual_btc:.6f} BTC (${result.avg_user_annual_usd:,.2f})")
        print(f"Treasury Cut:       {result.treasury_percentage:.2f}%")
        print(f"User ROI:           {result.roi_percentage:.2f}%")
        print(f"{'─'*70}")
    
    def run_multiple_scenarios(self) -> List[SimulationResult]:
        """
        Run predefined scenarios covering conservative to aggressive growth.
        
        Returns:
            List of SimulationResults
        """
        scenarios = [
            SimulationScenario(
                name="Conservative (5K users, $45K BTC)",
                num_users=5000,
                avg_stx_per_user=10000,
                phi_mean=650,
                phi_stddev=100,
                btc_price_usd=45000
            ),
            SimulationScenario(
                name="Moderate (15K users, $55K BTC)",
                num_users=15000,
                avg_stx_per_user=15000,
                phi_mean=700,
                phi_stddev=120,
                btc_price_usd=55000
            ),
            SimulationScenario(
                name="Aggressive (50K users, $70K BTC)",
                num_users=50000,
                avg_stx_per_user=20000,
                phi_mean=750,
                phi_stddev=150,
                btc_price_usd=70000
            ),
            SimulationScenario(
                name="Maximum (100K users, $100K BTC)",
                num_users=100000,
                avg_stx_per_user=25000,
                phi_mean=800,
                phi_stddev=150,
                btc_price_usd=100000
            )
        ]
        
        results = []
        for scenario in scenarios:
            result = self.simulate_scenario(scenario, verbose=True)
            results.append(result)
        
        return results
    
    def model_flywheel_effect(
        self,
        initial_users: int,
        growth_rate: float,
        periods: int
    ) -> List[Dict]:
        """
        Model the flywheel effect over time.
        
        Higher Φ → Higher NFT value → More STX → More BTC → 
        Higher Treasury → More Development → Higher Φ
        
        Args:
            initial_users: Starting user count
            growth_rate: User growth rate per period (e.g., 0.1 = 10% growth)
            periods: Number of periods to simulate
            
        Returns:
            List of period stats
        """
        results = []
        users = initial_users
        phi_base = 650
        
        for period in range(periods):
            # Φ increases with network effects
            phi_boost = math.log(users / initial_users + 1) * 50
            current_phi = phi_base + phi_boost
            
            scenario = SimulationScenario(
                name=f"Period {period}",
                num_users=int(users),
                avg_stx_per_user=10000 * (1 + period * 0.05),  # STX increases
                phi_mean=current_phi,
                phi_stddev=100,
                btc_price_usd=45000 * (1 + period * 0.03)  # BTC price appreciation
            )
            
            result = self.simulate_scenario(scenario, verbose=False)
            
            results.append({
                "period": period,
                "users": int(users),
                "phi_mean": current_phi,
                "treasury_usd": result.annual_treasury_usd,
                "user_yield_usd": result.annual_user_yield_usd
            })
            
            # Apply growth
            users *= (1 + growth_rate)
        
        return results


# CLI interface
if __name__ == "__main__":
    print("=" * 70)
    print("SPHINXOS ECONOMIC SIMULATOR")
    print("=" * 70)
    
    simulator = EconomicSimulator()
    
    # Run predefined scenarios
    print("\n🎯 RUNNING SCENARIO ANALYSIS")
    results = simulator.run_multiple_scenarios()
    
    # Summary comparison
    print("\n\n📈 SCENARIO COMPARISON")
    print("=" * 70)
    print(f"{'Scenario':<40} {'Treasury $':<15} {'User Yield $':<15}")
    print("─" * 70)
    for result in results:
        print(f"{result.scenario.name:<40} ${result.annual_treasury_usd:>13,.0f} ${result.annual_user_yield_usd:>13,.0f}")
    print("=" * 70)
    
    # Flywheel effect
    print("\n\n🔄 FLYWHEEL EFFECT PROJECTION (5 years, 20% annual growth)")
    print("=" * 70)
    flywheel_results = simulator.model_flywheel_effect(
        initial_users=5000,
        growth_rate=0.20,
        periods=5
    )
    print(f"{'Year':<10} {'Users':<15} {'Φ Mean':<12} {'Treasury $':<18} {'User Yield $'}")
    print("─" * 70)
    for r in flywheel_results:
        print(f"{r['period']:<10} {r['users']:<15,} {r['phi_mean']:<12.1f} ${r['treasury_usd']:<16,.0f} ${r['user_yield_usd']:,.0f}")
    print("=" * 70)
    
    print("\n✅ Simulation complete!")
    print("\nKey Insight: The flywheel effect creates exponential growth as higher")
    print("Φ scores attract more users, increasing treasury funds for development,")
    print("which further improves the protocol, creating a self-reinforcing cycle.")
