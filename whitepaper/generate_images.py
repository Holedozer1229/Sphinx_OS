#!/usr/bin/env python3
"""
Generate images for the NPTC whitepaper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Set style for scientific plots
plt.style.use('seaborn-v0_8-darkgrid')

def generate_icosahedron():
    """Generate icosahedron diagram"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Add center vertex
    center = np.array([[0, 0, 0]])
    all_vertices = np.vstack([vertices, center])
    
    # Plot vertices
    ax.scatter(all_vertices[:, 0], all_vertices[:, 1], all_vertices[:, 2], 
               c='gold', s=100, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Draw edges (icosahedron edges)
    edges = [
        (0, 1), (0, 4), (0, 5), (0, 8), (0, 9),
        (1, 6), (1, 7), (1, 8), (1, 9),
        (2, 3), (2, 4), (2, 5), (2, 10), (2, 11),
        (3, 6), (3, 7), (3, 10), (3, 11),
        (4, 5), (4, 8), (4, 10),
        (5, 9), (5, 11),
        (6, 7), (6, 8), (6, 10),
        (7, 9), (7, 11),
        (8, 10), (9, 11)
    ]
    
    for edge in edges:
        points = vertices[[edge[0], edge[1]]]
        ax.plot3D(*points.T, 'b-', alpha=0.3, linewidth=1)
    
    # Draw connections to center
    for i in range(12):
        points = np.vstack([vertices[i], center[0]])
        ax.plot3D(*points.T, 'r--', alpha=0.2, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Au$_{13}$ Icosahedral Cluster\n(12 surface + 1 central vertex)', fontsize=14, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Sphinx_OS/Sphinx_OS/whitepaper/images/icosahedron.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated icosahedron.png")

def generate_fano_plane():
    """Generate Fano plane diagram"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Seven points on a circle
    angles = np.linspace(0, 2*np.pi, 8)[:-1]
    points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Plot points
    for i, (x, y) in enumerate(points):
        circle = plt.Circle((x, y), 0.08, color='red', zorder=5)
        ax.add_patch(circle)
        ax.text(x*1.15, y*1.15, f'e$_{i+1}$', fontsize=16, ha='center', va='center', fontweight='bold')
    
    # Seven lines of the Fano plane
    lines = [
        [0, 1, 2],  # Line 1
        [0, 3, 4],  # Line 2
        [0, 5, 6],  # Line 3
        [1, 3, 6],  # Line 4
        [1, 4, 5],  # Line 5
        [2, 3, 5],  # Line 6
        [2, 4, 6],  # Line 7
    ]
    
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']
    
    for line_idx, line_points in enumerate(lines):
        if line_idx == 0:  # Circle for the inscribed line
            circle = plt.Circle((0, 0), 1.0, fill=False, color=colors[line_idx], linewidth=2, linestyle='--')
            ax.add_patch(circle)
        else:
            for i in range(len(line_points)):
                p1 = points[line_points[i]]
                p2 = points[line_points[(i+1) % len(line_points)]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[line_idx], linewidth=2, alpha=0.6)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Fano Plane: Seven Imaginary Octonions\n7 points, 7 lines, 3 points per line', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Sphinx_OS/Sphinx_OS/whitepaper/images/fano_plane.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fano_plane.png")

def generate_fibonacci_timing():
    """Generate Fibonacci timing diagram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Generate Fibonacci sequence
    fib = [1, 1]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    
    cumulative = np.cumsum(fib)
    
    # Plot Fibonacci sequence
    ax1.stem(range(len(fib)), fib, basefmt=' ')
    ax1.set_xlabel('Index n', fontsize=12)
    ax1.set_ylabel('F$_n$', fontsize=12)
    ax1.set_title('Fibonacci Sequence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot control timing
    ax2.plot(cumulative[:10], np.ones(10), 'ro', markersize=10)
    ax2.vlines(cumulative[:10], 0, 1, colors='blue', alpha=0.5, linewidth=2)
    
    # Add periodic comparison
    periodic = np.arange(0, cumulative[9], cumulative[9]/10)
    ax2.plot(periodic, np.ones(10)*0.5, 'gx', markersize=10, label='Periodic (for comparison)')
    
    ax2.set_xlabel('Time τ (arbitrary units)', fontsize=12)
    ax2.set_ylabel('Control Events', fontsize=12)
    ax2.set_title('Non-Periodic Control Timing (Fibonacci) vs Periodic', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Sphinx_OS/Sphinx_OS/whitepaper/images/fibonacci_timing.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fibonacci_timing.png")

def generate_spectral_gap():
    """Generate spectral gap visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Eigenvalues of L_13
    eigenvalues = [0, 1.08333, 1.67909, 1.67909, 1.67909, 3.54743, 4.26108, 5.10247]
    
    ax.stem(range(len(eigenvalues)), eigenvalues, basefmt=' ')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Highlight spectral gap
    ax.annotate('Spectral Gap\nγ₁₃ = 1.08333', 
                xy=(1, 1.08333), xytext=(2, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=12, fontweight='bold', color='red')
    
    # Highlight degeneracy
    ax.plot([2, 3, 4], [1.67909, 1.67909, 1.67909], 'ro', markersize=8)
    ax.annotate('3-fold degeneracy\nλ₂,₃,₄ = 1.67909', 
                xy=(3, 1.67909), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=12, fontweight='bold', color='blue')
    
    ax.set_xlabel('Eigenvalue Index', fontsize=12)
    ax.set_ylabel('λ (Eigenvalue)', fontsize=12)
    ax.set_title('Discrete Laplacian Spectrum of Icosahedral L₁₃', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Sphinx_OS/Sphinx_OS/whitepaper/images/spectral_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated spectral_gap.png")

def generate_xi_invariant():
    """Generate Xi invariant diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a conceptual diagram showing the three components
    components = ['ℏω^eff', 'kT_eff', 'C_geom']
    colors_comp = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # Central circle for Xi
    circle = plt.Circle((0.5, 0.5), 0.15, color='gold', alpha=0.8, zorder=10)
    ax.add_patch(circle)
    ax.text(0.5, 0.5, 'Ξ ≈ 1', fontsize=20, ha='center', va='center', fontweight='bold')
    
    # Three components around it
    positions = [(0.5, 0.85), (0.2, 0.25), (0.8, 0.25)]
    
    for i, (comp, pos, col) in enumerate(zip(components, positions, colors_comp)):
        rect = patches.FancyBboxPatch((pos[0]-0.1, pos[1]-0.05), 0.2, 0.1, 
                                       boxstyle="round,pad=0.01", 
                                       facecolor=col, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], comp, fontsize=12, ha='center', va='center', fontweight='bold')
        
        # Arrow to center
        ax.annotate('', xy=(0.5, 0.5), xytext=pos,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('NPTC Invariant Ξ: Unification of Three Scales\n' + 
                 'Quantum Energy / Thermal Energy × Geometric Complexity',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Sphinx_OS/Sphinx_OS/whitepaper/images/xi_invariant.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated xi_invariant.png")

def generate_cross_chain():
    """Generate cross-chain network diagram"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Seven chains in Fano plane configuration
    chains = ['Ethereum', 'Arbitrum', 'Optimism', 'zkSync', 'Scroll', 'Polygon zkEVM', 'Linea']
    
    # Positions on circle
    angles = np.linspace(0, 2*np.pi, 8)[:-1]
    positions = {i: (np.cos(a)*2, np.sin(a)*2) for i, a in enumerate(angles)}
    
    # Draw nodes
    for i, (pos, chain) in enumerate(zip(positions.values(), chains)):
        circle = plt.Circle(pos, 0.4, color='lightblue', alpha=0.8, zorder=5, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], chain, fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Draw Fano plane edges
    lines = [
        [0, 1, 2], [0, 3, 4], [0, 5, 6],
        [1, 3, 6], [1, 4, 5], [2, 3, 5], [2, 4, 6]
    ]
    
    for line in lines:
        for i in range(len(line)):
            p1 = positions[line[i]]
            p2 = positions[line[(i+1) % len(line)]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1.5, alpha=0.3, zorder=1)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Cross-Chain Verification Network\nFano Topology (N=7, λ₁=1)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Sphinx_OS/Sphinx_OS/whitepaper/images/cross_chain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated cross_chain.png")

def main():
    """Generate all images"""
    print("Generating whitepaper images...")
    generate_icosahedron()
    generate_fano_plane()
    generate_fibonacci_timing()
    generate_spectral_gap()
    generate_xi_invariant()
    generate_cross_chain()
    print("All images generated successfully!")

if __name__ == "__main__":
    main()
