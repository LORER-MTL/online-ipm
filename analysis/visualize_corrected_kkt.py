"""
Visualization: Corrected KKT Bound Analysis

Shows how K varies across the feasible region and the impact on 
the constraint variation bound.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D


def visualize_k_across_feasible_region():
    """
    Show how K varies as we move across the feasible region.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Grid for feasible region: x1 + x2 = 5, x1, x2 > 0
    x1_vals = np.linspace(0.1, 4.9, 50)
    x2_vals = 5.0 - x1_vals
    
    mu = 1.0
    A = np.ones((1, 2))
    
    K_inv_norms = []
    conditions = []
    
    for x1, x2 in zip(x1_vals, x2_vals):
        x = np.array([x1, x2])
        
        # KKT matrix
        H = mu * np.diag(1.0 / (x**2))
        K_mat = np.zeros((3, 3))
        K_mat[:2, :2] = H
        K_mat[:2, 2] = A.T.ravel()
        K_mat[2, :2] = A.ravel()
        
        K_inv = np.linalg.inv(K_mat)
        K_inv_norm = np.linalg.norm(K_inv, ord=2)
        cond = np.linalg.cond(K_mat)
        
        K_inv_norms.append(K_inv_norm)
        conditions.append(cond)
    
    # Plot 1: K variation along feasible line
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(x1_vals, K_inv_norms, 'b-', linewidth=2.5)
    ax1.axhline(y=np.max(K_inv_norms), color='r', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Max K = {np.max(K_inv_norms):.2f}')
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$K = ||K^{-1}||_2$', fontsize=12)
    ax1.set_title('K Varies Across Feasible Region\n($x_1 + x_2 = 5$)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Condition number
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x1_vals, conditions, 'g-', linewidth=2.5)
    ax2.set_xlabel('$x_1$', fontsize=12)
    ax2.set_ylabel('Condition Number', fontsize=12)
    ax2.set_title('Conditioning of KKT Matrix', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Bound variation
    ax3 = fig.add_subplot(2, 3, 3)
    alpha = 0.1  # Stability constant
    bounds = alpha / np.array(K_inv_norms)
    ax3.plot(x1_vals, bounds, 'purple', linewidth=2.5)
    ax3.axhline(y=np.min(bounds), color='r', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Min bound = {np.min(bounds):.4f}')
    ax3.set_xlabel('$x_1$', fontsize=12)
    ax3.set_ylabel(r'Bound $\alpha/K$', fontsize=12)
    ax3.set_title('Constraint Change Bound\n(Tightest at center)', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Counterexample - constraint sequence
    ax4 = fig.add_subplot(2, 3, 4)
    T = 100
    t_vals = np.arange(T)
    b_sequence = 5.0 + 0.5 * np.sin(2 * np.pi * t_vals / 20)
    
    ax4.plot(t_vals, b_sequence, 'b-', linewidth=2)
    ax4.set_xlabel('Time $t$', fontsize=12)
    ax4.set_ylabel('$b_t$', fontsize=12)
    ax4.set_title('Simple Sinusoidal Constraint Sequence', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Violations
    ax5 = fig.add_subplot(2, 3, 5)
    changes = np.array([abs(b_sequence[t] - b_sequence[t-1]) for t in range(1, T)])
    bound = np.min(bounds)  # Most restrictive bound
    
    t_changes = np.arange(1, T)
    ax5.plot(t_changes, changes, 'r-', linewidth=2, label='Actual $||\\Delta b||$')
    ax5.axhline(y=bound, color='g', linestyle='--', linewidth=2, 
                label=f'Bound = {bound:.4f}')
    ax5.fill_between(t_changes, 0, bound, color='green', alpha=0.15)
    ax5.fill_between(t_changes, bound, changes, where=(changes > bound),
                     color='red', alpha=0.2, label='Violation region')
    
    ax5.set_xlabel('Time $t$', fontsize=12)
    ax5.set_ylabel(r'$||b_t - b_{t-1}||$', fontsize=12)
    ax5.set_title('Constraint Changes vs Bound\n(100% violation)', 
                  fontsize=13, fontweight='bold')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=9)
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    violations = changes > bound
    summary_text = f"""
    CORRECTED ANALYSIS SUMMARY
    ═══════════════════════════════════
    
    Problem Conditioning:
    • K_max = {np.max(K_inv_norms):.2f}
    • K_min = {np.min(K_inv_norms):.2f}
    • K at center = {K_inv_norms[len(K_inv_norms)//2]:.2f}
    
    Constraint Bound:
    • Most restrictive: {bound:.5f}
    • Least restrictive: {np.max(bounds):.5f}
    
    Counterexample Results:
    • Max change: {np.max(changes):.5f}
    • Avg change: {np.mean(changes):.5f}
    • Violations: {np.sum(violations)}/{len(changes)} steps
    • Violation rate: {100*np.mean(violations):.0f}%
    
    Violation Factors:
    • Maximum: {np.max(changes)/bound:.1f}×
    • Average: {np.mean(changes)/bound:.1f}×
    
    CONCLUSION:
    Even with corrected K bound,
    simple dynamics still violate by 6-10×!
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/willyzz/Documents/online-ipm/analysis/corrected_kkt_analysis.png', 
                dpi=150, bbox_inches='tight')
    print("📊 Corrected analysis saved to: analysis/corrected_kkt_analysis.png")
    
    return fig


def compare_bounds():
    """
    Compare different K values and their impact on bounds.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    T = 100
    t_vals = np.arange(T)
    b_sequence = 5.0 + 0.5 * np.sin(2 * np.pi * t_vals / 20)
    changes = np.array([abs(b_sequence[t] - b_sequence[t-1]) for t in range(1, T)])
    t_changes = np.arange(1, T)
    
    scenarios = [
        ("Perfect Conditioning (K=1)", 1.0),
        ("Well-Conditioned (K=6.25)", 6.25),
        ("Moderate (K=10)", 10.0),
        ("Ill-Conditioned (K=100)", 100.0),
    ]
    
    for idx, (title, K) in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        
        bound = 0.1 / K
        violations = changes > bound
        
        ax.plot(t_changes, changes, 'r-', linewidth=1.5, label='Actual changes')
        ax.axhline(y=bound, color='g', linestyle='--', linewidth=2, 
                   label=f'Bound = {bound:.5f}')
        ax.fill_between(t_changes, 0, bound, color='green', alpha=0.15)
        ax.fill_between(t_changes, bound, changes, where=(changes > bound),
                        color='red', alpha=0.2)
        
        ax.set_xlabel('Time $t$', fontsize=11)
        ax.set_ylabel(r'$||b_t - b_{t-1}||$', fontsize=11)
        ax.set_title(f'{title}\nViolations: {100*np.mean(violations):.0f}%, ' +
                     f'Factor: {np.mean(changes)/bound:.1f}×',
                     fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/willyzz/Documents/online-ipm/analysis/k_sensitivity_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("📊 Sensitivity comparison saved to: analysis/k_sensitivity_comparison.png")
    
    return fig


def main():
    print("=" * 80)
    print("CREATING VISUALIZATIONS FOR CORRECTED KKT ANALYSIS")
    print("=" * 80)
    print()
    
    visualize_k_across_feasible_region()
    compare_bounds()
    
    print()
    print("✓ All visualizations created successfully!")
    print()


if __name__ == "__main__":
    main()
