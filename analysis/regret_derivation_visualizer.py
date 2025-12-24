"""
Visual diagram showing the step-by-step regret derivation flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_regret_derivation_diagram():
    """Create a flowchart showing the regret bound derivation."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(5, 19, 'Regret Bound Derivation: Step-by-Step', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Helper function for boxes
    def add_box(x, y, width, height, text, color='lightblue', title=None):
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        
        if title:
            ax.text(x + width/2, y + height - 0.3, title,
                   fontsize=10, fontweight='bold', ha='center', va='top')
            ax.text(x + width/2, y + height/2 - 0.1, text,
                   fontsize=9, ha='center', va='center', wrap=True)
        else:
            ax.text(x + width/2, y + height/2, text,
                   fontsize=9, ha='center', va='center', wrap=True)
    
    # Helper function for arrows
    def add_arrow(x1, y1, x2, y2, label='', style='->'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=style, color='black', 
                               linewidth=2, mutation_scale=20)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
                   style='italic', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
    
    # Step 1: Original Problem
    add_box(0.5, 17, 4, 1.3, 
            'Original Problem:\nmin c^T x  s.t.  Ax = b_t, Fx ≤ g_t',
            'lightcyan', 'Step 1')
    
    add_box(5.5, 17, 4, 1.3,
            'Define Regret:\nR_d(T) = Σ[c^T x_t - c^T x*_t]',
            'lightcyan', 'Step 1')
    
    # Step 2: Slack Transformation
    add_arrow(2.5, 17, 2.5, 15.8)
    add_arrow(7.5, 17, 7.5, 15.8)
    
    add_box(0.5, 14.5, 9, 1.3,
            'Introduce slacks: s_i = g_t,i - (Fx)_i  ⟹  z = [x; s],  ̃c = [c; 0],  Ã = [A 0; F I]',
            'lightyellow', 'Step 2: Slack Transform')
    
    # Step 3: Cost Equivalence
    add_arrow(5, 14.5, 5, 13.3)
    
    add_box(1, 12, 8, 1.3,
            'KEY INSIGHT: ̃c^T z = c^T x + 0^T s = c^T x\n⟹ R_d^aug(T) = R_d(T)',
            'lightgreen', 'Step 3: Regret Equivalence')
    
    # Step 4: Apply Theorem 1
    add_arrow(5, 12, 5, 10.8)
    
    add_box(0.5, 9.5, 9, 1.3,
            'Apply Theorem 1 to augmented problem:\nR_d^aug(T) ≤ (11pβ)/(5η₀(β-1)) + ||c|| · V_T^aug',
            'lightcoral', 'Step 4: Theorem 1')
    
    # Step 5: Path Variation Decomposition
    add_arrow(5, 9.5, 5, 8.3)
    
    add_box(0.5, 7, 9, 1.3,
            'Decompose: ||z*_t - z*_{t-1}|| = ||[Δx*; Δs*]|| ≤ ||Δx*|| + ||Δs*||\n⟹ V_T^aug ≤ V_T^x + V_T^s',
            'lavender', 'Step 5: Triangle Inequality')
    
    # Step 6: Slack Coupling
    add_arrow(5, 7, 3.5, 5.8)
    add_arrow(5, 7, 6.5, 5.8)
    
    add_box(0.5, 4.5, 4.5, 1.3,
            'Slack relation:\ns*_t = g_t - Fx*_t\nΔs* = Δg - F·Δx*',
            'mistyrose', 'Step 6a: Slack Formula')
    
    add_box(5.5, 4.5, 4, 1.3,
            'Bound slack change:\n||Δs*|| ≤ ||Δg|| + ||F||·||Δx*||',
            'mistyrose', 'Step 6b: Triangle Ineq')
    
    # Step 7: Sum over time
    add_arrow(3.5, 4.5, 3.5, 3.3)
    add_arrow(7.5, 4.5, 7.5, 3.3)
    
    add_box(1, 2, 8, 1.3,
            'Sum over t=1..T:\nV_T^s ≤ V_g^ineq + ||F|| · V_T^x',
            'wheat', 'Step 7: Sum Slack Bound')
    
    # Step 8: Combine everything
    add_arrow(5, 2, 5, 0.8)
    
    add_box(0.5, -0.5, 9, 1.3,
            'Substitute:\nV_T^aug ≤ V_T^x + V_T^s ≤ V_T^x + V_g^ineq + ||F||·V_T^x = (1+||F||)·V_T^x + V_g^ineq',
            'lightgoldenrodyellow', 'Step 8: Combine')
    
    # Final Result
    add_arrow(5, -0.5, 5, -1.7)
    
    add_box(0.5, -3, 9, 1.3,
            'FINAL BOUND:\nR_d(T) ≤ (11pβ)/(5η₀(β-1)) + ||c||·[(1+||F||)·V_T^x + V_g^ineq]',
            'lightgreen', 'Final Result')
    
    # Add legend boxes on the right
    ax.text(10.5, 18, 'Legend:', fontsize=10, fontweight='bold')
    
    legend_items = [
        ('Problem Setup', 'lightcyan'),
        ('Transformation', 'lightyellow'),
        ('Key Insight', 'lightgreen'),
        ('Theorem Application', 'lightcoral'),
        ('Mathematical Step', 'lavender'),
        ('Coupling Analysis', 'mistyrose'),
    ]
    
    y_pos = 17
    for label, color in legend_items:
        box = FancyBboxPatch((10.2, y_pos), 1.5, 0.4,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color, linewidth=1)
        ax.add_patch(box)
        ax.text(11.8, y_pos + 0.2, label, fontsize=8, va='center')
        y_pos -= 0.6
    
    plt.tight_layout()
    return fig

def create_coupling_visualization():
    """Create a diagram showing how x and s are coupled."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Understanding the Slack Coupling: Why (1 + ||F||) Factor?',
            fontsize=16, fontweight='bold', ha='center')
    
    # Original space
    ax.text(2, 6.5, 'Original Space (x)', fontsize=12, fontweight='bold', ha='center')
    circle1 = plt.Circle((2, 5), 0.8, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle1)
    ax.text(2, 5, 'x*_{t-1}', fontsize=10, ha='center', fontweight='bold')
    
    circle2 = plt.Circle((3.5, 4), 0.8, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle2)
    ax.text(3.5, 4, 'x*_t', fontsize=10, ha='center', fontweight='bold')
    
    # Arrow showing movement
    arrow1 = FancyArrowPatch((2.6, 4.6), (3.1, 4.4),
                            arrowstyle='->', color='red', linewidth=3,
                            mutation_scale=20)
    ax.add_patch(arrow1)
    ax.text(2.85, 5.2, 'Δx*', fontsize=11, color='red', fontweight='bold')
    ax.text(2.85, 4.8, '||Δx*|| = V_T^x', fontsize=9, style='italic')
    
    # Transformation arrow
    arrow_transform = FancyArrowPatch((4.5, 5), (6.5, 5),
                                     arrowstyle='->', color='black', 
                                     linewidth=3, mutation_scale=25)
    ax.add_patch(arrow_transform)
    ax.text(5.5, 5.5, 'Slack\nTransform', fontsize=10, ha='center',
           fontweight='bold')
    ax.text(5.5, 4.5, 's = g - Fx', fontsize=9, ha='center', style='italic')
    
    # Augmented space
    ax.text(9, 6.5, 'Augmented Space (z = [x; s])', fontsize=12, 
           fontweight='bold', ha='center')
    
    # z_{t-1}
    circle3 = plt.Circle((8, 4.5), 0.8, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(circle3)
    ax.text(8, 4.5, 'z*_{t-1}', fontsize=10, ha='center', fontweight='bold')
    ax.text(8, 4, '[x*_{t-1}; s*_{t-1}]', fontsize=8, ha='center')
    
    # z_t
    circle4 = plt.Circle((10, 3), 0.8, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(circle4)
    ax.text(10, 3, 'z*_t', fontsize=10, ha='center', fontweight='bold')
    ax.text(10, 2.5, '[x*_t; s*_t]', fontsize=8, ha='center')
    
    # Arrow showing augmented movement
    arrow2 = FancyArrowPatch((8.6, 4), (9.4, 3.5),
                            arrowstyle='->', color='purple', linewidth=3,
                            mutation_scale=20)
    ax.add_patch(arrow2)
    ax.text(9.5, 4.3, 'Δz*', fontsize=11, color='purple', fontweight='bold')
    
    # Explanation box
    explanation = (
        "Slack coupling: Δs* = Δg - F·Δx*\n\n"
        "By triangle inequality:\n"
        "||Δz*|| = ||[Δx*; Δs*]|| ≤ ||Δx*|| + ||Δs*||\n"
        "                           ≤ ||Δx*|| + ||Δg|| + ||F||·||Δx*||\n"
        "                           = (1 + ||F||)·||Δx*|| + ||Δg||\n\n"
        "Summing over time:\n"
        "V_T^aug ≤ (1 + ||F||)·V_T^x + V_g^ineq"
    )
    
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                     edgecolor='black', linewidth=2)
    ax.text(6, 1.5, explanation, fontsize=9, ha='center', va='center',
           bbox=bbox_props, family='monospace')
    
    plt.tight_layout()
    return fig

def create_bound_breakdown():
    """Create a visual breakdown of the regret bound terms."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Bound structure
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Regret Bound Structure', fontsize=14, fontweight='bold', pad=20)
    
    # Main bound
    main_box = FancyBboxPatch((0.5, 7), 9, 2,
                             boxstyle="round,pad=0.2",
                             edgecolor='black', facecolor='lightblue',
                             linewidth=3)
    ax1.add_patch(main_box)
    ax1.text(5, 8, 'R_d(T) ≤', fontsize=12, ha='center', fontweight='bold')
    
    # Constant term
    const_box = FancyBboxPatch((0.5, 4.5), 4, 2,
                              boxstyle="round,pad=0.2",
                              edgecolor='red', facecolor='lightcoral',
                              linewidth=2)
    ax1.add_patch(const_box)
    ax1.text(2.5, 5.8, 'Constant Term', fontsize=11, ha='center', fontweight='bold')
    ax1.text(2.5, 5.3, '11pβ', fontsize=14, ha='center')
    ax1.text(2.5, 4.95, '―――――――', fontsize=12, ha='center')
    ax1.text(2.5, 4.7, '5η₀(β-1)', fontsize=14, ha='center')
    
    # Plus sign
    ax1.text(4.8, 5.5, '+', fontsize=20, ha='center', fontweight='bold')
    
    # Path-dependent term
    path_box = FancyBboxPatch((5.5, 4.5), 4, 2,
                             boxstyle="round,pad=0.2",
                             edgecolor='blue', facecolor='lightgreen',
                             linewidth=2)
    ax1.add_patch(path_box)
    ax1.text(7.5, 5.8, 'Path-Dependent', fontsize=11, ha='center', fontweight='bold')
    ax1.text(7.5, 5.2, '||c|| · [(1+||F||)·V_T^x', fontsize=10, ha='center')
    ax1.text(7.5, 4.8, '+ V_g^ineq]', fontsize=10, ha='center')
    
    # Arrows connecting
    arrow1 = FancyArrowPatch((2.5, 6.5), (2.5, 7),
                            arrowstyle='->', color='black', linewidth=2)
    ax1.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((7.5, 6.5), (7.5, 7),
                            arrowstyle='->', color='black', linewidth=2)
    ax1.add_patch(arrow2)
    
    # Annotations
    annotations = [
        (2.5, 3.8, 'Barrier initialization\nLinear in p'),
        (7.5, 3.8, 'Tracking cost\nDepends on dynamics'),
        (1, 2.5, 'p = # inequalities\nβ = barrier update rate\nη₀ = initial barrier param'),
        (6, 2.5, 'V_T^x = Σ||x*_t - x*_{t-1}||\nV_g^ineq = Σ||g_t - g_{t-1}||\n||F|| = constraint matrix norm'),
    ]
    
    for x, y, text in annotations:
        bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax1.text(x, y, text, fontsize=8, ha='left', bbox=bbox)
    
    # Right plot: Factor breakdown
    ax2.set_title('Path-Dependent Term Breakdown', fontsize=14, fontweight='bold', pad=20)
    
    # Sample values for visualization
    c_norm = 1.75
    F_norm = 2.67
    V_x = 1.06
    V_g = 2.73
    
    components = [
        '||c|| · V_T^x',
        '||c|| · ||F|| · V_T^x',
        '||c|| · V_g^ineq'
    ]
    
    values = [
        c_norm * V_x,
        c_norm * F_norm * V_x,
        c_norm * V_g
    ]
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    bars = ax2.barh(components, values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Contribution to Regret', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, max(values) * 1.3)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center', fontsize=10,
                fontweight='bold')
    
    # Add total
    total = sum(values)
    ax2.axvline(total, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(total, -0.6, f'Total: {total:.2f}', ha='center', fontsize=10,
            fontweight='bold', color='red')
    
    # Add example parameters box
    params_text = (
        'Example Parameters:\n'
        f'||c|| = {c_norm:.2f}\n'
        f'||F|| = {F_norm:.2f}\n'
        f'V_T^x = {V_x:.2f}\n'
        f'V_g^ineq = {V_g:.2f}\n'
        f'Coupling factor = {1+F_norm:.2f}'
    )
    
    bbox = dict(boxstyle='round', facecolor='lightyellow', 
               edgecolor='black', linewidth=1.5)
    ax2.text(0.98, 0.97, params_text, transform=ax2.transAxes,
            fontsize=9, ha='right', va='top', bbox=bbox,
            family='monospace')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all visualizations."""
    
    print("Generating regret derivation visualizations...")
    
    # 1. Main flowchart
    print("1. Creating derivation flowchart...")
    fig1 = create_regret_derivation_diagram()
    fig1.savefig('analysis/regret_derivation_flowchart.png', dpi=300, bbox_inches='tight')
    print("   Saved: analysis/regret_derivation_flowchart.png")
    
    # 2. Coupling visualization
    print("2. Creating coupling visualization...")
    fig2 = create_coupling_visualization()
    fig2.savefig('analysis/regret_coupling_diagram.png', dpi=300, bbox_inches='tight')
    print("   Saved: analysis/regret_coupling_diagram.png")
    
    # 3. Bound breakdown
    print("3. Creating bound breakdown...")
    fig3 = create_bound_breakdown()
    fig3.savefig('analysis/regret_bound_breakdown.png', dpi=300, bbox_inches='tight')
    print("   Saved: analysis/regret_bound_breakdown.png")
    
    plt.close('all')
    print("\nAll visualizations generated successfully!")
    print("\nFiles created:")
    print("  - regret_derivation_flowchart.png (complete step-by-step flow)")
    print("  - regret_coupling_diagram.png (explains the (1 + ||F||) factor)")
    print("  - regret_bound_breakdown.png (breaks down the bound terms)")

if __name__ == "__main__":
    main()
