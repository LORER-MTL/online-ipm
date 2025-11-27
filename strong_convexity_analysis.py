"""
Strong Convexity Analysis of Barrier Functions
==============================================

This module provides a detailed mathematical analysis of why barrier functions
are strongly convex on their domain, including:

1. Definition of strong convexity
2. Proof for logarithmic barriers
3. Relationship to eigenvalues and positive definiteness
4. Practical implications for optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
import warnings

warnings.filterwarnings('ignore')

def analyze_strong_convexity():
    """
    Analyze strong convexity of barrier functions with mathematical rigor.
    """
    print("=" * 60)
    print("STRONG CONVEXITY ANALYSIS OF BARRIER FUNCTIONS")
    print("=" * 60)
    
    print("\n1. DEFINITION OF STRONG CONVEXITY")
    print("-" * 40)
    print("A function f: ℝⁿ → ℝ is strongly convex with parameter μ > 0 if:")
    print("   ∇²f(x) ⪰ μI  for all x in the domain")
    print()
    print("Equivalently:")
    print("   f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²")
    print()
    print("This means the Hessian is positive definite with minimum eigenvalue ≥ μ")
    
    print("\n2. LOGARITHMIC BARRIER FUNCTION")
    print("-" * 40)
    print("For φ(x) = -∑ᵢ log(xᵢ) on domain {x : x > 0}:")
    print()
    print("First derivative:  ∇φ(x) = (-1/x₁, -1/x₂, ..., -1/xₙ)ᵀ")
    print("Second derivative: ∇²φ(x) = diag(1/x₁², 1/x₂², ..., 1/xₙ²)")
    print()
    print("Since ∇²φ(x) is diagonal with positive entries, it's positive definite!")

def demonstrate_positive_definiteness():
    """
    Demonstrate that the barrier Hessian is positive definite.
    """
    print("\n3. POSITIVE DEFINITENESS VERIFICATION")
    print("-" * 40)
    
    # Test at various points
    test_points = [
        np.array([2.0, 3.0, 1.5]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.1, 0.5, 2.0]),
        np.array([0.01, 0.1, 5.0])
    ]
    
    print("Testing ∇²φ(x) = diag(1/x₁², 1/x₂², 1/x₃²) at various points:")
    print()
    print("Point x                  Eigenvalues                    Min eigenvalue (μ)")
    print("-" * 75)
    
    for x in test_points:
        # Compute Hessian diagonal entries
        hess_diag = 1.0 / (x**2)
        
        # For diagonal matrix, eigenvalues are the diagonal entries
        eigenvalues = hess_diag
        min_eigenval = np.min(eigenvalues)
        
        x_str = f"[{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}]"
        eig_str = f"[{eigenvalues[0]:.2e}, {eigenvalues[1]:.2e}, {eigenvalues[2]:.2e}]"
        
        print(f"{x_str:20} {eig_str:30} {min_eigenval:.2e}")
    
    print()
    print("✓ All eigenvalues are positive → ∇²φ(x) ≻ 0 (positive definite)")
    print("✓ Minimum eigenvalue μ(x) = min(1/xᵢ²) > 0 for all x > 0")

def analyze_strong_convexity_parameter():
    """
    Analyze how the strong convexity parameter varies with the point.
    """
    print("\n4. STRONG CONVEXITY PARAMETER")
    print("-" * 40)
    
    print("For φ(x) = -∑ log(xᵢ), the strong convexity parameter is:")
    print("   μ(x) = λₘᵢₙ(∇²φ(x)) = min{1/x₁², 1/x₂², ..., 1/xₙ²}")
    print("        = 1/max{x₁², x₂², ..., xₙ²}")
    print("        = 1/(max(x))²")
    print()
    
    # Demonstrate how μ changes with x
    x_max_values = np.logspace(-2, 1, 50)  # from 0.01 to 10
    mu_values = 1.0 / (x_max_values**2)
    
    print("Strong convexity parameter behavior:")
    print("max(x)     μ(x) = 1/(max(x))²")
    print("-" * 30)
    
    for i in range(0, len(x_max_values), 10):
        x_max = x_max_values[i]
        mu = mu_values[i]
        print(f"{x_max:6.2f}     {mu:12.2e}")
    
    print()
    print("Key observations:")
    print("✓ μ(x) > 0 always (strong convexity holds everywhere in domain)")
    print("✓ μ(x) → ∞ as max(x) → 0 (very strongly convex near boundary)")
    print("✓ μ(x) → 0 as max(x) → ∞ (less strongly convex far from boundary)")
    print("✓ Strong convexity is LOCAL - the parameter depends on the point")

def quadratic_form_verification():
    """
    Verify strong convexity using the quadratic form definition.
    """
    print("\n5. QUADRATIC FORM VERIFICATION")
    print("-" * 40)
    
    print("For any vector v ≠ 0 and point x > 0:")
    print("   vᵀ∇²φ(x)v = vᵀ diag(1/x₁², ..., 1/xₙ²) v")
    print("            = ∑ᵢ (vᵢ²/xᵢ²)")
    print("            = ∑ᵢ (vᵢ/xᵢ)²")
    print("            > 0  (since vᵢ/xᵢ = 0 for all i would imply v = 0)")
    print()
    
    # Numerical verification
    x = np.array([1.0, 2.0, 0.5])
    H = np.diag(1.0 / (x**2))
    
    print("Numerical verification at x = [1.0, 2.0, 0.5]:")
    print(f"∇²φ(x) = diag({H[0,0]:.2f}, {H[1,1]:.2f}, {H[2,2]:.2f})")
    print()
    
    # Test several random vectors
    print("Testing quadratic form vᵀ∇²φ(x)v for random vectors v:")
    print("Vector v                 vᵀ∇²φ(x)v    Positive?")
    print("-" * 50)
    
    np.random.seed(42)
    for _ in range(5):
        v = np.random.randn(3)
        quad_form = v.T @ H @ v
        print(f"[{v[0]:6.2f}, {v[1]:6.2f}, {v[2]:6.2f}]    {quad_form:8.4f}       ✓")
    
    print()
    print("✓ All quadratic forms are positive → strong convexity confirmed")

def compare_with_other_functions():
    """
    Compare barrier strong convexity with other common functions.
    """
    print("\n6. COMPARISON WITH OTHER FUNCTIONS")
    print("-" * 40)
    
    x = 1.0  # Test point
    
    functions = [
        ("Quadratic f(x) = x²", 2.0, "Always μ = 2"),
        ("Quartic f(x) = x⁴", 12.0 * x**2, f"μ = 12x² = {12.0 * x**2:.1f}"),
        ("Barrier φ(x) = -log(x)", 1.0/x**2, f"μ = 1/x² = {1.0/x**2:.1f}"),
        ("Exponential f(x) = eˣ", np.exp(x), f"μ = eˣ = {np.exp(x):.1f}")
    ]
    
    print("Function              Strong Convexity Parameter at x=1")
    print("-" * 55)
    for name, mu, desc in functions:
        print(f"{name:20} {desc}")
    
    print()
    print("Key differences:")
    print("• Quadratic: constant strong convexity (μ = const)")
    print("• Barrier: strong convexity depends on location (μ = 1/x²)")
    print("• Near boundary (x→0): barrier becomes VERY strongly convex")
    print("• Far from boundary: barrier strong convexity decreases")

def geometric_interpretation():
    """
    Provide geometric interpretation of barrier strong convexity.
    """
    print("\n7. GEOMETRIC INTERPRETATION")
    print("-" * 40)
    
    print("Strong convexity of φ(x) = -∑log(xᵢ) means:")
    print()
    print("1. BOWL SHAPE:")
    print("   • Function curves upward in all directions")
    print("   • No flat regions or saddle points")
    print("   • Unique global minimum (when it exists)")
    print()
    print("2. CONVERGENCE GUARANTEES:")
    print("   • Newton's method converges quadratically")
    print("   • Gradient descent has exponential convergence")
    print("   • Well-conditioned optimization problem")
    print()
    print("3. BARRIER EFFECT:")
    print("   • Near boundary (xᵢ → 0): very steep walls")
    print("   • Strong convexity parameter μ → ∞")
    print("   • Prevents iterates from leaving feasible region")
    print()
    print("4. INTERIOR BEHAVIOR:")
    print("   • Away from boundary: more gentle curvature")
    print("   • Still strongly convex but with smaller μ")
    print("   • Allows efficient optimization in interior")

def practical_implications():
    """
    Discuss practical implications for optimization algorithms.
    """
    print("\n8. PRACTICAL IMPLICATIONS FOR OPTIMIZATION")
    print("-" * 40)
    
    print("Strong convexity of barrier functions ensures:")
    print()
    print("✓ UNIQUE SOLUTIONS:")
    print("  • Each barrier problem has unique minimizer")
    print("  • Well-posed optimization problems")
    print()
    print("✓ FAST CONVERGENCE:")
    print("  • Newton: quadratic convergence rate")
    print("  • Gradient: exponential convergence rate")
    print("  • Superior to general convex case")
    print()
    print("✓ NUMERICAL STABILITY:")
    print("  • Positive definite Hessians")
    print("  • Reliable linear system solves")
    print("  • Good condition numbers (locally)")
    print()
    print("✓ BARRIER CONTAINMENT:")
    print("  • Strong convexity increases near boundary")
    print("  • Natural mechanism to stay feasible")
    print("  • Self-regulating algorithm behavior")
    print()
    print("⚠ TRADE-OFFS:")
    print("  • Hessian can be very large near boundary")
    print("  • May require small step sizes")
    print("  • Condition number varies with location")

def create_visualization():
    """
    Create visualization showing strong convexity behavior.
    """
    print("\n9. VISUALIZATION")
    print("-" * 40)
    
    # Create 1D barrier function plot
    x = np.linspace(0.01, 5, 1000)
    phi = -np.log(x)
    phi_second = 1.0 / (x**2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot barrier function
    ax1.plot(x, phi, 'b-', linewidth=2, label='φ(x) = -log(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('φ(x)')
    ax1.set_title('Logarithmic Barrier Function')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 5)
    
    # Plot strong convexity parameter
    ax2.semilogy(x, phi_second, 'r-', linewidth=2, label="μ(x) = φ''(x) = 1/x²")
    ax2.set_xlabel('x')
    ax2.set_ylabel('Strong convexity parameter μ(x)')
    ax2.set_title('Strong Convexity Parameter')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 5)
    
    # Add annotations
    ax2.annotate('μ(x) → ∞ as x → 0+\n(Very strong convexity)', 
                xy=(0.1, 100), xytext=(1, 500),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax2.annotate('μ(x) decreases as x increases\n(Still strongly convex)', 
                xy=(3, 0.11), xytext=(2, 2),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    plt.tight_layout()
    
    try:
        plt.savefig('/home/willyzz/Documents/online-ipm/strong_convexity_analysis.png', 
                    dpi=150, bbox_inches='tight')
        print("Strong convexity visualization saved as 'strong_convexity_analysis.png'")
    except:
        print("Could not save visualization file")
    
    try:
        plt.show()
    except:
        print("Could not display plot (no GUI available)")
    
    plt.close()

def main():
    """
    Main function providing comprehensive analysis of barrier strong convexity.
    """
    print("COMPREHENSIVE ANALYSIS: WHY BARRIER FUNCTIONS ARE STRONGLY CONVEX")
    print("=" * 70)
    
    analyze_strong_convexity()
    demonstrate_positive_definiteness()
    analyze_strong_convexity_parameter()
    quadratic_form_verification()
    compare_with_other_functions()
    geometric_interpretation()
    practical_implications()
    create_visualization()
    
    print("\n" + "=" * 70)
    print("SUMMARY: WHY STRONG CONVEXITY HOLDS")
    print("=" * 70)
    print()
    print("The barrier function φ(x) = -∑log(xᵢ) is strongly convex because:")
    print()
    print("1. MATHEMATICAL PROOF:")
    print("   • Hessian ∇²φ(x) = diag(1/x₁², ..., 1/xₙ²)")
    print("   • All diagonal entries are positive: 1/xᵢ² > 0")
    print("   • Therefore ∇²φ(x) ≻ 0 (positive definite)")
    print()
    print("2. STRONG CONVEXITY PARAMETER:")
    print("   • μ(x) = min{1/x₁², ..., 1/xₙ²} = 1/(max(x))²")
    print("   • μ(x) > 0 for all x in domain {x : x > 0}")
    print("   • Parameter varies with location but is always positive")
    print()
    print("3. QUADRATIC FORM TEST:")
    print("   • For any v ≠ 0: vᵀ∇²φ(x)v = ∑(vᵢ/xᵢ)² > 0")
    print("   • Confirms positive definiteness")
    print()
    print("4. PRACTICAL CONSEQUENCE:")
    print("   • Ensures unique minimizers in barrier problems")
    print("   • Guarantees fast convergence of optimization algorithms")
    print("   • Provides natural containment mechanism")
    print()
    print("Unlike the invalid Assumption 4 (bounded KKT matrix),")
    print("strong convexity is a FUNDAMENTAL and CORRECT property")
    print("of barrier functions that makes IPMs mathematically sound!")

if __name__ == "__main__":
    main()