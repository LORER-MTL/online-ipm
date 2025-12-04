"""
Analysis of KKT Matrix Inverse Bounds
=====================================

This module analyzes Assumption 4 when interpreted as a bound on the INVERSE
of the KKT matrix rather than the matrix itself. This interpretation might
make more mathematical sense given the properties of barrier functions.

The KKT matrix is:
K = [∇²φ(x)  A^T]
    [A        0 ]

We examine whether ||K^(-1)|| ≤ 1/m might be valid, and what this would mean.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags, hstack, vstack
from scipy.linalg import inv, norm, eigvals
from numpy.linalg import cond
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def analyze_kkt_matrix_structure():
    """
    Analyze the structure and properties of the KKT matrix.
    """
    print("=" * 70)
    print("KKT MATRIX STRUCTURE ANALYSIS")
    print("=" * 70)
    
    print("\nThe KKT matrix for barrier problems has the form:")
    print("    K = [∇²φ(x)  A^T]")
    print("        [A        0 ]")
    print()
    print("Where:")
    print("  • ∇²φ(x) = diag(1/x₁², 1/x₂², ..., 1/xₙ²) ≻ 0")
    print("  • A is the m×n constraint matrix") 
    print("  • 0 is the m×m zero block")
    print()
    print("This is a symmetric indefinite matrix of size (n+m)×(n+m)")
    print("The indefinite structure comes from the zero block in the bottom-right")

def compute_kkt_inverse_analytically():
    """
    Derive the analytical form of the KKT matrix inverse using block matrix inversion.
    """
    print("\n" + "=" * 70)
    print("ANALYTICAL KKT MATRIX INVERSE")
    print("=" * 70)
    
    print("\nFor a 2×2 block matrix:")
    print("    K = [B   C]")
    print("        [D   E]")
    print()
    print("The inverse (when it exists) is:")
    print("    K⁻¹ = [F₁₁  F₁₂]")
    print("          [F₂₁  F₂₂]")
    print()
    print("Where (assuming E is invertible):")
    print("  S = B - CE⁻¹D  (Schur complement)")
    print("  F₁₁ = S⁻¹")
    print("  F₁₂ = -S⁻¹CE⁻¹") 
    print("  F₂₁ = -E⁻¹DS⁻¹")
    print("  F₂₂ = E⁻¹ + E⁻¹DS⁻¹CE⁻¹")
    print()
    print("For our KKT matrix:")
    print("  B = ∇²φ(x), C = A^T, D = A, E = 0")
    print()
    print("PROBLEM: E = 0 is not invertible!")
    print("We need to use the other Schur complement formula...")
    print()
    print("Using S = E - DB⁻¹C = 0 - A(∇²φ(x))⁻¹A^T = -A(∇²φ(x))⁻¹A^T")
    print()
    print("This gives us:")
    print("  K⁻¹ = [(∇²φ)⁻¹ + (∇²φ)⁻¹A^T S⁻¹ A(∇²φ)⁻¹    -(∇²φ)⁻¹A^T S⁻¹]")
    print("        [           -S⁻¹ A(∇²φ)⁻¹                      S⁻¹       ]")
    print()
    print("Where S = -A(∇²φ)⁻¹A^T")

def analyze_schur_complement():
    """
    Analyze the properties of the Schur complement in the KKT inverse.
    """
    print("\n" + "=" * 70)
    print("SCHUR COMPLEMENT ANALYSIS")
    print("=" * 70)
    
    print("\nThe key component is the Schur complement:")
    print("    S = -A(∇²φ(x))⁻¹A^T")
    print()
    print("Properties:")
    print("  • (∇²φ(x))⁻¹ = diag(x₁², x₂², ..., xₙ²) ≻ 0")
    print("  • A(∇²φ(x))⁻¹A^T is positive semidefinite")  
    print("  • S = -A(∇²φ(x))⁻¹A^T ≼ 0 (negative semidefinite)")
    print()
    print("For S to be invertible, we need A to have full row rank m.")
    print("Assuming this, S ≺ 0 (negative definite) and S⁻¹ exists.")
    print()
    print("CRITICAL OBSERVATION:")
    print("As x_i → 0, we have:")
    print("  • (∇²φ(x))⁻¹ = diag(x₁², ..., xₙ²) → 0")
    print("  • A(∇²φ(x))⁻¹A^T → 0") 
    print("  • S = -A(∇²φ(x))⁻¹A^T → 0")
    print("  • ||S⁻¹|| → ∞ as S → 0")
    print()
    print("Therefore, ||K⁻¹|| → ∞ as any x_i → 0!")

def numerical_kkt_inverse_analysis():
    """
    Numerical analysis of KKT matrix inverse norms.
    """
    print("\n" + "=" * 70)
    print("NUMERICAL KKT INVERSE ANALYSIS")
    print("=" * 70)
    
    # Simple test case
    n, m = 3, 2
    A = np.array([[1.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0]])
    
    print(f"Test case: {n} variables, {m} constraints")
    print("Constraint matrix A:")
    print(A)
    print()
    
    # Test points approaching boundary
    test_points = [
        np.array([2.0, 2.0, 2.0]),      # Interior
        np.array([1.0, 1.0, 1.0]),      # Closer  
        np.array([0.5, 0.5, 0.5]),      # Nearer
        np.array([0.1, 0.1, 0.1]),      # Near boundary
        np.array([0.01, 0.01, 0.01]),   # Very close
    ]
    
    print("Analysis of KKT matrix and its inverse:")
    print("Point x              ||K||₂        ||K⁻¹||₂       Condition κ(K)")
    print("-" * 75)
    
    results = []
    for x in test_points:
        # Form KKT matrix
        H = np.diag(1.0 / (x**2))  # Barrier Hessian
        zeros_mm = np.zeros((m, m))
        
        # KKT matrix
        K_top = np.hstack([H, A.T])
        K_bottom = np.hstack([A, zeros_mm])
        K = np.vstack([K_top, K_bottom])
        
        # Compute norms and condition number
        try:
            K_inv = inv(K)
            norm_K = norm(K, 2)
            norm_K_inv = norm(K_inv, 2)
            cond_K = cond(K)
            
            x_str = f"[{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}]"
            print(f"{x_str:20} {norm_K:10.2e} {norm_K_inv:13.2e} {cond_K:15.2e}")
            
            results.append({
                'x': x.copy(),
                'norm_K': norm_K,
                'norm_K_inv': norm_K_inv,
                'condition': cond_K
            })
            
        except np.linalg.LinAlgError:
            print(f"{x_str:20} {'SINGULAR':>10} {'SINGULAR':>13} {'SINGULAR':>15}")
    
    print()
    return results

def analyze_inverse_bound_validity(results):
    """
    Analyze whether ||K⁻¹|| ≤ 1/m could be valid.
    """
    print("INVERSE BOUND ANALYSIS:")
    print("=" * 40)
    
    m = 2  # From our test case
    bound = 1.0 / m
    
    print(f"Claimed bound: ||K⁻¹|| ≤ 1/m = 1/{m} = {bound}")
    print()
    print("Checking if bound is satisfied:")
    
    satisfied_count = 0
    for i, result in enumerate(results):
        norm_inv = result['norm_K_inv']
        satisfied = norm_inv <= bound
        satisfied_count += satisfied
        
        x_str = f"[{result['x'][0]:.2f}, {result['x'][1]:.2f}, {result['x'][2]:.2f}]"
        status = "✓" if satisfied else "✗"
        print(f"  {x_str}: ||K⁻¹|| = {norm_inv:.2e} {status}")
    
    print()
    print(f"Bound satisfied: {satisfied_count}/{len(results)} cases")
    
    if satisfied_count == 0:
        print("❌ CONCLUSION: ||K⁻¹|| ≤ 1/m is ALSO INVALID!")
        print("   The inverse norm grows unboundedly as x approaches boundary.")
    elif satisfied_count < len(results):
        print("⚠️  CONCLUSION: ||K⁻¹|| ≤ 1/m is SOMETIMES valid but fails near boundary.")
    else:
        print("✓ CONCLUSION: ||K⁻¹|| ≤ 1/m might be valid for this specific case.")

def theoretical_inverse_behavior():
    """
    Theoretical analysis of why the inverse norm blows up.
    """
    print("\n" + "=" * 70)
    print("THEORETICAL ANALYSIS OF INVERSE BEHAVIOR")
    print("=" * 70)
    
    print("\nWhy does ||K⁻¹|| → ∞ as x_i → 0?")
    print()
    print("1. SCHUR COMPLEMENT BEHAVIOR:")
    print("   S = -A diag(x₁², ..., xₙ²) A^T")
    print("   As min(x_i) → 0: S → 0")
    print("   Therefore: ||S⁻¹|| → ∞")
    print()
    print("2. INVERSE BLOCK STRUCTURE:")
    print("   K⁻¹ contains S⁻¹ as a block")
    print("   ||K⁻¹|| ≥ ||S⁻¹|| → ∞")
    print()
    print("3. EIGENVALUE ANALYSIS:")
    print("   The smallest eigenvalue λ_min(K) → 0 as x_i → 0")
    print("   Since ||K⁻¹|| ≥ 1/λ_min(K), we have ||K⁻¹|| → ∞")
    print()
    print("4. PHYSICAL INTERPRETATION:")
    print("   As we approach boundary, the KKT system becomes ill-conditioned")
    print("   Small changes in RHS lead to large changes in solution")
    print("   This is INHERENT to barrier methods!")

def compare_matrix_vs_inverse():
    """
    Compare the behavior of ||K|| vs ||K⁻¹||.
    """
    print("\n" + "=" * 70)
    print("MATRIX vs INVERSE NORM COMPARISON")
    print("=" * 70)
    
    print("\nKey differences between ||K|| and ||K⁻¹|| bounds:")
    print()
    print("||K|| BOUND (Original Assumption 4):")
    print("  • Claims: ||K|| ≤ 1/m")
    print("  • Reality: ||K|| → ∞ as x_i → 0 (due to barrier Hessian)")
    print("  • Status: ❌ INVALID")
    print()
    print("||K⁻¹|| BOUND (Alternative interpretation):")
    print("  • Would claim: ||K⁻¹|| ≤ 1/m")
    print("  • Reality: ||K⁻¹|| → ∞ as x_i → 0 (due to Schur complement)")
    print("  • Status: ❌ ALSO INVALID")
    print()
    print("MATHEMATICAL REASON:")
    print("  Both norms blow up near the boundary due to the barrier structure")
    print("  ||K|| → ∞     because diagonal barrier terms 1/x_i² → ∞")
    print("  ||K⁻¹|| → ∞   because Schur complement S → 0, so S⁻¹ → ∞")
    print()
    print("CONCLUSION:")
    print("  Neither interpretation of Assumption 4 is mathematically valid!")

def potential_valid_interpretations():
    """
    Explore what the authors might have actually meant.
    """
    print("\n" + "=" * 70)
    print("POTENTIAL VALID INTERPRETATIONS")
    print("=" * 70)
    
    print("\nWhat could the authors have meant instead?")
    print()
    print("1. CONDITION NUMBER BOUND:")
    print("   Maybe: κ(K) = ||K|| ||K⁻¹|| ≤ C for some constant C")
    print("   Status: Still problematic as both norms diverge")
    print()
    print("2. WEIGHTED NORM BOUND:")
    print("   Maybe: ||D⁻¹/² K D⁻¹/²|| ≤ 1/m for some scaling D")
    print("   Status: Could be valid with appropriate scaling")
    print()
    print("3. RESTRICTED DOMAIN BOUND:")
    print("   Maybe: ||K|| ≤ 1/m for x ∈ {x : x_i ≥ δ > 0}")
    print("   Status: Could be valid away from boundary")
    print()
    print("4. DIFFERENT MATRIX ENTIRELY:")
    print("   Maybe they meant the constraint matrix A, not the full KKT matrix")
    print("   Status: ||A|| could be bounded, but that's a different assumption")
    print()
    print("5. MATRIX BLOCK BOUND:")
    print("   Maybe: ||A(∇²φ)⁻¹A^T|| ≤ 1/m")
    print("   Status: Could be related to constraint qualification")
    print()
    print("MOST LIKELY:")
    print("  This appears to be a fundamental error in the paper's assumptions.")
    print("  Standard barrier theory does not support any uniform boundedness")
    print("  of KKT matrices or their inverses.")

def create_visualization(results):
    """
    Create visualization showing norm behavior.
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    
    # Extract data for plotting
    min_x_vals = [np.min(r['x']) for r in results]
    norm_K_vals = [r['norm_K'] for r in results]
    norm_K_inv_vals = [r['norm_K_inv'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot matrix norms
    ax1.loglog(min_x_vals, norm_K_vals, 'b-o', label='||K||', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Claimed bound 1/m = 0.5')
    ax1.set_xlabel('min(x)')
    ax1.set_ylabel('Matrix norm')
    ax1.set_title('KKT Matrix Norm vs Distance to Boundary')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot inverse norms  
    ax2.loglog(min_x_vals, norm_K_inv_vals, 'r-s', label='||K⁻¹||', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Claimed bound 1/m = 0.5')
    ax2.set_xlabel('min(x)')
    ax2.set_ylabel('Inverse norm')
    ax2.set_title('KKT Inverse Norm vs Distance to Boundary')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add annotations
    ax1.annotate('||K|| → ∞ as x → boundary', 
                xy=(min_x_vals[0], norm_K_vals[0]),
                xytext=(min_x_vals[2], norm_K_vals[0]*10),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    ax2.annotate('||K⁻¹|| → ∞ as x → boundary', 
                xy=(min_x_vals[0], norm_K_inv_vals[0]),
                xytext=(min_x_vals[2], norm_K_inv_vals[0]*10),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    try:
        plt.savefig('/home/willyzz/Documents/online-ipm/kkt_inverse_analysis.png', 
                    dpi=150, bbox_inches='tight')
        print("KKT inverse analysis plot saved as 'kkt_inverse_analysis.png'")
    except:
        print("Could not save visualization file")
    
    try:
        plt.show()
    except:
        print("Could not display plot (no GUI available)")
    
    plt.close()

def main():
    """
    Main analysis of KKT matrix inverse bounds.
    """
    print("KKT MATRIX INVERSE BOUND ANALYSIS")
    print("Investigating whether Assumption 4 makes sense for ||K⁻¹|| instead of ||K||")
    print("=" * 70)
    
    # Analyze matrix structure
    analyze_kkt_matrix_structure()
    
    # Derive analytical inverse form
    compute_kkt_inverse_analytically()
    
    # Analyze Schur complement
    analyze_schur_complement()
    
    # Numerical analysis
    results = numerical_kkt_inverse_analysis()
    
    # Check bound validity
    analyze_inverse_bound_validity(results)
    
    # Theoretical analysis
    theoretical_inverse_behavior()
    
    # Compare matrix vs inverse
    compare_matrix_vs_inverse()
    
    # Potential interpretations
    potential_valid_interpretations()
    
    # Create visualization
    create_visualization(results)
    
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION ON KKT INVERSE BOUNDS")
    print("=" * 70)
    print()
    print("VERDICT: The inverse interpretation is ALSO INVALID!")
    print()
    print("Reasons:")
    print("1. ||K⁻¹|| → ∞ as x approaches boundary (due to Schur complement)")
    print("2. This is fundamental to barrier method structure")
    print("3. The KKT system becomes ill-conditioned near boundary BY DESIGN")
    print("4. No finite uniform bound can hold for ||K⁻¹||")
    print()
    print("Whether interpreted as ||K|| ≤ 1/m or ||K⁻¹|| ≤ 1/m,")
    print("Assumption 4 remains mathematically invalid.")
    print()
    print("The paper's theoretical foundation needs significant revision.")

if __name__ == "__main__":
    main()