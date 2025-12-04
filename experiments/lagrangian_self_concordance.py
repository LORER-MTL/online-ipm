"""
Self-Concordance Analysis of the Lagrangian
===========================================

This module analyzes whether the Lagrangian function in barrier methods
is self-concordant, including:

1. Definition of self-concordance for general functions
2. Tensor calculus for third derivatives
3. Verification of the self-concordance condition
4. Analysis of why the Lagrangian may NOT be self-concordant

Key insight: Self-concordant functions must be convex, but the Lagrangian
has indefinite structure due to the KKT system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
import warnings

warnings.filterwarnings('ignore')

def self_concordance_definition():
    """
    Define self-concordance rigorously with tensor notation.
    """
    print("=" * 70)
    print("SELF-CONCORDANCE DEFINITION AND TENSOR ANALYSIS")
    print("=" * 70)
    
    print("\n1. SELF-CONCORDANCE DEFINITION:")
    print("-" * 40)
    print("A function f: ℝⁿ → ℝ is self-concordant with parameter ν if:")
    print()
    print("For all x in domain and all directions h ∈ ℝⁿ:")
    print("   |∇³f(x)[h,h,h]| ≤ 2(∇²f(x)[h,h])^(3/2)")
    print()
    print("Where:")
    print("• ∇³f(x)[h,h,h] is the third derivative tensor applied to (h,h,h)")
    print("• ∇²f(x)[h,h] = h^T ∇²f(x) h is the quadratic form")
    print("• The bound must hold for ALL directions h")
    print()
    
    print("2. TENSOR NOTATION CLARIFICATION:")
    print("-" * 40)
    print("Third derivative tensor ∇³f(x) has components:")
    print("   (∇³f)_{ijk} = ∂³f/∂x_i ∂x_j ∂x_k")
    print()
    print("Applied to direction h:")
    print("   ∇³f(x)[h,h,h] = Σ_{i,j,k} (∇³f)_{ijk} h_i h_j h_k")
    print()
    print("This is a SCALAR (trilinear form applied to same vector 3 times)")
    print()
    
    print("3. WHY SELF-CONCORDANT FUNCTIONS MUST BE CONVEX:")
    print("-" * 50)
    print("Key theorem: If f is self-concordant, then ∇²f(x) ≽ 0")
    print()
    print("Proof sketch:")
    print("• Self-concordance gives local control of curvature")
    print("• The condition prevents 'negative curvature directions'")
    print("• Mathematical analysis shows this forces convexity")

def lagrangian_structure():
    """
    Analyze the structure of the barrier Lagrangian.
    """
    print("\n" + "=" * 70)
    print("BARRIER LAGRANGIAN STRUCTURE")
    print("=" * 70)
    
    print("\nThe barrier Lagrangian for linear programming:")
    print()
    print("L(x,λ) = c^T x - μ Σ log(x_i) + λ^T (Ax - b)")
    print()
    print("Where:")
    print("• x ∈ ℝⁿ are primal variables")
    print("• λ ∈ ℝᵐ are dual variables (Lagrange multipliers)")
    print("• μ > 0 is barrier parameter")
    print()
    
    print("First-order conditions (∇L = 0):")
    print("∇_x L = c - μ diag(1/x₁, ..., 1/xₙ) e + A^T λ = 0")
    print("∇_λ L = Ax - b = 0")
    print()
    
    print("Hessian of Lagrangian:")
    print("∇²L = [∇²_xx L    ∇²_xλ L ]")
    print("      [∇²_λx L    ∇²_λλ L ]")
    print()
    print("    = [μ diag(1/x₁², ..., 1/xₙ²)   A^T]")
    print("      [A                             0  ]")
    print()
    print("This is the KKT matrix we analyzed before!")

def compute_third_derivatives():
    """
    Compute third derivatives of the Lagrangian analytically.
    """
    print("\n" + "=" * 70)
    print("THIRD DERIVATIVE COMPUTATION")
    print("=" * 70)
    
    print("\n1. BARRIER COMPONENT THIRD DERIVATIVES:")
    print("-" * 45)
    print("For barrier φ(x) = -Σ log(x_i):")
    print()
    print("First derivatives:  ∂φ/∂x_i = -1/x_i")
    print("Second derivatives: ∂²φ/∂x_i² = 1/x_i²")
    print("                   ∂²φ/∂x_i∂x_j = 0  (i ≠ j)")
    print()
    print("Third derivatives:  ∂³φ/∂x_i³ = -2/x_i³")
    print("                   ∂³φ/∂x_i²∂x_j = 0  (i ≠ j)")
    print("                   ∂³φ/∂x_i∂x_j∂x_k = 0  (distinct i,j,k)")
    print()
    
    print("2. LAGRANGIAN THIRD DERIVATIVES:")
    print("-" * 35)
    print("L(x,λ) = c^T x - μ Σ log(x_i) + λ^T (Ax - b)")
    print()
    print("Since only the barrier term has x-dependence beyond linear:")
    print()
    print("∂³L/∂x_i³ = μ · 2/x_i³")
    print("∂³L/∂x_i²∂x_j = 0  (i ≠ j)")
    print("∂³L/∂x_i∂x_j∂x_k = 0  (distinct i,j,k)")
    print()
    print("Mixed derivatives with λ:")
    print("∂³L/∂x_i²∂λ_j = 0")
    print("∂³L/∂x_i∂λ_j∂λ_k = 0")
    print("∂³L/∂λ_i∂λ_j∂λ_k = 0")

def verify_self_concordance_condition():
    """
    Check if the Lagrangian satisfies self-concordance condition.
    """
    print("\n" + "=" * 70)
    print("SELF-CONCORDANCE VERIFICATION")
    print("=" * 70)
    
    print("\n1. SETTING UP THE VERIFICATION:")
    print("-" * 40)
    print("We need to check if for all directions h = [h_x; h_λ]:")
    print()
    print("|∇³L(x,λ)[h,h,h]| ≤ 2(∇²L(x,λ)[h,h])^(3/2)")
    print()
    
    print("2. COMPUTING THE THIRD DERIVATIVE TENSOR APPLICATION:")
    print("-" * 55)
    print("Direction h = [h_x₁, h_x₂, ..., h_xₙ, h_λ₁, ..., h_λₘ]")
    print()
    print("∇³L[h,h,h] = Σᵢ (∂³L/∂x_i³) h_xᵢ³ + [cross terms]")
    print()
    print("Since only ∂³L/∂x_i³ = 2μ/x_i³ is non-zero:")
    print()
    print("∇³L[h,h,h] = Σᵢ (2μ/x_i³) h_xᵢ³ = 2μ Σᵢ h_xᵢ³/x_i³")
    print()
    
    print("3. COMPUTING THE QUADRATIC FORM:")
    print("-" * 35)
    print("∇²L[h,h] = h^T ∇²L h")
    print()
    print("∇²L = [μ diag(1/x₁², ..., 1/xₙ²)   A^T]")
    print("      [A                             0  ]")
    print()
    print("∇²L[h,h] = μ Σᵢ h_xᵢ²/x_i² + 2h_x^T A^T h_λ")
    print()
    print("The cross-term 2h_x^T A^T h_λ can be positive or negative!")

def numerical_self_concordance_test():
    """
    Numerical test of self-concordance condition for specific examples.
    """
    print("\n" + "=" * 70)
    print("NUMERICAL SELF-CONCORDANCE TEST")
    print("=" * 70)
    
    # Simple 2D example: n=2 variables, m=1 constraint
    n, m = 2, 1
    A = np.array([[1.0, 1.0]])  # Sum constraint
    mu = 1.0
    
    print(f"\nTest case: {n} variables, {m} constraint")
    print(f"Constraint matrix A = {A}")
    print(f"Barrier parameter μ = {mu}")
    print()
    
    # Test point
    x = np.array([1.0, 2.0])
    lam = np.array([0.5])
    
    print(f"Test point: x = {x}, λ = {lam}")
    print()
    
    # Form Hessian
    H_xx = mu * np.diag(1.0 / (x**2))
    H_xλ = A.T
    H_λx = A
    H_λλ = np.zeros((m, m))
    
    # Full Hessian
    H_top = np.hstack([H_xx, H_xλ])
    H_bottom = np.hstack([H_λx, H_λλ])
    H = np.vstack([H_top, H_bottom])
    
    print("Lagrangian Hessian:")
    print(H)
    print()
    
    # Check if Hessian is positive semidefinite
    eigenvals = eigvals(H)
    print(f"Hessian eigenvalues: {eigenvals}")
    print(f"Positive semidefinite: {np.all(eigenvals >= -1e-10)}")
    print()
    
    # Test several directions
    np.random.seed(42)
    print("Testing self-concordance condition for random directions:")
    print("Direction h           ∇³L[h,h,h]    ∇²L[h,h]    (∇²L[h,h])^(3/2)   Satisfied?")
    print("-" * 80)
    
    for i in range(5):
        # Random direction
        h_x = np.random.randn(n)
        h_lam = np.random.randn(m)
        h = np.concatenate([h_x, h_lam])
        
        # Third derivative trilinear form
        third_deriv = 2 * mu * np.sum(h_x**3 / (x**3))
        
        # Quadratic form
        quad_form = h.T @ H @ h
        
        # Self-concordance condition
        if quad_form > 1e-12:  # Avoid numerical issues
            rhs = 2 * (quad_form)**(3/2)
            satisfied = abs(third_deriv) <= rhs
            
            h_str = f"[{h_x[0]:5.2f},{h_x[1]:5.2f},{h_lam[0]:5.2f}]"
            print(f"{h_str:15} {third_deriv:10.4f} {quad_form:12.4f} {rhs:15.4f}   {satisfied}")
        else:
            print(f"Direction {i+1}: Quadratic form too small, skipping...")

def analyze_indefinite_issue():
    """
    Analyze why the indefinite Hessian breaks self-concordance.
    """
    print("\n" + "=" * 70)
    print("WHY LAGRANGIAN CANNOT BE SELF-CONCORDANT")
    print("=" * 70)
    
    print("\n1. THE FUNDAMENTAL ISSUE:")
    print("-" * 30)
    print("Self-concordant functions MUST be convex (∇²f ≽ 0)")
    print("But the Lagrangian Hessian is INDEFINITE by design!")
    print()
    
    print("Lagrangian Hessian eigenvalue structure:")
    print("• Has POSITIVE eigenvalues (from barrier Hessian block)")
    print("• Has NEGATIVE eigenvalues (from saddle-point structure)")
    print("• This makes it indefinite: some eigenvalues > 0, some < 0")
    print()
    
    print("2. MATHEMATICAL PROOF OF NON-SELF-CONCORDANCE:")
    print("-" * 50)
    print("Theorem: If ∇²f has negative eigenvalues, then f is NOT self-concordant")
    print()
    print("Proof idea:")
    print("• Let v be an eigenvector with ∇²f v = λv, λ < 0")
    print("• Then ∇²f[v,v] = v^T ∇²f v = λ ||v||² < 0")
    print("• But self-concordance requires ∇²f[h,h] ≥ 0 for the condition to make sense")
    print("• Therefore, indefinite functions cannot be self-concordant")
    print()
    
    print("3. SPECIFIC TO BARRIER LAGRANGIANS:")
    print("-" * 40)
    print("The Lagrangian Hessian:")
    print("H = [μ diag(1/x²)   A^T]")
    print("    [A              0  ]")
    print()
    print("Always has:")
    print("• Positive eigenvalues from the barrier block")
    print("• Negative eigenvalues from the constraint structure")
    print("• This indefinite structure is ESSENTIAL for KKT systems")
    print("• Removing it would break the optimization method")

def demonstrate_failure_mode():
    """
    Demonstrate specific directions where self-concordance fails.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATING SELF-CONCORDANCE FAILURE")
    print("=" * 70)
    
    # Simple example
    n, m = 2, 1
    A = np.array([[1.0, 1.0]])
    x = np.array([1.0, 1.0])
    mu = 1.0
    
    # Form Hessian
    H_xx = mu * np.diag(1.0 / (x**2))  # [1, 1]
    H_xλ = A.T                         # [[1], [1]]
    H_λx = A                           # [1, 1]
    H_λλ = np.zeros((1, 1))           # [0]
    
    H_top = np.hstack([H_xx, H_xλ])
    H_bottom = np.hstack([H_λx, H_λλ])
    H = np.vstack([H_top, H_bottom])
    
    print("Example Lagrangian Hessian:")
    print(H)
    print()
    
    # Find eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(H)
    print("Eigenvalues and eigenvectors:")
    for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
        print(f"λ_{i+1} = {val:8.4f}, v_{i+1} = {vec}")
    
    print()
    
    # Find a direction with negative curvature
    negative_idx = np.argmin(eigenvals)
    if eigenvals[negative_idx] < 0:
        h = eigenvecs[:, negative_idx]
        
        print(f"Direction with negative curvature: h = {h}")
        print(f"∇²L[h,h] = {h.T @ H @ h:.6f} < 0")
        
        # This makes (∇²L[h,h])^(3/2) complex!
        quad_form = h.T @ H @ h
        if quad_form < 0:
            print()
            print("CRITICAL ISSUE:")
            print(f"Since ∇²L[h,h] = {quad_form:.6f} < 0,")
            print("the term (∇²L[h,h])^(3/2) is not even real!")
            print("This definitively proves the Lagrangian is NOT self-concordant.")

def alternative_disproof():
    """
    Provide an alternative disproof using convexity requirement.
    """
    print("\n" + "=" * 70)
    print("ALTERNATIVE DISPROOF: CONVEXITY REQUIREMENT")
    print("=" * 70)
    
    print("\nSIMPLE DISPROOF:")
    print("-" * 20)
    print("1. Self-concordant functions must be convex")
    print("2. Convex functions have ∇²f ≽ 0 (positive semidefinite)")
    print("3. The Lagrangian Hessian is indefinite (has negative eigenvalues)")
    print("4. Therefore, the Lagrangian cannot be self-concordant")
    print()
    print("QED. ■")
    print()
    
    print("DEEPER MATHEMATICAL INSIGHT:")
    print("-" * 35)
    print("The indefinite structure of the Lagrangian Hessian is not a bug—")
    print("it's a FEATURE that enables the KKT system to solve constrained")
    print("optimization problems. Self-concordance is a property of")
    print("unconstrained convex functions, not constrained optimization")
    print("Lagrangians.")

def main():
    """
    Main analysis of Lagrangian self-concordance.
    """
    print("SELF-CONCORDANCE ANALYSIS OF THE BARRIER LAGRANGIAN")
    print("Investigating whether the Lagrangian satisfies self-concordance")
    print("=" * 70)
    
    # Basic definitions and theory
    self_concordance_definition()
    
    # Analyze Lagrangian structure
    lagrangian_structure()
    
    # Compute third derivatives
    compute_third_derivatives()
    
    # Verify the condition
    verify_self_concordance_condition()
    
    # Numerical tests
    numerical_self_concordance_test()
    
    # Analyze why it fails
    analyze_indefinite_issue()
    
    # Demonstrate failure
    demonstrate_failure_mode()
    
    # Alternative disproof
    alternative_disproof()
    
    print("\n" + "=" * 70)
    print("CONCLUSION: LAGRANGIAN IS NOT SELF-CONCORDANT")
    print("=" * 70)
    print()
    print("DEFINITIVE ANSWER: NO, the Lagrangian is NOT self-concordant.")
    print()
    print("REASONS:")
    print("1. Self-concordant functions must be convex")
    print("2. The Lagrangian Hessian is indefinite (not positive semidefinite)")
    print("3. Directions with negative curvature make (∇²L[h,h])^(3/2) complex")
    print("4. This violates the basic requirements for self-concordance")
    print()
    print("IMPLICATION:")
    print("Any analysis relying on self-concordance of the full Lagrangian")
    print("is mathematically invalid. Self-concordance applies to the")
    print("BARRIER COMPONENT alone, not the complete constrained system.")

if __name__ == "__main__":
    main()