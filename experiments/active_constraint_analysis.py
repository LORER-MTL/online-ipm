"""
Analysis of Numerical Issues When Inequality Constraints Become Active
====================================================================

This module analyzes the numerical challenges that arise when inequality constraints
become active (i.e., when slack variables approach zero) in the reformulated problem,
and provides practical strategies for handling these issues.

Key Issue: When Fx <= g_t becomes Fx + z = g_t with z -> 0+
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags, block_diag
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import warnings

class ActiveConstraintAnalyzer:
    """
    Analyzes numerical issues when inequality constraints become active.
    """
    
    def __init__(self):
        self.examples = {}
        self.solutions = {}
    
    def explain_the_problem(self):
        """
        Explain why active constraints cause numerical issues.
        """
        print("=" * 80)
        print("NUMERICAL ISSUES WITH ACTIVE INEQUALITY CONSTRAINTS")
        print("=" * 80)
        
        print("\n1. THE FUNDAMENTAL ISSUE")
        print("-" * 50)
        
        print("When inequality constraint Fx <= g_t becomes active:")
        print("• Original: Fx = g_t (constraint is tight)")
        print("• Reformulated: Fx + z = g_t with z ≈ 0")
        print()
        print("Problems arise because:")
        print("a) Interior Point Methods require z > 0 (strict inequality)")
        print("b) Barrier function -log(z_i) → +∞ as z_i → 0+")
        print("c) Newton system becomes ill-conditioned")
        print("d) Step sizes become very small near boundary")
        
        print("\n2. MATHEMATICAL MANIFESTATION")
        print("-" * 50)
        
        print("In the KKT system, when z_i ≈ 0:")
        print()
        print("Standard KKT matrix has structure:")
        print("┌─────────────────────────┐")
        print("│ 0    A^T   I    0      │")  
        print("│ A     0    0    0      │")
        print("│ I     0    X    0      │ <- Original variables")
        print("│ 0     0    0    Z      │ <- Slack variables")
        print("└─────────────────────────┘")
        print()
        print("Where Z = diag(z_1, ..., z_m₂)")
        print("As z_i → 0: Z becomes singular → KKT matrix becomes ill-conditioned")
        
        print("\n3. CONCRETE MANIFESTATIONS")
        print("-" * 50)
        
        print("• Condition number: κ(KKT) ≈ O(1/min(z_i)) → ∞")
        print("• Newton steps: ||Δx|| becomes unreliable")
        print("• Line search: Step sizes α → 0 to maintain z > 0")
        print("• Convergence: Algorithm stalls near optimum")
        print("• Accuracy: Solution accuracy degrades")
        
    def create_example_problem(self):
        """
        Create a concrete example where inequality becomes active.
        """
        print("\n4. CONCRETE EXAMPLE: INEQUALITY BECOMING ACTIVE")
        print("-" * 50)
        
        print("Consider a simple 2D problem with time-varying constraints:")
        print()
        print("minimize    x₁ + x₂")
        print("subject to  x₁ + x₂ = 3")
        print("            x₁ ≤ g_t      <- This inequality becomes active")
        print("            x₁, x₂ ≥ 0")
        print()
        
        # Problem data
        A = np.array([[1, 1]])  # Equality constraint
        b = np.array([3])
        F = np.array([[1, 0]])  # Inequality constraint  
        c = np.array([1, 1])
        
        print("Problem data:")
        print(f"A = {A.tolist()}")
        print(f"b = {b.tolist()}")
        print(f"F = {F.tolist()}")
        print(f"c = {c.tolist()}")
        
        # Time-varying scenarios
        g_values = [5.0, 2.0, 1.5, 1.01, 1.001]  # g_t approaches optimal value
        
        print(f"\nTime-varying g_t values: {g_values}")
        print("(Notice g_t → 1.0, making inequality x₁ ≤ g_t active)")
        
        print("\nReformulated problem with slack variable z:")
        print("minimize    x₁ + x₂")
        print("subject to  x₁ + x₂ = 3")
        print("            x₁ + z = g_t")
        print("            x₁, x₂, z ≥ 0")
        
        # Analyze what happens
        print("\nAnalysis for each time step:")
        print("g_t  | Optimal x₁ | Optimal x₂ | Slack z | Condition")
        print("-----|------------|------------|---------|----------")
        
        results = []
        for g_t in g_values:
            # Optimal solution: minimize x₁ + x₂ = 3 subject to x₁ ≤ g_t
            if g_t >= 1.5:
                # Inequality inactive: x₁ = x₂ = 1.5 
                x1_opt = 1.5
                x2_opt = 1.5
                z_opt = g_t - x1_opt
                active = "Inactive"
            else:
                # Inequality active: x₁ = g_t, x₂ = 3 - g_t
                x1_opt = min(g_t, 1.5)
                x2_opt = 3 - x1_opt  
                z_opt = g_t - x1_opt
                active = "ACTIVE" if z_opt < 0.1 else "Nearly Active"
            
            results.append((g_t, x1_opt, x2_opt, z_opt, active))
            print(f"{g_t:4.3f} | {x1_opt:10.3f} | {x2_opt:10.3f} | {z_opt:7.3f} | {active}")
        
        return {
            'A': A, 'b': b, 'F': F, 'c': c,
            'g_values': g_values,
            'results': results
        }
    
    def demonstrate_numerical_issues(self, example_data):
        """
        Demonstrate the actual numerical issues that arise.
        """
        print("\n5. NUMERICAL ISSUES DEMONSTRATION")
        print("-" * 50)
        
        # Extract data
        results = example_data['results']
        
        print("As the slack variable z approaches zero:")
        print()
        
        for i, (g_t, x1, x2, z, status) in enumerate(results):
            print(f"Time step {i+1}: g_t = {g_t}")
            
            if z > 0:
                # Barrier function value
                barrier_val = -np.log(z)
                
                # Condition number estimate (simplified)
                # In practice, this would be computed from actual KKT matrix
                cond_estimate = max(1.0, 1.0 / z)
                
                # Step size limitation
                # Newton method step would be limited by: z + α*dz >= δ > 0
                # So α <= (δ - z) / |dz| if dz < 0
                max_step = min(1.0, z / 2)  # Simplified estimate
                
                print(f"  Slack variable: z = {z:.6f}")
                print(f"  Barrier term: -log(z) = {barrier_val:.2f}")
                print(f"  Condition estimate: {cond_estimate:.1e}")
                print(f"  Max step size: α ≤ {max_step:.6f}")
                
                if z < 0.01:
                    print("  ⚠️  WARNING: Approaching numerical instability!")
                if z < 0.001:
                    print("  🚨 CRITICAL: Severe ill-conditioning!")
                    
            else:
                print("  💥 INFEASIBLE: Negative slack variable!")
                
            print()
    
    def kkt_conditioning_analysis(self):
        """
        Analyze KKT matrix conditioning as constraints become active.
        """
        print("6. KKT MATRIX CONDITIONING ANALYSIS")
        print("-" * 50)
        
        print("For the reformulated problem, the KKT matrix structure is:")
        print()
        print("┌─────────────────────────────────────────┐")
        print("│  0   0   1   1   0  │ A^T  I_n  0_m₂ │")
        print("│  0   0   1   0   1  │      0    0    │")  
        print("│  1   1   0   0   0  │      0    0    │")
        print("│  1   0   0   0   0  │      0    0    │")
        print("│  0   1   0   0   0  │      0    0    │")
        print("├─────────────────────┼────────────────┤")
        print("│  A   0   0   0   0  │      0    0    │")
        print("│  I   0   X   0   0  │      0    0    │")
        print("│  0  I_m₂ 0   0   Z  │      0    0    │")
        print("└─────────────────────────────────────────┘")
        print()
        print("Where:")
        print("• X = diag(x₁, x₂, ..., xₙ)")
        print("• Z = diag(z₁, z₂, ..., z_m₂)  <- PROBLEM HERE!")
        print("• As z_i → 0: Z becomes singular")
        
        # Demonstrate with actual matrices
        print("\nNumerical demonstration:")
        print("Consider Z = diag(z) for different z values:")
        
        z_values = [1.0, 0.1, 0.01, 0.001, 0.0001]
        
        for z in z_values:
            # Simple 2x2 block matrix to illustrate
            X = np.array([[1.0, 0.0], [0.0, 1.0]])  # Well-conditioned
            Z = np.array([[z]])  # Becomes ill-conditioned
            
            # Simplified KKT block
            KKT_block = np.block([[X, np.zeros((2, 1))],
                                  [np.zeros((1, 2)), Z]])
            
            cond_num = np.linalg.cond(KKT_block)
            print(f"z = {z:8.4f} → cond(KKT block) = {cond_num:.2e}")
    
    def mitigation_strategies(self):
        """
        Present strategies to mitigate active constraint issues.
        """
        print("\n7. MITIGATION STRATEGIES")
        print("-" * 50)
        
        print("🛠️  STRATEGY 1: REGULARIZATION")
        print("-" * 30)
        print("Add small regularization to prevent z → 0:")
        print()
        print("Modified barrier: φ(x,z) = -Σlog(x_i) - Σlog(z_j + ε)")
        print("where ε > 0 is a small regularization parameter")
        print()
        print("Pros: Prevents singularity, easy to implement")
        print("Cons: Introduces bias, may slow convergence")
        print("Typical choice: ε = 1e-8 to 1e-12")
        
        print("\n🛠️  STRATEGY 2: ADAPTIVE BARRIER PARAMETER")
        print("-" * 30)
        print("Adjust barrier parameter μ based on slack values:")
        print()
        print("if min(z_i) < threshold:")
        print("    μ = max(μ_min, β * μ)  # Reduce more slowly")
        print("else:")
        print("    μ = α * μ              # Normal reduction")
        print()
        print("Pros: Problem-adaptive, maintains convergence")
        print("Cons: More complex parameter tuning")
        
        print("\n🛠️  STRATEGY 3: CONSTRAINT DROPPING")
        print("-" * 30)
        print("Temporarily remove nearly-active constraints:")
        print()
        print("if z_i < ε_drop:")
        print("    # Fix x at boundary: F_i x = g_t[i]")
        print("    # Remove z_i from optimization")
        print("    # Solve reduced problem")
        print()
        print("Pros: Avoids ill-conditioning completely")
        print("Cons: Complex logic, may miss optimal solution")
        
        print("\n🛠️  STRATEGY 4: PREDICTOR-CORRECTOR METHODS")
        print("-" * 30)
        print("Use predictor-corrector approach:")
        print()
        print("1. Predictor: Solve with current μ")
        print("2. Corrector: Adjust to maintain z_i ≥ σμ")
        print("3. Centering: Apply Mehrotra's strategy")
        print()
        print("Pros: Well-established, robust")
        print("Cons: More complex implementation")
        
        print("\n🛠️  STRATEGY 5: WARM-START RECENTERING")
        print("-" * 30)
        print("For online problems, recenter when needed:")
        print()
        print("if g_t changes significantly:")
        print("    z_new = g_t - F x_old")
        print("    if min(z_new) <= 0:")
        print("        # Move to interior")
        print("        x_init = arg min ||x - x_old||")
        print("        s.t. F x ≤ g_t - δ")
        print()
        print("Pros: Maintains feasibility in online setting")
        print("Cons: Additional computational overhead")
    
    def implement_practical_example(self):
        """
        Implement a practical example showing mitigation strategies.
        """
        print("\n8. PRACTICAL IMPLEMENTATION EXAMPLE")
        print("-" * 50)
        
        print("Let's implement Strategy 1 (Regularization) for our example:")
        
        # Example data
        A = np.array([[1, 1, 0]])  # [x₁ + x₂ + 0*z = 3]
        F_ext = np.array([[1, 0, 1]])  # [x₁ + 0*x₂ + z = g_t]
        A_full = np.vstack([A, F_ext])
        c_full = np.array([1, 1, 0])  # [minimize x₁ + x₂ + 0*z]
        
        print("\nFormulated system:")
        print("minimize   [1, 1, 0] * [x₁, x₂, z]")
        print("subject to [1, 1, 0] * [x₁, x₂, z] = 3")
        print("           [1, 0, 1] * [x₁, x₂, z] = g_t")
        print("           x₁, x₂, z ≥ 0")
        
        # Simulate what happens with and without regularization
        g_t_values = [2.0, 1.5, 1.1, 1.01, 1.001]
        epsilon = 1e-6  # Regularization parameter
        
        print(f"\nComparison (with regularization ε = {epsilon}):")
        print("g_t    | z_exact | z_regularized | barrier_exact | barrier_reg")
        print("-------|---------|---------------|---------------|------------")
        
        for g_t in g_t_values:
            # Optimal solution: x₁ = min(1.5, g_t), x₂ = 3 - x₁
            x1_opt = min(1.5, g_t)
            x2_opt = 3 - x1_opt
            z_exact = g_t - x1_opt
            
            # Regularized version
            z_reg = max(z_exact, epsilon)
            
            # Barrier function values
            if z_exact > 0:
                barrier_exact = -np.log(z_exact)
            else:
                barrier_exact = float('inf')
                
            barrier_reg = -np.log(z_reg)
            
            print(f"{g_t:6.3f} | {z_exact:7.4f} | {z_reg:13.6f} | {barrier_exact:13.2f} | {barrier_reg:11.2f}")
        
        print("\nObservations:")
        print("• Regularization prevents infinite barrier values")
        print("• Small bias introduced when z_exact < ε")
        print("• Numerical stability maintained throughout")
        
    def online_algorithm_recommendations(self):
        """
        Specific recommendations for online IPM algorithms.
        """
        print("\n9. ONLINE IPM ALGORITHM RECOMMENDATIONS")
        print("-" * 50)
        
        print("For online algorithms with time-varying inequalities:")
        print()
        
        print("📋 ALGORITHM TEMPLATE:")
        print("─" * 25)
        print("""
def online_ipm_with_inequalities(A, F, c, b_sequence, g_sequence):
    # Setup
    n, m1 = A.shape
    m2 = F.shape[0]
    A_full = [[A, 0], [F, I]]  # Augmented constraint matrix
    c_full = [c, zeros(m2)]    # Augmented cost vector
    
    # Initialize
    x, z, y, s = find_initial_interior_point(A_full, [b[0], g[0]], c_full)
    
    for t in range(T):
        b_t = [b_sequence[t], g_sequence[t]]
        
        # Check slack variables
        if min(z) < threshold:
            # Apply mitigation strategy
            x, z, y, s = handle_active_constraints(x, z, y, s, b_t)
        
        # Solve IPM step
        x, z, y, s = ipm_newton_step(A_full, b_t, c_full, x, z, y, s)
        
        # Extract original solution
        solution[t] = x[:n]  # Ignore slack variables z
    
    return solution
        """)
        
        print("\n🔧 KEY IMPLEMENTATION DETAILS:")
        print("─" * 35)
        
        print("\n1. Threshold Selection:")
        print("   threshold = max(1e-8, μ/100)")
        print("   • Adaptive based on current barrier parameter")
        print("   • Prevents premature triggering")
        
        print("\n2. Active Constraint Detection:")
        print("   active_mask = (z < threshold)")
        print("   • Boolean mask for easy indexing")
        print("   • Monitor which constraints are problematic")
        
        print("\n3. Recentering Strategy:")
        print("   if any(z_new <= 0):  # After b_t update")
        print("       x = recenter_to_interior(x, A_full, b_t)")
        print("       z = g_t - F @ x")
        
        print("\n4. Regularization Implementation:")
        print("   barrier_value = -sum(log(maximum(z, epsilon)))")
        print("   • Use maximum() to avoid log(0)")
        print("   • Maintains differentiability")
        
        print("\n5. Step Size Control:")
        print("   alpha = min(alpha_primal, alpha_dual)")
        print("   alpha_primal = 0.99 * min(-z[i]/dz[i] for dz[i] < 0)")
        print("   • Ensures z + α*dz ≥ 0.01*z")
        
    def run_complete_analysis(self):
        """
        Run the complete analysis of active constraint issues.
        """
        # Problem explanation
        self.explain_the_problem()
        
        # Concrete example
        example_data = self.create_example_problem()
        
        # Demonstrate issues
        self.demonstrate_numerical_issues(example_data)
        
        # Matrix conditioning
        self.kkt_conditioning_analysis()
        
        # Solutions
        self.mitigation_strategies()
        
        # Practical implementation
        self.implement_practical_example()
        
        # Online algorithm recommendations
        self.online_algorithm_recommendations()
        
        print("\n" + "=" * 80)
        print("SUMMARY: HANDLING ACTIVE CONSTRAINTS")
        print("=" * 80)
        
        print("\n🎯 THE CORE ISSUE:")
        print("   When Fx ≤ g_t becomes tight → slack z ≈ 0 → numerical instability")
        
        print("\n🛡️  RECOMMENDED SOLUTION (Simple & Effective):")
        print("   Use regularization: barrier = -Σlog(max(z_i, ε))")
        print("   with ε = 1e-8 to 1e-10")
        
        print("\n⚡ FOR ONLINE ALGORITHMS:")
        print("   1. Monitor min(z) at each time step")  
        print("   2. Recenter when g_t changes significantly")
        print("   3. Use adaptive barrier parameter reduction")
        print("   4. Implement robust step size control")
        
        print("\n✅ PRACTICAL IMPACT:")
        print("   These strategies maintain numerical stability while preserving")
        print("   the theoretical guarantees of your online optimization algorithm.")

if __name__ == "__main__":
    analyzer = ActiveConstraintAnalyzer()
    analyzer.run_complete_analysis()