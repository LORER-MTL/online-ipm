"""
Manual 2x2 KKT System Solver for Time-Varying Problems
=====================================================

This module provides a clean interface for manually solving 2x2 KKT systems
with time-varying right-hand side vectors. It's designed for scenarios where
you want direct control over the solving process.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve
import qdldl
from typing import Tuple, Optional, Dict, Any

class Manual2x2KKTSolver:
    """
    Manual solver for 2x2 KKT systems with time-varying b vectors.
    
    This class provides direct access to form and solve the reduced KKT system:
    [-X^{-1}S  A^T] [dx]   [rd + X^{-1}rc]
    [A         0  ] [dy] = [rp           ]
    
    Where:
    - X = diag(x), S = diag(s)
    - rd = A^T y + s - c (dual residual)
    - rp = A x - b (primal residual)
    - rc = X S e - μ e (complementarity residual)
    """
    
    def __init__(self, A: np.ndarray, c: np.ndarray):
        """
        Initialize the manual KKT solver.
        
        Args:
            A: Constraint matrix (m x n) - fixed
            c: Cost vector (n,) - fixed
        """
        self.A = csc_matrix(A) if not sp.issparse(A) else A.tocsc()
        self.c = np.array(c)
        self.m, self.n = self.A.shape
        
        print(f"Manual 2x2 KKT Solver initialized:")
        print(f"  Problem size: {self.n} variables, {self.m} constraints")
        print(f"  Constraint matrix A: {self.A.shape}")
        print(f"  Cost vector c: {self.c.shape}")
    
    def compute_residuals(self, x: np.ndarray, s: np.ndarray, y: np.ndarray, 
                         b: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute KKT residuals for current iterate.
        
        Args:
            x: Primal variables (n,)
            s: Dual slack variables (n,)
            y: Dual variables (m,)
            b: Right-hand side vector (m,) - can be time-varying
            mu: Barrier parameter
            
        Returns:
            rd: Dual residual (n,)
            rp: Primal residual (m,)
            rc: Complementarity residual (n,)
        """
        rd = self.A.T @ y + s - self.c  # Dual feasibility residual
        rp = self.A @ x - b             # Primal feasibility residual  
        rc = s * x - mu                 # Complementarity residual
        
        return rd, rp, rc
    
    def form_2x2_kkt_matrix(self, x: np.ndarray, s: np.ndarray) -> sp.csc_matrix:
        """
        Form the 2x2 KKT matrix.
        
        [-X^{-1}S  A^T]
        [A         0  ]
        
        Args:
            x: Current primal variables (n,)
            s: Current dual slack variables (n,)
            
        Returns:
            K2x2: 2x2 KKT matrix ((n+m) x (n+m))
        """
        # Check for positive variables
        if np.any(x <= 0) or np.any(s <= 0):
            print("Warning: Non-positive variables detected")
            x = np.maximum(x, 1e-8)
            s = np.maximum(s, 1e-8)
        
        # Form blocks
        XinvS = -diags(s / x, format='csc')  # -X^{-1}S
        AT = self.A.T                         # A^T
        A = self.A                           # A
        zeros_mm = csc_matrix((self.m, self.m))  # 0
        
        # Assemble 2x2 matrix
        top_row = hstack([XinvS, AT])
        bottom_row = hstack([A, zeros_mm])
        K2x2 = vstack([top_row, bottom_row])
        
        return K2x2
    
    def form_2x2_rhs(self, x: np.ndarray, rd: np.ndarray, rp: np.ndarray, rc: np.ndarray) -> np.ndarray:
        """
        Form the 2x2 right-hand side vector.
        
        [rd + X^{-1}rc]
        [rp           ]
        
        Args:
            x: Current primal variables (n,)
            rd: Dual residual (n,)
            rp: Primal residual (m,)
            rc: Complementarity residual (n,)
            
        Returns:
            rhs2x2: Modified right-hand side ((n+m),)
        """
        # Check for positive variables
        x_safe = np.maximum(x, 1e-8)
        
        rhs_top = rd + rc / x_safe  # rd + X^{-1}rc
        rhs_bottom = rp             # rp
        
        return np.concatenate([rhs_top, rhs_bottom])
    
    def solve_2x2_system(self, K2x2: sp.csc_matrix, rhs2x2: np.ndarray, 
                        method: str = 'qdldl') -> np.ndarray:
        """
        Solve the 2x2 KKT system.
        
        Args:
            K2x2: 2x2 KKT matrix
            rhs2x2: Right-hand side vector
            method: 'qdldl' or 'scipy'
            
        Returns:
            solution: [dx, dy] where dx is (n,) and dy is (m,)
        """
        if method == 'qdldl':
            try:
                solver = qdldl.Solver(K2x2)
                solution = solver.solve(rhs2x2)
                return solution
            except Exception as e:
                print(f"QDLDL failed: {e}")
                print("Falling back to scipy solver")
                method = 'scipy'
        
        if method == 'scipy':
            solution = spsolve(K2x2, rhs2x2)
            return solution
        
        raise ValueError(f"Unknown method: {method}")
    
    def recover_dual_slack_step(self, dx: np.ndarray, x: np.ndarray, 
                               s: np.ndarray, rc: np.ndarray) -> np.ndarray:
        """
        Recover dual slack step from 2x2 solution.
        
        ds = -X^{-1}(S dx + rc)
        
        Args:
            dx: Primal step from 2x2 system (n,)
            x: Current primal variables (n,)
            s: Current dual slack variables (n,)
            rc: Complementarity residual (n,)
            
        Returns:
            ds: Dual slack step (n,)
        """
        x_safe = np.maximum(x, 1e-8)
        return -(s * dx + rc) / x_safe
    
    def solve_newton_step(self, x: np.ndarray, s: np.ndarray, y: np.ndarray,
                         b: np.ndarray, mu: float, method: str = 'qdldl') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete Newton step solution using 2x2 system.
        
        Args:
            x: Current primal variables (n,)
            s: Current dual slack variables (n,)
            y: Current dual variables (m,)
            b: Right-hand side vector (m,) - time-varying
            mu: Barrier parameter
            method: Solver method
            
        Returns:
            dx: Primal step (n,)
            dy: Dual step (m,)
            ds: Dual slack step (n,)
        """
        # Step 1: Compute residuals
        rd, rp, rc = self.compute_residuals(x, s, y, b, mu)
        
        # Step 2: Form 2x2 KKT matrix
        K2x2 = self.form_2x2_kkt_matrix(x, s)
        
        # Step 3: Form 2x2 RHS
        rhs2x2 = self.form_2x2_rhs(x, rd, rp, rc)
        
        # Step 4: Solve 2x2 system
        solution = self.solve_2x2_system(K2x2, rhs2x2, method)
        
        # Step 5: Extract and recover steps
        dx = solution[:self.n]
        dy = solution[self.n:]
        ds = self.recover_dual_slack_step(dx, x, s, rc)
        
        return dx, dy, ds
    
    def compute_step_sizes(self, x: np.ndarray, s: np.ndarray, 
                          dx: np.ndarray, ds: np.ndarray, tau: float = 0.99) -> Tuple[float, float]:
        """
        Compute step sizes to maintain positivity.
        
        Args:
            x: Current primal variables
            s: Current dual slack variables
            dx: Primal step
            ds: Dual slack step
            tau: Safety factor (< 1)
            
        Returns:
            alpha_p: Primal step size
            alpha_d: Dual step size
        """
        # Primal step size
        alpha_p = 1.0
        neg_indices = dx < 0
        if np.any(neg_indices):
            ratios = -x[neg_indices] / dx[neg_indices]
            alpha_p = min(tau * np.min(ratios), 1.0)
        
        # Dual step size
        alpha_d = 1.0
        neg_indices = ds < 0
        if np.any(neg_indices):
            ratios = -s[neg_indices] / ds[neg_indices]
            alpha_d = min(tau * np.min(ratios), 1.0)
        
        return alpha_p, alpha_d
    
    def display_system_info(self, x: np.ndarray, s: np.ndarray, b: np.ndarray, mu: float) -> None:
        """
        Display information about the current 2x2 system.
        """
        print(f"\n--- 2x2 KKT System Information ---")
        print(f"Current b: {b}")
        print(f"Barrier parameter μ: {mu}")
        print(f"Current x (first 5): {x[:5]}")
        print(f"Current s (first 5): {s[:5]}")
        
        # Compute and display residuals
        rd, rp, rc = self.compute_residuals(x, s, np.zeros(self.m), b, mu)
        print(f"Residual norms:")
        print(f"  ||rd||: {np.linalg.norm(rd):.2e}")
        print(f"  ||rp||: {np.linalg.norm(rp):.2e}")
        print(f"  ||rc||: {np.linalg.norm(rc):.2e}")
        
        # Matrix info
        K2x2 = self.form_2x2_kkt_matrix(x, s)
        print(f"2x2 matrix size: {K2x2.shape}")
        print(f"2x2 matrix density: {K2x2.nnz / (K2x2.shape[0] * K2x2.shape[1]):.3f}")

def demo_manual_solver():
    """
    Demonstrate manual solving of 2x2 KKT systems with time-varying b.
    """
    print("=" * 60)
    print("MANUAL 2x2 KKT SOLVER DEMONSTRATION")
    print("=" * 60)
    
    # Setup problem
    np.random.seed(123)
    n, m = 6, 3
    A = np.random.randn(m, n)
    c = np.random.randn(n)
    
    # Initialize solver
    solver = Manual2x2KKTSolver(A, c)
    
    # Initial feasible point
    x = np.random.rand(n) + 0.5
    s = np.random.rand(n) + 0.5  
    y = np.random.randn(m)
    
    print(f"\nInitial point:")
    print(f"  x: {x}")
    print(f"  s: {s}")
    print(f"  y: {y}")
    
    # Time-varying scenarios
    b_scenarios = [
        np.array([1.0, 2.0, 0.5]),      # Scenario 1
        np.array([1.5, 1.8, 0.7]),      # Scenario 2  
        np.array([0.8, 2.2, 0.3]),      # Scenario 3
    ]
    
    mu = 0.1
    
    for i, b in enumerate(b_scenarios):
        print(f"\n{'='*40}")
        print(f"SCENARIO {i+1}: Time-varying b = {b}")
        print(f"{'='*40}")
        
        # Display system info
        solver.display_system_info(x, s, b, mu)
        
        # Manual step-by-step solving
        print(f"\n--- Manual Step-by-Step Solution ---")
        
        # Step 1: Compute residuals manually
        rd, rp, rc = solver.compute_residuals(x, s, y, b, mu)
        print(f"1. Residuals computed:")
        print(f"   rd = {rd}")
        print(f"   rp = {rp}") 
        print(f"   rc = {rc}")
        
        # Step 2: Form 2x2 matrix manually
        K2x2 = solver.form_2x2_kkt_matrix(x, s)
        print(f"2. 2x2 KKT matrix formed: {K2x2.shape}")
        
        # Step 3: Form RHS manually  
        rhs2x2 = solver.form_2x2_rhs(x, rd, rp, rc)
        print(f"3. 2x2 RHS formed: {rhs2x2}")
        
        # Step 4: Solve manually
        solution = solver.solve_2x2_system(K2x2, rhs2x2)
        dx = solution[:n]
        dy = solution[n:]
        ds = solver.recover_dual_slack_step(dx, x, s, rc)
        
        print(f"4. Solution obtained:")
        print(f"   dx = {dx}")
        print(f"   dy = {dy}")
        print(f"   ds = {ds}")
        
        # Step 5: Compute step sizes
        alpha_p, alpha_d = solver.compute_step_sizes(x, s, dx, ds)
        print(f"5. Step sizes: α_p = {alpha_p:.3f}, α_d = {alpha_d:.3f}")
        
        # Step 6: Update variables
        x_new = x + alpha_p * dx
        y_new = y + alpha_d * dy
        s_new = s + alpha_d * ds
        
        print(f"6. Updated variables:")
        print(f"   x_new = {x_new}")
        print(f"   y_new = {y_new}")
        print(f"   s_new = {s_new}")
        
        # Verify solution quality
        rd_new, rp_new, rc_new = solver.compute_residuals(x_new, s_new, y_new, b, mu)
        print(f"7. New residual norms:")
        print(f"   ||rd_new||: {np.linalg.norm(rd_new):.2e}")
        print(f"   ||rp_new||: {np.linalg.norm(rp_new):.2e}")
        print(f"   ||rc_new||: {np.linalg.norm(rc_new):.2e}")
        
        # Update for next scenario
        x, y, s = x_new, y_new, s_new
        mu *= 0.5  # Reduce barrier parameter
    
    print(f"\n{'='*60}")
    print("Manual solving demonstration completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    demo_manual_solver()