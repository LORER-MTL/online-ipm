"""
Fixed 2x2 KKT System for Online Interior Point Methods
=====================================================

This module provides a corrected implementation of the 2x2 KKT system 
formulation for linear programming with proper numerical stability
and correct mathematical formulation.

Standard form LP:
minimize    c^T x
subject to  A x = b
            x >= 0
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict, Any
import time

class ImprovedKKT2x2Solver:
    """
    Numerically stable 2x2 KKT solver for interior point methods.
    """
    
    def __init__(self, A: np.ndarray):
        """Initialize solver with constraint matrix."""
        self.A = csc_matrix(A) if not sp.issparse(A) else A.tocsc()
        self.m, self.n = self.A.shape
        
    def solve_newton_step(self, x: np.ndarray, s: np.ndarray, y: np.ndarray,
                         c: np.ndarray, b: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Newton step using numerically stable 2x2 elimination.
        
        The KKT system is:
        [0   A^T  I ] [dx]   [-rd]
        [A   0    0 ] [dy] = [-rp]  
        [S   0    X ] [ds]   [-rc]
        
        Where:
        rd = A^T y + s - c      (dual residual)
        rp = A x - b            (primal residual)  
        rc = X S e - mu e       (centrality residual)
        """
        # Compute residuals (with correct signs)
        rd = self.A.T @ y + s - c
        rp = self.A @ x - b
        rc = x * s - mu
        
        # Ensure numerical stability
        x_safe = np.maximum(x, 1e-12)
        s_safe = np.maximum(s, 1e-12)
        
        # Form reduced system by eliminating ds
        # From third equation: ds = -X^{-1}(S dx + rc)
        # Substitute into first: A^T dy + ds = -rd
        # A^T dy - X^{-1}S dx = -rd - X^{-1}rc
        
        # System matrix blocks
        D = diags(s_safe / x_safe, format='csc')  # X^{-1}S
        AT = self.A.T
        A = self.A
        zeros = csc_matrix((self.m, self.m))
        
        # Assemble system: [D    A^T] [dx] = [rhs1]
        #                  [A    0  ] [dy]   [rhs2]
        K = vstack([hstack([D, AT]), hstack([A, zeros])])
        
        # Right-hand side
        rhs1 = -rd - rc / x_safe  # -rd - X^{-1}rc
        rhs2 = -rp
        rhs = np.concatenate([rhs1, rhs2])
        
        # Solve system
        try:
            solution = spsolve(K, rhs)
        except Exception as e:
            print(f"Linear solve failed: {e}")
            # Return zero step as fallback
            return np.zeros(self.n), np.zeros(self.m), np.zeros(self.n)
        
        # Extract solution
        dx = solution[:self.n]
        dy = solution[self.n:]
        
        # Recover ds = -X^{-1}(S dx + rc)
        ds = -(s_safe * dx + rc) / x_safe
        
        return dx, dy, ds

class ImprovedOnlineIPM:
    """
    Improved online IPM with better numerical stability.
    """
    
    def __init__(self, A: np.ndarray, c: np.ndarray):
        """Initialize with problem data."""
        self.A = A
        self.c = c
        self.m, self.n = A.shape
        self.solver = ImprovedKKT2x2Solver(A)
        
        # State variables
        self.x = None
        self.y = None
        self.s = None
        self.mu = None
        
    def find_initial_point(self, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find initial feasible point using a more robust method.
        """
        # Phase I: Find feasible x such that Ax = b, x > 0
        try:
            # Try least squares solution
            x0 = np.linalg.lstsq(self.A, b, rcond=None)[0]
            
            # If any components are negative, shift to make positive
            if np.any(x0 <= 0):
                x0 = x0 - np.min(x0) + 1.0
                
            # Project back to feasibility: Ax = b
            if self.m < self.n:  # Underdetermined system
                # Find correction in null space
                residual = b - self.A @ x0
                if np.linalg.norm(residual) > 1e-10:
                    # Use pseudoinverse to correct
                    correction = np.linalg.pinv(self.A) @ residual
                    x0 = x0 + correction
                    
            # Ensure strict positivity
            x0 = np.maximum(x0, 1.0)
            
        except Exception:
            # Fallback: use all-ones vector and project
            x0 = np.ones(self.n)
            try:
                residual = b - self.A @ x0
                correction = np.linalg.pinv(self.A) @ residual
                x0 = x0 + correction
                x0 = np.maximum(x0, 1.0)
            except:
                pass
        
        # Initialize dual variables
        # Solve min ||A^T y - c||^2 for reasonable dual initialization
        try:
            y0 = np.linalg.lstsq(self.A.T, self.c, rcond=None)[0]
        except:
            y0 = np.zeros(self.m)
            
        # Dual slack: s = c - A^T y, ensure s > 0
        s0 = self.c - self.A.T @ y0
        s0 = np.maximum(s0, 1.0)
        
        return x0, y0, s0
    
    def compute_step_sizes(self, x: np.ndarray, s: np.ndarray, 
                          dx: np.ndarray, ds: np.ndarray) -> Tuple[float, float]:
        """Compute safe step sizes maintaining positivity."""
        
        # Maximum step size to boundary
        alpha_p_max = 1.0
        alpha_d_max = 1.0
        
        # Primal step size
        negative_dx = dx < 0
        if np.any(negative_dx):
            ratios = x[negative_dx] / (-dx[negative_dx])
            alpha_p_max = np.min(ratios)
            
        # Dual step size  
        negative_ds = ds < 0
        if np.any(negative_ds):
            ratios = s[negative_ds] / (-ds[negative_ds])
            alpha_d_max = np.min(ratios)
        
        # Apply safety factor and damping
        tau = 0.995  # Stay away from boundary
        alpha_p = min(tau * alpha_p_max, 0.9)  # Additional damping
        alpha_d = min(tau * alpha_d_max, 0.9)
        
        return alpha_p, alpha_d
    
    def solve_for_b(self, b: np.ndarray, max_iter: int = 50, 
                    tol: float = 1e-6, verbose: bool = True) -> Dict[str, Any]:
        """
        Solve LP for given b with improved algorithm.
        """
        start_time = time.time()
        
        # Initialize if needed
        if self.x is None:
            self.x, self.y, self.s = self.find_initial_point(b)
            self.mu = np.dot(self.x, self.s) / self.n
        
        # Verify initial feasibility
        prim_feas = np.linalg.norm(self.A @ self.x - b)
        if prim_feas > 1e-8:
            if verbose:
                print(f"Warning: Initial point not feasible, residual = {prim_feas:.2e}")
            # Try to repair
            self.x, self.y, self.s = self.find_initial_point(b)
        
        for iteration in range(max_iter):
            # Current residuals
            gap = np.dot(self.x, self.s)
            prim_res = np.linalg.norm(self.A @ self.x - b)
            dual_res = np.linalg.norm(self.A.T @ self.y + self.s - self.c)
            
            if verbose and iteration % 5 == 0:
                print(f"Iter {iteration:2d}: gap={gap:.2e}, prim={prim_res:.2e}, dual={dual_res:.2e}")
            
            # Check convergence
            if gap < tol and prim_res < tol and dual_res < tol:
                solve_time = time.time() - start_time
                return {
                    'x': self.x.copy(),
                    'y': self.y.copy(),
                    's': self.s.copy(),
                    'iterations': iteration,
                    'gap': gap,
                    'prim_res': prim_res,
                    'dual_res': dual_res,
                    'solve_time': solve_time,
                    'converged': True
                }
            
            # Update barrier parameter
            sigma = 0.1  # Centering parameter
            self.mu = sigma * gap / self.n
            self.mu = max(self.mu, 1e-12)  # Prevent mu from becoming too small
            
            # Solve Newton step
            dx, dy, ds = self.solver.solve_newton_step(
                self.x, self.s, self.y, self.c, b, self.mu)
            
            # Check for numerical issues
            if (np.any(np.isnan(dx)) or np.any(np.isnan(dy)) or np.any(np.isnan(ds)) or
                np.linalg.norm(dx) > 1e6 or np.linalg.norm(dy) > 1e6 or np.linalg.norm(ds) > 1e6):
                if verbose:
                    print(f"Numerical issues detected at iteration {iteration}")
                break
            
            # Compute step sizes
            alpha_p, alpha_d = self.compute_step_sizes(self.x, self.s, dx, ds)
            
            # Update variables
            self.x = self.x + alpha_p * dx
            self.y = self.y + alpha_d * dy  
            self.s = self.s + alpha_d * ds
            
            # Safety clamp to maintain positivity
            self.x = np.maximum(self.x, 1e-12)
            self.s = np.maximum(self.s, 1e-12)
        
        # Did not converge
        solve_time = time.time() - start_time
        gap = np.dot(self.x, self.s)
        prim_res = np.linalg.norm(self.A @ self.x - b)
        dual_res = np.linalg.norm(self.A.T @ self.y + self.s - self.c)
        
        return {
            'x': self.x.copy(),
            'y': self.y.copy(),
            's': self.s.copy(),
            'iterations': max_iter,
            'gap': gap,
            'prim_res': prim_res,
            'dual_res': dual_res,
            'solve_time': solve_time,
            'converged': False
        }

def demo_improved_system():
    """Demo the improved 2x2 KKT system."""
    print("=" * 70)
    print("IMPROVED 2x2 KKT SYSTEM FOR ONLINE INTERIOR POINT METHODS")
    print("=" * 70)
    
    # Create a well-conditioned test problem
    np.random.seed(42)
    n, m = 8, 4
    
    # Generate well-conditioned constraint matrix
    A = np.random.randn(m, n) 
    A = A / np.linalg.norm(A, axis=1, keepdims=True)  # Normalize rows
    
    # Create cost vector
    c = np.random.randn(n)
    
    print(f"Problem: minimize c^T x subject to A x = b, x >= 0")
    print(f"Dimensions: {n} variables, {m} constraints")
    print(f"Condition number of A: {np.linalg.cond(A @ A.T):.2f}")
    
    # Initialize solver
    solver = ImprovedOnlineIPM(A, c)
    
    # Test sequence of problems
    num_steps = 5
    results = []
    
    print(f"\nSolving {num_steps} sequential problems...")
    
    for t in range(num_steps):
        print(f"\n--- Problem {t+1} ---")
        
        # Generate feasible b
        if t == 0:
            # Initial problem
            x_feas = np.random.rand(n) + 0.1
            b = A @ x_feas
        else:
            # Perturb previous b
            delta_b = 0.2 * np.random.randn(m)
            b = results[-1]['b'] + delta_b
        
        print(f"Right-hand side b = {b}")
        
        # Solve
        result = solver.solve_for_b(b, max_iter=30, verbose=False)
        result['b'] = b.copy()  # Store b for next iteration
        results.append(result)
        
        # Print results
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Solve time: {result['solve_time']:.4f}s")
        print(f"Final gap: {result['gap']:.2e}")
        print(f"Primal residual: {result['prim_res']:.2e}")
        print(f"Dual residual: {result['dual_res']:.2e}")
        print(f"Objective value: {np.dot(c, result['x']):.6f}")
        
        # Verify solution quality
        x_opt = result['x']
        print(f"Min x component: {np.min(x_opt):.2e}")
        print(f"Constraint violation: {np.linalg.norm(A @ x_opt - b):.2e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    converged = [r['converged'] for r in results]
    avg_iter = np.mean([r['iterations'] for r in results])
    avg_time = np.mean([r['solve_time'] for r in results])
    
    print(f"Convergence rate: {np.sum(converged)}/{len(results)}")
    print(f"Average iterations: {avg_iter:.1f}")
    print(f"Average solve time: {avg_time:.4f}s")
    
    if np.sum(converged) > 0:
        final_gaps = [r['gap'] for r in results if r['converged']]
        final_prim = [r['prim_res'] for r in results if r['converged']]
        final_dual = [r['dual_res'] for r in results if r['converged']]
        
        print(f"Final gap (converged): {np.mean(final_gaps):.2e}")
        print(f"Final primal residual: {np.mean(final_prim):.2e}")  
        print(f"Final dual residual: {np.mean(final_dual):.2e}")

if __name__ == "__main__":
    demo_improved_system()
