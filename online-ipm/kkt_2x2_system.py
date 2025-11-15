"""
2x2 KKT System for Online Interior Point Methods
===============================================

This module implements a 2x2 KKT system formulation for linear programming
with time-varying right-hand side vectors. The approach eliminates the dual
slack variables to obtain a reduced system that can be efficiently solved
for online optimization scenarios.

The standard 3x3 KKT system:
[0   A^T  I ] [dx]   [rd]
[A   0    0 ] [dy] = [rp]
[S   0    X ] [ds]   [rc]

Is reduced to a 2x2 system by eliminating ds:
[A X A^T] [dy]   [rp_mod]
[0   I  ] [dx] = [rd_mod]

Where the elimination uses: ds = -X^{-1}(S dx + rc)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve
import qdldl
from typing import Tuple, Dict, Any
import time

class KKT2x2Solver:
    """
    Solver for 2x2 KKT systems in online interior point methods.
    
    This class implements the reduced KKT system approach where the dual slack
    variables are eliminated to obtain a smaller system that's more efficient
    for warm-starting and online updates.
    """
    
    def __init__(self, A: np.ndarray, factorization_method: str = 'qdldl'):
        """
        Initialize the 2x2 KKT solver.
        
        Args:
            A: Constraint matrix (m x n)
            factorization_method: 'qdldl', 'ldl', or 'lu'
        """
        self.A = csc_matrix(A) if not sp.issparse(A) else A.tocsc()
        self.m, self.n = self.A.shape
        self.factorization_method = factorization_method
        
        # Cache for factorizations
        self._cached_solver = None
        self._cached_x = None
        self._cached_s = None
        
    def form_2x2_system(self, x: np.ndarray, s: np.ndarray, 
                        rd: np.ndarray, rp: np.ndarray, rc: np.ndarray) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Form the 2x2 KKT system by eliminating dual slack variables.
        
        From the 3x3 system:
        [0   A^T  I ] [dx]   [rd]
        [A   0    0 ] [dy] = [rp]
        [S   0    X ] [ds]   [rc]
        
        We eliminate ds using: ds = -X^{-1}(S dx + rc)
        
        Substituting into the first equation:
        A^T dy + ds = rd
        A^T dy + (-X^{-1}(S dx + rc)) = rd
        A^T dy - X^{-1}S dx = rd + X^{-1}rc
        
        This gives the 2x2 system:
        [-X^{-1}S  A^T] [dx]   [rd + X^{-1}rc]
        [A         0  ] [dy] = [rp           ]
        
        Args:
            x: Current primal variables (n,)
            s: Current dual slack variables (n,)
            rd: Dual residual (n,)
            rp: Primal residual (m,)
            rc: Complementarity residual (n,)
            
        Returns:
            K2x2: 2x2 KKT matrix
            rhs2x2: Modified right-hand side
        """
        # Ensure numerical stability
        x_safe = np.maximum(np.abs(x), 1e-12)
        s_safe = np.maximum(np.abs(s), 1e-12)
        
        # Form the 2x2 blocks
        # Top-left: -X^{-1}S = -diag(s/x)
        XinvS = -diags(s_safe / x_safe, format='csc')
        
        # Top-right: A^T
        AT = self.A.T
        
        # Bottom-left: A
        A = self.A
        
        # Bottom-right: 0
        zeros_mm = csc_matrix((self.m, self.m))
        
        # Assemble 2x2 system
        top_row = hstack([XinvS, AT])
        bottom_row = hstack([A, zeros_mm])
        K2x2 = vstack([top_row, bottom_row])
        
        # Modified right-hand side with numerical stability
        rhs_top = rd + (rc / x_safe)  # rd + X^{-1}rc
        rhs_bottom = rp
        rhs2x2 = np.concatenate([rhs_top, rhs_bottom])
        
        return K2x2, rhs2x2
    
    def solve_2x2_system(self, K2x2: sp.csc_matrix, rhs2x2: np.ndarray) -> np.ndarray:
        """
        Solve the 2x2 KKT system.
        
        Args:
            K2x2: 2x2 KKT matrix
            rhs2x2: Right-hand side vector
            
        Returns:
            solution: [dx, dy] vector
        """
        if self.factorization_method == 'qdldl':
            try:
                solver = qdldl.Solver(K2x2)
                return solver.solve(rhs2x2)
            except Exception as e:
                print(f"QDLDL failed: {e}, using scipy fallback")
                return spsolve(K2x2, rhs2x2)
        else:
            return spsolve(K2x2, rhs2x2)
    
    def recover_dual_slack(self, dx: np.ndarray, x: np.ndarray, s: np.ndarray, rc: np.ndarray) -> np.ndarray:
        """
        Recover the dual slack step from the eliminated system.
        
        ds = -X^{-1}(S dx + rc)
        
        Args:
            dx: Primal step from 2x2 system
            x: Current primal variables
            s: Current dual slack variables
            rc: Complementarity residual
            
        Returns:
            ds: Dual slack step
        """
        x_safe = np.maximum(np.abs(x), 1e-12)
        return -(s * dx + rc) / x_safe
    
    def solve_step(self, x: np.ndarray, s: np.ndarray, y: np.ndarray, 
                   c: np.ndarray, b: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve for Newton step using 2x2 formulation.
        
        Args:
            x: Current primal variables
            s: Current dual slack variables  
            y: Current dual variables
            c: Cost vector
            b: Right-hand side (can be time-varying)
            mu: Barrier parameter
            
        Returns:
            dx, dy, ds: Step directions
        """
        # Compute residuals
        rd = self.A.T @ y + s - c
        rp = self.A @ x - b
        rc = s * x - mu
        
        # Form and solve 2x2 system
        K2x2, rhs2x2 = self.form_2x2_system(x, s, rd, rp, rc)
        solution = self.solve_2x2_system(K2x2, rhs2x2)
        
        # Extract solutions
        dx = solution[:self.n]
        dy = solution[self.n:]
        
        # Recover dual slack step
        ds = self.recover_dual_slack(dx, x, s, rc)
        
        return dx, dy, ds

class OnlineIPM:
    """
    Online Interior Point Method with 2x2 KKT system for time-varying b.
    """
    
    def __init__(self, A: np.ndarray, c: np.ndarray, factorization_method: str = 'qdldl'):
        """
        Initialize online IPM.
        
        Args:
            A: Constraint matrix (fixed)
            c: Cost vector (fixed)
            factorization_method: Solver method for KKT system
        """
        self.A = A
        self.c = c
        self.m, self.n = A.shape
        self.kkt_solver = KKT2x2Solver(A, factorization_method)
        
        # Current state
        self.x = None
        self.y = None
        self.s = None
        self.mu = 1.0
        
    def initialize(self, b: np.ndarray, x0: np.ndarray = None) -> None:
        """
        Initialize the method with starting point.
        
        Args:
            b: Initial right-hand side
            x0: Initial primal point (if None, computed automatically)
        """
        if x0 is None:
            # Find feasible starting point
            if self.m > 0:
                # Use a more robust initialization
                try:
                    x0 = np.linalg.lstsq(self.A, b, rcond=None)[0]
                    x0 = np.maximum(x0, 1.0)  # Larger initial values for stability
                except:
                    x0 = np.ones(self.n)
                
                # Ensure strict feasibility
                residual = b - self.A @ x0
                if np.linalg.norm(residual) > 1e-10:
                    try:
                        AtA_inv = np.linalg.pinv(self.A @ self.A.T)
                        correction = self.A.T @ AtA_inv @ residual
                        x0 = x0 + correction
                    except:
                        pass
                x0 = np.maximum(x0, 1.0)
            else:
                x0 = np.ones(self.n)
        
        self.x = x0
        
        # Better dual initialization - solve least squares problem
        # min ||A^T y - c||^2 subject to y
        try:
            self.y = np.linalg.lstsq(self.A.T, self.c, rcond=None)[0]
        except:
            self.y = np.zeros(self.m)
        
        # Initialize slack variables to maintain complementarity
        self.s = np.maximum(self.c - self.A.T @ self.y, 1.0)
        
        # Reset barrier parameter
        self.mu = 1.0
        
    def solve_one_step(self, b: np.ndarray, max_inner_iter: int = 20, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Solve for new b using warm start from previous solution.
        
        Args:
            b: New right-hand side vector
            max_inner_iter: Maximum inner iterations
            tol: Convergence tolerance
            
        Returns:
            result: Dictionary with solution info
        """
        if self.x is None:
            self.initialize(b)
            
        start_time = time.time()
        
        # Adaptive barrier parameter update
        gap = np.dot(self.x, self.s)
        if gap < 1e-3:
            self.mu = max(gap * 0.1, 1e-8)
        else:
            self.mu = max(gap * 0.2, 1e-6)
        
        for iter_count in range(max_inner_iter):
            # Solve Newton step using 2x2 system
            dx, dy, ds = self.kkt_solver.solve_step(self.x, self.s, self.y, self.c, b, self.mu)
            
            # Compute step sizes with more conservative approach
            alpha_p = 1.0
            alpha_d = 1.0
            
            # Primal step size
            neg_indices = dx < 0
            if np.any(neg_indices):
                ratios = -self.x[neg_indices] / dx[neg_indices]
                alpha_p = min(0.95 * np.min(ratios), 1.0)
            
            # Dual step size  
            neg_indices = ds < 0
            if np.any(neg_indices):
                ratios = -self.s[neg_indices] / ds[neg_indices]
                alpha_d = min(0.95 * np.min(ratios), 1.0)
            
            # Additional damping for stability
            alpha_p = min(alpha_p, 0.9)
            alpha_d = min(alpha_d, 0.9)
            
            # Update variables
            x_new = self.x + alpha_p * dx
            y_new = self.y + alpha_d * dy
            s_new = self.s + alpha_d * ds
            
            # Ensure positivity
            x_new = np.maximum(x_new, 1e-10)
            s_new = np.maximum(s_new, 1e-10)
            
            # Check convergence
            gap = np.dot(x_new, s_new)
            prim_res = np.linalg.norm(self.A @ x_new - b)
            dual_res = np.linalg.norm(self.A.T @ y_new + s_new - self.c)
            
            if gap < tol and prim_res < tol and dual_res < tol:
                self.x, self.y, self.s = x_new, y_new, s_new
                solve_time = time.time() - start_time
                return {
                    'x': self.x.copy(),
                    'y': self.y.copy(), 
                    's': self.s.copy(),
                    'iterations': iter_count + 1,
                    'gap': gap,
                    'prim_res': prim_res,
                    'dual_res': dual_res,
                    'solve_time': solve_time,
                    'converged': True
                }
            
            # Update state
            self.x, self.y, self.s = x_new, y_new, s_new
            
            # More conservative barrier parameter update
            sigma = 0.1  # Centering parameter
            self.mu = sigma * gap / self.n
        
        solve_time = time.time() - start_time
        return {
            'x': self.x.copy(),
            'y': self.y.copy(),
            's': self.s.copy(), 
            'iterations': max_inner_iter,
            'gap': gap,
            'prim_res': prim_res,
            'dual_res': dual_res,
            'solve_time': solve_time,
            'converged': False
        }

def demo_online_2x2_system():
    """
    Demonstrate the 2x2 KKT system with time-varying right-hand side.
    """
    print("=" * 60)
    print("2x2 KKT SYSTEM FOR ONLINE INTERIOR POINT METHODS")
    print("=" * 60)
    
    # Problem setup
    np.random.seed(42)
    n, m = 8, 4
    A = np.random.randn(m, n)
    c = np.random.randn(n)
    
    print(f"Problem dimensions: {n} variables, {m} constraints")
    print(f"Constraint matrix A: {A.shape}")
    
    # Initialize online IPM
    online_ipm = OnlineIPM(A, c)
    
    # Simulate time-varying scenarios
    num_time_steps = 5
    b_sequence = []
    
    print(f"\nSimulating {num_time_steps} time steps with varying b...")
    
    results = []
    for t in range(num_time_steps):
        # Generate time-varying b
        if t == 0:
            # Initial feasible b
            x_feas = np.random.rand(n) + 0.1
            b_t = A @ x_feas
        else:
            # Modify previous b slightly
            delta_b = 0.1 * np.random.randn(m)
            b_t = b_sequence[-1] + delta_b
        
        b_sequence.append(b_t)
        
        print(f"\n--- Time Step {t+1} ---")
        print(f"b = {b_t}")
        
        # Solve using 2x2 system
        result = online_ipm.solve_one_step(b_t)
        results.append(result)
        
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Solve time: {result['solve_time']:.4f}s")
        print(f"Duality gap: {result['gap']:.2e}")
        print(f"Primal residual: {result['prim_res']:.2e}")
        print(f"Dual residual: {result['dual_res']:.2e}")
        print(f"Optimal value: {np.dot(c, result['x']):.6f}")
    
    # Performance summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    avg_iterations = np.mean([r['iterations'] for r in results])
    avg_time = np.mean([r['solve_time'] for r in results])
    converged_count = sum([r['converged'] for r in results])
    
    print(f"Average iterations per time step: {avg_iterations:.1f}")
    print(f"Average solve time per time step: {avg_time:.4f}s")
    print(f"Convergence rate: {converged_count}/{num_time_steps}")
    
    # Demonstrate warm-starting benefit
    print(f"\n{'='*60}")
    print("DEMONSTRATING 2x2 SYSTEM STRUCTURE")
    print(f"{'='*60}")
    
    # Show the system for last time step
    x, s, y = results[-1]['x'], results[-1]['s'], results[-1]['y']
    b = b_sequence[-1]
    mu = 0.01
    
    # Form 2x2 system
    kkt_solver = KKT2x2Solver(A)
    rd = A.T @ y + s - c
    rp = A @ x - b
    rc = s * x - mu
    
    K2x2, rhs2x2 = kkt_solver.form_2x2_system(x, s, rd, rp, rc)
    
    print(f"Original 3x3 KKT system size: {m+n+n} x {m+n+n}")
    print(f"Reduced 2x2 KKT system size: {n+m} x {n+m}")
    print(f"Size reduction: {((m+n+n)**2 - (n+m)**2) / (m+n+n)**2 * 100:.1f}%")
    print(f"2x2 KKT matrix density: {K2x2.nnz / (K2x2.shape[0] * K2x2.shape[1]):.3f}")
    
    print("\n2x2 System structure:")
    print("[-X^{-1}S  A^T] [dx]   [rd + X^{-1}rc]")
    print("[A         0  ] [dy] = [rp           ]")
    
    print(f"\nActual matrix shape: {K2x2.shape}")
    print(f"Block sizes: dx({n}), dy({m})")

if __name__ == "__main__":
    demo_online_2x2_system()