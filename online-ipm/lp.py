"""
Standard Form Linear Program with Primal-Barrier Method
======================================================

This module demonstrates:
1. Formulating a standard form LP using CVXPY
2. Extracting problem data (A, b, c, K)
3. Forming the KKT system for the primal-barrier method
4. Solving the KKT system using QDLDL factorization
"""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, vstack, hstack, diags
import qdldl

def formulate_standard_lp():
    """
    Formulate a standard form linear program:
    minimize    c^T x
    subject to  A x = b
                x >= 0
    
    Returns:
        prob: CVXPY problem object
        x: CVXPY variable
        A, b, c: Problem data matrices/vectors
    """
    np.random.seed(42)  # For reproducibility
    
    # Problem dimensions
    n = 10  # number of variables
    m = 5   # number of equality constraints
    
    # Generate random problem data
    # Ensure A has full row rank for feasibility
    A = np.random.randn(m, n)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    A = U @ Vt  # This ensures full row rank
    
    # Generate a feasible point to ensure problem is feasible
    x_feas = np.random.rand(n) + 0.1  # Ensure x > 0
    b = A @ x_feas
    
    # Random cost vector
    c = np.random.randn(n)
    
    # Define CVXPY problem
    x = cp.Variable(n)
    constraints = [A @ x == b, x >= 0]
    objective = cp.Minimize(c.T @ x)
    prob = cp.Problem(objective, constraints)
    
    return prob, x, A, b, c

def extract_problem_data(prob):
    """
    Extract problem data in standard conic form from CVXPY problem.
    
    Returns:
        data: Dictionary containing problem matrices
    """
    # Get problem data in conic form
    data, chain, inverse_data = prob.get_problem_data(solver=cp.CLARABEL)
    
    return data

def form_kkt_system(A, b, c, x, s, y, mu):
    """
    Form the KKT system for the primal-barrier method.
    
    The primal-barrier method solves:
    minimize    c^T x - mu * sum(log(x_i))
    subject to  A x = b
    
    The KKT conditions are:
    [A^T y + s - c = 0  ]  (dual feasibility)
    [A x - b = 0        ]  (primal feasibility)  
    [X S e - mu e = 0   ]  (complementarity)
    
    Where X = diag(x), S = diag(s), e = ones vector
    
    The Newton system is:
    [0   A^T  I ] [dx]   [rd]
    [A   0    0 ] [dy] = [rp]
    [S   0    X ] [ds]   [rc]
    
    Args:
        A: Constraint matrix (m x n)
        b: RHS vector (m,)
        c: Cost vector (n,)
        x: Current primal variables (n,)
        s: Current dual slack variables (n,)
        y: Current dual variables (m,)
        mu: Barrier parameter
    
    Returns:
        K: KKT matrix
        rhs: Right-hand side vector
    """
    m, n = A.shape
    
    # Convert to sparse if needed
    if not sp.issparse(A):
        A = csc_matrix(A)
    
    # Residuals
    rd = A.T @ y + s - c  # dual residual
    rp = A @ x - b        # primal residual
    rc = s * x - mu       # complementarity residual
    
    # Form KKT matrix
    # [0   A^T  I ]
    # [A   0    0 ]
    # [S   0    X ]
    
    zeros_nn = csc_matrix((n, n))
    zeros_mm = csc_matrix((m, m))
    zeros_mn = csc_matrix((m, n))
    I_n = sp.eye(n, format='csc')
    
    # Create diagonal matrices
    X = diags(x, format='csc')
    S = diags(s, format='csc')
    
    # Assemble KKT matrix
    row1 = hstack([zeros_nn, A.T, I_n])
    row2 = hstack([A, zeros_mm, zeros_mn])
    zeros_mn_t = csc_matrix((n, m))  # n x m matrix for third row
    row3 = hstack([S, zeros_mn_t, X])
    
    K = vstack([row1, row2, row3])
    
    # Right-hand side
    rhs = np.concatenate([-rd, -rp, -rc])
    
    return K, rhs

def solve_kkt_system(K, rhs, use_qdldl=True):
    """
    Solve the KKT system using QDLDL factorization or scipy sparse solver.
    
    Args:
        K: KKT matrix (sparse)
        rhs: Right-hand side vector
        use_qdldl: Whether to use QDLDL (if False, uses scipy)
    
    Returns:
        sol: Solution vector [dx, dy, ds]
    """
    # Convert to CSC format if needed
    if not isinstance(K, csc_matrix):
        K = K.tocsc()
    
    if use_qdldl:
        try:
            # Use QDLDL solver
            solver = qdldl.Solver(K)
            sol = solver.solve(rhs)
            return sol
        except Exception as e:
            print(f"QDLDL failed, falling back to scipy: {e}")
    
    # Fallback to scipy sparse solver
    from scipy.sparse.linalg import spsolve
    sol = spsolve(K, rhs)
    
    return sol

def primal_barrier_method(A, b, c, mu=1.0, tol=1e-6, max_iter=50):
    """
    Solve the linear program using the primal-barrier method.
    
    Args:
        A: Constraint matrix
        b: RHS vector  
        c: Cost vector
        mu: Initial barrier parameter
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        x: Optimal primal variables
        y: Optimal dual variables
        s: Optimal dual slack variables
        history: Convergence history
    """
    m, n = A.shape
    
    # Initialize variables
    # Start with a feasible interior point
    if m > 0:
        # Find a feasible point using least squares
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        # Ensure x > 0 (project to positive orthant)
        x = np.maximum(x, 0.1)
        
        # Re-project to satisfy Ax = b
        residual = b - A @ x
        # Find minimum norm correction
        AtA_inv = np.linalg.pinv(A @ A.T)
        correction = A.T @ AtA_inv @ residual
        x = x + correction
        x = np.maximum(x, 0.01)  # Ensure strict positivity
    else:
        x = np.ones(n) * 0.1
    
    y = np.zeros(m)
    s = np.maximum(c - A.T @ y, 0.1)  # Ensure s > 0
    
    history = []
    
    for iteration in range(max_iter):
        # Form and solve KKT system
        K, rhs = form_kkt_system(A, b, c, x, s, y, mu)
        
        try:
            sol = solve_kkt_system(K, rhs, use_qdldl=True)
        except Exception as e:
            print(f"Warning: KKT solve failed at iteration {iteration}: {e}")
            try:
                sol = solve_kkt_system(K, rhs, use_qdldl=False)
                print("  Fallback scipy solver succeeded")
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                break
            
        # Extract step directions
        dx = sol[:n]
        dy = sol[n:n+m] if m > 0 else np.array([])
        ds = sol[n+m:]
        
        # Compute step sizes (ensuring x, s > 0)
        alpha_p = 1.0
        alpha_d = 1.0
        
        # Primal step size
        neg_indices = dx < 0
        if np.any(neg_indices):
            alpha_p = min(0.99 * np.min(-x[neg_indices] / dx[neg_indices]), 1.0)
        
        # Dual step size
        neg_indices = ds < 0
        if np.any(neg_indices):
            alpha_d = min(0.99 * np.min(-s[neg_indices] / ds[neg_indices]), 1.0)
        
        # Update variables
        x_new = x + alpha_p * dx
        y_new = y + alpha_d * dy if m > 0 else y
        s_new = s + alpha_d * ds
        
        # Check convergence
        gap = np.dot(x_new, s_new)
        prim_res = np.linalg.norm(A @ x_new - b) if m > 0 else 0
        dual_res = np.linalg.norm(A.T @ y_new + s_new - c)
        
        history.append({
            'iteration': iteration,
            'gap': gap,
            'prim_res': prim_res,
            'dual_res': dual_res,
            'mu': mu
        })
        
        print(f"Iter {iteration:2d}: gap={gap:.2e}, prim_res={prim_res:.2e}, dual_res={dual_res:.2e}")
        
        if gap < tol and prim_res < tol and dual_res < tol:
            print(f"Converged in {iteration} iterations!")
            return x_new, y_new, s_new, history
        
        # Update variables
        x, y, s = x_new, y_new, s_new
        
        # Update barrier parameter
        mu *= 0.1
    
    print(f"Maximum iterations ({max_iter}) reached")
    return x, y, s, history

def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("=" * 60)
    print("Standard Form Linear Program with Primal-Barrier Method")
    print("=" * 60)
    
    # Step 1: Formulate standard form LP
    print("\n1. Formulating standard form linear program...")
    prob, x_var, A, b, c = formulate_standard_lp()
    
    print(f"Problem dimensions: {A.shape[1]} variables, {A.shape[0]} constraints")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    print(f"c shape: {c.shape}")
    
    # Step 2: Extract problem data
    print("\n2. Extracting problem data...")
    try:
        data = extract_problem_data(prob)
        print("Problem data extracted successfully")
        print("Available keys:", list(data.keys()))
    except Exception as e:
        print(f"Note: CVXPY data extraction failed ({e}), using direct matrices")
    
    # Step 3: Solve using CVXPY for comparison
    print("\n3. Solving with CVXPY (for comparison)...")
    try:
        prob.solve(verbose=False)
        if prob.status == cp.OPTIMAL:
            print(f"CVXPY optimal value: {prob.value:.6f}")
            print(f"CVXPY optimal x (first 5): {x_var.value[:5]}")
        else:
            print(f"CVXPY status: {prob.status}")
    except Exception as e:
        print(f"CVXPY solve failed: {e}")
    
    # Step 4: Solve using primal-barrier method
    print("\n4. Solving using primal-barrier method...")
    try:
        x_opt, y_opt, s_opt, history = primal_barrier_method(A, b, c)
        
        print(f"\nPrimal-barrier optimal value: {np.dot(c, x_opt):.6f}")
        print(f"Optimal x (first 5): {x_opt[:5]}")
        print(f"Final duality gap: {np.dot(x_opt, s_opt):.2e}")
        
        # Verify KKT conditions
        print("\n5. Verifying KKT conditions...")
        prim_feasibility = np.linalg.norm(A @ x_opt - b)
        dual_feasibility = np.linalg.norm(A.T @ y_opt + s_opt - c)
        complementarity = np.dot(x_opt, s_opt)
        
        print(f"Primal feasibility error: {prim_feasibility:.2e}")
        print(f"Dual feasibility error: {dual_feasibility:.2e}")  
        print(f"Complementarity error: {complementarity:.2e}")
        print(f"All variables positive: x_min={np.min(x_opt):.2e}, s_min={np.min(s_opt):.2e}")
        
    except Exception as e:
        print(f"Primal-barrier method failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
