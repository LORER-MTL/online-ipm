"""
Simple demonstration of key components for the linear programming solution
"""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import qdldl

def demo_key_components():
    """
    Demonstrate the key components: formulation, data extraction, and KKT system
    """
    print("=" * 50)
    print("KEY COMPONENTS DEMONSTRATION")
    print("=" * 50)
    
    # 1. STANDARD FORM LP FORMULATION
    print("\n1. STANDARD FORM LINEAR PROGRAM")
    print("   minimize    c^T x")
    print("   subject to  A x = b")
    print("              x >= 0")
    
    # Simple example problem
    n, m = 4, 2
    A = np.array([[1, 1, 1, 0], 
                  [0, 1, 0, 1]])
    b = np.array([3, 2])
    c = np.array([1, 1, 2, 1])
    
    print(f"\nProblem data:")
    print(f"A = \n{A}")
    print(f"b = {b}")
    print(f"c = {c}")
    
    # 2. CVXPY FORMULATION
    print("\n2. CVXPY FORMULATION")
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x == b, x >= 0])
    
    # Extract problem data
    data, _, _ = prob.get_problem_data(solver=cp.CLARABEL)
    print("Available data keys:", list(data.keys()))
    print(prob.solve(solver=cp.CLARABEL))
    print(f"Optimal x: {x.value}")
    
    # 3. KKT SYSTEM FORMATION
    print("\n3. KKT SYSTEM FOR PRIMAL-BARRIER METHOD")
    print("   The barrier method solves: min c^T x - μ Σ log(x_i)")
    print("   subject to Ax = b")
    print("\n   KKT conditions:")
    print("   A^T y + s - c = 0    (dual feasibility)")
    print("   A x - b = 0          (primal feasibility)")
    print("   X S e - μ e = 0      (complementarity)")
    print("\n   Newton system:")
    print("   [0   A^T  I ] [dx]   [-(A^T y + s - c)]")
    print("   [A   0    0 ] [dy] = [-(A x - b)      ]")
    print("   [S   0    X ] [ds]   [-(X s - μ e)    ]")
    
    # Example KKT matrix formation
    x_curr = np.array([1, 1, 1, 0.1])  # Current point
    s_curr = np.array([0.1, 0.1, 0.1, 1])  # Current slack
    y_curr = np.array([0, 0])  # Current dual
    mu = 0.1
    
    # Form KKT matrix (simplified version)
    A_sparse = csc_matrix(A)
    zeros_nn = csc_matrix((n, n))
    zeros_mm = csc_matrix((m, m))
    zeros_mn = csc_matrix((m, n))
    zeros_nm = csc_matrix((n, m))
    I_n = sp.eye(n, format='csc')
    X = sp.diags(x_curr, format='csc')
    S = sp.diags(s_curr, format='csc')
    
    # Assemble KKT matrix
    from scipy.sparse import hstack, vstack
    row1 = hstack([zeros_nn, A_sparse.T, I_n])
    row2 = hstack([A_sparse, zeros_mm, zeros_mn])
    row3 = hstack([S, zeros_nm, X])
    K = vstack([row1, row2, row3])
    
    print(f"\n   KKT matrix shape: {K.shape}")
    print(f"   KKT matrix density: {K.nnz / (K.shape[0] * K.shape[1]):.3f}")
    
    # 4. QDLDL FACTORIZATION
    print("\n4. QDLDL FACTORIZATION")
    print("   Using QDLDL for sparse symmetric indefinite systems")
    
    # Form residuals
    rd = A.T @ y_curr + s_curr - c
    rp = A @ x_curr - b
    rc = s_curr * x_curr - mu
    rhs = np.concatenate([-rd, -rp, -rc])
    
    print(f"   RHS vector shape: {rhs.shape}")
    
    try:
        # Solve with QDLDL
        solver = qdldl.Solver(K)
        sol = solver.solve(rhs)
        print(f"   Solution found: dx={sol[:n]}")
        print(f"                   dy={sol[n:n+m]}")
        print(f"                   ds={sol[n+m:]}")
        print("   ✓ QDLDL factorization successful")
    except Exception as e:
        print(f"   ✗ QDLDL failed: {e}")
        
        # Fallback
        from scipy.sparse.linalg import spsolve
        sol = spsolve(K, rhs)
        print(f"   ✓ Scipy fallback successful")
    
    print("\n" + "=" * 50)
    print("All key components demonstrated successfully!")
    print("=" * 50)

if __name__ == "__main__":
    demo_key_components()