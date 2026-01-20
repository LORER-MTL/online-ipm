"""OPEN-M utilities for projection and KKT system solving."""

import numpy as np


def project_onto_equality(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project x onto the affine subspace {z : Az = b}.

    Uses the formula: x_proj = x + A^T (A A^T)^{-1} (b - Ax)

    Args:
        x: Point to project
        A: Constraint matrix (p, n)
        b: Constraint RHS (p,)

    Returns:
        Projected point satisfying Az = b
    """
    if A.size == 0:
        return x.copy()
    residual = b - A @ x
    correction = A.T @ np.linalg.solve(A @ A.T, residual)
    return x + correction


def build_kkt_matrix(H: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Build KKT matrix [H, A^T; A, 0].

    The KKT system is:
        [H   A^T] [dx]   = [-grad]
        [A   0  ] [nu]     [0    ]

    Args:
        H: Hessian matrix (n, n)
        A: Equality constraint matrix (p, n)

    Returns:
        KKT matrix of shape (n+p, n+p)
    """
    n = H.shape[0]
    p = A.shape[0] if A.size > 0 else 0
    K = np.zeros((n + p, n + p))
    K[:n, :n] = H
    if p > 0:
        K[:n, n:] = A.T
        K[n:, :n] = A
    return K


def solve_kkt_system(grad: np.ndarray, H: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Solve KKT system for Newton direction.

    Solves: [H, A^T; A, 0] [dx; nu] = -[grad; 0]

    Args:
        grad: Gradient vector (n,)
        H: Hessian matrix (n, n)
        A: Equality constraint matrix (p, n)

    Returns:
        Newton direction dx (the primal part of the solution)
    """
    n = len(grad)
    p = A.shape[0] if A.size > 0 else 0

    K = build_kkt_matrix(H, A)
    rhs = np.zeros(n + p)
    rhs[:n] = -grad

    sol = np.linalg.solve(K, rhs)
    return sol[:n]
