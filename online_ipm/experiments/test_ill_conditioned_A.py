"""Investigate numerical issues with ill-conditioned A_t in OPEN-M.

Test 7a showed that when A_t has condition number ~10^6, tracking errors
can be huge (~400). This script investigates the root cause.

Hypothesis: The issue is in the projection step or null space computation
when A_t is nearly rank-deficient.
"""

import numpy as np


def project_onto_equality(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project x onto {z : Az = b}."""
    if A.size == 0:
        return x.copy()
    residual = b - A @ x
    # This uses A @ A^T which squares the condition number!
    correction = A.T @ np.linalg.solve(A @ A.T, residual)
    return x + correction


def project_onto_equality_stable(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """More stable projection using least squares."""
    if A.size == 0:
        return x.copy()
    # Find minimum-norm solution to: find dx such that A(x + dx) = b
    # i.e., A dx = b - Ax
    residual = b - A @ x
    # Use lstsq which handles ill-conditioning better
    dx, _, _, _ = np.linalg.lstsq(A.T @ A, A.T @ residual, rcond=None)
    # Actually, we want min ||dx|| s.t. A dx = residual
    # This is dx = A^T (A A^T)^{-1} residual, but use pseudoinverse
    dx = A.T @ np.linalg.lstsq(A @ A.T, residual, rcond=None)[0]
    return x + dx


def compute_null_space_basis(A: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for null(A) using SVD."""
    if A.size == 0:
        return np.eye(A.shape[1] if A.ndim > 1 else 0)

    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
    rank = np.sum(s > tol)
    F = Vh[rank:].T
    return F


def solve_quadratic_with_equality(Q: np.ndarray, c: np.ndarray,
                                   A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min (1/2)x^T Q x + c^T x  s.t. Ax = b exactly via KKT."""
    n = len(c)
    p = A.shape[0] if A.size > 0 else 0

    K = np.zeros((n + p, n + p))
    K[:n, :n] = Q
    if p > 0:
        K[:n, n:] = A.T
        K[n:, :n] = A

    rhs = np.zeros(n + p)
    rhs[:n] = -c
    if p > 0:
        rhs[n:] = b

    sol = np.linalg.solve(K, rhs)
    return sol[:n]


def main():
    np.random.seed(42)

    print("=" * 70)
    print("INVESTIGATING ILL-CONDITIONED A_t IN OPEN-M")
    print("=" * 70)
    print()

    n = 10
    p = 3

    # Create well-conditioned Q
    M = np.random.randn(n, n)
    Q = M.T @ M + 0.5 * np.eye(n)

    # Test different condition numbers for A
    cond_numbers = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]

    print("Testing projection and Newton step accuracy vs cond(A):")
    print()
    print("| cond(A) | cond(A A^T) | Proj Error | Newton Error | Constraint Viol |")
    print("|---------|-------------|------------|--------------|-----------------|")

    for target_cond in cond_numbers:
        # Create A with specific condition number
        U = np.random.randn(p, p)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(n, p)
        V, _ = np.linalg.qr(V)

        # Singular values with target condition number
        s = np.array([1.0, 1.0/np.sqrt(target_cond), 1.0/target_cond])
        A = U @ np.diag(s) @ V.T

        actual_cond = np.linalg.cond(A)
        cond_AAT = np.linalg.cond(A @ A.T)

        c = np.random.randn(n)
        b = np.random.randn(p)

        # True solution
        x_star = solve_quadratic_with_equality(Q, c, A, b)

        # Start from a random point
        x_prev = np.random.randn(n)

        # Step 1: Project
        x_proj = project_onto_equality(x_prev, A, b)
        proj_error = np.linalg.norm(A @ x_proj - b)

        # Step 2: Null space
        F = compute_null_space_basis(A)

        # Step 3: Reduced Newton
        Q_reduced = F.T @ Q @ F
        c_reduced = F.T @ (Q @ x_proj + c)

        try:
            z_star = -np.linalg.solve(Q_reduced, c_reduced)
            x_new = x_proj + F @ z_star
            newton_error = np.linalg.norm(x_new - x_star)
            constraint_viol = np.linalg.norm(A @ x_new - b)
        except:
            newton_error = np.inf
            constraint_viol = np.inf

        print(f"| {actual_cond:7.1e} | {cond_AAT:11.1e} | {proj_error:10.2e} | {newton_error:12.2e} | {constraint_viol:15.2e} |")

    print()
    print("-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    print()
    print("The projection formula x_proj = x + A^T (A A^T)^{-1} (b - Ax)")
    print("requires solving (A A^T) w = (b - Ax).")
    print()
    print("Problem: cond(A A^T) = cond(A)^2 !")
    print()
    print("When cond(A) = 10^6, cond(A A^T) = 10^12, causing:")
    print("  - Large projection errors")
    print("  - These propagate to the Newton step")
    print("  - Final solution can be far from true optimum")
    print()

    # Now test if using pseudoinverse helps
    print("-" * 70)
    print("POTENTIAL FIX: Use SVD-based pseudoinverse for projection")
    print("-" * 70)
    print()

    print("| cond(A) | Standard Proj | SVD Proj | Newton (SVD) |")
    print("|---------|---------------|----------|--------------|")

    for target_cond in cond_numbers:
        U = np.random.randn(p, p)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(n, p)
        V, _ = np.linalg.qr(V)

        s = np.array([1.0, 1.0/np.sqrt(target_cond), 1.0/target_cond])
        A = U @ np.diag(s) @ V.T

        actual_cond = np.linalg.cond(A)

        c = np.random.randn(n)
        b = np.random.randn(p)

        x_star = solve_quadratic_with_equality(Q, c, A, b)
        x_prev = np.random.randn(n)

        # Standard projection
        x_proj_std = project_onto_equality(x_prev, A, b)
        proj_err_std = np.linalg.norm(A @ x_proj_std - b)

        # SVD-based projection using pseudoinverse
        A_pinv = np.linalg.pinv(A)
        x_proj_svd = x_prev + A_pinv @ (b - A @ x_prev)
        proj_err_svd = np.linalg.norm(A @ x_proj_svd - b)

        # Newton with SVD projection
        F = compute_null_space_basis(A)
        Q_reduced = F.T @ Q @ F
        c_reduced = F.T @ (Q @ x_proj_svd + c)

        try:
            z_star = -np.linalg.solve(Q_reduced, c_reduced)
            x_new = x_proj_svd + F @ z_star
            newton_err_svd = np.linalg.norm(x_new - x_star)
        except:
            newton_err_svd = np.inf

        print(f"| {actual_cond:7.1e} | {proj_err_std:13.2e} | {proj_err_svd:8.2e} | {newton_err_svd:12.2e} |")

    print()
    print("-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    print()
    print("For ill-conditioned A_t:")
    print("  1. Standard projection squares the condition number → numerically unstable")
    print("  2. Using pseudoinverse (SVD-based) projection is more stable")
    print("  3. OPEN-M paper assumes well-conditioned A_t (Assumption 3: ‖A_t‖ ≤ a)")
    print("     but doesn't mention condition number requirements")
    print()
    print("PRACTICAL IMPLICATION:")
    print("  OPEN-M with time-varying A_t requires A_t to be well-conditioned,")
    print("  not just bounded in norm. This is an additional hidden requirement.")


def investigate_reduced_system():
    """Deep dive into why Newton fails even with good projection."""
    np.random.seed(42)

    print()
    print("=" * 70)
    print("DEEP DIVE: Why does Newton fail even with SVD projection?")
    print("=" * 70)
    print()

    n = 10
    p = 3

    M = np.random.randn(n, n)
    Q = M.T @ M + 0.5 * np.eye(n)

    target_cond = 1e6

    # Create ill-conditioned A
    U = np.random.randn(p, p)
    U, _ = np.linalg.qr(U)
    V = np.random.randn(n, p)
    V, _ = np.linalg.qr(V)
    s = np.array([1.0, 1.0/np.sqrt(target_cond), 1.0/target_cond])
    A = U @ np.diag(s) @ V.T

    print(f"cond(A) = {np.linalg.cond(A):.2e}")
    print()

    c = np.random.randn(n)
    b = np.random.randn(p)

    # True solution via KKT
    x_star = solve_quadratic_with_equality(Q, c, A, b)
    print(f"True optimum x*: constraint violation = {np.linalg.norm(A @ x_star - b):.2e}")

    # Start from true optimum and see what happens
    x_prev = x_star.copy()

    print()
    print("Starting from TRUE OPTIMUM x*:")

    # Project (should be identity)
    A_pinv = np.linalg.pinv(A)
    x_proj = x_prev + A_pinv @ (b - A @ x_prev)
    print(f"  After projection: ‖x_proj - x*‖ = {np.linalg.norm(x_proj - x_star):.2e}")

    # Null space
    F = compute_null_space_basis(A)
    print(f"  F shape: {F.shape}, ‖AF‖ = {np.linalg.norm(A @ F):.2e}")

    # Reduced gradient at x_proj
    grad_x = Q @ x_proj + c
    print(f"  Full gradient at x_proj: ‖∇f‖ = {np.linalg.norm(grad_x):.2e}")

    # The KKT conditions say: ∇f(x*) + A^T ν* = 0
    # So ∇f(x*) = -A^T ν* (in row space of A)
    # And F^T ∇f(x*) = -F^T A^T ν* = 0 (since A F = 0)
    c_reduced = F.T @ grad_x
    print(f"  Reduced gradient F^T ∇f: ‖c_reduced‖ = {np.linalg.norm(c_reduced):.2e}")

    # The reduced gradient should be zero at optimum!
    # If it's not zero, that's the issue

    # Let's check what z* corresponds to x*
    # x* = x_particular + F z*, so z* = F^T (x* - x_particular)
    # where x_particular is any point satisfying Ax = b

    # Use x_proj as x_particular
    z_at_xstar = F.T @ (x_star - x_proj)  # Should be ~0 since x_star ≈ x_proj
    print(f"  z at x*: ‖z*‖ = {np.linalg.norm(z_at_xstar):.2e}")

    print()
    print("Now starting from a RANDOM point (far from optimum):")

    x_prev = 10 * np.random.randn(n)
    print(f"  Starting point: ‖x_prev - x*‖ = {np.linalg.norm(x_prev - x_star):.2f}")

    # Project
    x_proj = x_prev + A_pinv @ (b - A @ x_prev)
    proj_error = np.linalg.norm(A @ x_proj - b)
    print(f"  After projection: constraint violation = {proj_error:.2e}")
    print(f"  After projection: ‖x_proj - x*‖ = {np.linalg.norm(x_proj - x_star):.2f}")

    # Null space computation - this is where things go wrong!
    # The issue: for ill-conditioned A, the null space is numerically sensitive

    # Check if x* - x_proj is really in null(A)
    diff = x_star - x_proj
    print(f"  ‖A(x* - x_proj)‖ = {np.linalg.norm(A @ diff):.2e}")  # Should be ~0
    print(f"  ‖x* - x_proj‖ = {np.linalg.norm(diff):.2f}")

    # Express diff in terms of F
    z_diff = F.T @ diff
    reconstruction = F @ z_diff
    print(f"  ‖(x* - x_proj) - F F^T (x* - x_proj)‖ = {np.linalg.norm(diff - reconstruction):.2e}")

    # The reduced problem
    Q_reduced = F.T @ Q @ F
    c_reduced = F.T @ (Q @ x_proj + c)

    print(f"  cond(Q_reduced) = {np.linalg.cond(Q_reduced):.2e}")

    # True z*
    z_star_true = F.T @ (x_star - x_proj)

    # Computed z* from Newton
    z_star_computed = -np.linalg.solve(Q_reduced, c_reduced)

    print(f"  ‖z*_true - z*_computed‖ = {np.linalg.norm(z_star_true - z_star_computed):.2e}")
    print(f"  ‖z*_true‖ = {np.linalg.norm(z_star_true):.2f}")
    print(f"  ‖z*_computed‖ = {np.linalg.norm(z_star_computed):.2f}")

    # Reconstruct
    x_newton = x_proj + F @ z_star_computed
    print(f"  Final Newton error: ‖x_newton - x*‖ = {np.linalg.norm(x_newton - x_star):.2e}")

    print()
    print("-" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 70)
    print()

    # The issue is that when A is ill-conditioned, even though F is orthonormal,
    # the null space is "nearly" the full space (A is almost rank-deficient).
    # This means small errors in projection get amplified.

    # Let's verify: singular values of A
    U_A, s_A, Vh_A = np.linalg.svd(A)
    print(f"Singular values of A: {s_A}")
    print(f"Ratio s_max/s_min = {s_A[0]/s_A[-1]:.2e}")
    print()

    # The smallest singular value is ~10^-6, meaning A is almost rank-deficient
    # The null space F corresponds to singular values below tolerance

    # Check: what's the "effective" null space dimension?
    tol = max(A.shape) * np.finfo(float).eps * s_A[0]
    print(f"SVD tolerance for rank: {tol:.2e}")
    print(f"Smallest singular value: {s_A[-1]:.2e}")
    print(f"Is s_min > tol? {s_A[-1] > tol}")
    print()

    # AH! The issue: when s_min is very small but still > tol,
    # the corresponding "constraint direction" is poorly determined
    # but we treat it as a constraint, not part of null space

    # The projection amplifies errors in the direction of smallest singular value
    # Because (A A^T)^{-1} has eigenvalue 1/s_min^2 in that direction

    print("THE ROOT CAUSE:")
    print("  When A has a very small singular value s_min:")
    print(f"    1. Projection uses (A A^T)^(-1) which has eigenvalue 1/s_min^2 = {1/s_A[-1]**2:.2e}")
    print("    2. Small errors in (b - Ax) get amplified by this factor")
    print("    3. Even though F is orthonormal, the projected point x_proj has large error")
    print("    4. This error propagates to the Newton step")
    print()
    print("  The OPEN-M paper's assumption '‖A_t‖ ≤ a' bounds the LARGEST singular value,")
    print("  but says nothing about the SMALLEST. A bound like σ_min(A_t) ≥ σ_lower")
    print("  (i.e., bounded condition number) is needed for numerical stability.")


if __name__ == "__main__":
    main()
    investigate_reduced_system()
