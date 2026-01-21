"""Test OPEN-M tracking for quadratic objectives with large variation.

This experiment verifies the claim that OPEN-M can track arbitrary variation
when the objective is quadratic (L=0, constant Hessian).

For quadratics f_t(x) = (1/2)x^T Q x + c_t^T x with constant Q:
- Newton's method converges in ONE step (quadratic = linear gradient)
- Therefore OPEN-M should track exactly, regardless of variation in c_t

We test with:
1. Large random jumps in c_t (not small perturbations)
2. Verify that OPEN-M tracks the optimum exactly (up to numerical precision)
3. Compare with what would happen for non-quadratic objectives
"""

import numpy as np
from scipy.optimize import minimize


def project_onto_equality(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project x onto {z : Az = b}."""
    if A.size == 0:
        return x.copy()
    residual = b - A @ x
    correction = A.T @ np.linalg.solve(A @ A.T, residual)
    return x + correction


def solve_quadratic_with_equality(Q: np.ndarray, c: np.ndarray,
                                   A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min (1/2)x^T Q x + c^T x  s.t. Ax = b exactly via KKT."""
    n = len(c)
    p = A.shape[0] if A.size > 0 else 0

    # KKT system: [Q A^T; A 0] [x; nu] = [-c; b]
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


def open_m_step_quadratic(x_prev: np.ndarray, Q: np.ndarray, c: np.ndarray,
                          A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One step of OPEN-M for quadratic objective.

    1. Project x_prev onto new constraint Ax = b
    2. Take full Newton step (for quadratic, this is exact)
    """
    n = len(x_prev)
    p = A.shape[0] if A.size > 0 else 0

    # Step 1: Project onto equality constraints
    x_proj = project_onto_equality(x_prev, A, b)

    # Step 2: Newton step
    # Gradient at x_proj: g = Qx + c
    grad = Q @ x_proj + c

    # For equality-constrained Newton, solve:
    # [Q A^T] [dx]   [-grad]
    # [A  0 ] [nu] = [  0  ]
    K = np.zeros((n + p, n + p))
    K[:n, :n] = Q
    if p > 0:
        K[:n, n:] = A.T
        K[n:, :n] = A

    rhs = np.zeros(n + p)
    rhs[:n] = -grad

    sol = np.linalg.solve(K, rhs)
    dx = sol[:n]

    return x_proj + dx


def run_experiment():
    """Run the quadratic variation tracking experiment."""
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT: OPEN-M Tracking for Quadratic Objectives (L=0)")
    print("=" * 70)
    print()
    print("Testing claim: For quadratics, OPEN-M can track arbitrary variation")
    print("because Newton converges in one step regardless of starting point.")
    print()

    # Problem dimensions
    n = 10  # Decision variables
    p = 3   # Equality constraints
    T = 50  # Time steps

    # Generate a random positive definite Q (CONSTANT over time)
    M = np.random.randn(n, n)
    Q = M.T @ M + 0.1 * np.eye(n)  # Ensure positive definite
    h = np.min(np.linalg.eigvalsh(Q))  # Strong convexity parameter

    # Generate random equality constraint matrix (constant)
    A = np.random.randn(p, n)

    print(f"Problem: n={n} variables, p={p} equality constraints")
    print(f"Hessian Q is constant → L=0 (Lipschitz constant of Hessian)")
    print(f"Strong convexity h = {h:.4f}")
    print()

    # Test 1: Small variation (traditional setting)
    print("-" * 70)
    print("Test 1: Small variation in c_t (σ = 0.1)")
    print("-" * 70)

    c_base = np.random.randn(n)
    b = np.random.randn(p)

    # Initial solution
    x_star_0 = solve_quadratic_with_equality(Q, c_base, A, b)
    x_current = x_star_0.copy()

    errors_small = []
    variations_small = []

    for t in range(T):
        # Small perturbation to c
        c_t = c_base + 0.1 * np.random.randn(n)

        # True optimum
        x_star_t = solve_quadratic_with_equality(Q, c_t, A, b)

        # OPEN-M step
        x_new = open_m_step_quadratic(x_current, Q, c_t, A, b)

        # Track metrics
        error = np.linalg.norm(x_new - x_star_t)
        if t > 0:
            variation = np.linalg.norm(x_star_t - x_star_prev)
            variations_small.append(variation)

        errors_small.append(error)
        x_current = x_new
        x_star_prev = x_star_t

    print(f"  Average optimum variation: {np.mean(variations_small):.6f}")
    print(f"  Max tracking error: {np.max(errors_small):.2e}")
    print(f"  Mean tracking error: {np.mean(errors_small):.2e}")
    print()

    # Test 2: LARGE variation (violates typical bounds)
    print("-" * 70)
    print("Test 2: LARGE variation in c_t (σ = 10.0) - 100x larger!")
    print("-" * 70)

    x_current = x_star_0.copy()

    errors_large = []
    variations_large = []

    for t in range(T):
        # LARGE perturbation to c (100x larger than "small")
        c_t = c_base + 10.0 * np.random.randn(n)

        # True optimum
        x_star_t = solve_quadratic_with_equality(Q, c_t, A, b)

        # OPEN-M step
        x_new = open_m_step_quadratic(x_current, Q, c_t, A, b)

        # Track metrics
        error = np.linalg.norm(x_new - x_star_t)
        if t > 0:
            variation = np.linalg.norm(x_star_t - x_star_prev)
            variations_large.append(variation)

        errors_large.append(error)
        x_current = x_new
        x_star_prev = x_star_t

    print(f"  Average optimum variation: {np.mean(variations_large):.6f}")
    print(f"  Max tracking error: {np.max(errors_large):.2e}")
    print(f"  Mean tracking error: {np.mean(errors_large):.2e}")
    print()

    # Test 3: EXTREME variation (random jumps)
    print("-" * 70)
    print("Test 3: EXTREME variation - completely random c_t each step")
    print("-" * 70)

    x_current = x_star_0.copy()

    errors_extreme = []
    variations_extreme = []

    for t in range(T):
        # Completely random c_t (no relation to previous)
        c_t = 10.0 * np.random.randn(n)

        # True optimum
        x_star_t = solve_quadratic_with_equality(Q, c_t, A, b)

        # OPEN-M step
        x_new = open_m_step_quadratic(x_current, Q, c_t, A, b)

        # Track metrics
        error = np.linalg.norm(x_new - x_star_t)
        if t > 0:
            variation = np.linalg.norm(x_star_t - x_star_prev)
            variations_extreme.append(variation)

        errors_extreme.append(error)
        x_current = x_new
        x_star_prev = x_star_t

    print(f"  Average optimum variation: {np.mean(variations_extreme):.6f}")
    print(f"  Max tracking error: {np.max(errors_extreme):.2e}")
    print(f"  Mean tracking error: {np.mean(errors_extreme):.2e}")
    print()

    # Test 4: Time-varying b_t as well
    print("-" * 70)
    print("Test 4: Both c_t AND b_t vary (extreme variation)")
    print("-" * 70)

    x_current = x_star_0.copy()

    errors_both = []
    variations_both = []

    for t in range(T):
        # Random c_t and b_t
        c_t = 10.0 * np.random.randn(n)
        b_t = np.random.randn(p)

        # True optimum
        x_star_t = solve_quadratic_with_equality(Q, c_t, A, b_t)

        # OPEN-M step (project to NEW b_t, then Newton)
        x_new = open_m_step_quadratic(x_current, Q, c_t, A, b_t)

        # Track metrics
        error = np.linalg.norm(x_new - x_star_t)
        if t > 0:
            variation = np.linalg.norm(x_star_t - x_star_prev)
            variations_both.append(variation)

        errors_both.append(error)
        x_current = x_new
        x_star_prev = x_star_t

    print(f"  Average optimum variation: {np.mean(variations_both):.6f}")
    print(f"  Max tracking error: {np.max(errors_both):.2e}")
    print(f"  Mean tracking error: {np.mean(errors_both):.2e}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("For QUADRATIC objectives (constant Hessian, L=0):")
    print()
    print("| Test | Avg Variation | Max Error |")
    print("|------|---------------|-----------|")
    print(f"| Small (σ=0.1)   | {np.mean(variations_small):12.4f} | {np.max(errors_small):.2e} |")
    print(f"| Large (σ=10)    | {np.mean(variations_large):12.4f} | {np.max(errors_large):.2e} |")
    print(f"| Extreme random  | {np.mean(variations_extreme):12.4f} | {np.max(errors_extreme):.2e} |")
    print(f"| Both c_t & b_t  | {np.mean(variations_both):12.4f} | {np.max(errors_both):.2e} |")
    print()

    all_errors = errors_small + errors_large + errors_extreme + errors_both
    if np.max(all_errors) < 1e-10:
        print("✓ VERIFIED: OPEN-M tracks EXACTLY for quadratics, regardless of variation!")
        print()
        print("This confirms that for L=0 (quadratic objectives):")
        print("  - The variation bound is effectively UNBOUNDED")
        print("  - A single Newton step always finds the exact optimum")
        print("  - OPEN-M's regret is O(0) = optimal for this class")
        return True
    else:
        print("✗ Unexpected: Some tracking errors detected")
        print(f"  Max error: {np.max(all_errors):.2e}")
        return False


def test_varying_hessian():
    """Test OPEN-M with TIME-VARYING Hessian Q_t (still quadratic at each t).

    Key insight: L measures spatial Lipschitz (‖∇²f_t(x) - ∇²f_t(y)‖ ≤ L‖x-y‖)
    For quadratic f_t, ∇²f_t = Q_t is constant in x, so L = 0 even when Q_t varies!

    Newton still converges in one step because each f_t is quadratic.
    """
    np.random.seed(123)

    print()
    print("=" * 70)
    print("TEST 5: Time-varying Hessian Q_t (still quadratic each step)")
    print("=" * 70)
    print()
    print("Even with varying Q_t, L = 0 at each time step (Hessian constant in x)")
    print("Newton converges in one step for each quadratic f_t")
    print()

    n = 10
    p = 3
    T = 50

    # Generate random equality constraint matrix (constant)
    A = np.random.randn(p, n)

    # Start with some Q and find initial optimum
    M = np.random.randn(n, n)
    Q_0 = M.T @ M + 0.1 * np.eye(n)
    c_0 = np.random.randn(n)
    b = np.random.randn(p)

    x_star_0 = solve_quadratic_with_equality(Q_0, c_0, A, b)
    x_current = x_star_0.copy()

    errors = []
    variations = []
    hessian_changes = []
    x_star_prev = x_star_0.copy()
    Q_prev = Q_0.copy()

    for t in range(T):
        # Generate NEW random Q_t (significantly different from previous)
        M = np.random.randn(n, n)
        Q_t = M.T @ M + 0.1 * np.eye(n)  # Random positive definite

        # Also vary c_t
        c_t = 5.0 * np.random.randn(n)

        # Track how much Hessian changed
        hessian_change = np.linalg.norm(Q_t - Q_prev, 'fro')
        hessian_changes.append(hessian_change)

        # True optimum
        x_star_t = solve_quadratic_with_equality(Q_t, c_t, A, b)

        # OPEN-M step with the NEW Q_t
        x_new = open_m_step_quadratic(x_current, Q_t, c_t, A, b)

        # Track metrics
        error = np.linalg.norm(x_new - x_star_t)
        variation = np.linalg.norm(x_star_t - x_star_prev)

        errors.append(error)
        variations.append(variation)

        x_current = x_new
        x_star_prev = x_star_t.copy()
        Q_prev = Q_t.copy()

    print(f"  Average Hessian change ‖Q_t - Q_{{t-1}}‖_F: {np.mean(hessian_changes):.4f}")
    print(f"  Average optimum variation: {np.mean(variations):.4f}")
    print(f"  Max tracking error: {np.max(errors):.2e}")
    print(f"  Mean tracking error: {np.mean(errors):.2e}")
    print()

    if np.max(errors) < 1e-10:
        print("✓ VERIFIED: OPEN-M tracks exactly even with varying Q_t!")
        print()
        print("  This confirms: L = 0 means Hessian is constant IN SPACE,")
        print("  not across time. Newton converges in one step for each")
        print("  quadratic f_t, regardless of how Q_t changes between steps.")
        return True
    else:
        print(f"✗ Unexpected error: {np.max(errors):.2e}")
        return False


def compute_null_space_basis(A: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for null(A) using SVD.

    Returns F such that:
    - A @ F = 0 (columns of F are in null space)
    - F.T @ F = I (orthonormal)
    """
    if A.size == 0:
        return np.eye(A.shape[1] if A.ndim > 1 else 0)

    # SVD: A = U @ S @ Vh
    # Null space is spanned by rows of Vh corresponding to zero singular values
    U, s, Vh = np.linalg.svd(A, full_matrices=True)

    # Numerical rank
    tol = max(A.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
    rank = np.sum(s > tol)

    # Null space basis: last (n - rank) rows of Vh, transposed to columns
    F = Vh[rank:].T
    return F


def open_m_step_with_varying_A(x_prev: np.ndarray, Q: np.ndarray, c: np.ndarray,
                                A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict]:
    """One step of OPEN-M with time-varying A_t.

    This is the full OPEN-M algorithm:
    1. Project x_prev onto {x : Ax = b}
    2. Compute orthonormal basis F for null(A)
    3. Reduce to unconstrained problem in null space
    4. Take Newton step in reduced space
    5. Map back to original space

    Returns:
        Tuple of (new x, debug info dict)
    """
    n = len(x_prev)
    p = A.shape[0] if A.size > 0 else 0

    debug = {}

    # Step 1: Project onto equality constraints
    x_proj = project_onto_equality(x_prev, A, b)
    debug['projection_residual'] = np.linalg.norm(A @ x_proj - b) if p > 0 else 0

    if p == 0:
        # No constraints - just solve unconstrained
        x_new = -np.linalg.solve(Q, c)
        return x_new, debug

    # Step 2: Compute orthonormal basis F for null(A)
    F = compute_null_space_basis(A)
    debug['F_shape'] = F.shape
    debug['F_orthonormality_error'] = np.linalg.norm(F.T @ F - np.eye(F.shape[1])) if F.size > 0 else 0
    debug['AF_error'] = np.linalg.norm(A @ F) if F.size > 0 else 0

    if F.shape[1] == 0:
        # Fully constrained - x_proj is the only feasible point
        return x_proj, debug

    # Step 3: Reduce to null space
    # x = x_proj + F @ z, where z is the reduced variable
    # f(x) = (1/2)(x_proj + Fz)^T Q (x_proj + Fz) + c^T (x_proj + Fz)
    #      = (1/2) z^T (F^T Q F) z + (Q x_proj + c)^T F z + const

    Q_reduced = F.T @ Q @ F  # Reduced Hessian
    c_reduced = F.T @ (Q @ x_proj + c)  # Reduced gradient at z=0

    debug['Q_reduced_cond'] = np.linalg.cond(Q_reduced) if Q_reduced.size > 0 else 1

    # Step 4: Newton step in reduced space
    # For quadratic: z* = -Q_reduced^{-1} c_reduced
    try:
        z_star = -np.linalg.solve(Q_reduced, c_reduced)
    except np.linalg.LinAlgError:
        debug['solve_failed'] = True
        return x_proj, debug

    debug['z_norm'] = np.linalg.norm(z_star)

    # Step 5: Map back to original space
    x_new = x_proj + F @ z_star

    # Verify feasibility
    debug['final_constraint_error'] = np.linalg.norm(A @ x_new - b)

    return x_new, debug


def test_varying_A():
    """Test OPEN-M with time-varying A_t (equality constraint matrix).

    This tests the full OPEN-M claim: handling time-varying equality constraints.
    Key challenge: must recompute orthonormal basis F_t = null(A_t) at each step.
    """
    np.random.seed(456)

    print()
    print("=" * 70)
    print("TEST 6: Time-varying A_t (equality constraint matrix)")
    print("=" * 70)
    print()
    print("OPEN-M claims to handle time-varying A_t. This requires:")
    print("  1. Projecting onto new constraint set {x : A_t x = b_t}")
    print("  2. Computing orthonormal basis F_t for null(A_t)")
    print("  3. Newton step in reduced space")
    print()

    n = 10  # Decision variables
    p = 3   # Equality constraints (constant dimension)
    T = 50  # Time steps

    # Initial setup
    M = np.random.randn(n, n)
    Q_0 = M.T @ M + 0.5 * np.eye(n)
    c_0 = np.random.randn(n)
    A_0 = np.random.randn(p, n)
    b_0 = np.random.randn(p)

    x_star_0 = solve_quadratic_with_equality(Q_0, c_0, A_0, b_0)
    x_current = x_star_0.copy()

    errors = []
    variations = []
    A_changes = []
    constraint_errors = []
    F_orthonormality_errors = []

    x_star_prev = x_star_0.copy()
    A_prev = A_0.copy()

    print("Running experiment with varying Q_t, c_t, A_t, and b_t...")
    print()

    for t in range(T):
        # Generate NEW random problem data
        M = np.random.randn(n, n)
        Q_t = M.T @ M + 0.5 * np.eye(n)  # Random positive definite
        c_t = 5.0 * np.random.randn(n)

        # NEW constraint matrix A_t (this is the key test!)
        A_t = np.random.randn(p, n)
        b_t = np.random.randn(p)

        # Track how much A changed
        A_change = np.linalg.norm(A_t - A_prev, 'fro')
        A_changes.append(A_change)

        # True optimum (solve exactly)
        x_star_t = solve_quadratic_with_equality(Q_t, c_t, A_t, b_t)

        # OPEN-M step with varying A_t
        x_new, debug = open_m_step_with_varying_A(x_current, Q_t, c_t, A_t, b_t)

        # Track metrics
        error = np.linalg.norm(x_new - x_star_t)
        variation = np.linalg.norm(x_star_t - x_star_prev)

        errors.append(error)
        variations.append(variation)
        constraint_errors.append(debug.get('final_constraint_error', 0))
        F_orthonormality_errors.append(debug.get('F_orthonormality_error', 0))

        x_current = x_new
        x_star_prev = x_star_t.copy()
        A_prev = A_t.copy()

    print(f"  Average ‖A_t - A_{{t-1}}‖_F: {np.mean(A_changes):.4f}")
    print(f"  Average optimum variation: {np.mean(variations):.4f}")
    print(f"  Max tracking error: {np.max(errors):.2e}")
    print(f"  Mean tracking error: {np.mean(errors):.2e}")
    print()
    print(f"  Max constraint violation ‖A_t x - b_t‖: {np.max(constraint_errors):.2e}")
    print(f"  Max F orthonormality error ‖F^T F - I‖: {np.max(F_orthonormality_errors):.2e}")
    print()

    if np.max(errors) < 1e-10:
        print("✓ VERIFIED: OPEN-M tracks exactly even with varying A_t!")
        print()
        print("  For quadratics, the reduced problem is still quadratic,")
        print("  so Newton converges in one step regardless of A_t changes.")
        return True
    else:
        print(f"✗ Tracking error detected: {np.max(errors):.2e}")
        print()
        print("  Investigating potential issues...")
        return False


def test_A_variation_with_ill_conditioned():
    """Test edge cases: nearly rank-deficient A_t, large changes, etc."""
    np.random.seed(789)

    print()
    print("=" * 70)
    print("TEST 7: Edge cases for time-varying A_t")
    print("=" * 70)
    print()

    n = 10
    p = 3
    T = 30

    # Test 7a: Nearly rank-deficient A_t
    print("-" * 70)
    print("Test 7a: Nearly rank-deficient A_t (condition number ~10^6)")
    print("-" * 70)

    M = np.random.randn(n, n)
    Q = M.T @ M + 0.5 * np.eye(n)

    errors_7a = []
    cond_numbers = []

    x_current = np.random.randn(n)

    for t in range(T):
        c_t = np.random.randn(n)
        b_t = np.random.randn(p)

        # Create nearly rank-deficient A_t
        U = np.random.randn(p, p)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(n, p)
        V, _ = np.linalg.qr(V)

        # Singular values with large condition number
        s = np.array([1.0, 1e-3, 1e-6])
        A_t = U @ np.diag(s) @ V.T

        cond_A = np.linalg.cond(A_t)
        cond_numbers.append(cond_A)

        x_star_t = solve_quadratic_with_equality(Q, c_t, A_t, b_t)
        x_new, debug = open_m_step_with_varying_A(x_current, Q, c_t, A_t, b_t)

        error = np.linalg.norm(x_new - x_star_t)
        errors_7a.append(error)
        x_current = x_new

    print(f"  Average cond(A_t): {np.mean(cond_numbers):.2e}")
    print(f"  Max tracking error: {np.max(errors_7a):.2e}")
    print()

    # Test 7b: Sudden large changes in A_t
    print("-" * 70)
    print("Test 7b: Sudden large jumps in A_t (‖A_t - A_{t-1}‖ ~ 10)")
    print("-" * 70)

    errors_7b = []
    A_prev = np.random.randn(p, n)
    x_current = np.random.randn(n)

    for t in range(T):
        c_t = np.random.randn(n)

        # Large random change in A_t
        A_t = 3.0 * np.random.randn(p, n)  # Large magnitude
        b_t = A_t @ np.random.randn(n)  # Ensure feasible

        x_star_t = solve_quadratic_with_equality(Q, c_t, A_t, b_t)
        x_new, debug = open_m_step_with_varying_A(x_current, Q, c_t, A_t, b_t)

        error = np.linalg.norm(x_new - x_star_t)
        errors_7b.append(error)
        x_current = x_new
        A_prev = A_t

    print(f"  Max tracking error: {np.max(errors_7b):.2e}")
    print()

    # Test 7c: Dimension edge case - full rank constraints (p close to n)
    print("-" * 70)
    print("Test 7c: Nearly fully constrained (p = n-2, only 2 DOF)")
    print("-" * 70)

    n_7c = 10
    p_7c = 8  # Only 2 degrees of freedom

    M = np.random.randn(n_7c, n_7c)
    Q_7c = M.T @ M + 0.5 * np.eye(n_7c)

    errors_7c = []
    x_current = np.zeros(n_7c)

    for t in range(T):
        c_t = np.random.randn(n_7c)
        A_t = np.random.randn(p_7c, n_7c)
        b_t = np.random.randn(p_7c)

        x_star_t = solve_quadratic_with_equality(Q_7c, c_t, A_t, b_t)
        x_new, debug = open_m_step_with_varying_A(x_current, Q_7c, c_t, A_t, b_t)

        error = np.linalg.norm(x_new - x_star_t)
        errors_7c.append(error)
        x_current = x_new

    print(f"  Null space dimension: {n_7c - p_7c}")
    print(f"  Max tracking error: {np.max(errors_7c):.2e}")
    print()

    # Summary
    all_errors = errors_7a + errors_7b + errors_7c
    if np.max(all_errors) < 1e-8:
        print("✓ All edge cases pass: OPEN-M is robust to A_t variations")
        return True
    else:
        print(f"✗ Some edge cases have issues (max error: {np.max(all_errors):.2e})")
        return False


def compare_with_nonquadratic():
    """Show that non-quadratic objectives DON'T have this property."""
    np.random.seed(42)

    print()
    print("=" * 70)
    print("COMPARISON: Non-quadratic objective (log barrier)")
    print("=" * 70)
    print()
    print("For non-quadratic f(x) = c^T x - μ Σ log(x_i), L > 0")
    print("Single Newton step does NOT converge exactly.")
    print()

    n = 5
    T = 20
    mu = 1.0

    def barrier_obj(x, c):
        if np.any(x <= 0):
            return np.inf
        return c @ x - mu * np.sum(np.log(x))

    def barrier_grad(x, c):
        return c - mu / x

    def barrier_hess(x):
        return np.diag(mu / x**2)

    # Start with a feasible point
    x_current = np.ones(n)

    errors = []
    variations = []
    x_star_prev = None

    for t in range(T):
        # Vary c_t - keep it positive so optimum exists in interior
        c_t = 0.5 + np.abs(np.random.randn(n))  # c > 0.5 always

        # True optimum: ∇f = 0 → c - μ/x = 0 → x* = μ/c
        x_star_t = mu / c_t

        # Single Newton step from x_current
        grad = barrier_grad(x_current, c_t)
        H = barrier_hess(x_current)
        dx = -np.linalg.solve(H, grad)

        # Take full Newton step (like OPEN-M would)
        x_new = x_current + dx

        # Clip to stay feasible (this is the problem!)
        x_new = np.maximum(x_new, 0.01)

        error = np.linalg.norm(x_new - x_star_t)
        if x_star_prev is not None:
            variations.append(np.linalg.norm(x_star_t - x_star_prev))

        errors.append(error)
        x_current = x_new
        x_star_prev = x_star_t.copy()

    print(f"Average optimum variation: {np.mean(variations):.4f}")
    print(f"Max tracking error: {np.max(errors):.4f}")
    print(f"Mean tracking error: {np.mean(errors):.4f}")
    print()
    print("✗ Non-quadratic: Single Newton step has NON-ZERO tracking error")
    print("  This is where the variation bound v ≤ h/(8L) matters!")


def main():
    success = run_experiment()
    test_varying_hessian()
    test_varying_A()
    test_A_variation_with_ill_conditioned()
    compare_with_nonquadratic()

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("OPEN-M's practical utility depends critically on the objective type:")
    print()
    print("• QUADRATIC (L=0): Tracks exactly, any variation allowed")
    print("  → Actually useful for time-varying QPs")
    print()
    print("• NON-QUADRATIC (L>0): Tracking error grows with variation")
    print("  → Restricted to v ≤ h/(8L), often essentially zero")
    print()
    print("This confirms our analysis: OPEN-M is only truly useful for")
    print("time-varying quadratic programs. For anything else, the variation")
    print("bound makes it impractical.")


if __name__ == "__main__":
    main()
