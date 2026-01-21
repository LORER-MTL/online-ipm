"""Feasible Space Projection Algorithm.

From feasible_projection_analysis.md:
Instead of clipping slack variables, project onto the FULL feasible set:
    F = {(x, s) : Ax = b, Cx + s = d, s >= 0}

This is a convex QP at each step:
    minimize   ||y - y_newton||^2
    subject to [A, 0; C, I] @ y = [b; d]
               s >= 0

Key differences from SlackProjectionAlgorithm:
1. Uses QP projection instead of clipping
2. Maintains Cx + s = d after projection
3. Distance preservation property holds (convex set projection)

However, this approach still has problems:
- Active set changes destroy Newton structure
- Computational cost doubles (QP per iteration)
- Newton convergence breaks at constraint boundaries
"""

import numpy as np
from scipy.optimize import minimize

from .base import OnlineAlgorithm, StepMetrics
from ..problems import OnlineLPInstance
from ..open_m import build_kkt_matrix


def project_onto_feasible_set(x_curr: np.ndarray, s_curr: np.ndarray,
                              A: np.ndarray, b: np.ndarray,
                              F: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (x, s) onto the feasible set F = {Ax=b, Fx+s=d, s>=0}.

    Solves the QP:
        minimize   0.5 * ||(x,s) - (x_curr, s_curr)||^2
        subject to Ax = b
                   Fx + s = g
                   s >= 0

    Args:
        x_curr: Current x point (n,)
        s_curr: Current slack variables (m,)
        A: Equality constraint matrix (p, n)
        b: Equality RHS (p,)
        F: Inequality constraint matrix (m, n)
        g: Inequality RHS (m,)

    Returns:
        Tuple (x_proj, s_proj) satisfying all constraints
    """
    n = len(x_curr)
    m = len(s_curr)
    p = A.shape[0] if A.size > 0 else 0

    y_curr = np.concatenate([x_curr, s_curr])

    def objective(y):
        return 0.5 * np.sum((y - y_curr)**2)

    def grad(y):
        return y - y_curr

    # Build equality constraints: [A, 0; F, I] @ y = [b; g]
    def eq_constraints(y):
        x, s = y[:n], y[n:]
        constraints = []
        if p > 0:
            constraints.append(A @ x - b)
        constraints.append(F @ x + s - g)
        return np.concatenate(constraints) if len(constraints) > 1 else constraints[0]

    def eq_jacobian(y):
        # Jacobian of equality constraints
        jac_rows = []
        if p > 0:
            jac_rows.append(np.hstack([A, np.zeros((p, m))]))
        jac_rows.append(np.hstack([F, np.eye(m)]))
        return np.vstack(jac_rows)

    # Bounds: x free (or x >= 0 if enforcing), s >= 0
    # For this implementation we enforce x >= 0 (standard LP form) and s >= 0
    bounds = [(0, None)] * n + [(0, None)] * m

    result = minimize(
        objective, y_curr, jac=grad,
        constraints={'type': 'eq', 'fun': eq_constraints, 'jac': eq_jacobian},
        bounds=bounds,
        method='SLSQP',
        options={'ftol': 1e-10, 'maxiter': 500}
    )

    if not result.success:
        # Fall back to trust-constr if SLSQP fails
        from scipy.optimize import LinearConstraint, Bounds

        # Build linear constraint matrix for [A, 0; F, I]
        A_aug = np.zeros((p + m, n + m))
        if p > 0:
            A_aug[:p, :n] = A
        A_aug[p:, :n] = F
        A_aug[p:, n:] = np.eye(m)
        b_aug = np.concatenate([b, g]) if p > 0 else g

        result = minimize(
            objective, y_curr, jac=grad,
            constraints=LinearConstraint(A_aug, b_aug, b_aug),
            bounds=Bounds([0]*n + [0]*m, [np.inf]*(n+m)),
            method='trust-constr',
            options={'xtol': 1e-10, 'maxiter': 500}
        )

    return result.x[:n], result.x[n:]


class FeasibleProjectionAlgorithm(OnlineAlgorithm):
    """Full feasible set projection method for inequality constraints.

    This implements the FULL PROJECTION approach that projects onto
    F = {Ax=b, Fx+s=d, s>=0} instead of clipping.

    Advantages over clipping:
    - Distance preservation property holds (convex set projection)
    - Constraints Fx + s = g and s >= 0 are both satisfied

    Disadvantages:
    - Computational cost ~2x (QP solve per iteration)
    - Active set changes still cause Newton convergence issues
    - Projection can land on boundary kinks

    The algorithm:
    1. Augments variables: y = [x; s] where s = g - Fx (slack)
    2. Uses log barrier on x >= 0 and s >= 0 for Newton step
    3. Projects onto F = {Ax=b, Fx+s=g, x>=0, s>=0} via QP
    4. Takes Newton step in projected space
    5. Projects result back onto F (instead of clipping)
    """

    def __init__(self, n: int, m: int, mu: float = 1.0):
        """Initialize feasible projection algorithm.

        Args:
            n: Number of primal variables
            m: Number of inequality constraints
            mu: Barrier parameter (larger = closer to true LP)
        """
        super().__init__(n, m)
        self.mu = mu
        self.x: np.ndarray = None
        self.s: np.ndarray = None  # Slack variables

    def initialize(self, instance: OnlineLPInstance, x_init: np.ndarray) -> None:
        """Initialize primal and slack variables.

        Args:
            instance: Initial problem instance
            x_init: Initial primal solution (should be strictly feasible)
        """
        self.x = x_init.copy()
        self.s = instance.g - instance.F @ x_init  # s = g - Fx (should be > 0)

    def get_current_x(self) -> np.ndarray:
        """Return current primal solution."""
        return self.x.copy()

    def _barrier_grad_hess(self, x: np.ndarray, s: np.ndarray, c: np.ndarray):
        """Compute gradient and Hessian of barrier-augmented objective.

        Objective: c^T x + (1/mu) * (-sum(log(x_i)) - sum(log(s_j)))

        Returns:
            Tuple of (gradient, Hessian diagonal)
        """
        # Safeguard against log(0)
        x_safe = np.maximum(x, 1e-10)
        s_safe = np.maximum(s, 1e-10)

        # Gradient: [c - (1/mu)/x; -(1/mu)/s]
        grad_x = c - (1/self.mu) / x_safe
        grad_s = -(1/self.mu) / s_safe
        grad = np.concatenate([grad_x, grad_s])

        # Hessian diagonal: [(1/mu)/x^2; (1/mu)/s^2]
        hess_diag = np.concatenate([
            (1/self.mu) / (x_safe ** 2),
            (1/self.mu) / (s_safe ** 2)
        ])

        return grad, hess_diag

    def step(self, instance: OnlineLPInstance, x_star: np.ndarray,
             f_star: float) -> StepMetrics:
        """One step of feasible space projection.

        Algorithm:
        1. Project current (x, s) onto F (handles time-varying constraints)
        2. Compute Newton step with barrier Hessian
        3. Take Newton step
        4. Project result onto F via QP (instead of clipping!)

        Args:
            instance: Current problem instance
            x_star: True optimal solution (for metrics)
            f_star: True optimal objective value

        Returns:
            StepMetrics including algorithm-specific metrics
        """
        n, m = instance.n, instance.m
        p = instance.p

        # Current point
        y = np.concatenate([self.x, self.s])

        # Compute optimal slack for comparison
        s_star = instance.g - instance.F @ x_star
        y_star = np.concatenate([x_star, s_star])

        # Track distance before projection
        dist_before_proj = np.linalg.norm(y - y_star)

        # Step 1: Project onto F to handle time-varying constraints
        x_proj, s_proj = project_onto_feasible_set(
            self.x, self.s, instance.A, instance.b,
            instance.F, instance.g
        )

        # Track distance after initial projection
        y_proj = np.concatenate([x_proj, s_proj])
        dist_after_initial_proj = np.linalg.norm(y_proj - y_star)

        # Verify distance preservation property (should hold for convex set)
        initial_proj_preserved_dist = dist_after_initial_proj <= dist_before_proj + 1e-8

        # Step 2: Compute barrier gradient and Hessian at projected point
        grad, hess_diag = self._barrier_grad_hess(x_proj, s_proj, instance.c)
        H = np.diag(hess_diag)

        # Build augmented equality constraint matrix: [A, 0; F, I]
        A_aug = np.zeros((p + m, n + m))
        if p > 0:
            A_aug[:p, :n] = instance.A
        A_aug[p:, :n] = instance.F
        A_aug[p:, n:] = np.eye(m)

        # Step 3: Solve KKT system for Newton direction
        K = build_kkt_matrix(H, A_aug)
        rhs = np.zeros(n + m + p + m)
        rhs[:n+m] = -grad

        try:
            sol = np.linalg.solve(K, rhs)
            dy = sol[:n+m]
        except np.linalg.LinAlgError:
            # If KKT system is singular, skip Newton step
            dy = np.zeros(n + m)

        # Step 4: Take full Newton step
        y_newton = y_proj + dy
        x_newton = y_newton[:n]
        s_newton = y_newton[n:]

        # Track how many components violated bounds BEFORE final projection
        num_x_violated = int(np.sum(x_newton < 0))
        num_s_violated = int(np.sum(s_newton < 0))
        newton_exited_F = num_x_violated > 0 or num_s_violated > 0

        # Track distance after Newton (before final projection)
        dist_after_newton = np.linalg.norm(y_newton - y_star)

        # Step 5: Project back onto F via QP (THE KEY DIFFERENCE FROM CLIPPING!)
        x_new, s_new = project_onto_feasible_set(
            x_newton, s_newton, instance.A, instance.b,
            instance.F, instance.g
        )

        # Track distance after final projection
        y_new = np.concatenate([x_new, s_new])
        dist_after_final_proj = np.linalg.norm(y_new - y_star)

        # Verify distance preservation (should hold)
        final_proj_preserved_dist = dist_after_final_proj <= dist_after_newton + 1e-8

        # Track active set: which constraints are tight (s_i ≈ 0)?
        active_tol = 1e-6
        active_before = np.sum(s_proj < active_tol)
        active_after = np.sum(s_new < active_tol)
        active_set_changed = active_before != active_after

        # Update state
        self.x = x_new
        self.s = s_new

        # Compute metrics
        metrics = self.compute_common_metrics(x_new, instance, x_star, f_star)

        # Check constraint satisfaction
        slack_eq_violation = np.linalg.norm(
            instance.F @ x_new + s_new - instance.g
        )

        # Add algorithm-specific metrics
        metrics.extra = {
            # Distance tracking
            'distance_before_proj': dist_before_proj,
            'distance_after_initial_proj': dist_after_initial_proj,
            'distance_after_newton': dist_after_newton,
            'distance_after_final_proj': dist_after_final_proj,

            # Distance preservation verification
            'initial_proj_preserved_dist': initial_proj_preserved_dist,
            'final_proj_preserved_dist': final_proj_preserved_dist,

            # Newton step analysis
            'newton_exited_F': newton_exited_F,
            'num_x_violated': num_x_violated,
            'num_s_violated': num_s_violated,

            # Active set tracking
            'active_constraints_before': int(active_before),
            'active_constraints_after': int(active_after),
            'active_set_changed': active_set_changed,

            # Constraint satisfaction (should be ~0 for full projection)
            'slack_equality_violation': slack_eq_violation,
        }

        self.history.append(metrics)
        return metrics
