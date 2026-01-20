"""Slack Variable Projection Algorithm.

From slack_variable_analysis.md:
1. Reformulate Fx ≤ g as Fx + s = g with s ≥ 0
2. Use log barrier on x >= 0 and s >= 0
3. Apply OPEN-M style projection + full Newton step
4. Clip s to s ≥ 0 when it goes negative (the problematic step)

This approach fails because:
- Clipping can increase distance to optimum
- Equality Fx + s = g is violated after clipping
- Information about constraint violations is destroyed
"""

import numpy as np

from .base import OnlineAlgorithm, StepMetrics
from ..problems import OnlineLPInstance
from ..open_m import project_onto_equality, build_kkt_matrix


class SlackProjectionAlgorithm(OnlineAlgorithm):
    """Slack variable projection method for inequality constraints.

    This implements the PROBLEMATIC approach that clips slack variables,
    demonstrating why naive extension to inequalities fails.

    The algorithm:
    1. Augments variables: y = [x; s] where s = g - Fx (slack)
    2. Uses log barrier on x >= 0 and s >= 0
    3. Projects onto equality constraints [A, 0; F, I] @ y = [b; g]
    4. Takes FULL Newton step (no line search, like OPEN-M)
    5. Clips s to s >= 0 (THIS IS THE FAILURE POINT)
    """

    def __init__(self, n: int, m: int, mu: float = 1.0):
        """Initialize slack projection algorithm.

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
        n, m = len(x), len(s)

        # Gradient: [c - (1/mu)/x; -(1/mu)/s]
        # Handle infeasible points gracefully
        x_safe = np.maximum(x, 1e-10)
        s_safe = np.maximum(s, 1e-10)

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
        """One step of slack variable projection.

        Algorithm:
        1. Form augmented system: [A, 0; F, I] @ [x; s] = [b; g]
        2. Project onto equality constraints (new b_t, g_t)
        3. Full Newton step with barrier Hessian (no line search!)
        4. Clip s to s >= 0 (THE FAILURE POINT)

        Args:
            instance: Current problem instance
            x_star: True optimal solution (for metrics)
            f_star: True optimal objective value

        Returns:
            StepMetrics including algorithm-specific failure indicators
        """
        n, m = instance.n, instance.m
        p = instance.p

        # Build augmented system: y = [x; s]
        y = np.concatenate([self.x, self.s])

        # Augmented equality constraint matrix: [A, 0; F, I]
        A_aug = np.zeros((p + m, n + m))
        if p > 0:
            A_aug[:p, :n] = instance.A
        A_aug[p:, :n] = instance.F
        A_aug[p:, n:] = np.eye(m)
        b_aug = np.concatenate([instance.b, instance.g])

        # Step 1: Project onto equality constraints (handle new b_t, g_t)
        y_proj = project_onto_equality(y, A_aug, b_aug)
        x_proj = y_proj[:n]
        s_proj = y_proj[n:]

        # Step 2: Compute barrier gradient and Hessian at projected point
        grad, hess_diag = self._barrier_grad_hess(x_proj, s_proj, instance.c)
        H = np.diag(hess_diag)

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

        # Take FULL Newton step (no line search - this can cause issues!)
        y_new = y_proj + dy

        x_new = y_new[:n]
        s_before_clip = y_new[n:]

        # Track how many x components went negative
        num_x_negative = int(np.sum(x_new < 0))

        # Compute distance BEFORE clipping (for tracking)
        s_star = instance.g - instance.F @ x_star
        dist_before = np.linalg.norm(
            np.concatenate([x_new - x_star, s_before_clip - s_star])
        )

        # Step 4: CLIP slack variables to s >= 0 (THIS CAUSES FAILURES!)
        # In a proper IPM, we'd use line search to stay feasible
        num_s_clipped = int(np.sum(s_before_clip < 0))
        s_after_clip = np.maximum(s_before_clip, 1e-8)  # Small positive to avoid log(0)

        # Also clip x to x >= 0 to maintain barrier feasibility
        num_x_clipped = int(np.sum(x_new < 0))
        x_after_clip = np.maximum(x_new, 1e-8)

        # Compute distance AFTER clipping
        dist_after = np.linalg.norm(
            np.concatenate([x_after_clip - x_star, s_after_clip - s_star])
        )

        # Update state (use clipped values)
        self.x = x_after_clip
        self.s = s_after_clip

        # Compute metrics using clipped x
        metrics = self.compute_common_metrics(x_after_clip, instance, x_star, f_star)

        # Add algorithm-specific metrics
        slack_eq_violation = np.linalg.norm(
            instance.F @ x_after_clip + s_after_clip - instance.g
        )
        metrics.extra = {
            'distance_before_clip': dist_before,
            'distance_after_clip': dist_after,
            'clip_increased_distance': dist_after > dist_before + 1e-10,
            'num_s_clipped': num_s_clipped,
            'num_x_clipped': num_x_clipped,
            'slack_equality_violation': slack_eq_violation,
        }

        self.history.append(metrics)
        return metrics
