"""Barrier Reformulation Algorithm.

From barrier_reformulation_analysis.md:
1. Add log barrier: min c^T x + (1/μ) * sum(-log(g_i - F_i x))
2. Apply OPEN-M with barrier objective (equality constraints Ax = b)
3. Take FULL Newton step (no line search)

This approach fails because:
- Newton step can exit feasible region (Fx ≥ g)
- Hessian conditioning blows up near boundary
- Lipschitz Hessian assumption fails
- Barrier optimum ≠ true optimum (O(1/μ) gap)
"""

import numpy as np

from .base import OnlineAlgorithm, StepMetrics
from ..problems import OnlineLPInstance
from ..open_m import build_kkt_matrix


class BarrierMethodAlgorithm(OnlineAlgorithm):
    """Barrier reformulation method with OPEN-M style full Newton steps.

    This demonstrates failures when taking full Newton steps without line search.
    The standard OPEN-M analysis requires bounded Newton decrement and Lipschitz
    Hessian, both of which fail for barrier functions near the boundary.

    Issues demonstrated:
    1. Newton step exits feasible region (no line search)
    2. Hessian condition number explodes near boundary
    3. Lipschitz Hessian assumption fails
    4. Barrier optimum has O(1/μ) gap from true LP optimum
    """

    def __init__(self, n: int, m: int, mu: float = 10.0):
        """Initialize barrier method algorithm.

        Args:
            n: Number of primal variables
            m: Number of inequality constraints
            mu: Barrier parameter (larger = closer to true problem, but worse conditioning)
        """
        super().__init__(n, m)
        self.mu = mu  # Barrier parameter
        self.x: np.ndarray = None
        self.feasible: bool = True

    def initialize(self, instance: OnlineLPInstance, x_init: np.ndarray) -> None:
        """Initialize with a strictly feasible point.

        Args:
            instance: Initial problem instance
            x_init: Initial primal solution (should be strictly feasible: Fx < g)
        """
        self.x = x_init.copy()
        # Check feasibility
        slack = instance.g - instance.F @ self.x
        self.feasible = np.all(slack > 0)

    def get_current_x(self) -> np.ndarray:
        """Return current primal solution."""
        return self.x.copy()

    def _barrier_gradient(self, x: np.ndarray, instance: OnlineLPInstance) -> np.ndarray:
        """Gradient of c^T x + (1/μ) * sum(-log(g_i - F_i x)).

        The barrier gradient is:
            c + (1/μ) * F^T @ (1/slack)

        where slack = g - Fx.

        Args:
            x: Current point
            instance: Problem instance

        Returns:
            Gradient vector, or inf if infeasible
        """
        slack = instance.g - instance.F @ x
        if np.any(slack <= 0):
            return np.full(len(x), np.inf)
        return instance.c + (1/self.mu) * instance.F.T @ (1/slack)

    def _barrier_hessian(self, x: np.ndarray, instance: OnlineLPInstance) -> np.ndarray:
        """Hessian of barrier function.

        The barrier Hessian is:
            (1/μ) * F^T diag(1/slack^2) F

        This has eigenvalues that blow up as slack -> 0.

        Args:
            x: Current point
            instance: Problem instance

        Returns:
            Hessian matrix, or inf if infeasible
        """
        slack = instance.g - instance.F @ x
        if np.any(slack <= 0):
            return np.full((len(x), len(x)), np.inf)
        weights = 1 / (slack ** 2)
        return (1/self.mu) * instance.F.T @ np.diag(weights) @ instance.F

    def step(self, instance: OnlineLPInstance, x_star: np.ndarray,
             f_star: float) -> StepMetrics:
        """One step of barrier method with FULL Newton step (no line search).

        This is what causes failures - standard IPM always uses line search
        to stay in the feasible region!

        Args:
            instance: Current problem instance
            x_star: True optimal solution (for metrics)
            f_star: True optimal objective value

        Returns:
            StepMetrics including algorithm-specific failure indicators
        """
        n = instance.n
        p = instance.p

        # Check current feasibility
        slack = instance.g - instance.F @ self.x
        if np.any(slack <= 0):
            self.feasible = False
            metrics = self.compute_common_metrics(self.x, instance, x_star, f_star)
            metrics.extra = {
                'min_slack': float(np.min(slack)),
                'feasible': False,
                'hessian_condition': np.inf,
                'newton_step_norm': 0.0,
            }
            self.history.append(metrics)
            return metrics

        # Compute barrier gradient and Hessian
        grad = self._barrier_gradient(self.x, instance)
        H = self._barrier_hessian(self.x, instance)
        H = H + 1e-10 * np.eye(n)  # Small regularization for numerical stability

        # Compute condition number (tracks Issue 2: conditioning catastrophe)
        try:
            eigvals = np.linalg.eigvalsh(H)
            cond = eigvals.max() / max(eigvals.min(), 1e-15)
        except np.linalg.LinAlgError:
            cond = np.inf

        # Solve KKT system for Newton direction
        try:
            K = build_kkt_matrix(H, instance.A)
            rhs = np.zeros(n + p)
            rhs[:n] = -grad
            sol = np.linalg.solve(K, rhs)
            dx = sol[:n]
        except np.linalg.LinAlgError:
            self.feasible = False
            metrics = self.compute_common_metrics(self.x, instance, x_star, f_star)
            metrics.extra = {
                'min_slack': float(np.min(slack)),
                'feasible': False,
                'hessian_condition': cond,
                'newton_step_norm': 0.0,
            }
            self.history.append(metrics)
            return metrics

        # Take FULL Newton step (THIS IS THE PROBLEM - no line search!)
        x_new = self.x + dx

        # Check if we exited feasible region
        new_slack = instance.g - instance.F @ x_new
        self.feasible = np.all(new_slack > 0)

        # Update state
        self.x = x_new

        # Compute metrics
        metrics = self.compute_common_metrics(x_new, instance, x_star, f_star)
        metrics.extra = {
            'min_slack': float(np.min(new_slack)),
            'feasible': self.feasible,
            'hessian_condition': cond,
            'newton_step_norm': np.linalg.norm(dx),
        }

        self.history.append(metrics)
        return metrics
