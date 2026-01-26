"""Primal-Dual IPM with Line Search Algorithm.

Approach 2 from the plan: Instead of clipping, use backtracking line search to ensure
x + α*Δx > 0 and s + α*Δs > 0. Maintains strict feasibility throughout.

The algorithm solves:
    minimize    c^T x
    subject to  Ax = b         (equality constraints)
                Fx + s = g     (slack form of Fx <= g)
                x >= 0, s >= 0

Algorithm (one iteration):
1. If constraints changed, project (x, s) back to feasible set
2. Compute Newton direction via full primal-dual KKT system
3. Line search: α_max = min over i of (-x_i/Δx_i for Δx_i < 0), similarly for s
4. Step: α = τ · min(α_x_max, α_s_max) where τ = 0.995
5. Update: x ← x + α·Δx, s ← s + α·Δs

Key advantage: Maintains strict feasibility (x > 0, s > 0) at all times by
backing off from the boundary. This is the standard approach in production IPM solvers.
"""

import numpy as np

from .base import OnlineAlgorithm, StepMetrics
from ..problems import OnlineLPInstance
from ..open_m import build_kkt_matrix


class PrimalDualLineSearchAlgorithm(OnlineAlgorithm):
    """Primal-dual interior point method with line search.

    This approach handles inequality constraints Fx <= g by converting to
    slack form Fx + s = g, s >= 0. Line search ensures we never violate
    the positivity constraints x > 0, s > 0.

    The method solves the primal-dual system and uses the fraction-to-boundary
    rule to determine step sizes that maintain strict feasibility.

    This is more conservative than clipping but provides better stability
    and guaranteed feasibility.
    """

    def __init__(self, n: int, m: int, mu: float = 1.0, tau: float = 0.995,
                 min_step: float = 1e-8):
        """Initialize primal-dual line search algorithm.

        Args:
            n: Number of primal variables
            m: Number of inequality constraints
            mu: Barrier parameter (fixed)
            tau: Step size reduction factor (typically 0.99 or 0.995)
            min_step: Minimum step size before declaring failure
        """
        super().__init__(n, m)
        self.mu = mu
        self.tau = tau
        self.min_step = min_step
        self.x: np.ndarray = None
        self.s: np.ndarray = None  # Slack variables for Fx + s = g
        self.lam: np.ndarray = None  # Dual for Ax = b
        self.nu: np.ndarray = None  # Dual for Fx + s = g (multiplier for s >= 0)

    def initialize(self, instance: OnlineLPInstance, x_init: np.ndarray) -> None:
        """Initialize with a strictly feasible starting point.

        Args:
            instance: Initial problem instance
            x_init: Initial primal solution (should satisfy Ax ≈ b)
        """
        eps = 1e-4
        self.x = np.maximum(x_init.copy(), eps)

        # Initialize slack: s = g - Fx (ensure s > 0)
        s_init = instance.g - instance.F @ self.x
        self.s = np.maximum(s_init, eps)

        # Initialize duals
        self.lam = np.zeros(instance.p)  # Dual for Ax = b
        self.nu = self.mu / self.s  # Complementarity: nu_i * s_i = mu

    def get_current_x(self) -> np.ndarray:
        """Return current primal solution."""
        return self.x.copy()

    def _compute_max_step(self, z: np.ndarray, dz: np.ndarray) -> float:
        """Compute maximum step size to maintain z + α*dz > 0.

        Uses the fraction-to-boundary rule:
            α_max = min over i of (-z_i / dz_i) for dz_i < 0

        Args:
            z: Current point (must be > 0)
            dz: Step direction

        Returns:
            Maximum step size that keeps z positive
        """
        neg_mask = dz < -1e-12  # Avoid division by tiny numbers
        if not np.any(neg_mask):
            return 1.0  # Can take full step

        ratios = -z[neg_mask] / dz[neg_mask]
        return float(np.min(ratios))

    def _project_to_feasible(self, instance: OnlineLPInstance) -> None:
        """Project (x, s) to satisfy Fx + s = g approximately.

        When constraints change, the current (x, s) may violate Fx + s = g.
        We adjust s to restore feasibility while maintaining s > 0.

        Args:
            instance: Current problem instance
        """
        eps = 1e-4
        # Compute what s should be
        s_target = instance.g - instance.F @ self.x
        # Blend toward target while maintaining positivity
        self.s = np.maximum(s_target, eps)
        # Update nu to maintain complementarity
        self.nu = self.mu / self.s

    def step(self, instance: OnlineLPInstance, x_star: np.ndarray,
             f_star: float) -> StepMetrics:
        """One step of primal-dual IPM with line search.

        The Newton system is the same as in infeasible-start, but we use
        line search instead of clipping to maintain feasibility.

        Args:
            instance: Current problem instance
            x_star: True optimal solution (for metrics)
            f_star: True optimal objective value

        Returns:
            StepMetrics including step sizes and duality gap
        """
        n = instance.n
        p = instance.p
        m = instance.m
        eps = 1e-6

        # Project to handle changing constraints
        self._project_to_feasible(instance)

        # Ensure strict positivity
        self.x = np.maximum(self.x, eps)
        self.s = np.maximum(self.s, eps)
        self.nu = np.maximum(self.nu, eps)

        # Compute residuals
        r_primal = instance.A @ self.x - instance.b if p > 0 else np.zeros(0)
        r_slack = instance.F @ self.x + self.s - instance.g

        # Dual residual: c - μ/x + A^T λ + F^T ν = 0
        grad_barrier_x = self.mu / self.x
        r_dual = instance.c - grad_barrier_x
        if p > 0:
            r_dual = r_dual + instance.A.T @ self.lam
        r_dual = r_dual + instance.F.T @ self.nu

        # Complementarity residual
        r_cent_s = self.s * self.nu - self.mu

        # Build reduced KKT system (same as infeasible-start)
        X_inv2 = self.mu / (self.x ** 2)
        S_inv = 1.0 / self.s
        nu_over_s = self.nu * S_inv

        H_red = np.diag(X_inv2) + instance.F.T @ np.diag(nu_over_s) @ instance.F
        rhs_x = -r_dual - instance.F.T @ (S_inv * (self.nu * r_slack - r_cent_s))

        K = build_kkt_matrix(H_red, instance.A)
        rhs = np.zeros(n + p)
        rhs[:n] = rhs_x
        if p > 0:
            rhs[n:] = -r_primal

        # Solve
        try:
            sol = np.linalg.solve(K, rhs)
            dx = sol[:n]
            dlam = sol[n:] if p > 0 else np.zeros(0)
        except np.linalg.LinAlgError:
            dx = np.zeros(n)
            dlam = np.zeros(p)

        # Recover Δs and Δν
        ds = -r_slack - instance.F @ dx
        F_dx = instance.F @ dx
        dnu = S_inv * (self.nu * F_dx + self.nu * r_slack - r_cent_s)

        # Line search: compute maximum step sizes
        alpha_x_max = self._compute_max_step(self.x, dx)
        alpha_s_max = self._compute_max_step(self.s, ds)
        alpha_nu_max = self._compute_max_step(self.nu, dnu)

        # Use separate step sizes for primal and dual (common in IPM)
        alpha_primal = self.tau * min(alpha_x_max, alpha_s_max)
        alpha_dual = self.tau * min(alpha_nu_max, 1.0)

        # Ensure minimum step
        alpha_primal = max(alpha_primal, self.min_step)
        alpha_dual = max(alpha_dual, self.min_step)

        # Take the step
        self.x = self.x + alpha_primal * dx
        self.s = self.s + alpha_primal * ds
        if p > 0:
            self.lam = self.lam + alpha_dual * dlam
        self.nu = self.nu + alpha_dual * dnu

        # Compute duality gap: sum(x_i * (c_i + barrier_grad)) + sum(s_i * nu_i)
        # For LP, duality gap ≈ μ * (n + m) at the barrier optimum
        duality_gap = np.sum(self.x * grad_barrier_x) + np.sum(self.s * self.nu)

        # Compute metrics
        metrics = self.compute_common_metrics(self.x, instance, x_star, f_star)
        metrics.extra = {
            'alpha_primal': float(alpha_primal),
            'alpha_dual': float(alpha_dual),
            'alpha_x_max': float(alpha_x_max),
            'alpha_s_max': float(alpha_s_max),
            'min_x': float(np.min(self.x)),
            'min_s': float(np.min(self.s)),
            'duality_gap': float(duality_gap),
            'r_primal_norm': float(np.linalg.norm(r_primal)),
            'r_slack_norm': float(np.linalg.norm(r_slack)),
            'r_dual_norm': float(np.linalg.norm(r_dual)),
            'step_norm': float(np.linalg.norm(dx)),
        }

        self.history.append(metrics)
        return metrics
