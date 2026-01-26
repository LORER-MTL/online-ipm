"""Infeasible-Start Newton with Clipping Algorithm.

Approach 1 from the plan: After Newton step, clip s and x to ensure s >= ε and x >= ε.
Uses infeasible-start Newton which explicitly handles constraint residuals, so clipping
doesn't permanently break the method (unlike standard feasible-start methods).

The algorithm solves:
    minimize    c^T x
    subject to  Ax = b         (equality constraints)
                Fx + s = g     (slack form of Fx <= g)
                x >= 0, s >= 0

Algorithm (one iteration):
1. Compute residuals: r_primal = Ax - b, r_slack = Fx + s - g, r_dual = ∇f - μ/x + A^T λ + F^T ν
2. Form reduced KKT system: H = X^{-2} + F^T S^{-2} F
3. Solve for (Δx, Δλ), recover Δs = -r_slack - F Δx
4. Full step: x ← x + Δx, s ← s + Δs
5. Clip: x ← max(x, ε), s ← max(s, ε)

Key advantage: Infeasible-start IPM explicitly tracks and reduces residuals, so clipping
introduces "just another source of infeasibility" that gets corrected over iterations.
"""

import numpy as np

from .base import OnlineAlgorithm, StepMetrics
from ..problems import OnlineLPInstance
from ..open_m import build_kkt_matrix


class InfeasibleStartClippingAlgorithm(OnlineAlgorithm):
    """Infeasible-start Newton method with clipping for positivity.

    This approach handles inequality constraints Fx <= g by converting to
    slack form Fx + s = g, s >= 0. After each Newton step, we clip x and s
    to ensure positivity, relying on the infeasible-start framework to
    correct the resulting constraint violations.

    The method tracks three residuals:
    - r_primal: Ax - b (equality constraint violation)
    - r_slack: Fx + s - g (slack constraint violation)
    - r_dual: gradient of Lagrangian (dual feasibility)

    Clipping introduces violations in r_slack, but infeasible-start Newton
    explicitly drives all residuals toward zero.
    """

    def __init__(self, n: int, m: int, mu: float = 1.0, eps: float = 1e-4):
        """Initialize infeasible-start clipping algorithm.

        Args:
            n: Number of primal variables
            m: Number of inequality constraints
            mu: Barrier parameter (fixed)
            eps: Clipping threshold (x, s clipped to >= eps)
        """
        super().__init__(n, m)
        self.mu = mu
        self.eps = eps
        self.x: np.ndarray = None
        self.s: np.ndarray = None  # Slack variables for Fx + s = g
        self.lam: np.ndarray = None  # Dual for Ax = b
        self.nu: np.ndarray = None  # Dual for Fx + s = g

    def initialize(self, instance: OnlineLPInstance, x_init: np.ndarray) -> None:
        """Initialize with a starting point.

        Args:
            instance: Initial problem instance
            x_init: Initial primal solution (should satisfy Ax ≈ b)
        """
        self.x = np.maximum(x_init.copy(), self.eps)

        # Initialize slack: s = g - Fx (ensure s > 0)
        s_init = instance.g - instance.F @ self.x
        self.s = np.maximum(s_init, self.eps)

        # Initialize duals
        self.lam = np.zeros(instance.p)  # Dual for Ax = b
        self.nu = self.mu / self.s  # Complementarity: nu_i * s_i = mu

    def get_current_x(self) -> np.ndarray:
        """Return current primal solution."""
        return self.x.copy()

    def step(self, instance: OnlineLPInstance, x_star: np.ndarray,
             f_star: float) -> StepMetrics:
        """One step of infeasible-start Newton with clipping.

        The Newton system for infeasible-start IPM is:

        [H    0    A^T   F^T ] [Δx ]   [r_dual  ]
        [0    S^2   0    I   ] [Δs ]   [S*ν - μ ]
        [A    0    0    0   ] [Δλ ] = -[r_primal]
        [F    I    0    0   ] [Δν ]   [r_slack ]

        where H = diag(μ/x_i^2) is the barrier Hessian for x >= 0.

        We use Schur complement to reduce this to a smaller system.

        Args:
            instance: Current problem instance
            x_star: True optimal solution (for metrics)
            f_star: True optimal objective value

        Returns:
            StepMetrics including residual norms and clipping counts
        """
        n = instance.n
        p = instance.p
        m = instance.m

        # Ensure positivity (may have been violated by previous step)
        self.x = np.maximum(self.x, self.eps)
        self.s = np.maximum(self.s, self.eps)

        # Compute residuals
        r_primal = instance.A @ self.x - instance.b if p > 0 else np.zeros(0)
        r_slack = instance.F @ self.x + self.s - instance.g

        # Dual residual: c - μ/x + A^T λ + F^T ν = 0 for optimality
        # (For LP, the barrier gradient replaces c with c + μ * sum(1/x_i))
        grad_barrier_x = self.mu / self.x  # Gradient of -μ*sum(log(x_i)) w.r.t. x
        r_dual = instance.c - grad_barrier_x
        if p > 0:
            r_dual = r_dual + instance.A.T @ self.lam
        r_dual = r_dual + instance.F.T @ self.nu

        # Complementarity residual for slack
        r_cent_s = self.s * self.nu - self.mu  # Should be zero at optimum

        # Build and solve the KKT system using Schur complement
        # We eliminate Δs and Δν to get a system in (Δx, Δλ)
        #
        # From the slack equation: Δs = -r_slack - F*Δx
        # From complementarity: S*Δν + ν*Δs = -r_cent_s
        #   => Δν = S^{-1}*(-r_cent_s - ν*Δs) = S^{-1}*(-r_cent_s - ν*(-r_slack - F*Δx))
        #   => Δν = S^{-1}*(ν*F*Δx + ν*r_slack - r_cent_s)
        #
        # Substituting into the Δx equation:
        #   H*Δx + A^T*Δλ + F^T*Δν = -r_dual
        #   H*Δx + A^T*Δλ + F^T*S^{-1}*(ν*F*Δx + ν*r_slack - r_cent_s) = -r_dual
        #   (H + F^T*S^{-1}*diag(ν)*F)*Δx + A^T*Δλ = -r_dual - F^T*S^{-1}*(ν*r_slack - r_cent_s)

        # Barrier Hessian for x >= 0
        X_inv2 = self.mu / (self.x ** 2)  # Diagonal of barrier Hessian
        S_inv = 1.0 / self.s
        nu_over_s = self.nu * S_inv  # = ν/s

        # Reduced Hessian: H_red = diag(μ/x^2) + F^T * diag(ν/s) * F
        H_red = np.diag(X_inv2) + instance.F.T @ np.diag(nu_over_s) @ instance.F

        # Reduced RHS for x-equation
        rhs_x = -r_dual - instance.F.T @ (S_inv * (self.nu * r_slack - r_cent_s))

        # Build KKT system: [H_red, A^T; A, 0] [Δx; Δλ] = [rhs_x; -r_primal]
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
            # Singular system - take no step
            dx = np.zeros(n)
            dlam = np.zeros(p)

        # Recover Δs and Δν
        ds = -r_slack - instance.F @ dx
        F_dx = instance.F @ dx
        dnu = S_inv * (self.nu * F_dx + self.nu * r_slack - r_cent_s)

        # Full Newton step
        x_new = self.x + dx
        s_new = self.s + ds
        lam_new = self.lam + dlam if p > 0 else np.zeros(0)
        nu_new = self.nu + dnu

        # Clip to maintain positivity
        num_clipped_x = np.sum(x_new < self.eps)
        num_clipped_s = np.sum(s_new < self.eps)

        self.x = np.maximum(x_new, self.eps)
        self.s = np.maximum(s_new, self.eps)
        self.lam = lam_new
        self.nu = np.maximum(nu_new, self.eps)  # Also clip dual to stay positive

        # Compute metrics
        metrics = self.compute_common_metrics(self.x, instance, x_star, f_star)
        metrics.extra = {
            'r_primal_norm': float(np.linalg.norm(r_primal)),
            'r_slack_norm': float(np.linalg.norm(r_slack)),
            'r_dual_norm': float(np.linalg.norm(r_dual)),
            'num_clipped_x': int(num_clipped_x),
            'num_clipped_s': int(num_clipped_s),
            'num_clipped': int(num_clipped_x + num_clipped_s),
            'step_norm': float(np.linalg.norm(dx)),
            'min_x': float(np.min(self.x)),
            'min_s': float(np.min(self.s)),
        }

        self.history.append(metrics)
        return metrics
