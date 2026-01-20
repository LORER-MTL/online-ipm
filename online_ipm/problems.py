"""Problem definitions and LP solving utilities for online optimization experiments."""

import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass


@dataclass
class OnlineLPInstance:
    """Single instance of the online LP at time t.

    Problem formulation:
        minimize    c^T x
        subject to  Ax = b
                    Fx ≤ g
                    x ≥ 0
    """
    c: np.ndarray      # Cost vector (n,)
    A: np.ndarray      # Equality constraint matrix (p, n)
    b: np.ndarray      # Equality RHS (p,)
    F: np.ndarray      # Inequality constraint matrix (m, n)
    g: np.ndarray      # Inequality RHS (m,)

    @property
    def n(self) -> int:
        """Number of decision variables."""
        return len(self.c)

    @property
    def p(self) -> int:
        """Number of equality constraints."""
        return self.A.shape[0] if self.A.size > 0 else 0

    @property
    def m(self) -> int:
        """Number of inequality constraints."""
        return self.F.shape[0]


def solve_lp(instance: OnlineLPInstance) -> tuple[np.ndarray, float]:
    """Solve LP exactly using scipy.linprog (HiGHS).

    Args:
        instance: The LP instance to solve

    Returns:
        Tuple of (optimal x, optimal objective value)

    Raises:
        ValueError: If LP solve fails
    """
    result = linprog(
        c=instance.c,
        A_eq=instance.A if instance.p > 0 else None,
        b_eq=instance.b if instance.p > 0 else None,
        A_ub=instance.F,
        b_ub=instance.g,
        bounds=(0, None),  # x >= 0
        method='highs'
    )
    if not result.success:
        raise ValueError(f"LP solve failed: {result.message}")
    return result.x, result.fun


class OnlineLPProblem:
    """Generator for time-varying LP instances.

    Stores the constant parts (c, A, F) and sequences of time-varying
    right-hand sides (b_t, g_t).
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, F: np.ndarray,
                 b_sequence: list[np.ndarray], g_sequence: list[np.ndarray]):
        """Initialize online LP problem.

        Args:
            c: Cost vector (constant over time)
            A: Equality constraint matrix (constant)
            F: Inequality constraint matrix (constant)
            b_sequence: List of equality RHS vectors b_t
            g_sequence: List of inequality RHS vectors g_t
        """
        self.c = c
        self.A = A
        self.F = F
        self.b_sequence = b_sequence
        self.g_sequence = g_sequence
        self.T = len(b_sequence)

        assert len(b_sequence) == len(g_sequence), "b and g sequences must have same length"

    def get_instance(self, t: int) -> OnlineLPInstance:
        """Get LP instance at time t."""
        return OnlineLPInstance(
            c=self.c, A=self.A, b=self.b_sequence[t],
            F=self.F, g=self.g_sequence[t]
        )


# ============ TEST PROBLEM GENERATORS ============

def create_simple_2d_problem(T: int = 50) -> OnlineLPProblem:
    """Create a simple 2D problem for visualization.

    minimize  -x1 - x2
    s.t.      x1 + x2 <= b_t   (time-varying upper bound)
              x1 <= g1_t       (time-varying)
              x2 <= g2_t       (time-varying)
              x >= 0

    The bounds have DECAYING variation over time to ensure sub-linear
    total variation V_T = O(sqrt(T)).

    Args:
        T: Number of time steps

    Returns:
        OnlineLPProblem instance
    """
    c = np.array([-1.0, -1.0])
    A = np.zeros((0, 2))  # No equality constraints

    # Inequality constraints: x1 + x2 <= b_t, x1 <= g1_t, x2 <= g2_t
    F = np.array([
        [1.0, 1.0],   # x1 + x2 <= b_t
        [1.0, 0.0],   # x1 <= g1_t
        [0.0, 1.0]    # x2 <= g2_t
    ])

    # Base values (feasible region)
    g_base = np.array([2.0, 1.2, 1.2])

    b_seq = [np.zeros(0)] * T  # Empty equality RHS
    g_seq = []
    for t in range(T):
        # Decaying amplitude: amplitude ~ 1/sqrt(t+1) gives V_T = O(sqrt(T))
        decay = 1.0 / np.sqrt(t + 1)
        g_t = g_base + decay * np.array([
            0.3 * np.sin(2*np.pi*t/20),           # x1 + x2 perturbation
            0.2 * np.sin(2*np.pi*t/20),           # x1 perturbation
            0.2 * np.sin(2*np.pi*t/20 + np.pi/3)  # x2 perturbation
        ])
        g_seq.append(g_t)

    return OnlineLPProblem(c, A, F, b_seq, g_seq)


def create_medium_problem(n: int = 10, m: int = 5, T: int = 100,
                          seed: int = 42) -> OnlineLPProblem:
    """Create a medium-sized random problem for realistic testing.

    Includes upper bounds on x to create a bounded feasible region.
    Uses DECAYING variation to ensure sub-linear total variation V_T = O(sqrt(T)).

    Args:
        n: Number of decision variables
        m: Number of general inequality constraints (in addition to x <= upper_bound)
        T: Number of time steps
        seed: Random seed for reproducibility

    Returns:
        OnlineLPProblem instance
    """
    rng = np.random.default_rng(seed)

    # Generate random problem data
    c = -np.abs(rng.standard_normal(n))  # Negative cost for minimization
    A = np.zeros((0, n))  # No equality constraints

    # F matrix: [random constraints; upper bound constraints x_i <= bound]
    F_random = rng.standard_normal((m, n))
    F_upper = np.eye(n)  # x_i <= upper_bound constraints
    F = np.vstack([F_random, F_upper])

    total_m = m + n  # Total number of inequality constraints

    # Generate a feasible interior point
    x_feas = np.abs(rng.standard_normal(n)) * 0.5 + 0.5  # x in [0.5, ~1.5]

    # Set up RHS:
    # - Random constraints: Fx + slack
    # - Upper bounds: generous but finite
    upper_bound = 3.0  # x_i <= 3
    g_random = F_random @ x_feas + 0.5  # Small slack for random constraints
    g_upper = np.full(n, upper_bound)
    g_base = np.concatenate([g_random, g_upper])

    # Generate random perturbation directions (only for random constraints)
    perturbation_dir = np.zeros(total_m)
    perturbation_dir[:m] = rng.standard_normal(m)
    perturbation_dir[:m] = perturbation_dir[:m] / np.linalg.norm(perturbation_dir[:m])

    b_seq = [np.zeros(0)] * T  # Empty equality RHS
    g_seq = []
    for t in range(T):
        # Decaying amplitude: amplitude ~ 1/sqrt(t+1) gives V_T = O(sqrt(T))
        decay = 1.0 / np.sqrt(t + 1)
        drift = 0.3 * decay * np.sin(2*np.pi*t/50)
        g_seq.append(g_base + drift * perturbation_dir)

    return OnlineLPProblem(c, A, F, b_seq, g_seq)
