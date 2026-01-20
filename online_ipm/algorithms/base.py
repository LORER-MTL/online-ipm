"""Abstract base class for online optimization algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from ..problems import OnlineLPInstance


@dataclass
class StepMetrics:
    """Metrics collected at each timestep.

    Common metrics are computed for all algorithms. Algorithm-specific
    metrics are stored in the `extra` dictionary.
    """
    # Common metrics
    regret: float = 0.0                  # c^T x_t - c^T x*_t
    cumulative_regret: float = 0.0       # Sum of regrets up to t
    distance_to_optimum: float = 0.0     # ||x_t - x*_t||
    equality_violation: float = 0.0      # ||Ax - b||
    inequality_violation: float = 0.0    # ||max(Fx - g, 0)||

    # Algorithm-specific metrics (filled by subclasses)
    extra: dict = field(default_factory=dict)


class OnlineAlgorithm(ABC):
    """Abstract base class for online optimization algorithms.

    All algorithms must implement:
    - initialize(): Set up initial state
    - step(): Perform one iteration given new problem data
    - get_current_x(): Return current solution
    """

    def __init__(self, n: int, m: int):
        """Initialize algorithm.

        Args:
            n: Number of primal variables
            m: Number of inequality constraints
        """
        self.n = n  # Number of primal variables
        self.m = m  # Number of inequality constraints
        self.cumulative_regret = 0.0
        self.history: list[StepMetrics] = []

    @abstractmethod
    def initialize(self, instance: OnlineLPInstance, x_init: np.ndarray) -> None:
        """Initialize the algorithm state.

        Args:
            instance: Initial problem instance
            x_init: Initial primal solution
        """
        pass

    @abstractmethod
    def step(self, instance: OnlineLPInstance, x_star: np.ndarray,
             f_star: float) -> StepMetrics:
        """Perform one step of the algorithm.

        Args:
            instance: Current problem instance
            x_star: True optimal solution (for metric computation)
            f_star: True optimal objective value

        Returns:
            StepMetrics for this timestep
        """
        pass

    @abstractmethod
    def get_current_x(self) -> np.ndarray:
        """Return current primal solution."""
        pass

    def compute_common_metrics(self, x: np.ndarray, instance: OnlineLPInstance,
                               x_star: np.ndarray, f_star: float) -> StepMetrics:
        """Compute metrics common to all algorithms.

        Args:
            x: Current solution
            instance: Current problem instance
            x_star: True optimal solution
            f_star: True optimal objective value

        Returns:
            StepMetrics with common fields filled in
        """
        # Regret
        regret = instance.c @ x - f_star
        self.cumulative_regret += regret

        # Violations
        eq_viol = np.linalg.norm(instance.A @ x - instance.b) if instance.p > 0 else 0.0
        ineq_viol = np.linalg.norm(np.maximum(instance.F @ x - instance.g, 0))

        return StepMetrics(
            regret=regret,
            cumulative_regret=self.cumulative_regret,
            distance_to_optimum=np.linalg.norm(x - x_star),
            equality_violation=eq_viol,
            inequality_violation=ineq_viol
        )
