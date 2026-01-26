from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np

from ..constants import *

class BaseProblem(ABC):
    """
    Abstract base class for optimization problems. It defines the structure and 
    necessary methods that all derived problem classes must implement. 
    This class handles the initialization of common decision variables, storage of results, 
    and provides abstract methods for problem-specific implementations.
    """

    def __init__(self, name, start_t, end_t):
        """
        Initializes the BaseProblem with given name, start, and end times.

        :param name: Name of the problem.
        :param start_t: Start time of the problem.
        :param end_t: End time of the problem.
        """
        self.name = name

        # Data structures for storing results
        self.weights = {t: {p: {m: [] for m in range(M)} for p in range(P)} for t in range(T)} # For plotting purposes 
        self.L_values = []
        self.Q_values = {0: {t: np.zeros((P, M)) for t in range(T)}} # Initial state for Q_values

        # Decision Variables
        self.f_in = {t: cp.Variable((P, M), name = "f_in", nonneg=True) for t in range(T)}  # Incoming flow
        self.f_out = {t: cp.Variable((P, M), name = "f_out", nonneg=True) for t in range(T)}  # Outgoing flow
        self.delta_Q = {t: cp.Variable((P, M), name = "delta_Q", nonneg=False) for t in range(T)}  # Queue flow
        self.Q = {t: cp.Variable((P, M), name = "Q", nonneg=True) for t in range(T)}  # Queue flow
        self.L = {t: cp.Variable((P, M), name = "L", nonneg=True) for t in range(T)}  # Total packet loss

    @abstractmethod
    def initialize_decision_variables(self):
        """Initializes decision variables specific to the problem."""
        pass

    @abstractmethod
    def create_objective(self, start_t, end_t):
        """Creates the objective function for the optimization problem."""
        pass

    @abstractmethod
    def create_constraints(self, start_t, end_t, F):
        """Creates constraints for the optimization problem."""
        pass
    
    @abstractmethod
    def initialize_problem(self, start_t, end_t):
        """Initializes the problem with the given start and end times."""
        pass

    @abstractmethod
    def solve(self):
        """Solves the optimization problem."""
        pass

    def store_results(self):
        """
        Stores the results of the optimization problem. This includes
        incoming and outgoing flows, queue values, and weights.
        """
        self.f_in_all = self.f_in
        self.f_out_all = self.f_out
        for t in range(T):
            self.L_values.append(self.L[t].value if isinstance(self.L[t], cp.Variable) else self.L[t])
            self.Q_values[t] = self.Q[t].value if isinstance(self.Q[t], cp.Variable) else self.Q[t]


    def get_f_in_values(self):
        """Getter method for f_in values."""
        return {key: value for key, value in self.f_in.items()}


    def __str__(self):
        """String representation of the problem."""
        return f"BaseProblem: {self.name}"
