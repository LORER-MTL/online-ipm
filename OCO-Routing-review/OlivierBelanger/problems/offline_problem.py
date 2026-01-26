import cvxpy as cp
import numpy as np
from .base_problem import BaseProblem
from ..constants import *
from ..solvers.solver import Solver


class OfflineProblem(BaseProblem):
    def __init__(self, name, start_t, end_t, flows, strategy="dynamic"):
        """Initialize an OfflineProblem with specified parameters."""
        super().__init__(name, start_t, end_t)
        self.strategy = strategy
        self.initialize_decision_variables()
        self.flows = flows
    
    def initialize_decision_variables(self):
        """Initialize decision variables based on the chosen strategy."""
        if self.strategy == "proportional":
            self.w = {t: np.zeros((P, M)) for t in range(T)}
        else:
            self.w = {t: cp.Variable((P, M), nonneg=True, name = "w") for t in range(T)}
        
        self.prev_w = {t: cp.Variable((P, M), name = "w", nonneg=True) for t in range(T)}  # Incoming flow

    def create_objective(self, start_t, end_t):
        """Create the objective function for the optimization problem."""
        return cp.Minimize(cp.sum([cp.sum(cp.multiply(cp.sum(self.L[t], axis=1), k_p)) for t in range(start_t, end_t)]))

    def create_constraints(self, start_t, end_t, F):
        """Create constraints for the optimization problem."""
        constraints = []
        
        constraints.append(self.Q[0][:, :] == Q_0)
        # constraints.append(self.delta_Q[0][:, :] >= Q_0) # Constraint 8

        for t in range(start_t, end_t):
            constraints.extend([cp.sum(self.w[t][:P, m]) == 1 for m in range(M)]) # Constraint 2
            constraints.extend([cp.sum(self.Q[t][:P, m] + self.delta_Q[t][:P, m]) <= Q_BAR for m in range(M)])
            # constraints.extend([cp.sum(self.Q[t][:P, m]) <= Q_BAR for m in range(M)])
            # constraints.extend([cp.sum(self.delta_Q[t][:P, m]) <= Q_BAR for m in range(M)])
            constraints.extend([cp.sum(self.f_in[t][p, :M]) == F[t, p] for p in range(P)])
            
            for p in range(P):
                if self.strategy == "proportional":
                    self.w[t][p, :] = k_p[p] / sum(k_p)
                    
                for m in range(M):
                    # Define shared constraints
                    constraints.append(self.f_out[t][p, m] * DELTA_S <= self.w[t][p, m])
                    constraints.append(cp.sum(self.f_in[t][p, m] - self.f_out[t][p, m] - self.delta_Q[t][p, m]) == self.L[t][p,m])
                    
                    # Constraints 5,6
                    constraints.append(self.Q[T-1][p, m] == Q_0)
                    constraints.append(self.delta_Q[T-1][p, m] == Q_0) # Constraint 7
                    
                    if t < T - 1:
                        constraints.append(self.Q[t+1][p, m] == self.Q[t][p, m] + self.delta_Q[t][p, m])
                        constraints.append(self.prev_w[t+1] == self.w[t])
                    if t > 0:
                        constraints.append(cp.abs(self.w[t][p, m] - self.w[t-1][p, m]) <= DELTA_W_BAR)
                    

        if self.strategy == "fixed":
            for t in range(1, T):
                for p in range(P):
                    for m in range(M):
                        constraints.append(self.w[t][p, m] == self.w[0][p, m])

        return constraints


    def initialize_problem(self, start_t =0, end_t =T):
        """Initialize the problem with objective and constraints."""
        F = self.flows[start_t:end_t, :]
        objective = self.create_objective(start_t, end_t)
        constraints = self.create_constraints(start_t, end_t, F)
        return objective, constraints

    def solve(self):
        """Solve the optimization problem using a Solver instance."""
        solver = Solver()
        objective, constraints = self.initialize_problem(start_t =0, end_t =T) 
        solver.solve_problem(objective,constraints)
        self.store_results()

    def get_Q_values(self):
        """Getter method for Q_values."""
        return {key: value for key, value in self.Q_values.items()}

    def get_L_values(self):
        """Getter method for L_values."""
        return {key: value for key, value in self.L_values.items()}

    def get_f_out_values(self):
        """Getter method for f_out values."""
        return {key: value for key, value in self.f_out.items()}