import cvxpy as cp
import numpy as np
from ..constants import *

class Solver:
    def __init__(self):
        pass

    def solve_problem(self, objective, constraints):
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        
        # Check if the problem is infeasible
        if problem.status == cp.INFEASIBLE:
            print("The problem is infeasible.")
            for i, constraint in enumerate(constraints):
                # print(f"Constraint {i}: {constraint}")
                dual_value = constraints[i].dual_value
                if dual_value is not None and dual_value > 0:
                    print(f"Constraint {i} is not satisfied. Dual value: {dual_value}")
            infeasibility_certificate = problem.value
            print("Infeasibility certificate:", infeasibility_certificate)