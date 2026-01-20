"""Online Interior Point Methods for Time-Varying Constraints.

This package provides numerical experiments demonstrating failures of
slack variable projection and barrier reformulation approaches when
extending OPEN-M to inequality constraints.
"""

from .problems import OnlineLPInstance, OnlineLPProblem, solve_lp
from .problems import create_simple_2d_problem, create_medium_problem

__all__ = [
    'OnlineLPInstance',
    'OnlineLPProblem',
    'solve_lp',
    'create_simple_2d_problem',
    'create_medium_problem',
]
