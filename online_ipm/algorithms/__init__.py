"""Algorithm implementations for online optimization with inequality constraints."""

from .base import OnlineAlgorithm, StepMetrics
from .slack_projection import SlackProjectionAlgorithm
from .barrier_method import BarrierMethodAlgorithm
from .feasible_projection import FeasibleProjectionAlgorithm

__all__ = [
    'OnlineAlgorithm',
    'StepMetrics',
    'SlackProjectionAlgorithm',
    'BarrierMethodAlgorithm',
    'FeasibleProjectionAlgorithm',
]
