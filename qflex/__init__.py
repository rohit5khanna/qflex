"""
QFlex Distributions

A flexible quantile-parameterized distribution family with support for
unbounded, semibounded, and bounded domains, plus optional constraints
to ensure valid probability densities.
"""

from .core import QFlex
from .transforms import LogQFlex, LogitQFlex
from .constraints import ConstraintType, QFlexError
from .mono_verification import check_proposition4, check_delta_p_monotonicity
from .utils import compute_w1

__all__ = [
    'QFlex',
    'LogQFlex',
    'LogitQFlex',
    'QFlexError',
    'ConstraintType',
    'check_proposition4',
    'check_delta_p_monotonicity',
    'compute_w1',
]

__version__ = '1.0.0'
