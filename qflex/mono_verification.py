"""
Monotonicity Verification for QFlex

Provides a posteriori verification methods:
    - check_proposition4(): Verify the tail-center magnitude condition (m_tail > M_center)
    - check_delta_p_monotonicity(): Verify strict monotonicity on a discrete grid
"""

import numpy as np
from typing import Dict, Optional, Callable

from .basis import get_term_structure, BasisType, evaluate_basis_derivative

PROB_EPS = 1e-12
PROP4_STRICT_TOLERANCE = 1e-10


def check_proposition4(coefficients: np.ndarray,
                      terms: int,
                      gamma: float,
                      p_grid: Optional[np.ndarray] = None,
                      num_points: int = 10000) -> Dict:
    """
    Verify Proposition 4 (tail-center magnitude condition).
    
    Proposition 4 guarantees that if m_tail > M_center, then the quantile
    derivative q(p) = dQ/dp is positive everywhere, ensuring a valid PDF.
    
    Here:
        m_tail = inf q_tail(p) over (0, 1)  (smallest tail contribution)
        M_center = sup q_center(p) over (0, 1)  (largest center contribution)
    
    Parameters
    ----------
    coefficients : array
        Fitted coefficient vector.
    terms : int
        Number of terms.
    gamma : float
        Center parameter.
    p_grid : array, optional
        Grid for evaluation. Defaults to [0.001, 0.002, ..., 0.999].
    num_points : int
        Unused if p_grid is provided.
        
    Returns
    -------
    result : dict
        'satisfied': True if m_tail > M_center
        'm_tail': Minimum tail derivative value
        'M_center': Maximum center derivative value
        'margin': m_tail - M_center
        'q_flex_min': Minimum of full q(p) on grid
        'q_flex_positive': True if q(p) > 0 everywhere
    """
    if p_grid is None:
        p_grid = 0.001 + np.arange(999) * 0.001
    else:
        p_grid = np.asarray(p_grid)
    
    p_grid = np.clip(p_grid, PROB_EPS, 1 - PROB_EPS)
    
    q_tail = np.zeros_like(p_grid)
    q_center = np.zeros_like(p_grid)
    
    structure = get_term_structure(terms)
    for idx, (basis_type, order) in enumerate(structure):
        if idx >= len(coefficients):
            break
        
        if basis_type == BasisType.F1_TAIL_RIGHT or basis_type == BasisType.F2_TAIL_LEFT:
            q_tail += coefficients[idx] * evaluate_basis_derivative(p_grid, basis_type, order, gamma)
        elif basis_type == BasisType.F3_CENTER:
            q_center += coefficients[idx] * evaluate_basis_derivative(p_grid, basis_type, order, gamma)
    
    m_tail = np.min(q_tail)
    M_center = np.max(np.abs(q_center))
    margin = m_tail - M_center
    satisfied = margin > 0
    
    q_flex = q_tail + q_center
    q_flex_min = np.min(q_flex)
    
    return {
        'satisfied': satisfied,
        'm_tail': float(m_tail),
        'M_center': float(M_center),
        'margin': float(margin),
        'q_flex_min': float(q_flex_min),
        'q_flex_positive': q_flex_min > 0
    }


def check_delta_p_monotonicity(quantile_func: Callable,
                               delta_p: float = 0.001,
                               p_start: float = 0.001,
                               p_end: float = 0.999) -> Dict:
    """
    Verify strict monotonicity by checking Q(p + Δp) > Q(p) for all p.
    
    This is a discrete approximation: if Q is monotonic on a grid with
    spacing Δp, it is likely (though not guaranteed) monotonic everywhere.
    
    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(p) -> x.
    delta_p : float
        Grid spacing (default 0.001).
    p_start : float
        Start of probability range (default 0.001).
    p_end : float
        End of probability range (default 0.999).
        
    Returns
    -------
    result : dict
        'satisfied': True if strictly monotonic on grid
        'min_difference': Smallest Q(p+Δp) - Q(p) value
        'num_violations': Number of non-increasing steps
        'violation_indices': Indices where violations occur
    """
    p_grid = np.arange(p_start, p_end + delta_p, delta_p)
    p_grid = np.clip(p_grid, PROB_EPS, 1 - PROB_EPS)
    
    quantile_vals = quantile_func(p_grid)
    differences = np.diff(quantile_vals)
    
    violations = differences <= 0
    num_violations = np.sum(violations)
    satisfied = num_violations == 0
    
    violation_indices = np.where(violations)[0] if num_violations > 0 else np.array([])
    min_difference = np.min(differences) if len(differences) > 0 else np.inf
    
    return {
        'satisfied': bool(satisfied),
        'min_difference': float(min_difference),
        'num_violations': int(num_violations),
        'violation_indices': violation_indices
    }
