"""
Monotonicity Verification for QFlex

Provides a posteriori verification methods (numbering as in the paper):
    - check_proposition3(): the grid tail-centre condition m_tail > M_center,
      together with Theorem 5 (leading tail coefficients > 0). Earlier
      releases called it check_proposition4(); that name still works.
    - check_delta_p_monotonicity(): strict monotonicity on a discrete grid.
"""

import numpy as np
from typing import Dict, Optional, Callable

from .basis import get_term_structure, BasisType, evaluate_basis_derivative, leading_tail_terms

PROB_EPS = 1e-12
PROP4_STRICT_TOLERANCE = 1e-10


def check_proposition3(coefficients: np.ndarray,
                       terms: int,
                       gamma: float,
                       p_grid: Optional[np.ndarray] = None,
                       num_points: Optional[int] = None) -> Dict:
    """
    Verify Proposition 3 (grid tail-centre condition) and Theorem 5.

    Proposition 3: if m_tail = inf q_tail(p) exceeds M_center = sup |q_center(p)|
    over (0, 1), Q is strictly increasing. Evaluated on a grid, as here, it
    certifies q(p) = dQ/dp > 0 at the grid points:
        m_tail   = min over the grid of the tail part of q(p)
        M_center = max over the grid of |centre part of q(p)|.
    A grid says nothing about p beyond its ends, so (as the paper's Remark 7
    requires) the leading, i.e. highest-order non-zero, right and left tail
    coefficients must also be > 0 (Theorem 5); otherwise Q eventually
    decreases as p -> 0 or p -> 1.

    Parameters
    ----------
    coefficients : array
        Fitted coefficient vector.
    terms : int
        Number of terms.
    gamma : float
        Centre parameter.
    p_grid : array, optional
        Grid for evaluation. Takes precedence over ``num_points``.
    num_points : int, optional
        Number of evenly spaced points on [0.001, 0.999] when ``p_grid`` is not
        given. Default: the paper's grid 0.001, 0.002, ..., 0.999.

    Returns
    -------
    result : dict
        'satisfied': grid condition and Theorem 5 both hold
        'grid_condition': m_tail > M_center
        'theorem5_satisfied': leading right and left tail coefficients > 0
        'm_tail', 'M_center', 'margin' (= m_tail - M_center)
        'q_flex_min': minimum of the full q(p) on the grid
        'q_flex_positive': q(p) > 0 at every grid point
        'leading_right', 'leading_left': (coefficient, order) or None
    """
    if p_grid is not None:
        p_grid = np.asarray(p_grid, dtype=float)
    elif num_points is not None:
        num_points = int(num_points)
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        p_grid = np.linspace(0.001, 0.999, num_points)
    else:
        p_grid = 0.001 + np.arange(999) * 0.001

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
    grid_condition = bool(margin > 0)

    lead = leading_tail_terms(coefficients, terms)
    theorem5 = all(term is None or term[0] > 0 for term in (lead['right'], lead['left']))

    q_flex = q_tail + q_center
    q_flex_min = np.min(q_flex)

    return {
        'satisfied': bool(grid_condition and theorem5),
        'grid_condition': grid_condition,
        'theorem5_satisfied': bool(theorem5),
        'm_tail': float(m_tail),
        'M_center': float(M_center),
        'margin': float(margin),
        'q_flex_min': float(q_flex_min),
        'q_flex_positive': bool(q_flex_min > 0),
        'leading_right': lead['right'],
        'leading_left': lead['left'],
    }


def check_proposition4(coefficients: np.ndarray,
                       terms: int,
                       gamma: float,
                       p_grid: Optional[np.ndarray] = None,
                       num_points: Optional[int] = None) -> Dict:
    """
    Backward-compatible name of :func:`check_proposition3`.

    Earlier releases numbered the grid tail-centre condition "Proposition 4";
    the paper numbers it Proposition 3.
    """
    return check_proposition3(coefficients, terms, gamma, p_grid=p_grid, num_points=num_points)


def check_delta_p_monotonicity(quantile_func: Callable,
                               delta_p: float = 0.001,
                               p_start: float = 0.001,
                               p_end: float = 0.999) -> Dict:
    """
    Verify strict monotonicity by checking Q(p + delta_p) > Q(p) on a grid
    (δ-p monotonicity, the paper's Definition 4; the paper uses δ = 0.001).

    A discrete check: monotone on a grid with spacing delta_p does not
    guarantee monotone in between or beyond the grid ends.

    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(p) -> x.
    delta_p : float
        Grid spacing (default 0.001).
    p_start, p_end : float
        Grid range (default 0.001 to 0.999).

    Returns
    -------
    result : dict
        'satisfied': strictly increasing on the grid
        'min_difference': smallest Q(p + delta_p) - Q(p)
        'num_violations': number of non-increasing steps
        'violation_indices': indices where they occur
    """
    n = int(round((p_end - p_start) / delta_p)) + 1
    p_grid = p_start + np.arange(n) * delta_p
    p_grid = np.clip(p_grid, PROB_EPS, 1 - PROB_EPS)

    quantile_vals = quantile_func(p_grid)
    differences = np.diff(quantile_vals)

    violations = ~(differences > 0)
    num_violations = int(np.sum(violations))
    satisfied = num_violations == 0

    violation_indices = np.where(violations)[0] if num_violations > 0 else np.array([], dtype=int)
    min_difference = np.min(differences) if len(differences) > 0 else np.inf

    return {
        'satisfied': bool(satisfied),
        'min_difference': float(min_difference),
        'num_violations': num_violations,
        'violation_indices': violation_indices
    }
