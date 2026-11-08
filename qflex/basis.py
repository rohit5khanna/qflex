"""
QFlex Basis Functions

Defines the three basis function families used in QFlex:
    - Constant term (a_1)
    - Right tail R_i (f1): [-ln(1-p)]^i for i = 1, 2, 3, ...
    - Left tail L_i (f2): (-1)^(i+1) × [ln(p)]^i for i = 1, 2, 3, ...
    - Center C_i (f3): (p - γ)^(2i-1) for i = 1, 2, 3, ... (odd powers)

Coefficients are numbered a_1, ..., a_K as in the paper (a_1 constant,
a_2 on R_1, a_3 on L_1, a_4 on C_1, a_5 on R_2, ...); in code they are
0-based, so a_k is coefficients[k - 1].

Derivatives are computed analytically (used by the PDF, which is analytical
by default, and by the Proposition 3 / Proposition 4 validity checks;
numbering as in Bickel et al.).
"""

import numpy as np
from typing import List, Tuple
from enum import Enum

# Small epsilon to avoid log(0) and log(1) at boundaries
PROB_EPS = 1e-12


class BasisType(Enum):
    """Identifies the basis function family."""
    CONSTANT = "constant"
    F1_TAIL_RIGHT = "f1"
    F2_TAIL_LEFT = "f2"
    F3_CENTER = "f3"


def get_term_structure(terms: int) -> List[Tuple[BasisType, int]]:
    """
    Build the sequence of (basis_type, order) pairs for a given number of terms.
    
    The pattern cycles through (constant, f1, f2, f3) at increasing orders:
        terms=4: constant, f1^1, f2^1, f3^1
        terms=7: constant, f1^1, f2^1, f3^1, f1^2, f2^2, f3^2
    
    Parameters
    ----------
    terms : int
        Total number of terms in the expansion.
        
    Returns
    -------
    structure : list of (BasisType, int)
        Ordered list of basis types and their orders.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    
    structure = [(BasisType.CONSTANT, 0)]
    order = 1
    idx = 1
    
    while idx < terms:
        if idx < terms:
            structure.append((BasisType.F1_TAIL_RIGHT, order))
            idx += 1
        if idx < terms:
            structure.append((BasisType.F2_TAIL_LEFT, order))
            idx += 1
        if idx < terms:
            structure.append((BasisType.F3_CENTER, order))
            idx += 1
        order += 1
    
    return structure


def evaluate_basis(y: np.ndarray, basis_type: BasisType, order: int, gamma: float) -> np.ndarray:
    """
    Evaluate a single basis function at the given probability values.
    
    Parameters
    ----------
    y : array
        Cumulative probabilities in (0, 1).
    basis_type : BasisType
        Which basis family to evaluate.
    order : int
        Order of the basis function (1, 2, 3, ...).
    gamma : float
        Center parameter for f3 basis (ignored for other types).
        
    Returns
    -------
    values : array
        Basis function evaluated at each y.
    """
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    
    if basis_type == BasisType.CONSTANT:
        return np.ones_like(y)
    
    elif basis_type == BasisType.F1_TAIL_RIGHT:
        # [-ln(1-y)]^order
        return (-np.log(1 - y)) ** order
    
    elif basis_type == BasisType.F2_TAIL_LEFT:
        # (-1)^(order+1) × [ln(y)]^order
        sign = (-1) ** (order + 1)
        return sign * (np.log(y) ** order)
    
    elif basis_type == BasisType.F3_CENTER:
        # (y - γ)^(2*order - 1)
        return (y - gamma) ** (2 * order - 1)
    
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def evaluate_basis_derivative(y: np.ndarray, basis_type: BasisType, order: int, gamma: float) -> np.ndarray:
    """
    Compute d/dy of a basis function (used for q(y) = dQ/dy).

    Exact, and cheaper than finite differences on the dense grids used by
    the PDF and the validity checks.
    
    Parameters
    ----------
    y : array
        Cumulative probabilities in (0, 1).
    basis_type : BasisType
        Which basis family.
    order : int
        Order of the basis function.
    gamma : float
        Center parameter for f3 basis.
        
    Returns
    -------
    derivatives : array
        Derivative of basis function at each y.
    """
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    
    if basis_type == BasisType.CONSTANT:
        return np.zeros_like(y)
    
    elif basis_type == BasisType.F1_TAIL_RIGHT:
        # d/dy [-ln(1-y)]^order = order × [-ln(1-y)]^(order-1) / (1-y)
        # y is already clipped to [PROB_EPS, 1 - PROB_EPS], the same clip the
        # quantile uses, so q(y) and Q(y) are always evaluated at the same point.
        denom = 1 - y
        if order == 1:
            return np.ones_like(y) / denom
        u = -np.log(denom)
        return order * (u ** (order - 1)) / denom
    
    elif basis_type == BasisType.F2_TAIL_LEFT:
        # d/dy [(-1)^(order+1) × ln(y)^order] = order × (-1)^(order+1) × ln(y)^(order-1) / y
        #
        # NO CLIP ON ln(y). ln(y) < 0 for every y in (0, 1), and `order - 1` is
        # an integer, so the power is well defined for negative values. A
        # previous version clipped ln(y) to [0, inf) whenever the sign factor
        # was positive -- that is, for every ODD order -- which forced the term
        # to 0 ** (order-1) = 0. Orders 3, 5, 7, ... therefore returned exactly
        # zero everywhere, and evaluate_quantile_derivative was wrong from
        # K = 9 on, where L3 first enters the term structure.
        sign = (-1) ** (order + 1)
        if order == 1:
            return sign / y
        return order * sign * (np.log(y) ** (order - 1)) / y
    
    elif basis_type == BasisType.F3_CENTER:
        # d/dy (y-γ)^(2*order-1) = (2*order-1) × (y-γ)^(2*order-2)
        return (2 * order - 1) * ((y - gamma) ** (2 * order - 2))
    
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def build_design_matrix(y_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Construct the design matrix Y where Y @ coefficients gives quantile values.
    
    Parameters
    ----------
    y_data : array of shape (m,)
        Cumulative probabilities.
    terms : int
        Number of basis terms.
    gamma : float
        Center parameter.
        
    Returns
    -------
    Y : array of shape (m, terms)
        Design matrix with Y[i, j] = basis_j(y_data[i]).
    """
    y_data = np.clip(np.asarray(y_data, dtype=float), PROB_EPS, 1 - PROB_EPS)
    structure = get_term_structure(terms)
    m = len(y_data)
    Y = np.zeros((m, terms))
    
    for col_idx, (basis_type, order) in enumerate(structure):
        Y[:, col_idx] = evaluate_basis(y_data, basis_type, order, gamma)
    
    return Y


def evaluate_quantile(y: np.ndarray, coefficients: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Evaluate the quantile function Q(y) = Σ a_k × basis_k(y).
    
    Parameters
    ----------
    y : array
        Cumulative probabilities.
    coefficients : array of shape (terms,)
        Fitted coefficients.
    terms : int
        Number of terms.
    gamma : float
        Center parameter.
        
    Returns
    -------
    quantiles : array
        Q(y) values.
    """
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    structure = get_term_structure(terms)
    result = np.zeros_like(y)
    
    for idx, (basis_type, order) in enumerate(structure):
        if idx < len(coefficients):
            result += coefficients[idx] * evaluate_basis(y, basis_type, order, gamma)
    
    return result


def evaluate_quantile_derivative(y: np.ndarray, coefficients: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Evaluate q(y) = dQ/dy = Σ a_k × basis_k'(y).
    
    This is the derivative of the quantile function, used for:
        - Proposition 3 / Proposition 4 verification
        - the PDF, f(Q(y)) = 1 / q(y) (the default method='analytical')
    
    Parameters
    ----------
    y : array
        Cumulative probabilities.
    coefficients : array of shape (terms,)
        Fitted coefficients.
    terms : int
        Number of terms.
    gamma : float
        Center parameter.
        
    Returns
    -------
    derivatives : array
        q(y) = dQ/dy values.
    """
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    structure = get_term_structure(terms)
    result = np.zeros_like(y)
    
    for idx, (basis_type, order) in enumerate(structure):
        if idx < len(coefficients):
            result += coefficients[idx] * evaluate_basis_derivative(y, basis_type, order, gamma)
    
    return result


def evaluate_quantile_logodds(u: np.ndarray, coefficients: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Evaluate Q at p = 1 / (1 + exp(-u)) WITHOUT the PROB_EPS clip.

    -ln(1 - p) = log(1 + e^u) and -ln(p) = log(1 + e^-u) are computed with
    ``np.logaddexp``, so the basis stays accurate however close p is to 0 or 1.
    Used for moments, which need the whole tail, not the clipped one.
    """
    u = np.asarray(u, dtype=float)
    r = np.logaddexp(0.0, u)          # -ln(1 - p)
    l = np.logaddexp(0.0, -u)         # -ln(p)
    p = 0.5 * (1.0 + np.tanh(0.5 * u))  # expit(u) without overflow
    structure = get_term_structure(terms)
    result = np.zeros_like(u)
    for idx, (basis_type, order) in enumerate(structure):
        if idx >= len(coefficients):
            break
        a = coefficients[idx]
        if basis_type == BasisType.CONSTANT:
            result = result + a
        elif basis_type == BasisType.F1_TAIL_RIGHT:
            result = result + a * r ** order
        elif basis_type == BasisType.F2_TAIL_LEFT:
            result = result - a * l ** order      # (-1)^(j+1) ln(p)^j = -(-ln p)^j
        else:
            result = result + a * (p - gamma) ** (2 * order - 1)
    return result


def leading_tail_terms(coefficients: np.ndarray, terms: int, rel_tol: float = 1e-12) -> dict:
    """
    Highest-order non-zero right-tail and left-tail coefficients.

    Coefficients within ``rel_tol`` x max|a_k| (k >= 2) of zero count as zero,
    so rounding residue from a constrained solver is not mistaken for a term.

    Returns
    -------
    dict with keys 'right' and 'left', each ``(coefficient, order)`` or
    ``None`` when the family has no non-zero term.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    structure = get_term_structure(terms)
    scale = np.max(np.abs(coefficients[1:])) if len(coefficients) > 1 else 0.0
    tol = rel_tol * scale
    out = {'right': None, 'left': None}
    for idx, (basis_type, order) in enumerate(structure):
        if idx >= len(coefficients) or abs(coefficients[idx]) <= tol:
            continue
        if basis_type == BasisType.F1_TAIL_RIGHT:
            out['right'] = (float(coefficients[idx]), order)
        elif basis_type == BasisType.F2_TAIL_LEFT:
            out['left'] = (float(coefficients[idx]), order)
    return out


def quantile_limits(coefficients: np.ndarray, terms: int, gamma: float):
    """
    Limits of Q(p) as p -> 0 and p -> 1 (on the fitting scale).

    A tail family with a non-zero leading coefficient diverges to +/- infinity
    with the sign of that coefficient (left tail: -inf for a positive one).
    Without tail terms the limit is the finite value of the remaining terms.
    """
    lead = leading_tail_terms(coefficients, terms)
    structure = get_term_structure(terms)
    def finite_part(p):
        total = 0.0
        for idx, (basis_type, order) in enumerate(structure):
            if idx >= len(coefficients):
                break
            if basis_type == BasisType.CONSTANT:
                total += coefficients[idx]
            elif basis_type == BasisType.F3_CENTER:
                total += coefficients[idx] * (p - gamma) ** (2 * order - 1)
            elif basis_type == BasisType.F1_TAIL_RIGHT and p == 0.0:
                pass                       # R_j(0) = 0
            elif basis_type == BasisType.F2_TAIL_LEFT and p == 1.0:
                pass                       # L_j(1) = 0
        return float(total)
    if lead['left'] is not None:
        q0 = -np.inf if lead['left'][0] > 0 else np.inf
    else:
        q0 = finite_part(0.0)
    if lead['right'] is not None:
        q1 = np.inf if lead['right'][0] > 0 else -np.inf
    else:
        q1 = finite_part(1.0)
    return q0, q1
