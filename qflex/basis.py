"""
QFlex Basis Functions

Defines the three basis function families used in QFlex:
    - Constant term (a_1)
    - Right tail: [-ln(1-p)]^i for i = 1, 2, 3, ...
    - Left tail: (-1)^(i+1) × [ln(p)]^i for i = 1, 2, 3, ...
    - Center: (p - γ)^(2i-1) for i = 1, 2, 3, ... (odd powers)

Derivatives are computed analytically for efficiency in Proposition 4/5 verification.
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
    
    Analytical derivatives are more efficient than finite differences
    when evaluating on dense grids for Proposition verification.
    
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
        u = -np.log(1 - y)
        denom = np.clip(1 - y, 1e-10, None)
        if order == 1:
            return np.ones_like(y) / denom
        else:
            u = np.clip(u, 0, None)
            return order * (u ** (order - 1)) / denom
    
    elif basis_type == BasisType.F2_TAIL_LEFT:
        # d/dy [(-1)^(order+1) × ln(y)^order] = order × (-1)^(order+1) × ln(y)^(order-1) / y
        sign = (-1) ** (order + 1)
        v = np.log(y)
        y_clipped = np.clip(y, 1e-10, None)
        if order == 1:
            return sign / y_clipped
        else:
            if sign < 0:
                v_clipped = np.clip(v, None, 0)
            else:
                v_clipped = np.clip(v, 0, None)
            return order * sign * (v_clipped ** (order - 1)) / y_clipped
    
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
        - Proposition 4/5 verification
        - Analytical PDF computation (as an alternative to numerical differentiation)
    
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
