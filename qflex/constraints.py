"""
Constraint Solvers for QFlex

Implements various constraint types for fitting QFlex coefficients:
    - A+: All coefficients non-negative (k >= 2)
    - TL+: Leading tail coefficients non-negative
    - TA+: All tail coefficients non-negative
    - TC: Proposition 5 tail-center margin constraint
    - TC_mag: Proposition 4 grid-based constraint (m_tail > M_center)
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, lsq_linear, LinearConstraint, Bounds
from enum import Enum
from typing import List
import warnings

from .basis import get_term_structure, BasisType, evaluate_basis_derivative

PROB_EPS = 1e-12
PROP4_STRICT_TOLERANCE = 1e-10


def _create_prop4_grid():
    """
    Create a grid for Proposition 4 evaluation with 0.001 spacing.
    Returns 999 points: [0.001, 0.002, ..., 0.999].
    """
    return 0.001 + np.arange(999) * 0.001


class ConstraintType(Enum):
    """Available constraint types for coefficient estimation."""
    NONE = "none"           # Unconstrained least squares
    A = "A+"                # All coefficients >= 0 for k >= 2
    TL = "TL+"              # Leading tail coefficients >= 0
    TA = "TA+"              # All tail coefficients >= 0
    TC = "TC"               # Proposition 5: tail-center margin
    TC_MAG = "TC_mag"       # Proposition 4: m_tail > M_center on grid


class QFlexError(Exception):
    """Raised when QFlex fitting or validation fails."""
    pass


def get_tail_indices(terms: int, leading_only: bool = False) -> List[int]:
    """
    Get coefficient indices corresponding to tail basis functions (f1 and f2).
    
    Parameters
    ----------
    terms : int
        Number of terms in the expansion.
    leading_only : bool
        If True, return only the highest-order tail coefficients.
        
    Returns
    -------
    indices : list of int
        0-based coefficient indices.
    """
    structure = get_term_structure(terms)
    f1_indices = []
    f2_indices = []
    
    for idx, (basis_type, order) in enumerate(structure):
        if basis_type == BasisType.F1_TAIL_RIGHT:
            f1_indices.append(idx)
        elif basis_type == BasisType.F2_TAIL_LEFT:
            f2_indices.append(idx)
    
    if leading_only:
        tail_indices = []
        if f1_indices:
            tail_indices.append(f1_indices[-1])
        if f2_indices:
            tail_indices.append(f2_indices[-1])
        return tail_indices
    else:
        return f1_indices + f2_indices


def center_M_j(order: int, gamma: float) -> float:
    """
    Compute M_j = sup |C_j'(p)| over (0,1) for center basis C_j(p) = (p-γ)^(2j-1).
    
    Since C_j'(p) = (2j-1)(p-γ)^(2j-2), and |p-γ| <= max(γ, 1-γ),
    the supremum is (2j-1) × max(γ, 1-γ)^(2j-2).
    
    Used in the Proposition 5 tail-center margin calculation.
    """
    r = max(gamma, 1 - gamma)
    return (2 * order - 1) * (r ** (2 * order - 2))


def tail_center_margin_coeff(a: np.ndarray, terms: int, gamma: float) -> float:
    """
    Compute the Proposition 5 tail-center margin in coefficient space.
    
    m_TC(a) = (a_2 + a_3) - Σ M_j × |a_k|  over center terms
    
    Here a_2 and a_3 are the first-order tail coefficients (indices 1, 2 in 0-based),
    and the sum runs over center basis coefficients weighted by their M_j values.
    
    Parameters
    ----------
    a : array
        Coefficient vector.
    terms : int
        Number of terms.
    gamma : float
        Center parameter.
        
    Returns
    -------
    margin : float
        Positive margin indicates Proposition 5 is satisfied.
    """
    if len(a) < 3:
        return -np.inf
    
    # Tail contribution: a_2 + a_3 (first-order f1 and f2)
    m_tail = a[1] + a[2]
    
    # Center contribution: weighted sum of |a_k| for f3 terms
    structure = get_term_structure(terms)
    M_center = 0.0
    
    for idx, (basis_type, order) in enumerate(structure):
        if basis_type == BasisType.F3_CENTER and idx < len(a):
            M_j = center_M_j(order, gamma)
            M_center += M_j * abs(a[idx])
    
    return m_tail - M_center


def solve_with_constraints(Y: np.ndarray,
                          x_data: np.ndarray,
                          terms: int,
                          gamma: float,
                          constraint_type: ConstraintType) -> np.ndarray:
    """
    Solve the constrained fitting problem.
    
    Routes to the appropriate solver based on constraint type.
    
    Parameters
    ----------
    Y : array of shape (m, terms)
        Design matrix.
    x_data : array of shape (m,)
        Target quantile values.
    terms : int
        Number of terms.
    gamma : float
        Center parameter.
    constraint_type : ConstraintType
        Which constraint to enforce.
        
    Returns
    -------
    coefficients : array of shape (terms,)
        Fitted coefficient vector.
    """
    solvers = {
        ConstraintType.A: solve_positive_all,
        ConstraintType.TL: solve_tail_leading,
        ConstraintType.TA: solve_tail_all,
        ConstraintType.TC_MAG: solve_proposition_4,
    }
    
    if constraint_type == ConstraintType.TC:
        return solve_tail_center(Y, x_data, terms, gamma)
    
    solver = solvers.get(constraint_type)
    if solver is None:
        raise QFlexError(f"Unknown constraint type: {constraint_type}")
    
    return solver(Y, x_data, terms, gamma)


def get_initial_guess(Y: np.ndarray, x_data: np.ndarray, terms: int) -> np.ndarray:
    """Compute an unconstrained least-squares solution to use as initial guess.

    Uses the SVD-based ``np.linalg.lstsq`` (stable for ill-conditioned design
    matrices) rather than an explicit inverse.
    """
    try:
        initial_guess, *_ = np.linalg.lstsq(Y, x_data, rcond=None)
    except np.linalg.LinAlgError:
        initial_guess = np.ones(terms) * 0.1

    return initial_guess


def solve_positive_all(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Solve with all coefficients non-negative for k >= 2.
    Uses scipy's lsq_linear for efficient bound-constrained least squares.
    """
    lb = np.full(terms, -np.inf)
    lb[1:] = 0.0
    ub = np.full(terms, np.inf)
    
    result = lsq_linear(Y, x_data, bounds=(lb, ub))
    if not result.success:
        raise QFlexError(f"Positive coefficients optimization failed: {result.message}")
    
    return result.x


def solve_tail_leading(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """Solve with only the leading (highest-order) tail coefficients non-negative."""
    tail_indices = get_tail_indices(terms, leading_only=True)
    return solve_with_bounds(Y, x_data, terms, tail_indices)


def solve_tail_all(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """Solve with all tail coefficients non-negative."""
    tail_indices = get_tail_indices(terms, leading_only=False)
    return solve_with_bounds(Y, x_data, terms, tail_indices)


def solve_with_bounds(Y: np.ndarray, x_data: np.ndarray, terms: int,
                     bound_indices: List[int]) -> np.ndarray:
    """
    Helper: minimize ||Y @ a - x||^2 with specified coefficients bounded >= 0.
    """
    lb = np.full(terms, -np.inf)
    for idx in bound_indices:
        lb[idx] = 0.0
    ub = np.full(terms, np.inf)
    
    initial_guess = get_initial_guess(Y, x_data, terms)
    initial_guess = np.maximum(initial_guess, -1e-3)
    for idx in bound_indices:
        if initial_guess[idx] < 0:
            initial_guess[idx] = 1e-6
    
    def objective(a):
        return np.sum((Y @ a - x_data) ** 2)
    
    result = minimize(objective, initial_guess, method='SLSQP',
                     bounds=list(zip(lb, ub)),
                     options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
    
    if not result.success:
        raise QFlexError(f"Constrained optimization failed: {result.message}")
    
    # Enforce bounds exactly (numerical cleanup)
    coefficients = result.x.copy()
    for idx in bound_indices:
        if coefficients[idx] < 0:
            coefficients[idx] = 0.0
    
    return coefficients


def solve_tail_center(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Solve Proposition 5 using a linear reformulation with auxiliary variables.

    For each center coefficient a_k, introduce u_k >= 0 and v_k >= 0 such that
    a_k = u_k - v_k and |a_k| = u_k + v_k. This makes the margin constraint linear:
        m_TC = (a_2 + a_3) - Σ M_j(u_k + v_k) >= 0

    Uses CVXPY if available, otherwise falls back to scipy.optimize.
    """
    all_tail_indices = get_tail_indices(terms, leading_only=False)
    
    structure = get_term_structure(terms)
    center_indices = [idx for idx, (bt, _) in enumerate(structure) 
                     if bt == BasisType.F3_CENTER]
    
    n_center = len(center_indices)
    
    if n_center == 0:
        return solve_tail_all(Y, x_data, terms, gamma)
    
    try:
        import cvxpy as cp
        
        a = cp.Variable(terms)
        u = cp.Variable(n_center)
        v = cp.Variable(n_center)
        a_modified = cp.Variable(terms)
        
        constraints_var = []
        for i in range(terms):
            if i not in center_indices:
                constraints_var.append(a_modified[i] == a[i])
        
        for i, center_idx in enumerate(center_indices):
            constraints_var.append(a_modified[center_idx] == u[i] - v[i])
        
        objective = cp.Minimize(cp.sum_squares(Y @ a_modified - x_data))
        
        all_constraints = constraints_var.copy()
        
        for idx in all_tail_indices:
            all_constraints.append(a[idx] >= 0)
        
        all_constraints.append(u >= 0)
        all_constraints.append(v >= 0)
        
        M_values = [center_M_j(structure[idx][1], gamma) for idx in center_indices]
        margin_expr = (a[1] + a[2]) - cp.sum([M_j * (u[i] + v[i]) 
                                              for i, M_j in enumerate(M_values)])
        all_constraints.append(margin_expr >= 1e-8)
        
        problem = cp.Problem(objective, all_constraints)
        problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise QFlexError(f"CVXPY optimization failed: {problem.status}")
        
        coefficients = a_modified.value
        
    except ImportError:
        warnings.warn("CVXPY not available, using scipy fallback for linear TC method")
        
        n_vars = terms + 2 * n_center
        
        def objective_expanded(vars_expanded):
            a_orig = vars_expanded[:terms]
            u_aux = vars_expanded[terms:terms+n_center]
            v_aux = vars_expanded[terms+n_center:]
            
            a_modified = a_orig.copy()
            for i, center_idx in enumerate(center_indices):
                a_modified[center_idx] = u_aux[i] - v_aux[i]
            
            return np.sum((Y @ a_modified - x_data) ** 2)
        
        bounds_list = [(None, None)] * n_vars
        for idx in all_tail_indices:
            bounds_list[idx] = (0, None)
        for i in range(n_center):
            bounds_list[terms + i] = (0, None)
            bounds_list[terms + n_center + i] = (0, None)
        
        M_values = [center_M_j(structure[idx][1], gamma) for idx in center_indices]
        
        A_margin = np.zeros((1, n_vars))
        A_margin[0, 1] = 1.0
        A_margin[0, 2] = 1.0
        for i, M_j in enumerate(M_values):
            A_margin[0, terms + i] = -M_j
            A_margin[0, terms + n_center + i] = -M_j
        
        margin_constraint = LinearConstraint(A_margin, [1e-8], [np.inf])
        
        initial_guess = np.zeros(n_vars)
        try:
            pos_all_lb = np.full(terms, -np.inf)
            pos_all_ub = np.full(terms, np.inf)
            pos_all_lb[1:] = 0.0
            pos_all_result = lsq_linear(Y, x_data, bounds=(pos_all_lb, pos_all_ub))
            if pos_all_result.success:
                a_init = pos_all_result.x
            else:
                a_init = get_initial_guess(Y, x_data, terms)
        except Exception:
            a_init = get_initial_guess(Y, x_data, terms)
        
        initial_guess[:terms] = a_init
        for i, center_idx in enumerate(center_indices):
            a_k = a_init[center_idx]
            if a_k >= 0:
                initial_guess[terms + i] = a_k
                initial_guess[terms + n_center + i] = 0.0
            else:
                initial_guess[terms + i] = 0.0
                initial_guess[terms + n_center + i] = -a_k
        
        result = minimize(objective_expanded, initial_guess, 
                         method='trust-constr',
                         bounds=Bounds([b[0] if b[0] is not None else -np.inf for b in bounds_list], 
                                      [b[1] if b[1] is not None else np.inf for b in bounds_list]),
                         constraints=[margin_constraint],
                         options={'maxiter': 5000, 'disp': False})
        
        if not result.success:
            raise QFlexError(f"Linear Proposition 5 optimization failed: {result.message}")
        
        vars_sol = result.x
        coefficients = vars_sol[:terms].copy()
        
        for i, center_idx in enumerate(center_indices):
            u_val = vars_sol[terms + i]
            v_val = vars_sol[terms + n_center + i]
            coefficients[center_idx] = u_val - v_val
    
    final_margin = tail_center_margin_coeff(coefficients, terms, gamma)
    if final_margin <= 0:
        raise QFlexError(
            f"Proposition 5 constraint not satisfied: margin = {final_margin:.2e} <= 0"
        )
    
    for idx in all_tail_indices:
        if coefficients[idx] < 0:
            coefficients[idx] = 0.0
    
    return coefficients


def solve_proposition_4(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Solve with Proposition 4 constraint: m_tail > M_center (strict inequality).
    
    Also enforces Proposition 3 (leading tail coefficients >= 0).
    The constraint is evaluated on a fixed grid with 0.001 spacing.
    """
    leading_tail_indices = get_tail_indices(terms, leading_only=True)
    
    bounds = [(None, None) for _ in range(terms)]
    for idx in leading_tail_indices:
        bounds[idx] = (0, None)
    
    initial_guess = get_initial_guess(Y, x_data, terms)
    initial_guess = np.maximum(initial_guess, 1e-6)
    for idx in leading_tail_indices:
        if initial_guess[idx] < 0:
            initial_guess[idx] = 1e-6
    
    p_grid = _create_prop4_grid()
    p_grid = np.clip(p_grid, PROB_EPS, 1 - PROB_EPS)
    
    def prop4_margin(a):
        """Compute m_tail - M_center on the fixed grid."""
        q_tail = np.zeros_like(p_grid)
        q_center = np.zeros_like(p_grid)
        
        structure = get_term_structure(terms)
        for idx, (basis_type, order) in enumerate(structure):
            if idx >= len(a):
                break
            
            if basis_type == BasisType.F1_TAIL_RIGHT or basis_type == BasisType.F2_TAIL_LEFT:
                q_tail += a[idx] * evaluate_basis_derivative(p_grid, basis_type, order, gamma)
            elif basis_type == BasisType.F3_CENTER:
                q_center += a[idx] * evaluate_basis_derivative(p_grid, basis_type, order, gamma)
        
        m_tail = np.min(q_tail)
        M_center = np.max(np.abs(q_center))
        return m_tail - M_center
    
    constraint_tolerance = PROP4_STRICT_TOLERANCE + PROB_EPS * 100
    nlc_prop4 = NonlinearConstraint(prop4_margin, constraint_tolerance, np.inf)
    
    def objective(a):
        return np.sum((Y @ a - x_data) ** 2)
    
    result = minimize(objective, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=[nlc_prop4],
                     options={'ftol': 1e-9, 'maxiter': 5000, 'disp': False})
    
    if not result.success:
        raise QFlexError(f"Proposition 4 constraint optimization failed: {result.message}")
    
    coefficients = result.x.copy()
    
    final_margin = prop4_margin(coefficients)
    if final_margin <= 0:
        raise QFlexError(
            f"Proposition 4 constraint not satisfied: margin = {final_margin:.2e} <= 0. "
            f"The constraint may be too strict for {terms} terms."
        )
    
    for idx in leading_tail_indices:
        if coefficients[idx] < 0:
            coefficients[idx] = 0.0
    
    return coefficients
