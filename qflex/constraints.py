"""
Constraint Solvers for QFlex

Implements the constraint types for fitting QFlex coefficients. Numbering
follows the paper (Bickel, Colombe and Leibowicz, "The QFlex Distribution"):

    - A+ (``ConstraintType.A``): all coefficients except the intercept >= 0.
      Valid everywhere (Theorem 1).
    - TL+ (``ConstraintType.TL``): leading (highest-order) right and left tail
      coefficients > 0. Necessary for valid tails (Theorem 5), not sufficient.
    - TA+ (``ConstraintType.TA``): all tail coefficients >= 0. Not sufficient.
    - TC (``ConstraintType.TC``): Proposition 4 tail-centre coefficient bound,
      all tails >= 0 and (a_2 + a_3) - sum_j M_j |c_j| > 0, where a_2 and a_3
      are the R_1 and L_1 coefficients and c_j the centre (C_j) coefficients.
      Valid everywhere.
    - TC_mag (``ConstraintType.TC_MAG``): Proposition 3 grid condition
      m_tail > M_center, with the leading tails > 0 (Theorem 5). Enforced
      until it holds on a fine grid that includes the tails, so the returned
      fit is valid on the whole range the library evaluates.

Earlier releases called TC "Proposition 5" and TC_mag "Proposition 4".

Coefficients a_1, ..., a_K are numbered as in the paper (a_1 = the constant);
array indices in the code are 0-based.
"""

import warnings
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.optimize import lsq_linear, minimize

from .basis import get_term_structure, BasisType, evaluate_basis_derivative

PROB_EPS = 1e-12
PROP4_STRICT_TOLERANCE = 1e-10          # kept for backward compatibility

#: Iteration cap for the bounded least-squares solvers.
BOUNDED_LS_MAX_ITER = 20000

#: TL+ and TC_mag keep the leading tail coefficients at least this fraction of
#: the data range above zero (Theorem 5 needs them strictly positive).
TL_FLOOR_REL = 1e-6

#: TC and TC_mag margin floor, as a fraction of the data range.
MARGIN_REL = 1e-8


def _create_prop4_grid():
    """The paper's evaluation grid: 999 points, p = 0.001, 0.002, ..., 0.999."""
    return 0.001 + np.arange(999) * 0.001


def _create_prop3_grid():
    """Alias of :func:`_create_prop4_grid` under the paper's numbering."""
    return _create_prop4_grid()


class ConstraintType(Enum):
    """Available constraint types for coefficient estimation."""
    NONE = "none"           # Unconstrained least squares (QFlex-U)
    A = "A+"                # a_k >= 0 for k >= 2, i.e. all but the constant (Theorem 1)
    TL = "TL+"              # Leading tail coefficients > 0 (Theorem 5)
    TA = "TA+"              # All tail coefficients >= 0
    TC = "TC"               # Proposition 4: tail-centre coefficient bound
    TC_MAG = "TC_mag"       # Proposition 3: m_tail > M_center on a grid

    @classmethod
    def resolve(cls, value) -> "ConstraintType":
        """
        Accept a ConstraintType, its value ("none", "A+", "TL+", "TA+", "TC",
        "TC_mag"), its name ("NONE", "A", "TL", "TA", "TC", "TC_MAG") or a common
        label ("U", "unconstrained", "T+", "QFlex-A+", ...), case-insensitively.
        None means unconstrained.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.NONE
        if isinstance(value, str):
            key = value.strip().lower().replace('qflex-', '').replace(' ', '')
            aliases = {
                'none': cls.NONE, 'u': cls.NONE, 'unconstrained': cls.NONE,
                'a': cls.A, 'a+': cls.A,
                'tl': cls.TL, 'tl+': cls.TL, 't+': cls.TL, 't': cls.TL,
                'ta': cls.TA, 'ta+': cls.TA,
                'tc': cls.TC,
                'tc_mag': cls.TC_MAG, 'tcmag': cls.TC_MAG, 'tc-mag': cls.TC_MAG,
            }
            if key in aliases:
                return aliases[key]
        raise QFlexError(
            f"Unknown constraint type {value!r}; use a ConstraintType or one of "
            "'none', 'A+', 'TL+', 'TA+', 'TC', 'TC_mag'")


class QFlexError(Exception):
    """Raised when QFlex fitting or validation fails."""
    pass


def get_tail_indices(terms: int, leading_only: bool = False) -> List[int]:
    """
    Coefficient indices of the tail basis functions (right R_j and left L_j).

    Parameters
    ----------
    terms : int
        Number of terms in the expansion.
    leading_only : bool
        If True, return only the highest-order right and left tail indices.

    Returns
    -------
    indices : list of int
        0-based coefficient indices.
    """
    structure = get_term_structure(terms)
    f1_indices = [i for i, (t, _) in enumerate(structure) if t == BasisType.F1_TAIL_RIGHT]
    f2_indices = [i for i, (t, _) in enumerate(structure) if t == BasisType.F2_TAIL_LEFT]
    if leading_only:
        return ([f1_indices[-1]] if f1_indices else []) + ([f2_indices[-1]] if f2_indices else [])
    return f1_indices + f2_indices


def get_center_indices(terms: int) -> List[int]:
    """Coefficient indices of the centre basis functions C_j."""
    return [i for i, (t, _) in enumerate(get_term_structure(terms)) if t == BasisType.F3_CENTER]


def center_M_j(order: int, gamma: float) -> float:
    """
    M_j = sup |C_j'(p)| over (0, 1) for the centre basis C_j(p) = (p - gamma)^(2j-1).

    C_j'(p) = (2j-1)(p - gamma)^(2j-2) and |p - gamma| <= max(gamma, 1 - gamma),
    so the supremum is (2j-1) max(gamma, 1 - gamma)^(2j-2). Used in the
    Proposition 4 (TC) tail-centre margin.
    """
    r = max(gamma, 1 - gamma)
    return (2 * order - 1) * (r ** (2 * order - 2))


def tail_center_margin_coeff(a: np.ndarray, terms: int, gamma: float) -> float:
    """
    Proposition 4 (TC) tail-centre margin in coefficient space.

        m_TC(a) = (a_2 + a_3) - sum_j M_j |c_j|

    a_2 and a_3 are the first-order right and left tail coefficients; the sum
    runs over the centre coefficients. With every tail coefficient >= 0,
    m_TC > 0 implies Q'(p) >= m_TC > 0 for all p, because R_1' = 1/(1-p) >= 1,
    L_1' = 1/p >= 1, the higher tail derivatives are >= 0, and |C_j'| <= M_j.

    Returns
    -------
    margin : float
        Positive when the Proposition 4 condition holds.
    """
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return -np.inf
    margin = a[1] + (a[2] if len(a) >= 3 else 0.0)
    for idx, (basis_type, order) in enumerate(get_term_structure(terms)):
        if basis_type == BasisType.F3_CENTER and idx < len(a):
            margin -= center_M_j(order, gamma) * abs(a[idx])
    return float(margin)


def _data_scale(x_data: np.ndarray) -> float:
    span = float(np.max(x_data) - np.min(x_data)) if len(x_data) else 0.0
    return span if span > 0 else 1.0


def solve_with_constraints(Y: np.ndarray,
                           x_data: np.ndarray,
                           terms: int,
                           gamma: float,
                           constraint_type: ConstraintType) -> np.ndarray:
    """
    Solve the constrained least-squares fit for the given constraint type.

    Parameters
    ----------
    Y : array of shape (m, terms)
        Design matrix.
    x_data : array of shape (m,)
        Target quantile values (on the fitting scale).
    terms : int
        Number of terms.
    gamma : float
        Centre parameter.
    constraint_type : ConstraintType
        Which constraint to enforce.

    Returns
    -------
    coefficients : array of shape (terms,)
    """
    constraint_type = ConstraintType.resolve(constraint_type)
    solvers = {
        ConstraintType.A: solve_positive_all,
        ConstraintType.TL: solve_tail_leading,
        ConstraintType.TA: solve_tail_all,
        ConstraintType.TC: solve_tail_center,
        ConstraintType.TC_MAG: solve_proposition_3,
    }
    solver = solvers.get(constraint_type)
    if solver is None:
        raise QFlexError(f"Unknown constraint type: {constraint_type}")
    return solver(Y, np.asarray(x_data, dtype=float), terms, gamma)


def get_initial_guess(Y: np.ndarray, x_data: np.ndarray, terms: int) -> np.ndarray:
    """Unconstrained least-squares solution (SVD-based ``np.linalg.lstsq``)."""
    try:
        initial_guess, *_ = np.linalg.lstsq(Y, x_data, rcond=None)
    except np.linalg.LinAlgError:
        initial_guess = np.ones(terms) * 0.1
    return initial_guess


def _bounded_lsq(Y: np.ndarray, x_data: np.ndarray, lb: np.ndarray, what: str) -> np.ndarray:
    """
    min ||Y a - x||^2 subject to a >= lb (entries of lb may be -inf).

    Uses the exact active-set BVLS method (Stark & Parker), which finishes
    in a finite number of steps; the iterative 'trf' method used before could
    stop at its iteration cap on ill-conditioned high-K designs. 'trf' remains
    as a fallback.
    """
    ub = np.full(len(lb), np.inf)
    try:
        res = lsq_linear(Y, x_data, bounds=(lb, ub), method='bvls',
                         tol=1e-14, max_iter=BOUNDED_LS_MAX_ITER)
        if res.success:
            return np.maximum(res.x, lb)
    except (ValueError, np.linalg.LinAlgError):
        pass
    res = lsq_linear(Y, x_data, bounds=(lb, ub), method='trf',
                     max_iter=BOUNDED_LS_MAX_ITER, lsmr_tol='auto')
    if not res.success:
        raise QFlexError(f"{what} optimization failed: {res.message}")
    return np.maximum(res.x, lb)


def solve_positive_all(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """A+: all coefficients except the intercept >= 0 (bounded least squares)."""
    lb = np.full(terms, -np.inf)
    lb[1:] = 0.0
    return _bounded_lsq(Y, x_data, lb, "Positive coefficients")


def solve_tail_leading(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    TL+: leading (highest-order) right and left tail coefficients strictly positive.

    They are kept at least TL_FLOOR_REL x (data range) above zero: Theorem 5
    needs them > 0, and a leading coefficient driven to exactly 0 hands the
    tail to the next-lower term, which may be negative.
    """
    floor = TL_FLOOR_REL * _data_scale(x_data)
    return solve_with_bounds(Y, x_data, terms, get_tail_indices(terms, leading_only=True), floor=floor)


def solve_tail_all(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """TA+: all tail coefficients >= 0."""
    return solve_with_bounds(Y, x_data, terms, get_tail_indices(terms, leading_only=False))


def solve_with_bounds(Y: np.ndarray, x_data: np.ndarray, terms: int,
                      bound_indices: List[int], floor: float = 0.0) -> np.ndarray:
    """
    min ||Y a - x||^2 with the coefficients in ``bound_indices`` >= ``floor``.

    A bound-constrained linear least-squares problem, solved exactly by BVLS.
    """
    lb = np.full(terms, -np.inf)
    for idx in bound_indices:
        lb[idx] = floor
    return _bounded_lsq(Y, np.asarray(x_data, dtype=float), lb, "Constrained")


def solve_tail_center(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    TC (Proposition 4): all tail coefficients >= 0 and
    m_TC = (a_2 + a_3) - sum_j M_j |c_j| >= MARGIN_REL x (data range).

    Exact solve with SciPy only (CVXPY is not needed):

    1. The TA+ fit is the answer when it already meets the margin.
    2. Otherwise the margin is active at the optimum (the problem is convex
       and the TA+ optimum is unique). With the signs of the centre terms held
       fixed, |c_j| = s_j c_j and the margin is one linear equality, handled
       by the method of multipliers: each step is a BVLS solve with the
       equality appended as a weighted row.
    3. A centre term that ends at zero has its sign flipped, and the fit
       repeated, when the optimality condition |df/dc_j| <= mu M_j fails
       (mu = the multiplier of the margin).
    4. If rounding leaves the margin a hair below the floor, the centre terms
       are shrunk to restore it.

    Checked against CVXPY/Clarabel on 144 fits (normal, t3, lognormal and
    bimodal samples, N = 30..1000, K = 4..16, the margin active in 135): the
    objective was never higher than Clarabel's (Clarabel's was up to 1e-5
    higher, within its own tolerance).
    """
    x_data = np.asarray(x_data, dtype=float)
    eps = MARGIN_REL * _data_scale(x_data)
    a0 = solve_tail_all(Y, x_data, terms, gamma)
    cent = get_center_indices(terms)
    if terms < 3 or not cent or tail_center_margin_coeff(a0, terms, gamma) >= eps:
        return a0

    structure = get_term_structure(terms)
    lb = np.full(terms, -np.inf)
    gv = np.zeros(terms)
    sign = np.ones(terms)
    for k, (basis_type, order) in enumerate(structure):
        if basis_type == BasisType.CONSTANT:
            continue
        lb[k] = 0.0
        if basis_type in (BasisType.F1_TAIL_RIGHT, BasisType.F2_TAIL_LEFT) and order == 1:
            gv[k] = 1.0
        elif basis_type == BasisType.F3_CENTER:
            gv[k] = -center_M_j(order, gamma)
            sign[k] = 1.0 if a0[k] >= 0 else -1.0

    half_rho = 1e4 * np.sum(Y ** 2) / np.sum(gv ** 2)
    rho = 2.0 * half_rho
    sr = np.sqrt(half_rho)
    a = a0.copy()
    for _outer in range(2 * len(cent) + 2):
        Ae = np.vstack([Y * sign, sr * gv])
        lam = 0.0
        w = np.zeros(terms)
        for _it in range(200):
            be = np.append(x_data, sr * (eps - lam / rho))
            res = lsq_linear(Ae, be, bounds=(lb, np.full(terms, np.inf)), method='bvls',
                             tol=1e-14, max_iter=BOUNDED_LS_MAX_ITER)
            if not res.success:
                raise QFlexError(f"Proposition 4 (TC) optimization failed: {res.message}")
            w = np.maximum(res.x, lb)
            h = gv @ w - eps
            dl = rho * h
            lam += dl
            if abs(h) <= 1e-13 * (np.abs(gv * w).sum() + eps) or abs(dl) <= 1e-9 * abs(lam):
                break
        a = w * sign
        mu = -lam
        grad = 2.0 * Y.T @ (Y @ a - x_data)
        wtol = 1e-12 * np.abs(w).max()
        gscale = np.abs(grad).max()
        flipped = False
        for k in cent:
            if w[k] <= wtol:
                order = structure[k][1]
                if abs(grad[k]) > mu * center_M_j(order, gamma) * (1 + 1e-9) + 1e-12 * gscale:
                    new_sign = -np.sign(grad[k])
                    if new_sign != sign[k]:
                        sign[k] = new_sign
                        flipped = True
        if not flipped:
            break

    margin = tail_center_margin_coeff(a, terms, gamma)
    if margin < eps:
        tail1 = a[1] + (a[2] if terms >= 3 else 0.0)
        centre_sum = tail1 - margin
        if centre_sum > 0:
            factor = (tail1 - eps) / centre_sum * (1 - 1e-12) if tail1 > eps else 0.0
            for k in cent:
                a[k] *= factor
        if not tail_center_margin_coeff(a, terms, gamma) > 0:
            raise QFlexError("Proposition 4 (TC) constraint cannot be satisfied: "
                             "the first-order tail terms are too small")
    return a


def _derivative_rows(p: np.ndarray, terms: int, gamma: float):
    """Tail and centre parts of dQ/dp as matrices: q_tail = Dt @ a, q_center = Dc @ a."""
    structure = get_term_structure(terms)
    Dt = np.zeros((len(p), terms))
    Dc = np.zeros((len(p), terms))
    for k, (basis_type, order) in enumerate(structure):
        if basis_type in (BasisType.F1_TAIL_RIGHT, BasisType.F2_TAIL_LEFT):
            Dt[:, k] = evaluate_basis_derivative(p, basis_type, order, gamma)
        elif basis_type == BasisType.F3_CENTER:
            Dc[:, k] = evaluate_basis_derivative(p, basis_type, order, gamma)
    return Dt, Dc


def _tail_grid(n_each: int) -> np.ndarray:
    lo = np.logspace(np.log10(PROB_EPS), -3, n_each, endpoint=False)
    return np.concatenate([lo, 1.0 - lo[::-1]])


def _check_grid() -> np.ndarray:
    """Fine verification grid: the paper grid refined 10x plus 300 points per tail."""
    return np.unique(np.concatenate([np.linspace(0.001, 0.999, 9981), _tail_grid(300)]))


def _have_cvxpy() -> bool:
    try:
        import cvxpy  # noqa: F401
        return True
    except ImportError:
        return False


def _solve_prop3_qp(Y, x_data, terms, Dt, Dc, lb, eps, start, use_cvxpy):
    """
    min ||Y a - x||^2  s.t.  Dt a >= t,  |Dc a| <= s,  t - s >= eps,  a >= lb.

    m_tail = min_i (Dt a)_i and M_center = max_i |(Dc a)_i| are linear-program
    epigraphs, so the Proposition 3 condition m_tail - M_center >= eps is a set
    of linear inequalities in (a, t, s): a convex quadratic program.
    """
    n = terms
    scale = max(float(np.sum(x_data ** 2)), 1e-300)
    rt = np.linalg.norm(np.hstack([Dt, np.ones((len(Dt), 1))]), axis=1)
    rc = np.linalg.norm(np.hstack([Dc, np.ones((len(Dc), 1))]), axis=1)
    if use_cvxpy:
        import cvxpy as cp
        a = cp.Variable(n)
        t = cp.Variable()
        s = cp.Variable()
        cons = [(Dt @ a - t) / rt >= 0, (s - Dc @ a) / rc >= 0, (s + Dc @ a) / rc >= 0, t - s >= eps]
        fin = np.isfinite(lb)
        if np.any(fin):
            cons.append(a[np.where(fin)[0]] >= lb[fin])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(Y @ a - x_data) / scale), cons)
        for solver in ('CLARABEL', 'ECOS', 'SCS'):
            try:
                prob.solve(solver=solver)
            except Exception:
                continue
            if prob.status in ('optimal', 'optimal_inaccurate') and a.value is not None:
                return np.asarray(a.value, dtype=float)
        return None

    # SciPy: SLSQP from a feasible start. The fit error is written in the
    # coordinates b = R a (Y = QR), where it is ||b - Q'x||^2: an identity
    # Hessian, so SLSQP's quasi-Newton steps are exact and an ill-conditioned
    # design (high K) no longer stalls it. All constraints are linear in b.
    Qm, R = np.linalg.qr(Y)
    c = Qm.T @ x_data
    try:
        Rinv = np.linalg.solve(R, np.eye(n))
    except np.linalg.LinAlgError:
        return None
    m_t, m_c = len(Dt), len(Dc)
    fin = np.where(np.isfinite(lb))[0]
    rows = m_t + 2 * m_c + 1 + len(fin)
    A = np.zeros((rows, n + 2))
    b_vec = np.zeros(rows)
    A[:m_t, :n] = (Dt / rt[:, None]) @ Rinv
    A[:m_t, n] = -1.0 / rt
    A[m_t:m_t + m_c, :n] = -(Dc / rc[:, None]) @ Rinv
    A[m_t:m_t + m_c, n + 1] = 1.0 / rc
    A[m_t + m_c:m_t + 2 * m_c, :n] = (Dc / rc[:, None]) @ Rinv
    A[m_t + m_c:m_t + 2 * m_c, n + 1] = 1.0 / rc
    k = m_t + 2 * m_c
    A[k, n], A[k, n + 1] = 1.0, -1.0
    b_vec[k] = eps
    for r_i, idx in enumerate(fin):
        A[k + 1 + r_i, :n] = Rinv[idx]
        b_vec[k + 1 + r_i] = lb[idx]
    cs = max(float(c @ c), 1e-300)

    def f(v):
        d = v[:n] - c
        return float(d @ d) / cs

    def g(v):
        out = np.zeros(n + 2)
        out[:n] = 2.0 * (v[:n] - c) / cs
        return out

    t0 = float(np.min(Dt @ start))
    s0 = float(np.max(np.abs(Dc @ start)))
    v0 = np.concatenate([R @ start, [t0, s0]])
    res = minimize(f, v0, jac=g, method='SLSQP',
                   bounds=[(None, None)] * n + [(None, None), (0.0, None)],
                   constraints=[{'type': 'ineq', 'fun': lambda v: A @ v - b_vec, 'jac': lambda v: A}],
                   options={'ftol': 1e-16, 'maxiter': 3000})
    if res.x is None:
        return None
    return np.asarray(Rinv @ res.x[:n], dtype=float)


def solve_proposition_3(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float,
                        use_cvxpy: Optional[bool] = None) -> np.ndarray:
    """
    TC_mag (Proposition 3): minimise the fit error subject to
    m_tail > M_center on a grid, with the leading tail coefficients > 0.

    m_tail = min q_tail(p) and M_center = max |q_center(p)| over the grid, where
    q_tail and q_center are the tail and centre parts of dQ/dp. The condition
    implies dQ/dp > 0 at every grid point.

    The problem is solved as a convex QP (CVXPY/Clarabel when installed,
    otherwise SciPy SLSQP, within about 0.2% in squared error in tests),
    started from the TC fit, which satisfies the condition. The result is
    never worse than that start (the TC fit with its leading tails lifted to
    the floor below). The grid starts coarse and grows: after each solve the condition
    is checked on a fine grid (the 0.001 grid refined 10x, plus 300 points in
    each tail down to 1e-12), the worst points are added, and the fit repeats
    until it holds there. The leading tail coefficients are kept above
    TL_FLOOR_REL x (data range), as Theorem 5 requires. If the refinement does
    not settle, the TC fit is returned with a warning.
    """
    x_data = np.asarray(x_data, dtype=float)
    if use_cvxpy is None:
        use_cvxpy = _have_cvxpy()
    scale = _data_scale(x_data)
    eps = MARGIN_REL * scale
    floor = TL_FLOOR_REL * scale

    lb = np.full(terms, -np.inf)
    for idx in get_tail_indices(terms, leading_only=True):
        lb[idx] = floor

    # Feasible start: the TC fit with its leading tails lifted to the floor
    # (raising a tail coefficient only raises q_tail, so it stays feasible).
    start = solve_tail_center(Y, x_data, terms, gamma)
    start = np.maximum(start, np.where(np.isfinite(lb), lb, -np.inf))

    p_fine = _check_grid()
    Dt_f, Dc_f = _derivative_rows(p_fine, terms, gamma)

    def margin_on_fine(a):
        return float(np.min(Dt_f @ a) - np.max(np.abs(Dc_f @ a)))

    p_work = np.unique(np.concatenate([np.linspace(0.001, 0.999, 99), [0.001, 0.999], _tail_grid(15)]))
    best = None
    eps_k = eps
    for _round in range(12):
        Dt, Dc = _derivative_rows(p_work, terms, gamma)
        a = _solve_prop3_qp(Y, x_data, terms, Dt, Dc, lb, eps_k, start, use_cvxpy)
        if a is None:
            break
        a = np.maximum(a, np.where(np.isfinite(lb), lb, -np.inf))
        if margin_on_fine(a) > 0:
            best = a
            break
        qt = Dt_f @ a
        qc = np.abs(Dc_f @ a)
        worst = np.unique(np.concatenate([np.argsort(qt)[:20], np.argsort(-qc)[:20]]))
        grown = np.unique(np.concatenate([p_work, p_fine[worst]]))
        if len(grown) == len(p_work):
            # the worst points are already constrained: the solver met them only
            # to its tolerance, so ask for a larger margin
            eps_k *= 10.0
        p_work = grown

    if best is None:
        warnings.warn("TC_mag (Proposition 3) fit did not settle; returning the TC "
                      "(Proposition 4) fit, which satisfies the same condition")
        best = start
    # never return something worse than the feasible TC start
    if np.sum((Y @ best - x_data) ** 2) > np.sum((Y @ start - x_data) ** 2) * (1 + 1e-12):
        best = start
    return best


def solve_proposition_4(Y: np.ndarray, x_data: np.ndarray, terms: int, gamma: float) -> np.ndarray:
    """
    Backward-compatible name of :func:`solve_proposition_3` (TC_mag).

    Earlier releases numbered the grid condition "Proposition 4"; the paper
    numbers it Proposition 3.
    """
    return solve_proposition_3(Y, x_data, terms, gamma)
