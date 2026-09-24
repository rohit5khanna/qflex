"""
Differential and round-trip tests.

These exist because the original suite checked that functions RUN and that
outputs look plausible, but never compared an analytical derivative against a
finite difference, and never inverted `cdf` back through `quantile`. Four
defects lived in exactly that gap:

  * evaluate_basis_derivative returned 0 for odd F2_TAIL_LEFT orders
  * evaluate_quantile_derivative was therefore wrong from K = 9 (where L3 enters)
  * compute_cdf_inverse bracketed across a non-monotone tail and converged to
    the wrong root, silently
  * compute_pdf_numerical took abs() of a negative gradient, which made the
    PDF-positivity half of _check_feasibility unreachable

Every test here is a property that must hold for any correct implementation,
not a pinned number, so they stay meaningful as the library changes.
"""
import numpy as np
import pytest

from qflex import QFlex, LogQFlex, LogitQFlex
from qflex.core import ConstraintType
from qflex.basis import (BasisType, evaluate_basis, evaluate_basis_derivative,
                         evaluate_quantile_derivative, get_term_structure)

FD_STEP = 1e-9
PROBS = np.array([1e-5, 1e-3, 0.01, 0.3, 0.7, 0.99])
ALL_BASES = [BasisType.CONSTANT, BasisType.F1_TAIL_RIGHT,
             BasisType.F2_TAIL_LEFT, BasisType.F3_CENTER]


def _central(f, y, h=FD_STEP):
    return (f(np.array([y + h]))[0] - f(np.array([y - h]))[0]) / (2 * h)


@pytest.mark.parametrize("basis_type", ALL_BASES)
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_basis_derivative_matches_finite_difference(basis_type, order):
    """d/dy of every basis function, at every order, against central differences."""
    gamma = 0.5
    for y in PROBS:
        analytic = float(evaluate_basis_derivative(np.array([y]), basis_type, order, gamma)[0])
        numeric = float(_central(lambda v: evaluate_basis(v, basis_type, order, gamma), y))
        scale = max(abs(numeric), 1e-10)
        assert abs(analytic - numeric) / scale < 1e-5, (
            f"{basis_type} order={order} at y={y:g}: "
            f"analytical {analytic:.6e} vs finite difference {numeric:.6e}")


@pytest.mark.parametrize("terms", list(range(3, 15)))
def test_quantile_derivative_matches_finite_difference(terms):
    """dQ/dp assembled from the basis must match a finite difference on Q."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.normal(size=300))
    y = np.arange(1, 301) / 301
    m = QFlex(x, y, terms=terms)

    grid = np.linspace(0.02, 0.98, 200)
    analytic = np.asarray(
        evaluate_quantile_derivative(grid, m.coefficients, m.terms, m.gamma), float)
    numeric = (np.asarray(m.quantile(grid + FD_STEP), float)
               - np.asarray(m.quantile(grid - FD_STEP), float)) / (2 * FD_STEP)

    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-10)
    assert rel.max() < 1e-4, (
        f"K={terms}: max relative error {rel.max():.3e} at "
        f"p={grid[int(np.argmax(rel))]:.4f}")


@pytest.mark.parametrize("terms", [3, 5, 7, 9, 11])
@pytest.mark.parametrize("cls,kwargs,sampler", [
    (QFlex, {}, lambda r: np.sort(r.normal(size=200))),
    (LogQFlex, {"lower_bound": 0}, lambda r: np.sort(r.lognormal(0, 0.4, 200))),
    (LogitQFlex, {"lower_bound": 0, "upper_bound": 1},
     lambda r: np.sort(r.uniform(0.05, 0.95, 200))),
])
def test_cdf_inverts_quantile(cls, kwargs, sampler, terms):
    """cdf(quantile(p)) must return p, for EVERY feasible fit.

    Swept over seeds rather than pinned to one: the failure needs a fit that is
    monotone across the data but inverted beyond it, which a single seed will
    usually miss. One such fit is enough to break the root-finder, so a test
    that samples only one draw gives false confidence."""
    p = np.linspace(0.05, 0.95, 40)
    checked = 0
    worst = (0.0, None)
    for seed in range(12):
        rng = np.random.default_rng(seed)
        x = sampler(rng)
        y = np.arange(1, len(x) + 1) / (len(x) + 1)
        m = cls(x, y, terms=terms, **kwargs)
        if not m.is_feasible:
            continue
        checked += 1
        back = np.asarray(m.cdf(np.asarray(m.quantile(p), float)), float)
        err = float(np.max(np.abs(back - p)))
        if err > worst[0]:
            worst = (err, seed)
    if checked == 0:
        pytest.skip("no feasible fit at this configuration")
    assert worst[0] < 1e-6, (
        f"{cls.__name__} K={terms}: max |cdf(Q(p)) - p| = {worst[0]:.3e} "
        f"(seed {worst[1]}, {checked} feasible fits checked)")


def test_pdf_sign_is_not_discarded():
    """A decreasing quantile function must NOT yield a positive density.

    compute_pdf_numerical used to apply abs() to a negative gradient, which made
    the PDF look valid wherever the fit was actually inverted -- and made the
    PDF-positivity half of _check_feasibility unreachable."""
    from qflex.utils import compute_pdf_numerical

    # A deliberately decreasing "quantile function".
    decreasing = lambda p: -np.asarray(p, float)
    pdf = np.asarray(compute_pdf_numerical(decreasing, np.linspace(0.1, 0.9, 9)), float)
    assert not np.any(pdf > 0), (
        "density reported positive for a strictly decreasing quantile function: "
        f"{pdf[:3]}")


def test_feasibility_pdf_condition_is_reachable():
    """_check_feasibility claims to require a strictly positive PDF. If the PDF
    can never be non-positive, that half of the test is dead code."""
    from qflex.utils import compute_pdf_numerical

    decreasing = lambda p: -np.asarray(p, float)
    vals = np.asarray(compute_pdf_numerical(decreasing, np.linspace(0.001, 0.999, 99)), float)
    assert not bool(np.all(vals > 0)), (
        "np.all(pdf > 0) is True even for a decreasing quantile function, so the "
        "feasibility check's PDF condition can never fail")


@pytest.mark.parametrize("terms", [4, 7, 10, 13])
def test_analytical_and_numerical_pdf_agree(terms):
    """The two PDF paths must agree where the fit is well behaved."""
    rng = np.random.default_rng(5)
    x = np.sort(rng.normal(size=300))
    y = np.arange(1, 301) / 301
    m = QFlex(x, y, terms=terms, constraint_type=ConstraintType.TA)
    if not m.is_feasible:
        pytest.skip("infeasible fit")

    grid = np.linspace(0.05, 0.95, 100)
    a = np.asarray(m.pdf(grid, method="analytical"), float)
    n = np.asarray(m.pdf(grid, method="numerical"), float)
    rel = np.abs(a - n) / np.maximum(np.abs(n), 1e-10)
    assert rel.max() < 1e-3, (
        f"K={terms}: analytical and numerical PDF differ by {rel.max():.3e} "
        f"at p={grid[int(np.argmax(rel))]:.4f}")
