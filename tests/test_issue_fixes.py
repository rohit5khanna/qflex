"""
Regression tests for the fixes in qflex 1.2.0 (issue IDs from the review log).
"""

import time
import warnings

import numpy as np
import pytest

from qflex import (QFlex, LogQFlex, LogitQFlex, ConstraintType, QFlexError,
                   check_proposition3, check_proposition4)
from qflex.basis import build_design_matrix, evaluate_quantile_derivative
from qflex.constraints import (solve_proposition_3, solve_tail_center,
                               tail_center_margin_coeff, TL_FLOOR_REL)
from qflex.utils import calculate_gamma, compute_w1

# Fine grid that includes the tails down to 1e-12 (the clip)
FINE = np.concatenate([np.logspace(-12, -3, 300, endpoint=False),
                       np.linspace(0.001, 0.999, 9981),
                       1 - np.logspace(-12, -3, 300, endpoint=False)[::-1]])


def _sample(dist, seed, n=200):
    r = np.random.default_rng(seed)
    x = {'normal': r.normal(size=n),
         'lognormal': r.lognormal(0, 0.8, n),
         't3': r.standard_t(3, n),
         'bimodal': np.where(r.random(n) < 0.4, r.normal(-2, 0.5, n), r.normal(1.5, 1, n))}[dist]
    x = np.sort(x)
    return x, np.arange(1, n + 1) / (n + 1)


def _qmin(qf):
    return float(np.min(evaluate_quantile_derivative(FINE, qf.coefficients, qf.terms, qf.gamma)))


# --- H1: TC_mag returns valid fits as good as TC (up to the leading-tail floor)

@pytest.mark.parametrize("dist", ['normal', 'lognormal', 't3', 'bimodal'])
@pytest.mark.parametrize("terms", [5, 7, 10])
def test_h1_tc_mag_valid_everywhere_and_not_worse_than_tc(dist, terms):
    x, y = _sample(dist, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qm = QFlex(x, y, terms=terms, constraint_type=ConstraintType.TC_MAG)
        qc = QFlex(x, y, terms=terms, constraint_type=ConstraintType.TC)
    assert _qmin(qm) > 0
    Y = build_design_matrix(y, terms, qm.gamma)
    sse_m = np.sum((Y @ qm.coefficients - x) ** 2)
    sse_c = np.sum((Y @ qc.coefficients - x) ** 2)
    assert sse_m <= sse_c * (1 + 1e-4)
    assert check_proposition3(qm.coefficients, terms, qm.gamma)['satisfied']


def test_h1_tc_mag_scipy_path_matches_cvxpy_path():
    pytest.importorskip("cvxpy")
    x, y = _sample('t3', 1)
    gamma = QFlex(x, y, terms=10).gamma
    Y = build_design_matrix(y, 10, gamma)
    a_cv = solve_proposition_3(Y, x, 10, gamma, use_cvxpy=True)
    a_sp = solve_proposition_3(Y, x, 10, gamma, use_cvxpy=False)
    s_cv = np.sum((Y @ a_cv - x) ** 2)
    s_sp = np.sum((Y @ a_sp - x) ** 2)
    assert s_sp <= s_cv * (1 + 2e-3)


# --- H2: the grid check also needs Theorem 5 --------------------------------

def test_h2_negative_leading_tail_is_rejected():
    res = check_proposition4(np.array([0, 1, 1, 0, -0.05]), 5, 0.5)
    assert res['grid_condition']            # the grid alone would pass it
    assert not res['theorem5_satisfied']
    assert not res['satisfied']
    assert res == check_proposition3(np.array([0, 1, 1, 0, -0.05]), 5, 0.5)


# --- H3: tail densities ------------------------------------------------------

def test_h3_numerical_and_analytical_pdf_agree_in_the_tails():
    x, y = _sample('normal', 1)
    qf = QFlex(x, y, terms=4, constraint_type=ConstraintType.TA)
    p = np.array([0.001, 0.002, 0.5, 0.998, 0.999])
    num = qf.pdf(p, method='numerical')
    ana = qf.pdf(p, method='analytical')
    assert np.allclose(num, ana, rtol=1e-4)
    assert np.allclose(qf.pdf(p), ana)       # default is analytical


# --- H4: moments that do not exist ------------------------------------------

def test_h4_heavy_tail_moments_are_infinite_not_negative():
    r = np.random.default_rng(2)
    x = np.sort((1 - r.random(300)) ** (-1 / 1.5))       # Pareto(1.5)
    y = np.arange(1, 301) / 301
    qf = LogQFlex(x, y, lower_bound=0, terms=3)
    assert 0.5 < qf.coefficients[1] < 1                 # mean exists, variance does not
    with pytest.warns(UserWarning, match="do not exist"):
        m = qf.moments()
    assert np.isfinite(m['mean'])
    assert m['variance'] == np.inf
    assert np.isnan(m['skewness'])


def test_h4_moments_match_exact_lognormal_like_values():
    # Q = exp(0.3 R1 + 0.3 L1) has E[X] = B(1.3, 0.7), E[X^2] = B(1.6, 0.4)
    from scipy.special import beta
    qf = LogQFlex([1.0, 2.0, 3.0], [0.1, 0.5, 0.9], lower_bound=0, terms=3)
    qf.qflex.coefficients = qf.coefficients = np.array([0.0, 0.3, 0.3])
    m = qf.moments(order=2)
    assert m['mean'] == pytest.approx(beta(1.3, 0.7), rel=1e-10)
    assert m['raw_2'] == pytest.approx(beta(1.6, 0.4), rel=1e-8)


# --- H5: gamma for symmetric assessments ------------------------------------

def test_h5_symmetric_gamma_and_full_rank_k7():
    assert calculate_gamma([10, 50, 90], [0.1, 0.5, 0.9]) == 0.45
    assert calculate_gamma([5, 5, 5], [0.1, 0.5, 0.9]) == 0.45
    y = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    x = np.array([-3, -2, -1, 0, 1, 2, 3.0])
    qf = QFlex(x, y, terms=7)
    assert qf.gamma == 0.45
    assert np.linalg.matrix_rank(build_design_matrix(y, 7, qf.gamma)) == 7
    assert np.allclose(qf.quantile(y), x, atol=1e-8)


# --- H6 / L11: logit back-transform never overflows --------------------------

def test_h6_logit_quantile_finite_for_extreme_z():
    qf = LogitQFlex([0.2, 0.5, 0.7], [0.1, 0.5, 0.9], 0, 1, terms=3)
    qf.qflex.coefficients = qf.coefficients = np.array([0.0, 60.0, 60.0])   # |z| ~ 1600 at the clip
    q = qf.quantile(np.array([1e-12, 1e-6, 0.5, 1 - 1e-12]))
    assert np.all(np.isfinite(q))
    assert q[0] == pytest.approx(0.0) and q[-1] == pytest.approx(1.0)
    d = qf.pdf(np.array([1e-6, 0.5]))
    assert not np.any(np.isnan(d)) and np.all(d > 0)   # inf at the saturated end, not NaN


# --- M1: TC no longer depends on cvxpy --------------------------------------

@pytest.mark.parametrize("dist,terms,seed", [('lognormal', 7, 3), ('t3', 7, 1), ('t3', 7, 3),
                                             ('lognormal', 12, 0), ('bimodal', 12, 2)])
def test_m1_tc_solves_and_is_valid(dist, terms, seed):
    x, y = _sample(dist, seed)
    qf = QFlex(x, y, terms=terms, constraint_type=ConstraintType.TC)
    assert tail_center_margin_coeff(qf.coefficients, terms, qf.gamma) > 0
    assert _qmin(qf) > 0


def test_m1_tc_matches_a_convex_solver():
    cp = pytest.importorskip("cvxpy")
    x, y = _sample('t3', 1)
    terms = 10
    qf = QFlex(x, y, terms=terms, constraint_type='TC')
    Y = build_design_matrix(y, terms, qf.gamma)
    from qflex.constraints import get_tail_indices, get_center_indices, center_M_j, MARGIN_REL
    from qflex.basis import get_term_structure
    a = cp.Variable(terms)
    st = get_term_structure(terms)
    cons = [a[i] >= 0 for i in get_tail_indices(terms)]
    cons.append(a[1] + a[2] - sum(center_M_j(st[i][1], qf.gamma) * cp.abs(a[i])
                                  for i in get_center_indices(terms)) >= MARGIN_REL * (x.max() - x.min()))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Y @ a - x)), cons)
    prob.solve(solver='CLARABEL')
    ours = np.sum((Y @ qf.coefficients - x) ** 2)
    assert ours <= prob.value * (1 + 1e-6)


# --- M3: paper numbering with the old names still working -------------------

def test_m3_old_and_new_names_agree():
    qf = QFlex([12, 18, 25, 34, 45], [0.1, 0.25, 0.5, 0.75, 0.9], terms=5)
    assert qf.check_proposition3() == qf.check_proposition4()
    lq = LogQFlex([12, 18, 25, 34, 45], [0.1, 0.25, 0.5, 0.75, 0.9], terms=5)
    assert lq.check_proposition3() == lq.check_proposition4()


# --- M4: W1 on uneven grids ---------------------------------------------------

def test_m4_w1_same_on_uniform_and_nonuniform_grids():
    x, y = _sample('normal', 4, 50)
    qf = QFlex(x, y, terms=5)
    g1 = np.linspace(0.01, 0.99, 4000)
    g2 = np.sort(np.concatenate([np.linspace(0.01, 0.5, 3000), np.linspace(0.5, 0.99, 600)[1:]]))
    w_a = compute_w1(qf.quantile, x, y, y_grid=g1)[0]
    w_b = compute_w1(qf.quantile, x, y, y_grid=g2)[0]
    assert w_a == pytest.approx(w_b, rel=1e-3)


# --- M5 / M6 ------------------------------------------------------------------

def test_m5_moments_order_one():
    m = QFlex([12, 25, 45], [0.1, 0.5, 0.9]).moments(order=1)
    assert set(m) == {'raw_1', 'mean'}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("cls,kw", [(QFlex, {}), (LogQFlex, {'lower_bound': 0}),
                                    (LogitQFlex, {'lower_bound': 0, 'upper_bound': 100})])
def test_m6_non_finite_inputs_raise(bad, cls, kw):
    with pytest.raises(QFlexError, match="finite"):
        cls([12, bad, 45], [0.1, 0.5, 0.9], **kw)
    with pytest.raises(QFlexError, match="finite"):
        cls([12, 25, 45], [0.1, bad, 0.9], **kw)


# --- Low severity -------------------------------------------------------------

def test_l1_num_points_is_used():
    qf = QFlex([12, 18, 25, 34, 45], [0.1, 0.25, 0.5, 0.75, 0.9], terms=5)
    assert qf.check_proposition3(num_points=10)['m_tail'] != qf.check_proposition3(num_points=100000)['m_tail']
    assert (qf.check_proposition3()['m_tail']
            == qf.check_proposition3(p_grid=0.001 + np.arange(999) * 0.001)['m_tail'])


def test_l4_one_warning_per_infeasible_fit():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LogQFlex([45, 25, 12], [0.1, 0.5, 0.9])
    assert [str(i.message) for i in w] == ["LogQFlex distribution may not be feasible"]


def test_l5_sample_is_reproducible():
    qf = QFlex([12, 25, 45], [0.1, 0.5, 0.9])
    assert np.array_equal(qf.sample(10, random_state=3), qf.sample(10, random_state=3))
    g1, g2 = np.random.default_rng(9), np.random.default_rng(9)
    assert np.array_equal(qf.sample((2, 3), random_state=g1), qf.sample((2, 3), random_state=g2))


def test_l6_cdf_is_fast_and_exact():
    x, y = _sample('normal', 5, 100)
    qf = QFlex(x, y, terms=5, constraint_type='A+')
    xs = np.linspace(-3, 3, 10000)
    t0 = time.time()
    c = qf.cdf(xs)
    assert time.time() - t0 < 1.0
    inside = (c > 0) & (c < 1)
    assert np.max(np.abs(qf.quantile(c[inside]) - xs[inside])) < 1e-9
    assert isinstance(qf.cdf(0.3), float)


def test_l7_quantile_limits_at_0_and_1():
    qf = QFlex([12, 25, 45], [0.1, 0.5, 0.9])
    q = qf.quantile([0.0, 0.5, 1.0])
    assert q[0] == -np.inf and q[2] == np.inf and np.isfinite(q[1])
    assert LogQFlex([12, 25, 45], [0.1, 0.5, 0.9]).quantile([0.0, 1.0]).tolist() == [0.0, np.inf]
    assert LogitQFlex([0.2, 0.5, 0.7], [0.1, 0.5, 0.9]).quantile([0.0, 1.0]).tolist() == [0.0, 1.0]


def test_l8_derivative_and_quantile_use_the_same_clip():
    qf = QFlex([12, 18, 25, 34, 45], [0.1, 0.25, 0.5, 0.75, 0.9], terms=5)
    p, h = 1e-11, 1e-13
    numeric = (qf.quantile(p + h) - qf.quantile(p - h)) / (2 * h)
    assert 1 / qf.pdf(p) == pytest.approx(numeric, rel=1e-3)


@pytest.mark.parametrize("args,match", [
    ((np.ones((3, 2)), np.full((3, 2), 0.5)), "one-dimensional"),
    (([1, 2, 3], [0.1, 0.5, 0.9], 2.5), "positive integer"),
    (([1, 2, 3], [0.1, 0.5, 0.9], 0), "positive integer"),
    (([1, 2, 3], [0.1, 0.5, 0.9], 3, 'Z'), "Unknown constraint type"),
])
def test_l9_clear_errors(args, match):
    with pytest.raises(QFlexError, match=match):
        QFlex(*args)


def test_l9_integer_valued_float_terms_and_string_constraints_accepted():
    qf = QFlex([12, 18, 25, 34, 45], [0.1, 0.25, 0.5, 0.75, 0.9], terms=5.0, constraint_type='ta+')
    assert qf.terms == 5 and qf.constraint_type is ConstraintType.TA


def test_l10_a_plus_high_k_does_not_hit_iteration_cap():
    r = np.random.default_rng(6)
    v = np.sort(np.concatenate([r.normal(2.0, 0.3, 600), r.normal(3.2, 0.2, 399)]))
    v = (v - v.min() + 0.1) / (v.max() - v.min() + 0.2)
    y = np.arange(1, 1000) / 1000
    for terms in (12, 14, 16):
        qf = LogitQFlex(v, y, 0, 1, terms=terms, constraint_type=ConstraintType.A)
        assert np.all(qf.coefficients[1:] >= 0)


def test_tl_leading_tails_strictly_positive():
    x, y = _sample('t3', 2)
    qf = QFlex(x, y, terms=8, constraint_type=ConstraintType.TL)
    from qflex.constraints import get_tail_indices
    floor = TL_FLOOR_REL * (x.max() - x.min())
    assert np.all(qf.coefficients[get_tail_indices(8, leading_only=True)] >= floor * (1 - 1e-12))
    assert qf.check_proposition3()['theorem5_satisfied']


def test_a_plus_and_tc_valid_on_the_whole_range():
    for dist in ('normal', 'lognormal', 't3', 'bimodal'):
        x, y = _sample(dist, 7)
        for terms in (5, 9, 13):
            for c in (ConstraintType.A, ConstraintType.TC):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    qf = QFlex(x, y, terms=terms, constraint_type=c)
                assert _qmin(qf) > 0, (dist, terms, c)
