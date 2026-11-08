"""
Utility Functions for QFlex

Standalone numerical methods for PDF/CDF computation, moment calculation,
gamma estimation, and Wasserstein distance. The PDF, CDF, moment and W1
functions take a quantile function Q(p) -> x, so they work with any QPD.
"""

import warnings

import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import interp1d
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .core import QFlexBase

PROB_EPS = 1e-12

#: Points (evenly spaced in log-odds) used to bracket Q before inverting it.
CDF_SCAN_POINTS = 4001

#: gamma for a symmetric assessment (skewness index 0), as in the paper.
GAMMA_SYMMETRIC = 0.45


def compute_pdf_numerical(quantile_func: Callable, y: np.ndarray, step_size: float = 1e-6) -> np.ndarray:
    """
    Compute PDF via numerical differentiation of the quantile function.
    
    Since PDF(y) = 1 / q(y) where q(y) = dQ/dy, we estimate the derivative
    using central differences and invert.
    
    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(y) -> x.
    y : array
        Cumulative probabilities in (0, 1).
    step_size : float
        Finite-difference step at p = 0.5; it scales with min(p, 1 - p).
        
    Returns
    -------
    pdf : array
        Probability density values.
    """
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)

    def derivative(y_vals, delta):
        # The step is relative to the distance from the nearer end: delta at
        # p = 0.5, shrinking in proportion towards 0 and 1. A fixed step of
        # 0.001 at p = 0.001 hit the 1e-12 clip and gave a one-sided difference
        # over the wrong interval (90% error in the tails).
        h = delta * 2.0 * np.minimum(y_vals, 1 - y_vals)
        y_plus = np.clip(y_vals + h, PROB_EPS, 1 - PROB_EPS)
        y_minus = np.clip(y_vals - h, PROB_EPS, 1 - PROB_EPS)
        
        q_plus = quantile_func(y_plus)
        q_minus = quantile_func(y_minus)
        
        denom = y_plus - y_minus
        denom = np.where(np.abs(denom) < PROB_EPS,
                        np.sign(denom) * PROB_EPS + (denom == 0) * PROB_EPS,
                        denom)
        
        return (q_plus - q_minus) / denom
    
    grad = derivative(y, step_size)
    
    # Retry with larger step size where needed
    bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
    if np.any(bad):
        grad[bad] = derivative(y[bad], step_size * 10)
    
    bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
    if np.any(bad):
        grad[bad] = derivative(y[bad], step_size * 100)
    
    # THE SIGN IS INFORMATION, NOT NOISE. This used to read
    #     grad = np.where(grad <= 0, np.abs(grad), grad)
    # which reported a positive density wherever the quantile function was
    # DECREASING -- i.e. exactly where the fit is invalid. It also made the
    # PDF-positivity half of QFlex._check_feasibility unreachable, since the
    # PDF could then never be non-positive. A negative gradient now yields a
    # negative density, and a zero gradient an undefined one, so callers can
    # see the problem instead of inheriting a plausible-looking number.
    grad = np.where(grad == 0, np.nan, grad)

    out = 1.0 / grad
    return out[0] if scalar else out


def compute_cdf_inverse(quantile_func: Callable, x: np.ndarray,
                        y_data: np.ndarray = None) -> np.ndarray:
    """
    Compute the CDF by inverting the quantile function.

    Finds y such that Q(y) = x for each input value: a scan on a log-odds grid
    brackets each x between two grid points, then vectorised bisection
    refines all points at once (to ~1e-15 in y). Values below Q(PROB_EPS)
    get 0, above Q(1 - PROB_EPS) get 1.

    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(y) -> x.
    x : array
        Values at which to evaluate the CDF.
    y_data : array, optional
        Unused; kept for backward compatibility.

    Returns
    -------
    cdf : array
        Cumulative probabilities.
    """
    x = np.asarray(x, dtype=float)
    x_scalar = x.ndim == 0
    x_flat = x.ravel()

    # INVERT ONLY WHERE Q IS ACTUALLY INCREASING.
    #
    # A fit can be monotone across the data and inverted beyond it -- on one
    # real example Q(1e-12) = +24.2 while Q(1-1e-12) = -26.0, with Q properly
    # increasing in between. Inverting over the whole range then returns
    # whichever root happens to be found (cdf(Q(0.5)) came back as 1.0).
    # So: scan once, take the longest strictly-increasing run, and invert
    # inside that. For a valid fit this is the whole interval.
    umax = np.log((1 - PROB_EPS) / PROB_EPS)
    u_grid = np.linspace(-umax, umax, CDF_SCAN_POINTS)
    grid = 0.5 * (1.0 + np.tanh(0.5 * u_grid))
    grid[0], grid[-1] = PROB_EPS, 1 - PROB_EPS
    q_grid = np.asarray(quantile_func(grid), dtype=float)

    finite = np.isfinite(q_grid)
    increasing = finite[:-1] & finite[1:] & (np.diff(q_grid) > 0)

    # longest run of True in `increasing`
    best_len = best_start = 0
    run_len = run_start = 0
    for i, inc in enumerate(increasing):
        if inc:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0

    if best_len == 0:
        warnings.warn("quantile function is nowhere increasing; cdf is undefined")
        out = np.full_like(x_flat, np.nan, dtype=float)
        return out[0] if x_scalar else out.reshape(x.shape)

    lo_i, hi_i = best_start, best_start + best_len
    if best_len < len(increasing):
        warnings.warn(
            "quantile function is not monotone over the full probability range; "
            f"cdf inverted on [{grid[lo_i]:.4g}, {grid[hi_i]:.4g}] only")

    g = grid[lo_i:hi_i + 1]
    qg = q_grid[lo_i:hi_i + 1]
    cdf_vals = np.full(x_flat.shape, np.nan)
    cdf_vals[x_flat <= qg[0]] = 0.0
    cdf_vals[x_flat >= qg[-1]] = 1.0
    inside = (x_flat > qg[0]) & (x_flat < qg[-1])
    if np.any(inside):
        xt = x_flat[inside]
        k = np.clip(np.searchsorted(qg, xt, side='right') - 1, 0, len(g) - 2)
        a, b = g[k].copy(), g[k + 1].copy()
        for _ in range(100):
            m = 0.5 * (a + b)
            below = np.asarray(quantile_func(m), dtype=float) < xt
            a = np.where(below, m, a)
            b = np.where(below, b, m)
            if np.all(b - a <= 4e-16 * np.maximum(b, 1e-300)):
                break
        cdf_vals[inside] = 0.5 * (a + b)
    cdf_vals[np.isnan(x_flat)] = np.nan

    if x_scalar:
        return float(cdf_vals[0])
    return cdf_vals.reshape(x.shape)


def compute_moments(quantile_func: Callable, order: int = 4,
                    quantile_logodds: Optional[Callable] = None,
                    moment_exists: Optional[Callable] = None) -> dict:
    """
    Compute moments by integrating the quantile function over p.

    E[X^k] = integral_0^1 Q(p)^k dp is computed in the log-odds u = ln(p/(1-p)),
    dp = p(1-p) du, by Simpson's rule on |u| <= 700 (step 0.01 for |u| <= 40,
    0.1 beyond). With ``quantile_logodds`` (Q as a function of u) the tails are
    integrated without the 1e-12 clip; otherwise ``quantile_func`` is used.

    ``moment_exists(k)`` may return False when E[X^k] diverges (for example a
    log-transformed fit whose right tail is too heavy). Those moments are
    reported as inf, the statistics that depend on them as inf or nan, and a
    warning says so, instead of returning a meaningless finite number.

    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(y) -> x.
    order : int
        Maximum moment order to compute (default 4; 1 gives the mean only).
    quantile_logodds : callable, optional
        Q as a function of the log-odds of p, accurate in the extreme tails.
    moment_exists : callable, optional
        k -> bool, whether E[X^k] is finite.

    Returns
    -------
    moments : dict
        'raw_k' (k = 1..order), 'central_k' (k = 2..order), 'mean', and when
        order allows: 'variance', 'std', 'skewness', 'kurtosis' (m4 / sigma^4,
        not excess).
    """
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    UC, UMAX, HC, HT = 40.0, 700.0, 0.01, 0.1
    nodes, weights = [], []
    for a, b, h in ((-UMAX, -UC, HT), (-UC, UC, HC), (UC, UMAX, HT)):
        n = int(round((b - a) / h))
        u = np.linspace(a, b, n + 1)
        w = np.full(n + 1, 2.0)
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        nodes.append(u)
        weights.append(w * (b - a) / n / 3.0)
    u = np.concatenate(nodes)
    w = np.concatenate(weights)
    p = 0.5 * (1.0 + np.tanh(0.5 * u))
    # dp/du = p (1 - p) = 1 / (4 cosh^2(u/2)), computed without forming 1 - p,
    # which rounds to 0 for u > 37 and would drop the right tail
    wt = w * 0.25 / np.cosh(0.5 * u) ** 2
    with np.errstate(over='ignore', invalid='ignore'):
        if quantile_logodds is not None:
            q = np.asarray(quantile_logodds(u), dtype=float)
        else:
            q = np.asarray(quantile_func(np.clip(p, PROB_EPS, 1 - PROB_EPS)), dtype=float)

    def exists(k):
        return True if moment_exists is None else bool(moment_exists(k))

    missing = [k for k in range(1, order + 1) if not exists(k)]
    if missing:
        warnings.warn(f"moments of order >= {missing[0]} do not exist for this fit "
                      "(the tail is too heavy); they are reported as inf")

    def integral(f):
        with np.errstate(over='ignore', invalid='ignore'):
            val = float(np.sum(wt * f))
        return val if np.isfinite(val) else np.inf

    moments = {}
    for k in range(1, order + 1):
        moments[f'raw_{k}'] = integral(q ** k) if exists(k) else np.inf
    mean = moments['raw_1']
    for k in range(2, order + 1):
        moments[f'central_{k}'] = (integral((q - mean) ** k)
                                   if exists(k) and np.isfinite(mean) else np.inf)

    moments['mean'] = mean
    if order >= 2:
        moments['variance'] = moments['central_2']
        moments['std'] = float(np.sqrt(moments['central_2']))
    finite_std = order >= 2 and np.isfinite(moments['std']) and moments['std'] > 0
    if order >= 3:
        moments['skewness'] = (moments['central_3'] / moments['std'] ** 3
                               if finite_std and np.isfinite(moments['central_3']) else np.nan)
    if order >= 4:
        moments['kurtosis'] = (moments['central_4'] / moments['std'] ** 4
                               if finite_std and np.isfinite(moments['central_4']) else np.nan)
    return moments


def calculate_gamma(x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Estimate the gamma (center) parameter from P10, P50, P90 (paper Eqs. 10-11).

    Gamma controls where the center basis functions are anchored. It lies
    in [0, 1]: below 0.5 for right-skewed and above 0.5 for left-skewed
    quantiles, and 0.45 for symmetric ones (as in the paper).

    The algorithm:
        1. Get P10, P50, P90 from the data by linear interpolation (linear
           extrapolation if the data do not reach 0.1 or 0.9)
        2. Compute the Kelly skewness index
           SI = (P90 - P50 - (P50 - P10)) / (P90 - P10)
        3. Set gamma = 0.5 - 0.5 × SI, clipped to [0, 1]; gamma = 0.45 when
           SI = 0 or P90 = P10
    
    Parameters
    ----------
    x_data : array
        Quantile values.
    y_data : array
        Cumulative probabilities in (0, 1).
        
    Returns
    -------
    gamma : float
        Center parameter in [0, 1].
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    
    sort_idx = np.argsort(y_data)
    y_sorted = y_data[sort_idx]
    x_sorted = x_data[sort_idx]
    
    def interpolate_percentile(target_p):
        """Get the quantile value at a given probability via interpolation."""
        exact_match = np.where(np.isclose(y_sorted, target_p, atol=1e-10))[0]
        if len(exact_match) > 0:
            return float(x_sorted[exact_match[0]])
        
        try:
            interp = interp1d(y_sorted, x_sorted,
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate',
                            assume_sorted=True)
            return float(interp(target_p))
        except Exception:
            if target_p < y_sorted[0]:
                if len(y_sorted) >= 2:
                    slope = (x_sorted[1] - x_sorted[0]) / (y_sorted[1] - y_sorted[0] + 1e-15)
                    return float(x_sorted[0] + slope * (target_p - y_sorted[0]))
                else:
                    return float(x_sorted[0])
            elif target_p > y_sorted[-1]:
                if len(y_sorted) >= 2:
                    slope = (x_sorted[-1] - x_sorted[-2]) / (y_sorted[-1] - y_sorted[-2] + 1e-15)
                    return float(x_sorted[-1] + slope * (target_p - y_sorted[-1]))
                else:
                    return float(x_sorted[-1])
            else:
                idx = np.searchsorted(y_sorted, target_p)
                if idx == 0:
                    return float(x_sorted[0])
                if idx >= len(y_sorted):
                    return float(x_sorted[-1])
                y_lower, y_upper = y_sorted[idx-1], y_sorted[idx]
                x_lower, x_upper = x_sorted[idx-1], x_sorted[idx]
                if np.isclose(y_upper, y_lower):
                    return float((x_lower + x_upper) / 2)
                t = (target_p - y_lower) / (y_upper - y_lower + 1e-15)
                return float(x_lower + t * (x_upper - x_lower))
    
    P_low = interpolate_percentile(0.10)
    P50 = interpolate_percentile(0.50)
    P_high = interpolate_percentile(0.90)
    
    UD = P_high - P50
    LD = P50 - P_low
    span = P_high - P_low
    
    # A symmetric assessment (SI = 0), or one with P90 = P10, gets gamma = 0.45
    # as in the paper (text after Eq. 11). gamma = 0.5 would put the centre
    # terms' anchor exactly at the middle of a symmetric probability grid,
    # which can make the design matrix singular (the paper's Theorem 3 example:
    # K = 7 on 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95 has rank 6 at
    # gamma = 0.5 and rank 7 at 0.45).
    if np.isclose(span, 0.0, atol=1e-10):
        return GAMMA_SYMMETRIC

    SI = (UD - LD) / span
    if abs(SI) <= 1e-12:
        return GAMMA_SYMMETRIC
    gamma = 0.5 - 0.5 * SI
    
    return float(np.clip(gamma, 0.0, 1.0))


def compute_w1(quantile_fitted: Callable,
                x_data: np.ndarray,
                y_data: np.ndarray,
                y_grid: Optional[np.ndarray] = None,
                num_points: int = 1000) -> tuple:
    """
    Wasserstein-1 distance between the fitted and the target quantile functions.

    W1 = integral of |Q_fit(p) - Q_target(p)| dp over the grid's range,
    computed with the trapezoid rule on the actual grid points (so uneven
    grids are handled correctly). Q_target linearly interpolates the
    (y_data, x_data) pairs, held flat outside them. The default grid is
    0.01..0.99 (``num_points`` points); pass ``y_grid`` to change the range.

    The normalized value divides by the interdecile range P90 - P10 of the
    data (interpolated). This is the paper's normalized W1 (Eq. 25), except
    that the integral runs over the grid's range rather than all of (0, 1).

    Parameters
    ----------
    quantile_fitted : callable
        Fitted quantile function Q_fit(p) -> x.
    x_data : array
        Target quantile values from the data.
    y_data : array
        Cumulative probabilities corresponding to x_data.
    y_grid : array, optional
        Probabilities to integrate over (sorted). Default: num_points points
        evenly spaced on [0.01, 0.99].
    num_points : int
        Number of grid points if y_grid is not provided.

    Returns
    -------
    w1 : float
        Absolute W1 distance over the grid range.
    w1_norm : float
        W1 divided by the interdecile range P90 - P10 of the data.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    sort_idx = np.argsort(y_data)
    y_sorted = y_data[sort_idx]
    x_sorted = x_data[sort_idx]

    if y_grid is None:
        y_grid = np.linspace(0.01, 0.99, num_points)
    else:
        y_grid = np.sort(np.asarray(y_grid, dtype=float))

    q_target = np.interp(y_grid, y_sorted, x_sorted)
    q_fitted = np.asarray(quantile_fitted(y_grid), dtype=float)
    abs_diff = np.abs(q_fitted - q_target)
    w1 = float(np.sum(0.5 * (abs_diff[1:] + abs_diff[:-1]) * np.diff(y_grid)))

    p10 = np.interp(0.10, y_sorted, x_sorted)
    p90 = np.interp(0.90, y_sorted, x_sorted)
    denom = p90 - p10
    w1_norm = w1 / denom if denom > 0 else np.nan
    return float(w1), float(w1_norm)
