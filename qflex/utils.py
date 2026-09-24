"""
Utility Functions for QFlex

Standalone numerical methods for PDF/CDF computation, moment calculation,
gamma estimation, and Wasserstein distance. These work with any object
that provides a quantile() method.
"""

import warnings

import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import interp1d
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .core import QFlexBase

PROB_EPS = 1e-12

#: Points used to locate the increasing region of Q before inverting it.
CDF_SCAN_POINTS = 4001


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
        Step size for finite differences.
        
    Returns
    -------
    pdf : array
        Probability density values.
    """
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    
    def derivative(y_vals, delta):
        y_plus = np.clip(y_vals + delta, PROB_EPS, 1 - PROB_EPS)
        y_minus = np.clip(y_vals - delta, PROB_EPS, 1 - PROB_EPS)
        
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

    return 1.0 / grad


def compute_cdf_inverse(quantile_func: Callable, x: np.ndarray, 
                        y_data: np.ndarray = None) -> np.ndarray:
    """
    Compute CDF by inverting the quantile function using Brent's method.
    
    Finds y such that Q(y) = x for each input value.
    
    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(y) -> x.
    x : array
        Values at which to evaluate the CDF.
    y_data : array, optional
        Original probability data (helps set better search bounds).
        
    Returns
    -------
    cdf : array
        Cumulative probabilities.
    """
    x = np.asarray(x, dtype=float)
    x_scalar = x.ndim == 0
    x_flat = x.flatten()

    # INVERT ONLY WHERE Q IS ACTUALLY INCREASING.
    #
    # A fit can be monotone across the data and inverted beyond it -- on one
    # real example Q(1e-12) = +24.2 while Q(1-1e-12) = -26.0, with Q properly
    # increasing in between. The previous implementation handed brentq the
    # whole [PROB_EPS, 1-PROB_EPS] bracket. Over a bracket containing several
    # sign changes brentq still converges, silently, to whichever root it
    # happens to find: cdf(Q(0.5)) came back as 1.0 instead of 0.5. It raised
    # nothing, so the caller had no way to know.
    #
    # So: scan once, take the longest strictly-increasing run, and invert
    # inside that. For a well-behaved fit this is the whole interval and the
    # behaviour is unchanged.
    grid = np.linspace(PROB_EPS, 1 - PROB_EPS, CDF_SCAN_POINTS)
    q_grid = np.asarray(quantile_func(grid), dtype=float)

    finite = np.isfinite(q_grid)
    increasing = np.zeros(len(grid) - 1, dtype=bool)
    increasing[:] = finite[:-1] & finite[1:] & (np.diff(q_grid) > 0)

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
        # Nowhere increasing: there is no CDF to report.
        warnings.warn("quantile function is nowhere increasing; cdf is undefined")
        out = np.full_like(x_flat, np.nan, dtype=float)
        return out[0] if x_scalar else out.reshape(x.shape)

    lo_i, hi_i = best_start, best_start + best_len
    y_lo, y_hi = float(grid[lo_i]), float(grid[hi_i])
    q_lo, q_hi = float(q_grid[lo_i]), float(q_grid[hi_i])

    if best_len < len(increasing):
        warnings.warn(
            "quantile function is not monotone over the full probability range; "
            f"cdf inverted on [{y_lo:.4g}, {y_hi:.4g}] only")

    def objective(y, x_target):
        return float(quantile_func(np.array([y]))[0]) - x_target

    cdf_vals = np.zeros_like(x_flat, dtype=float)
    for i, x_val in enumerate(x_flat):
        if x_val <= q_lo:
            cdf_vals[i] = 0.0
        elif x_val >= q_hi:
            cdf_vals[i] = 1.0
        else:
            try:
                cdf_vals[i] = optimize.brentq(objective, y_lo, y_hi, args=(x_val,),
                                              xtol=1e-14, rtol=8.9e-16, maxiter=200)
            except (ValueError, RuntimeError):
                cdf_vals[i] = np.nan

    if x_scalar:
        return cdf_vals[0]
    return cdf_vals.reshape(x.shape)


def compute_moments(quantile_func: Callable, order: int = 4) -> dict:
    """
    Compute moments via numerical integration over the quantile function.
    
    Parameters
    ----------
    quantile_func : callable
        Quantile function Q(y) -> x.
    order : int
        Maximum moment order to compute (default 4).
        
    Returns
    -------
    moments : dict
        Contains raw moments, central moments, mean, variance, std, skewness, kurtosis.
    """
    moments = {}
    
    for k in range(1, order + 1):
        integrand = lambda y: quantile_func(y) ** k
        moment, _ = integrate.quad(integrand, 0, 1)
        moments[f'raw_{k}'] = moment
    
    mean = moments['raw_1']
    for k in range(2, order + 1):
        integrand = lambda y: (quantile_func(y) - mean) ** k
        moment, _ = integrate.quad(integrand, 0, 1)
        moments[f'central_{k}'] = moment
    
    variance = moments['central_2']
    std = np.sqrt(variance)
    moments['mean'] = mean
    moments['variance'] = variance
    moments['std'] = std
    
    if order >= 3:
        moments['skewness'] = moments['central_3'] / (std ** 3)
    if order >= 4:
        moments['kurtosis'] = moments['central_4'] / (std ** 4)
    
    return moments


def calculate_gamma(x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Estimate the gamma (center) parameter from P10, P50, P90.
    
    Gamma controls where the center basis functions are anchored. It ranges
    from 0 to 1, with 0.5 corresponding to a symmetric distribution.
    
    The algorithm:
        1. Interpolate to get P10, P50, P90 from the data
        2. Compute skewness index SI = (P90 - P50 - (P50 - P10)) / (P90 - P10)
        3. Set gamma = 0.5 - 0.5 × SI
    
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
    
    if np.isclose(span, 0.0, atol=1e-10):
        return 0.5
    
    SI = (UD - LD) / span
    gamma = 0.5 - 0.5 * SI
    
    return float(np.clip(gamma, 0.0, 1.0))


def compute_w1(quantile_fitted: Callable,
                x_data: np.ndarray,
                y_data: np.ndarray,
                y_grid: Optional[np.ndarray] = None,
                num_points: int = 1000) -> tuple[float, float]:
    """
    Compute the Wasserstein-1 distance between fitted and target quantile functions.
    
    Uses a Riemann sum (rectangle rule) to integrate |Q_fit(p) - Q_target(p)| over p.
    The normalized version divides by the interquartile range (P90 - P10).
    
    Parameters
    ----------
    quantile_fitted : callable
        Fitted quantile function Q_fit(p) -> x.
    x_data : array
        Target quantile values from the data.
    y_data : array
        Cumulative probabilities corresponding to x_data.
    y_grid : array, optional
        Grid of probabilities for integration. If None, uses uniform spacing.
    num_points : int
        Number of grid points if y_grid is not provided.
        
    Returns
    -------
    w1 : float
        Absolute W1 distance.
    w1_norm : float
        Normalized W1 distance (relative to P90 - P10).
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    
    sort_idx = np.argsort(y_data)
    y_sorted = y_data[sort_idx]
    x_sorted = x_data[sort_idx]
    
    if y_grid is None:
        y_grid = np.linspace(0.01, 0.99, num_points)
    else:
        y_grid = np.asarray(y_grid)
    
    # Target quantile via linear interpolation
    q_target = np.interp(y_grid, y_sorted, x_sorted)
    q_fitted = quantile_fitted(y_grid)
    
    abs_diff = np.abs(q_fitted - q_target)
    
    dp = y_grid[1] - y_grid[0]
    w1 = np.sum(abs_diff) * dp
    
    p10 = np.interp(0.10, y_sorted, x_sorted)
    p90 = np.interp(0.90, y_sorted, x_sorted)
    denom = p90 - p10
    
    w1_norm = w1 / denom if denom > 0 else np.nan
    
    return float(w1), float(w1_norm)
