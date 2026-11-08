"""
Transform-based QFlex Distributions

Provides semibounded and bounded variants via log and logit transforms:
    - LogQFlex: For data with a lower bound (e.g., income, time-to-event)
    - LogitQFlex: For data bounded on both sides (e.g., proportions)
"""

import numpy as np
import warnings
from scipy.special import expit, log_expit

from .core import QFlex, QFlexBase, _validate_terms, _validate_xy
from .constraints import ConstraintType, QFlexError
from .basis import leading_tail_terms


def _inverse_scaled(dq, log_scale):
    """1 / (exp(log_scale) * dq), signed, computed in logs; NaN where dq == 0."""
    dq = np.asarray(dq, dtype=float)
    log_scale = np.asarray(log_scale, dtype=float)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        mag = np.exp(-(log_scale + np.log(np.abs(dq))))
    return np.where(dq > 0, mag, np.where(dq < 0, -mag, np.nan))


def _fit_inner(z_data, y_data, terms, constraint_type):
    """Fit QFlex on the transformed data without its own feasibility warning
    (the outer class issues one warning for the whole fit)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="QFlex distribution may not be feasible")
        return QFlex(z_data, y_data, terms, constraint_type)

PROB_EPS = 1e-12


class LogQFlex(QFlexBase):
    """
    Semibounded QFlex distribution using a log transform.
    
    Transforms the data as z = ln(x - L), fits an unbounded QFlex to z,
    then maps back: Q_X(p) = L + exp(Q_Z(p)).
    
    Parameters
    ----------
    x_data : array-like
        Observed quantile values (must be > lower_bound).
    y_data : array-like
        Corresponding cumulative probabilities in (0, 1).
    lower_bound : float, optional
        Lower bound of the distribution (default 0).
    terms : int, optional
        Number of terms in the expansion (default 3).
    constraint_type : ConstraintType or str, optional
        Constraint to apply during fitting (default: unconstrained). It acts
        on the coefficients of the QFlex fitted to z.
    """

    def __init__(self,
                 x_data,
                 y_data,
                 lower_bound: float = 0,
                 terms: int = 3,
                 constraint_type: ConstraintType = ConstraintType.NONE):
        self.lower_bound = float(lower_bound)
        if not np.isfinite(self.lower_bound):
            raise QFlexError("lower_bound must be finite")
        self.terms = _validate_terms(terms)
        self.constraint_type = ConstraintType.resolve(constraint_type)
        self.x_data, self.y_data = _validate_xy(x_data, y_data, self.terms)

        if np.any(self.x_data <= self.lower_bound):
            raise QFlexError(f"All x_data values must be > lower_bound ({self.lower_bound})")

        # Transform to unbounded scale
        self.z_data = np.log(self.x_data - self.lower_bound)

        # Fit QFlex on transformed data
        self.qflex = _fit_inner(self.z_data, self.y_data, self.terms, self.constraint_type)
        
        self.is_feasible = self.qflex.is_feasible and self._check_monotonicity()
        self.coefficients = self.qflex.coefficients
        self.gamma = self.qflex.gamma
        
        if not self.is_feasible:
            warnings.warn("LogQFlex distribution may not be feasible")
    
    def quantile(self, y):
        """
        Evaluate the quantile function on the original scale.
        
        Q_X(p) = L + exp(Q_Z(p)). At exactly p = 0 and p = 1 the limits are
        returned (L and +inf for a valid fit).
        """
        z = self.qflex.quantile(y)
        with np.errstate(over='ignore'):
            return self.lower_bound + np.exp(z)
    
    def pdf(self, y, step_size=0.001, method='analytical'):
        """
        Density on the original scale at Q(y); see QFlexBase.pdf.
        """
        return super().pdf(y, step_size=step_size, method=method)

    def _quantile_logodds(self, u):
        with np.errstate(over='ignore'):
            return self.lower_bound + np.exp(self.qflex._quantile_logodds(u))

    def _moment_exists(self, k: int) -> bool:
        """
        E[X^k] for Q = L + exp(z): as p -> 1, z ~ a t^j with t = -ln(1 - p)
        and a t^j the leading right-tail term, so X^k ~ (1 - p)^(-k a) when
        j = 1. With a positive leading term the moment is finite only if
        j = 1 (the term is R_1, with coefficient a = a_2) and k a < 1.
        """
        lead = leading_tail_terms(self.coefficients, self.terms)['right']
        if lead is None or lead[0] <= 0:
            return True
        coef, order = lead
        return order == 1 and k * coef < 1

    def pdf_analytical(self, y):
        """
        Compute PDF using analytical derivatives with exponential transform.

        Uses the chain rule:
        Q_X(p) = L + exp(Q_Z(p))
        dQ_X/dp = exp(Q_Z(p)) × dQ_Z/dp
        PDF_X(p) = 1 / dQ_X/dp

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).

        Returns
        -------
        pdf : array-like
            Probability density values; negative where the fit decreases.
        """
        from .basis import evaluate_quantile_derivative

        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)

        # Get Q_Z(p) and dQ_Z/dp analytically
        Q_Z = self.qflex.quantile(y)
        dQ_Z = evaluate_quantile_derivative(y, self.qflex.coefficients,
                                          self.qflex.terms, self.qflex.gamma)

        # Chain rule: dQ_X/dp = exp(Q_Z) × dQ_Z/dp, inverted in logs so that
        # exp(Q_Z) underflowing to 0 gives a (huge) density, not 0 * x = NaN.
        # The sign of dQ_Z is kept: a negative density marks where the fit
        # decreases (see QFlex.pdf_analytical).
        return _inverse_scaled(dQ_Z, Q_Z)
    
    def cdf(self, x):
        """CDF with lower bound handling (values <= L have CDF = 0)."""
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x, dtype=float)
        valid_mask = x > self.lower_bound
        if np.any(valid_mask):
            z = np.log(x[valid_mask] - self.lower_bound)
            cdf_vals[valid_mask] = self.qflex.cdf(z)
        cdf_vals[np.isnan(x)] = np.nan
        return float(cdf_vals[0]) if scalar else cdf_vals


class LogitQFlex(QFlexBase):
    """
    Bounded QFlex distribution using a logit transform.
    
    Transforms the data as z = ln((x - L)/(U - x)), fits an unbounded QFlex to z,
    then maps back: Q_X(p) = L + (U - L) × sigmoid(Q_Z(p)).
    
    Parameters
    ----------
    x_data : array-like
        Observed quantile values (must be in (lower_bound, upper_bound)).
    y_data : array-like
        Corresponding cumulative probabilities in (0, 1).
    lower_bound : float, optional
        Lower bound of the distribution (default 0).
    upper_bound : float, optional
        Upper bound of the distribution (default 1).
    terms : int, optional
        Number of terms in the expansion (default 3).
    constraint_type : ConstraintType or str, optional
        Constraint to apply during fitting (default: unconstrained). It acts
        on the coefficients of the QFlex fitted to z.
    """
    
    def __init__(self,
                 x_data,
                 y_data,
                 lower_bound: float = 0,
                 upper_bound: float = 1,
                 terms: int = 3,
                 constraint_type: ConstraintType = ConstraintType.NONE):
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        if not (np.isfinite(self.lower_bound) and np.isfinite(self.upper_bound)):
            raise QFlexError("lower_bound and upper_bound must be finite")
        self.terms = _validate_terms(terms)
        self.constraint_type = ConstraintType.resolve(constraint_type)
        self.x_data, self.y_data = _validate_xy(x_data, y_data, self.terms)

        if self.upper_bound <= self.lower_bound:
            raise QFlexError("upper_bound must be > lower_bound")
        
        if np.any(self.x_data <= self.lower_bound) or np.any(self.x_data >= self.upper_bound):
            raise QFlexError(f"All x_data values must be in ({self.lower_bound}, {self.upper_bound})")
        
        # Transform to unbounded scale via logit
        self.z_data = np.log((self.x_data - self.lower_bound) / (self.upper_bound - self.x_data))
        
        # Fit QFlex on transformed data
        self.qflex = _fit_inner(self.z_data, self.y_data, self.terms, self.constraint_type)

        self.is_feasible = self.qflex.is_feasible
        self.coefficients = self.qflex.coefficients
        self.gamma = self.qflex.gamma
        
        if not self.is_feasible:
            warnings.warn("LogitQFlex distribution may not be feasible")
    
    def quantile(self, y):
        """
        Evaluate the quantile function on the original scale.
        
        Q_X(p) = L + (U - L) × expit(z), z = Q_Z(p). ``expit`` stays finite for
        any z (exp(z)/(1+exp(z)) overflowed to inf/inf = NaN for z > 709).
        At exactly p = 0 and p = 1 the limits are returned (L and U for a
        valid fit).
        """
        z = self.qflex.quantile(y)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * expit(z)

    def pdf(self, y, step_size=0.001, method='analytical'):
        """
        Density on the original scale at Q(y); see QFlexBase.pdf.
        """
        return super().pdf(y, step_size=step_size, method=method)

    def _quantile_logodds(self, u):
        z = self.qflex._quantile_logodds(u)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * expit(z)

    def pdf_analytical(self, y):
        """
        Compute PDF using analytical derivatives with logit transform.

        Uses the chain rule:
        Q_X(p) = L + (U-L) × σ(Q_Z(p))  where σ(z) = exp(z)/(1+exp(z))
        dQ_X/dp = (U-L) × σ(Q_Z) × (1-σ(Q_Z)) × dQ_Z/dp
        PDF_X(p) = 1 / dQ_X/dp

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).

        Returns
        -------
        pdf : array-like
            Probability density values; negative where the fit decreases.
        """
        from .basis import evaluate_quantile_derivative

        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)

        # Get Q_Z(p) and dQ_Z/dp analytically
        Q_Z = self.qflex.quantile(y)
        dQ_Z = evaluate_quantile_derivative(y, self.qflex.coefficients,
                                          self.qflex.terms, self.qflex.gamma)

        # dQ_X/dp = (U - L) sigma(z) (1 - sigma(z)) dQ_Z/dp, inverted in logs:
        # log sigma(z) + log sigma(-z) never overflows or cancels, so a z deep
        # in the tail gives a (huge) density instead of 0 * x = NaN.
        range_scale = self.upper_bound - self.lower_bound
        log_scale = np.log(range_scale) + log_expit(Q_Z) + log_expit(-Q_Z)
        return _inverse_scaled(dQ_Z, log_scale)
    
    def cdf(self, x):
        """CDF with bounds handling (values <= L have CDF = 0, values >= U have CDF = 1)."""
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x, dtype=float)

        below_lower = x <= self.lower_bound
        above_upper = x >= self.upper_bound
        valid_mask = (x > self.lower_bound) & (x < self.upper_bound)
        
        cdf_vals[below_lower] = 0.0
        cdf_vals[above_upper] = 1.0
        
        if np.any(valid_mask):
            z = np.log((x[valid_mask] - self.lower_bound) / (self.upper_bound - x[valid_mask]))
            cdf_vals[valid_mask] = self.qflex.cdf(z)
        cdf_vals[np.isnan(x)] = np.nan
        return float(cdf_vals[0]) if scalar else cdf_vals
