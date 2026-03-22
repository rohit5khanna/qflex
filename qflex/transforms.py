"""
Transform-based QFlex Distributions

Provides semibounded and bounded variants via log and logit transforms:
    - LogQFlex: For data with a lower bound (e.g., income, time-to-event)
    - LogitQFlex: For data bounded on both sides (e.g., proportions)
"""

import numpy as np
import warnings

from .core import QFlex, QFlexBase
from .constraints import ConstraintType, QFlexError
from .utils import compute_cdf_inverse

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
        Number of terms in the expansion (default 5).
    constraint_type : ConstraintType, optional
        Constraint to apply during fitting (default: unconstrained).
    tc_method : str, optional
        For TC constraint: 'nonlinear' or 'linear'.
    """
    
    def __init__(self,
                 x_data,
                 y_data,
                 lower_bound: float = 0,
                 terms: int = 5,
                 constraint_type: ConstraintType = ConstraintType.NONE,
                 tc_method: str = 'nonlinear'):
        self.lower_bound = float(lower_bound)
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.terms = terms
        
        if np.any(self.x_data <= self.lower_bound):
            raise QFlexError(f"All x_data values must be > lower_bound ({self.lower_bound})")
        
        # Transform to unbounded scale
        self.z_data = np.log(self.x_data - self.lower_bound)
        
        # Fit QFlex on transformed data
        self.qflex = QFlex(self.z_data, self.y_data, terms, constraint_type, tc_method=tc_method)
        
        self.is_feasible = self.qflex.is_feasible and self._check_monotonicity()
        self.coefficients = self.qflex.coefficients
        self.gamma = self.qflex.gamma
        
        if not self.is_feasible:
            warnings.warn("LogQFlex distribution may not be feasible")
    
    def quantile(self, y):
        """
        Evaluate the quantile function on the original scale.
        
        Q_X(p) = L + exp(Q_Z(p))
        """
        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
        z = self.qflex.quantile(y)
        return self.lower_bound + np.exp(z)
    
    def pdf(self, y, step_size=0.001, method='numerical'):
        """
        Compute PDF on the original scale.

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).
        step_size : float, optional
            Step size for finite differences (default 0.001).
            Only used when method='numerical'.
        method : str, optional
            Computation method: 'numerical' or 'analytical' (default 'numerical').

        Returns
        -------
        pdf : array-like
            Probability density values.
        """
        return super().pdf(y, step_size=step_size, method=method)

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
            Probability density values.
        """
        from .basis import evaluate_quantile_derivative

        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)

        # Get Q_Z(p) and dQ_Z/dp analytically
        Q_Z = self.qflex.quantile(y)
        dQ_Z = evaluate_quantile_derivative(y, self.qflex.coefficients,
                                          self.qflex.terms, self.qflex.gamma)

        # Chain rule: dQ_X/dp = exp(Q_Z) × dQ_Z/dp
        exp_Q_Z = np.exp(Q_Z)
        dQ_X = exp_Q_Z * dQ_Z

        # PDF = 1 / dQ_X, ensuring dQ_X > 0
        dQ_X = np.clip(dQ_X, 1e-12, None)

        return 1.0 / dQ_X
    
    def cdf(self, x):
        """CDF with lower bound handling (values <= L have CDF = 0)."""
        x = np.asarray(x, dtype=float)
        valid_mask = x > self.lower_bound
        cdf_vals = np.zeros_like(x, dtype=float)
        
        if np.any(valid_mask):
            z = np.log(x[valid_mask] - self.lower_bound)
            cdf_vals[valid_mask] = self.qflex.cdf(z)
        
        return cdf_vals


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
        Number of terms in the expansion (default 5).
    constraint_type : ConstraintType, optional
        Constraint to apply during fitting (default: unconstrained).
    tc_method : str, optional
        For TC constraint: 'nonlinear' or 'linear'.
    """
    
    def __init__(self,
                 x_data,
                 y_data,
                 lower_bound: float = 0,
                 upper_bound: float = 1,
                 terms: int = 5,
                 constraint_type: ConstraintType = ConstraintType.NONE,
                 tc_method: str = 'nonlinear'):
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.terms = terms
        
        if self.upper_bound <= self.lower_bound:
            raise QFlexError("upper_bound must be > lower_bound")
        
        if np.any(self.x_data <= self.lower_bound) or np.any(self.x_data >= self.upper_bound):
            raise QFlexError(f"All x_data values must be in ({self.lower_bound}, {self.upper_bound})")
        
        # Transform to unbounded scale via logit
        self.z_data = np.log((self.x_data - self.lower_bound) / (self.upper_bound - self.x_data))
        
        # Fit QFlex on transformed data
        self.qflex = QFlex(self.z_data, self.y_data, terms, constraint_type, tc_method=tc_method)
        
        self.is_feasible = self.qflex.is_feasible
        self.coefficients = self.qflex.coefficients
        self.gamma = self.qflex.gamma
        
        if not self.is_feasible:
            warnings.warn("LogitQFlex distribution may not be feasible")
    
    def quantile(self, y):
        """
        Evaluate the quantile function on the original scale.
        
        Q_X(p) = L + (U - L) × exp(z) / (1 + exp(z))  where z = Q_Z(p)
        """
        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
        z = self.qflex.quantile(y)
        exp_z = np.exp(z)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * exp_z / (1 + exp_z)
    
    def pdf(self, y, step_size=0.001, method='numerical'):
        """
        Compute PDF on the original scale.

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).
        step_size : float, optional
            Step size for finite differences (default 0.001).
            Only used when method='numerical'.
        method : str, optional
            Computation method: 'numerical' or 'analytical' (default 'numerical').

        Returns
        -------
        pdf : array-like
            Probability density values.
        """
        return super().pdf(y, step_size=step_size, method=method)

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
            Probability density values.
        """
        from .basis import evaluate_quantile_derivative

        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)

        # Get Q_Z(p) and dQ_Z/dp analytically
        Q_Z = self.qflex.quantile(y)
        dQ_Z = evaluate_quantile_derivative(y, self.qflex.coefficients,
                                          self.qflex.terms, self.qflex.gamma)

        # Compute sigmoid(Q_Z)
        exp_Q_Z = np.exp(Q_Z)
        sigma = exp_Q_Z / (1 + exp_Q_Z)

        # Chain rule: dQ_X/dp = (U-L) × σ(Q_Z) × (1-σ(Q_Z)) × dQ_Z/dp
        range_scale = self.upper_bound - self.lower_bound
        dQ_X = range_scale * sigma * (1 - sigma) * dQ_Z

        # PDF = 1 / dQ_X, ensuring dQ_X > 0
        dQ_X = np.clip(dQ_X, 1e-12, None)

        return 1.0 / dQ_X
    
    def cdf(self, x):
        """CDF with bounds handling (values <= L have CDF = 0, values >= U have CDF = 1)."""
        x = np.asarray(x, dtype=float)
        cdf_vals = np.zeros_like(x, dtype=float)
        
        below_lower = x <= self.lower_bound
        above_upper = x >= self.upper_bound
        valid_mask = (x > self.lower_bound) & (x < self.upper_bound)
        
        cdf_vals[below_lower] = 0.0
        cdf_vals[above_upper] = 1.0
        
        if np.any(valid_mask):
            z = np.log((x[valid_mask] - self.lower_bound) / (self.upper_bound - x[valid_mask]))
            cdf_vals[valid_mask] = self.qflex.cdf(z)
        
        return cdf_vals
