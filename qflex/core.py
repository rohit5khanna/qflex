"""
QFlex Core Implementation

Provides the base class and unbounded QFlex distribution.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings

from .basis import (build_design_matrix, evaluate_quantile, evaluate_quantile_logodds,
                    quantile_limits)
from .utils import calculate_gamma
from .constraints import ConstraintType, solve_with_constraints, QFlexError
from .mono_verification import check_proposition3, check_delta_p_monotonicity
from .utils import compute_pdf_numerical, compute_cdf_inverse, compute_moments

PROB_EPS = 1e-12


def _validate_terms(terms) -> int:
    """terms must be a positive integer (an int, a NumPy integer, or e.g. 5.0)."""
    if isinstance(terms, bool):
        raise QFlexError("terms must be a positive integer")
    try:
        t = int(terms)
    except (TypeError, ValueError):
        raise QFlexError(f"terms must be a positive integer, got {terms!r}")
    if t != terms or t < 1:
        raise QFlexError(f"terms must be a positive integer, got {terms!r}")
    return t


def _validate_xy(x_data, y_data, terms: int):
    """Shared input checks for all QFlex classes; returns float arrays."""
    try:
        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)
    except (TypeError, ValueError):
        raise QFlexError("x_data and y_data must be numeric")
    if x.ndim != 1 or y.ndim != 1:
        raise QFlexError(f"x_data and y_data must be one-dimensional "
                         f"(got shapes {x.shape} and {y.shape})")
    if len(x) != len(y):
        raise QFlexError("x_data and y_data must have the same length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise QFlexError("x_data and y_data must be finite (no NaN or inf)")
    if len(x) < terms:
        raise QFlexError(f"Need at least {terms} data points for a {terms}-term QFlex, got {len(x)}")
    if np.any(y <= 0) or np.any(y >= 1):
        raise QFlexError("All y_data values must be in (0, 1)")
    if len(np.unique(y)) < terms:
        raise QFlexError(f"Need at least {terms} distinct y values")
    return x, y


class QFlexBase(ABC):
    """
    Abstract base class for all QFlex distribution variants.
    
    Subclasses implement quantile() and pdf_analytical() (the default PDF
    method). CDF, sampling, moments and the numerical PDF come from shared
    utility functions.
    """
    
    @abstractmethod
    def quantile(self, y):
        """
        Evaluate the quantile function Q(y).
        
        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).
            
        Returns
        -------
        x : array-like
            Corresponding quantile values.
        """
        pass
    
    def pdf(self, y, step_size=0.001, method='analytical'):
        """
        Probability density at the quantile Q(y), i.e. f(Q(y)) = 1 / Q'(y).

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).
        step_size : float, optional
            Largest finite-difference step (default 0.001); it shrinks near
            0 and 1. Only used when method='numerical'.
        method : str, optional
            'analytical' (default; exact derivative of the basis) or
            'numerical' (central differences).

        Returns
        -------
        pdf : array-like
            Probability density values.
        """
        if method == 'numerical':
            return compute_pdf_numerical(self.quantile, y, step_size)
        elif method == 'analytical':
            return self.pdf_analytical(y)
        else:
            raise ValueError(f"method must be 'numerical' or 'analytical', got '{method}'")
    
    def cdf(self, x):
        """
        Compute CDF by inverting the quantile function.
        
        Parameters
        ----------
        x : array-like
            Values at which to evaluate the CDF.
            
        Returns
        -------
        cdf : array-like
            Cumulative probabilities.
        """
        y_data = getattr(self, 'y_data', None)
        return compute_cdf_inverse(self.quantile, x, y_data)
    
    def sample(self, size=1, random_state=None):
        """
        Draw random samples by inverse transform sampling.

        Parameters
        ----------
        size : int or tuple
            Number (or shape) of samples.
        random_state : None, int or numpy.random.Generator, optional
            Seed or generator for reproducible draws (default: fresh entropy).
        """
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0.0, 1.0, size)
        return self.quantile(u)

    def moments(self, order=4, clipped=False):
        """
        Moments of the fitted distribution, by integrating Q(p) over p.

        Returns a dict with 'mean' and, as ``order`` allows, 'variance', 'std',
        'skewness', 'kurtosis' (not excess), plus 'raw_k' and 'central_k'.

        By default these are the moments of the fitted QPD itself. Moments that
        do not exist (a LogQFlex fit whose right tail is too heavy: a positive
        leading right-tail term of order >= 2, or leading term R_1 with
        coefficient a_2 and k * a_2 >= 1) are reported as inf with a warning. With ``clipped=True`` the moments are those of the
        distribution as quantile() and sample() evaluate it, with p held
        within [1e-12, 1 - 1e-12]; these are always finite.
        """
        if clipped:
            return compute_moments(self.quantile, order)
        return compute_moments(self.quantile, order,
                               quantile_logodds=self._quantile_logodds,
                               moment_exists=self._moment_exists)

    def _quantile_logodds(self, u):
        """Q as a function of the log-odds of p, without the 1e-12 clip."""
        return evaluate_quantile_logodds(u, self.coefficients, self.terms, self.gamma)

    def _moment_exists(self, k: int) -> bool:
        """Whether E[X^k] is finite (always, for unbounded and logit fits)."""
        return True

    def pdf_analytical(self, y):
        """
        Compute the PDF from the exact derivative of Q. pdf() uses this by
        default; every QFlex class overrides it, and a subclass that does
        not must be called with method='numerical'.

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).

        Returns
        -------
        pdf : array-like
            Probability density values.

        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement analytical derivatives.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement analytical PDF. "
            "Use method='numerical' instead."
        )

    def check_proposition3(self, p_grid=None, num_points=None):
        """
        Verify Proposition 3 (grid tail-centre condition m_tail > M_center) and
        Theorem 5 (leading tail coefficients > 0) for this fit.

        For the log and logit classes the check runs on the fitting scale z,
        where the coefficients live; the back-transform is increasing, so the
        result carries over.

        Returns a dict with 'satisfied', 'grid_condition',
        'theorem5_satisfied', 'm_tail', 'M_center', 'margin', 'q_flex_min',
        'q_flex_positive', 'leading_right', 'leading_left'.
        """
        return check_proposition3(self.coefficients, self.terms, self.gamma,
                                  p_grid=p_grid, num_points=num_points)

    def check_proposition4(self, p_grid=None, num_points=None):
        """Backward-compatible name of :meth:`check_proposition3` (the paper's Proposition 3)."""
        return self.check_proposition3(p_grid=p_grid, num_points=num_points)

    def _check_monotonicity(self) -> bool:
        """δ-p monotonicity (paper's Definition 4, δ = 0.001): Q strictly
        increasing on the grid 0.001, ..., 0.999."""
        result = check_delta_p_monotonicity(self.quantile, delta_p=0.001)
        return result['satisfied']

    @classmethod
    def fit_from_data(cls, data, terms=3,
                      constraint_type=ConstraintType.NONE, **kwargs):
        """
        Fit a QFlex distribution directly from raw data using Weibull plotting positions.

        Sorts the data and assigns cumulative probabilities using the Weibull formula:
            y_i = i / (n + 1)  for i = 1, 2, ..., n

        Parameters
        ----------
        data : array-like
            Raw observations. Will be sorted internally.
        terms : int, optional
            Number of basis terms (default 3).
        constraint_type : ConstraintType or str, optional
            Constraint to apply during fitting (default: unconstrained);
            strings such as "A+", "TL+", "TA+", "TC", "TC_mag" work too.
        **kwargs
            Additional arguments passed to the class constructor.
            Use lower_bound for LogQFlex, lower_bound and upper_bound for LogitQFlex.

        Returns
        -------
        fitted : instance of the calling class
            Fitted distribution object.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.lognormal(mean=3, sigma=0.5, size=100)
        >>> qf = QFlex.fit_from_data(data, terms=3)
        >>> log_qf = LogQFlex.fit_from_data(data, lower_bound=0, terms=3)
        """
        terms = _validate_terms(terms)
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise QFlexError(f"data must be one-dimensional (got shape {data.shape})")
        if not np.all(np.isfinite(data)):
            raise QFlexError("data must be finite (no NaN or inf)")
        data = np.sort(data)
        n = len(data)
        if n < terms:
            raise QFlexError(
                f"Need at least {terms} data points for a {terms}-term fit, got {n}"
            )
        y = np.arange(1, n + 1) / (n + 1)
        return cls(data, y, terms=terms, constraint_type=constraint_type, **kwargs)

    def summary(self):
        """
        Print a formatted summary of the fitted distribution.

        Includes class info, fit parameters, moments, and feasibility.
        """
        m = self.moments(order=4)
        p10 = self.quantile(0.10)
        p50 = self.quantile(0.50)
        p90 = self.quantile(0.90)
        if hasattr(p10, 'item'):
            p10, p50, p90 = p10.item(), p50.item(), p90.item()
        else:
            p10, p50, p90 = float(p10), float(p50), float(p90)

        w = 40
        sep = '-' * w
        print(sep)
        print(f"  {self.__class__.__name__} Summary")
        print(sep)
        print(f"  Terms            : {self.terms}")
        print(f"  Gamma (γ)        : {self.gamma:.4f}")
        print(f"  Feasible         : {self.is_feasible}")
        print(f"  Constraint       : {self.constraint_type.value}")
        print(sep)
        print(f"  Mean             : {m['mean']:.4f}")
        print(f"  Std Dev          : {m['std']:.4f}")
        print(f"  Variance         : {m['variance']:.4f}")
        print(f"  Skewness         : {m['skewness']:.4f}")
        print(f"  Kurtosis         : {m['kurtosis']:.4f}")
        print(sep)
        print(f"  P10              : {p10:.4f}")
        print(f"  P50 (median)     : {p50:.4f}")
        print(f"  P90              : {p90:.4f}")
        print(sep)
        print(f"  Coefficients     : {np.array2string(self.coefficients, precision=4, suppress_small=True)}")
        print(sep)

    def plot(self, p_grid=None, show_data=True, ax=None):
        """
        Plot the fitted PDF and quantile function.

        Parameters
        ----------
        p_grid : array-like, optional
            Probability grid for evaluation. Defaults to [0.01, ..., 0.99].
        show_data : bool, optional
            If True, overlay the input quantile points on the quantile plot (default True).
        ax : array of two matplotlib Axes, optional
            If provided, plots into these axes. Otherwise creates a new figure.

        Returns
        -------
        fig, axes : matplotlib Figure and array of two Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plot(). Install it with: pip install matplotlib")

        if p_grid is None:
            p_grid = np.linspace(0.01, 0.99, 500)
        p_grid = np.asarray(p_grid)

        x_vals  = self.quantile(p_grid)
        pdf_vals = self.pdf(p_grid)

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        else:
            fig = ax[0].figure
            axes = ax

        # Left: PDF
        axes[0].plot(x_vals, pdf_vals, color='steelblue', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{self.__class__.__name__} — PDF')
        axes[0].set_ylim(bottom=0)

        # Right: Quantile function
        axes[1].plot(p_grid, x_vals, color='steelblue', linewidth=2, label='Fitted Q(p)')
        if show_data and hasattr(self, 'x_data') and hasattr(self, 'y_data'):
            axes[1].scatter(self.y_data, self.x_data, color='firebrick', zorder=5,
                           s=40, label='Input data')
            axes[1].legend()
        axes[1].set_xlabel('Cumulative probability')
        axes[1].set_ylabel('x')
        axes[1].set_title(f'{self.__class__.__name__} — Quantile function')

        plt.tight_layout()
        return fig, axes


class QFlex(QFlexBase):
    """
    Unbounded QFlex distribution.
    
    The quantile function is a linear combination of basis functions:
        Q(y) = Σ a_k × basis_k(y),  k = 1..terms (a_1 = constant)

    Three basis families are used:
        - Right tail R_j: -ln(1-p) raised to increasing powers
        - Left tail L_j: -(-ln(p)) raised to increasing powers
        - Center C_j: odd powers of (p - γ)

    Parameters
    ----------
    x_data : array-like
        Observed quantile values.
    y_data : array-like
        Corresponding cumulative probabilities in (0, 1).
    terms : int, optional
        Number of terms in the expansion (default 3).
    constraint_type : ConstraintType or str, optional
        Constraint to apply during fitting (default: unconstrained); strings
        such as "A+", "TL+", "TA+", "TC", "TC_mag" work too.
    """
    
    def __init__(self,
                 x_data,
                 y_data,
                 terms: int = 3,
                 constraint_type: ConstraintType = ConstraintType.NONE):
        self.terms = _validate_terms(terms)
        self.constraint_type = ConstraintType.resolve(constraint_type)
        self.x_data, self.y_data = _validate_xy(x_data, y_data, self.terms)
        
        # Estimate gamma from the data (controls center basis location)
        self.gamma = calculate_gamma(self.x_data, self.y_data)
        
        # Fit coefficients (with or without constraints)
        self.coefficients = self._fit_coefficients()
        
        # Check if the fitted distribution is valid
        self.is_feasible = self._check_feasibility()
        
        if not self.is_feasible:
            warnings.warn("QFlex distribution may not be feasible (PDF not strictly positive)")
    
    def _validate_inputs(self):
        """Validate that inputs are consistent and sufficient for fitting."""
        self.x_data, self.y_data = _validate_xy(self.x_data, self.y_data, self.terms)
    
    def _fit_coefficients(self) -> np.ndarray:
        """Fit coefficients using least squares or constrained optimization."""
        Y = build_design_matrix(self.y_data, self.terms, self.gamma)
        
        if self.constraint_type == ConstraintType.NONE:
            return self._fit_unconstrained(Y)
        else:
            return solve_with_constraints(Y, self.x_data, self.terms,
                                         self.gamma, self.constraint_type)
    
    def _fit_unconstrained(self, Y: np.ndarray) -> np.ndarray:
        """Fit coefficients by solving Y @ a = x in a least-squares sense.

        Uses the SVD-based solver ``np.linalg.lstsq`` for both the square
        (exact interpolation) and overdetermined cases. Unlike forming an
        explicit inverse, this stays numerically stable when the design matrix
        is ill-conditioned -- for example when the assessed probabilities are
        symmetric about 0.5 with gamma = 0.5, which makes the matrix nearly
        rank-deficient. In that case ``inv(Y) @ x`` silently returns a vector
        that does not interpolate the data, whereas ``lstsq`` recovers the
        minimum-norm exact (or least-squares) solution. (Since 1.2.0 an exactly
        symmetric assessment gets gamma = 0.45, which avoids that particular
        case, but other near-singular designs remain possible.)
        """
        try:
            coefficients, *_ = np.linalg.lstsq(Y, self.x_data, rcond=None)
        except np.linalg.LinAlgError:
            raise QFlexError("Least-squares fit failed. Check for collinear or degenerate data.")

        return coefficients
    
    def quantile(self, y):
        """
        Evaluate the quantile function Q(y).

        Probabilities strictly inside (0, 1) are clipped to [1e-12, 1 - 1e-12].
        At exactly 0 and 1 the limits are returned (-inf / +inf when the
        corresponding tail has a positive leading coefficient).

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in [0, 1].

        Returns
        -------
        x : array-like
            Corresponding quantile values.
        """
        y_in = np.asarray(y, dtype=float)
        out = evaluate_quantile(np.clip(y_in, PROB_EPS, 1 - PROB_EPS),
                                self.coefficients, self.terms, self.gamma)
        if np.any((y_in == 0.0) | (y_in == 1.0)):
            q0, q1 = quantile_limits(self.coefficients, self.terms, self.gamma)
            out = np.where(y_in == 0.0, q0, np.where(y_in == 1.0, q1, out))
            if np.ndim(out) == 0:
                out = float(out)
        return out

    def pdf_analytical(self, y):
        """
        Compute PDF using exact analytical derivatives.

        Uses closed-form derivatives of the QFlex basis functions:
        - dR_j/dy = j × [-ln(1-y)]^(j-1) / (1-y)
        - dL_j/dy = j × (-1)^(j+1) × ln(y)^(j-1) / y
        - dC_j/dy = (2j-1) × (y-γ)^(2j-2)

        Parameters
        ----------
        y : array-like
            Cumulative probabilities in (0, 1).

        Returns
        -------
        pdf : array-like
            Probability density values; negative where the fit decreases
            (an invalid fit), NaN where dQ/dy = 0.
        """
        from .basis import evaluate_quantile_derivative

        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)

        # Compute q(y) = dQ/dy analytically
        q = evaluate_quantile_derivative(y, self.coefficients, self.terms, self.gamma)

        # A NEGATIVE q MEANS THE FIT IS DECREASING THERE, NOT THAT THE DENSITY
        # IS ENORMOUS. Clipping q up to 1e-12 turned every such point into a
        # density of 1e12 -- the largest representable -- which manufactured a
        # spike out of infeasibility and capped genuine spikes at the same
        # value. The sign is preserved instead.
        q = np.where(q == 0, np.nan, q)

        return 1.0 / q
    
    def _check_feasibility(self) -> bool:
        """
        Feasibility in the paper's sense, on the grid p = 0.001, ..., 0.999:
        1. Q strictly increasing from one grid point to the next
        2. a strictly positive density at every grid point
        It does not look beyond the grid; use check_proposition3() for the
        Theorem 5 tail condition.
        """
        monotonic = self._check_monotonicity()
        
        # Test the PDF on the same grid
        y_test = 0.001 + np.arange(999) * 0.001
        pdf_vals = self.pdf(y_test)
        pdf_positive = np.all(pdf_vals > 0)
        
        return monotonic and pdf_positive
