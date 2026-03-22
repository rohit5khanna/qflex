"""
QFlex Core Implementation

Provides the base class and unbounded QFlex distribution.
"""

import numpy as np
from scipy.linalg import inv, LinAlgError
from abc import ABC, abstractmethod
import warnings

from .basis import build_design_matrix, evaluate_quantile
from .utils import calculate_gamma
from .constraints import ConstraintType, solve_with_constraints, QFlexError
from .mono_verification import check_proposition4, check_delta_p_monotonicity
from .utils import compute_pdf_numerical, compute_cdf_inverse, compute_moments

PROB_EPS = 1e-12


class QFlexBase(ABC):
    """
    Abstract base class for all QFlex distribution variants.
    
    Subclasses only need to implement quantile(). PDF, CDF, sampling, and
    moments are computed using shared utility functions.
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
    
    def pdf(self, y, step_size=0.001, method='numerical'):
        """
        Compute probability density function.

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
    
    def sample(self, size=1):
        """Generate random samples using inverse transform sampling."""
        u = np.random.uniform(0, 1, size)
        return self.quantile(u)
    
    def moments(self, order=4):
        """Compute moments up to the specified order via numerical integration."""
        return compute_moments(self.quantile, order)

    def pdf_analytical(self, y):
        """
        Compute PDF using analytical derivatives (only for QFlex subclasses).

        This is a placeholder that should be overridden by subclasses
        that have analytical derivative implementations.

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

    def _check_monotonicity(self) -> bool:
        """Check if the quantile function is strictly increasing."""
        result = check_delta_p_monotonicity(self.quantile, delta_p=0.001)
        return result['satisfied']

    @classmethod
    def fit_from_data(cls, data, terms=5,
                      constraint_type=ConstraintType.NONE,
                      tc_method='nonlinear', **kwargs):
        """
        Fit a QFlex distribution directly from raw data using Weibull plotting positions.

        Sorts the data and assigns cumulative probabilities using the Weibull formula:
            y_i = i / (n + 1)  for i = 1, 2, ..., n

        Parameters
        ----------
        data : array-like
            Raw observations. Will be sorted internally.
        terms : int, optional
            Number of basis terms (default 5).
        constraint_type : ConstraintType, optional
            Constraint to apply during fitting (default: unconstrained).
        tc_method : str, optional
            For TC constraint: 'nonlinear' or 'linear' (default 'nonlinear').
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
        >>> qf = QFlex.fit_from_data(data, terms=5)
        >>> log_qf = LogQFlex.fit_from_data(data, lower_bound=0, terms=5)
        """
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        if n < terms:
            raise QFlexError(
                f"Need at least {terms} data points for a {terms}-term fit, got {n}"
            )
        y = np.arange(1, n + 1) / (n + 1)
        return cls(data, y, terms=terms, constraint_type=constraint_type,
                   tc_method=tc_method, **kwargs)

    def summary(self):
        """
        Print a formatted summary of the fitted distribution.

        Includes class info, fit parameters, moments, and feasibility.
        """
        m = self.moments(order=4)
        p10 = float(self.quantile(np.array([0.10])))
        p50 = float(self.quantile(np.array([0.50])))
        p90 = float(self.quantile(np.array([0.90])))

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
        Q(y) = Σ a_k × basis_k(y)
    
    Three basis families are used:
        - Right tail: -ln(1-p) raised to increasing powers
        - Left tail: alternating-sign powers of ln(p)
        - Center: odd powers of (p - γ)
    
    Parameters
    ----------
    x_data : array-like
        Observed quantile values.
    y_data : array-like
        Corresponding cumulative probabilities in (0, 1).
    terms : int, optional
        Number of terms in the expansion (default 5).
    constraint_type : ConstraintType, optional
        Constraint to apply during fitting (default: unconstrained).
    tc_method : str, optional
        For TC constraint: 'nonlinear' (SLSQP) or 'linear' (auxiliary variables).
    """
    
    def __init__(self,
                 x_data,
                 y_data,
                 terms: int = 5,
                 constraint_type: ConstraintType = ConstraintType.NONE,
                 tc_method: str = 'nonlinear'):
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.terms = terms
        self.constraint_type = constraint_type
        self.tc_method = tc_method
        
        self._validate_inputs()
        
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
        if len(self.x_data) != len(self.y_data):
            raise QFlexError("x_data and y_data must have the same length")
        
        if len(self.x_data) < self.terms:
            raise QFlexError(f"Need at least {self.terms} data points for {self.terms}-term QFlex")
        
        if np.any(self.y_data <= 0) or np.any(self.y_data >= 1):
            raise QFlexError("All y_data values must be in (0, 1)")
        
        if len(np.unique(self.y_data)) < self.terms:
            raise QFlexError(f"Need at least {self.terms} distinct y values")
    
    def _fit_coefficients(self) -> np.ndarray:
        """Fit coefficients using least squares or constrained optimization."""
        Y = build_design_matrix(self.y_data, self.terms, self.gamma)
        
        if self.constraint_type == ConstraintType.NONE:
            return self._fit_unconstrained(Y)
        else:
            return solve_with_constraints(Y, self.x_data, self.terms,
                                         self.gamma, self.constraint_type,
                                         tc_method=self.tc_method)
    
    def _fit_unconstrained(self, Y: np.ndarray) -> np.ndarray:
        """Solve the linear system exactly or via least squares."""
        try:
            if len(self.x_data) == self.terms:
                # Exact fit: solve Y @ a = x directly
                coefficients = inv(Y) @ self.x_data
            else:
                # Overdetermined: use normal equations
                YTY_inv = inv(Y.T @ Y)
                coefficients = YTY_inv @ Y.T @ self.x_data
        except LinAlgError:
            raise QFlexError("Design matrix is singular. Check for collinear data.")
        
        return coefficients
    
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
        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
        return evaluate_quantile(y, self.coefficients, self.terms, self.gamma)

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
            Probability density values.
        """
        from .basis import evaluate_quantile_derivative

        y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)

        # Compute q(y) = dQ/dy analytically
        q = evaluate_quantile_derivative(y, self.coefficients, self.terms, self.gamma)

        # PDF = 1 / q(y), ensuring q > 0
        q = np.clip(q, 1e-12, None)

        return 1.0 / q
    
    def _check_feasibility(self) -> bool:
        """
        Check if the distribution produces a valid PDF.
        
        A feasible distribution has:
        1. A strictly increasing quantile function
        2. A strictly positive PDF everywhere
        """
        monotonic = self._check_monotonicity()
        
        # Test PDF on a fine grid
        y_test = 0.001 + np.arange(999) * 0.001
        pdf_vals = self.pdf(y_test)
        pdf_positive = np.all(pdf_vals > 0)
        
        return monotonic and pdf_positive
    
    def check_proposition4(self, p_grid=None, num_points=10000):
        """
        Verify Proposition 4 conditions (m_tail > M_center).
        
        Returns a dict with 'satisfied', 'm_tail', 'M_center', 'margin', etc.
        """
        return check_proposition4(self.coefficients, self.terms, self.gamma, 
                                 p_grid=p_grid, num_points=num_points)
