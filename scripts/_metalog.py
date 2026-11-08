"""
Minimal unbounded Metalog (Keelin, 2016) used by the example scripts.

Only what the scripts need: least-squares fit from quantile pairs, the
quantile function, and a grid-based feasibility flag. This is NOT part of the
qflex package API.

Basis (y in (0,1), L(y) = ln(y / (1 - y))):
    k = 1 : 1
    k = 2 : L(y)
    k = 3 : (y - 0.5) L(y)
    k = 4 : (y - 0.5)
    k >= 5, odd  : (y - 0.5) ** ((k - 1) / 2)
    k >= 6, even : (y - 0.5) ** (k / 2 - 1) * L(y)
"""

import numpy as np

PROB_EPS = 1e-12


def _basis(y: np.ndarray, terms: int) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), PROB_EPS, 1 - PROB_EPS)
    L = np.log(y / (1 - y))
    c = y - 0.5
    cols = []
    for k in range(1, terms + 1):
        if k == 1:
            cols.append(np.ones_like(y))
        elif k == 2:
            cols.append(L)
        elif k == 3:
            cols.append(c * L)
        elif k == 4:
            cols.append(c)
        elif k % 2 == 1:
            cols.append(c ** ((k - 1) // 2))
        else:
            cols.append(c ** (k // 2 - 1) * L)
    return np.column_stack(cols)


class Metalog:
    """Unbounded Metalog fitted by ordinary least squares."""

    def __init__(self, x_data, y_data, terms: int = 3):
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.terms = int(terms)
        if len(self.x_data) != len(self.y_data):
            raise ValueError("x_data and y_data must have the same length")
        if len(self.x_data) < self.terms:
            raise ValueError(f"Need at least {self.terms} points for a {self.terms}-term Metalog")
        if np.any(self.y_data <= 0) or np.any(self.y_data >= 1):
            raise ValueError("All y_data values must be in (0, 1)")
        Y = _basis(self.y_data, self.terms)
        self.coefficients, *_ = np.linalg.lstsq(Y, self.x_data, rcond=None)
        self.is_feasible = self._check_feasibility()

    def quantile(self, y):
        y = np.asarray(y, dtype=float)
        out = _basis(np.atleast_1d(y), self.terms) @ self.coefficients
        return out.reshape(y.shape) if y.ndim else float(out[0])

    def _check_feasibility(self, delta: float = 0.001) -> bool:
        """Strictly increasing Q on a Δp grid (same test qflex uses)."""
        p = np.arange(delta, 1.0, delta)
        return bool(np.all(np.diff(self.quantile(p)) > 0))
