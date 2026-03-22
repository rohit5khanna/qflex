# QFlex Distributions

A modular Python implementation of QFlex quantile-parameterized distributions with flexible basis functions.

## Overview

QFlex is a distribution family that uses custom basis functions to fit quantile data. It supports unbounded, semibounded, and bounded distributions through log and logit transforms, with optional constraints to ensure valid probability densities.

## Module Structure

| File | Purpose |
|------|---------|
| `basis.py` | Basis function definitions and evaluation (constant, tail, center families) |
| `constraints.py` | Constraint solvers for coefficient estimation (Propositions 3–5) |
| `core.py` | Main `QFlex` class for unbounded distributions |
| `transforms.py` | `LogQFlex` (semibounded) and `LogitQFlex` (bounded) variants |
| `utils.py` | Gamma calculation, PDF/CDF computation, moments, and W1 distance |
| `mono_verification.py` | Proposition 4 verification and monotonicity checks |
| `__init__.py` | Public API exports |

## Quick Start

```python
from qflex import QFlex, LogQFlex, LogitQFlex, ConstraintType

# Fit an unbounded distribution
qflex = QFlex(x_data, y_data, terms=5)

# With non-negativity constraints on coefficients
qflex = QFlex(x_data, y_data, terms=5, constraint_type=ConstraintType.A)

# Semibounded distribution (e.g., income, time-to-event)
log_qflex = LogQFlex(x_data, y_data, lower_bound=0, terms=5)

# Bounded distribution (e.g., proportions, percentages)
logit_qflex = LogitQFlex(x_data, y_data, lower_bound=0, upper_bound=1, terms=5)

# Verify Proposition 4 conditions
result = qflex.check_proposition4()
print(f"Satisfied: {result['satisfied']}, Margin: {result['margin']:.4f}")
```

## Constraint Types

| Type | Description |
|------|-------------|
| `NONE` | Unconstrained least squares |
| `A` | All coefficients non-negative (k ≥ 2) |
| `TL` | Leading tail coefficients non-negative |
| `TA` | All tail coefficients non-negative |
| `TC` | Proposition 5: tail-center margin constraint |
| `TC_MAG` | Proposition 4: m_tail > M_center on grid |

## Basis Functions

QFlex uses three basis function families:

- **Right tail (f1)**: `-ln(1-p)` raised to powers 1, 2, 3, ...
- **Left tail (f2)**: `(-1)^(i+1) × [ln(p)]^i` for orders 1, 2, 3, ...
- **Center (f3)**: `(p - γ)^(2i-1)` for odd powers 1, 3, 5, ...

The gamma (γ) parameter controls the center of the distribution and is estimated from the data using P10, P50, and P90 quantiles.

## References

See the QFlex paper for theoretical background on the basis functions, propositions, and constraint formulations.
