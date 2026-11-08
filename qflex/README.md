# QFlex Distributions

A modular Python implementation of QFlex quantile-parameterized distributions with flexible basis functions.

## Overview

QFlex is a distribution family that uses custom basis functions to fit quantile data. It supports unbounded, semibounded, and bounded distributions through log and logit transforms, with optional constraints to ensure valid probability densities.

## Module Structure

| File | Purpose |
|------|---------|
| `basis.py` | Basis function definitions and evaluation (constant, tail, center families) |
| `constraints.py` | Constraint solvers for coefficient estimation (Theorems 1 and 5, Propositions 3 and 4) |
| `core.py` | Main `QFlex` class for unbounded distributions |
| `transforms.py` | `LogQFlex` (semibounded) and `LogitQFlex` (bounded) variants |
| `utils.py` | Gamma calculation, PDF/CDF computation, moments, and W1 distance |
| `mono_verification.py` | Proposition 3 / Theorem 5 verification and monotonicity checks |
| `__init__.py` | Public API exports |

## Quick Start

```python
from qflex import QFlex, LogQFlex, LogitQFlex, ConstraintType

# 3-point expert elicitation (default terms=3)
qflex = QFlex([12, 25, 45], [0.10, 0.50, 0.90])

# With non-negativity constraints on coefficients
qflex = QFlex(x_data, y_data, constraint_type=ConstraintType.A)

# Semibounded distribution (e.g., income, time-to-event)
log_qflex = LogQFlex(x_data, y_data, lower_bound=0)

# Bounded distribution (e.g., proportions, percentages)
logit_qflex = LogitQFlex(x_data, y_data, lower_bound=0, upper_bound=1)

# Verify Proposition 3 (grid tail-centre condition) and Theorem 5
result = qflex.check_proposition3()
print(f"Satisfied: {result['satisfied']}, Margin: {result['margin']:.4f}")
```

## Constraint Types

| Type | Description |
|------|-------------|
| `NONE` | Unconstrained least squares |
| `A` | All coefficients except the constant a₁ non-negative; Theorem 1, always valid |
| `TL` | Leading tail coefficients > 0 (Theorem 5; necessary only) |
| `TA` | All tail coefficients non-negative (not sufficient) |
| `TC` | Proposition 4: tail-centre coefficient bound; always valid |
| `TC_MAG` | Proposition 3: m_tail > M_center, with leading tails > 0; valid on the whole evaluated range |

## Basis Functions

QFlex uses three basis function families:

- **Right tail R_j (`f1` in code)**: `[-ln(1-p)]^j` for j = 1, 2, 3, ...
- **Left tail L_j (`f2`)**: `(-1)^(j+1) × [ln(p)]^j` for j = 1, 2, 3, ...
- **Center C_j (`f3`)**: `(p - γ)^(2j-1)`, i.e. odd powers 1, 3, 5, ...

Terms cycle constant → R_j → L_j → C_j at increasing orders j. The default `terms=3` fits the constant a₁, R₁ and L₁ (suitable for P10/P50/P90). Increase `terms` to add centre and higher-order tail terms. Coefficients are numbered a₁, …, a_K as in the paper; in code they are 0-based (`coefficients[k-1]`).

The gamma (γ) parameter anchors the centre terms and is estimated from P10, P50 and P90 of the data: γ = 0.5(1 − SI) with the Kelly skewness index SI, and γ = 0.45 for a symmetric assessment (SI = 0), as in the paper.

## References

Bickel, Colombe and Leibowicz, "The QFlex Distribution" (SSRN 5930859). Theorem and proposition numbers in this package follow that paper; see the top-level README for details.
