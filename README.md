# QFlex

[![PyPI version](https://badge.fury.io/py/qflex.svg)](https://badge.fury.io/py/qflex)
[![Python](https://img.shields.io/pypi/pyversions/qflex.svg)](https://pypi.org/project/qflex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library implementing **QFlex quantile-parameterized distributions** — a flexible family of probability distributions fit directly from quantile data, with support for unbounded, semibounded, and bounded domains.

QFlex is especially well suited for **expert elicitation** (fitting from P10/P50/P90 assessments), **flexible tail modelling**, and any setting where a distribution is most naturally expressed through its quantile function rather than a PDF or CDF.

---

## Installation

```bash
pip install qflex
```

To enable the linear Proposition 5 solver (requires CVXPY):

```bash
pip install qflex[linear]
```

---

## Quick Start

### From quantile pairs (expert elicitation)

Provide cumulative probabilities and their corresponding quantile values — exactly as you would collect them from an expert or an empirical summary.

```python
import numpy as np
from qflex import QFlex, LogQFlex, LogitQFlex, ConstraintType

y_data = [0.10, 0.25, 0.50, 0.75, 0.90]   # cumulative probabilities
x_data = [12.0, 18.0, 25.0, 34.0, 45.0]   # corresponding quantile values

qf = QFlex(x_data, y_data, terms=5)

print(qf.quantile([0.1, 0.5, 0.9]))        # → quantile values at given probabilities
print(qf.pdf([0.1, 0.5, 0.9]))             # → density at those cumulative probabilities
print(qf.cdf([20.0, 25.0, 30.0]))          # → cumulative probabilities at given x values

samples = qf.sample(size=1000)

m = qf.moments(order=4)
print(m['mean'], m['std'], m['skewness'], m['kurtosis'])
```

### From raw data (Weibull plotting positions)

If you have raw observations rather than pre-computed quantiles, use `fit_from_data`. It sorts the data and assigns cumulative probabilities using Weibull plotting positions `y_i = i / (n + 1)`.

```python
data = np.random.lognormal(mean=3, sigma=0.5, size=200)

qf       = QFlex.fit_from_data(data, terms=5)
log_qf   = LogQFlex.fit_from_data(data, lower_bound=0, terms=5)
```

### Summarise and plot

```python
qf.summary()              # prints a formatted table: moments, P10/P50/P90, coefficients

fig, axes = qf.plot()     # two-panel figure: PDF (left) + quantile function (right)
fig.savefig('fit.png')
```

---

## Distribution Variants

### Unbounded: `QFlex`

For data with no natural bounds (e.g. log-returns, temperature anomalies).

```python
qf = QFlex(x_data, y_data, terms=5)
```

### Semibounded: `LogQFlex`

For data with a lower bound (e.g. income, asset prices, durations).
Internally fits QFlex to `ln(x - lower_bound)` and maps all outputs back to the original scale.

```python
qf = LogQFlex(x_data, y_data, lower_bound=0, terms=5)
```

### Bounded: `LogitQFlex`

For data bounded on both sides (e.g. proportions, test scores, rates).
Internally fits QFlex to `logit((x - L) / (U - L))`.

```python
qf = LogitQFlex(x_data, y_data, lower_bound=0, upper_bound=1, terms=5)
```

---

## Constraint Types

Unconstrained least-squares fitting does not guarantee a valid (positive) PDF. The following constraints enforce feasibility with varying degrees of strictness:

| Constraint | Description | Restrictiveness |
|---|---|---|
| `ConstraintType.NONE` | Unconstrained least squares (default) | — |
| `ConstraintType.A` | All coefficients ≥ 0 for k ≥ 2 (Prop 3) | Most restrictive |
| `ConstraintType.TL` | Leading tail coefficients ≥ 0 | High |
| `ConstraintType.TA` | All tail coefficients ≥ 0 | Medium |
| `ConstraintType.TC` | Prop 5 tail-centre margin > 0 via SLSQP | Low |
| `ConstraintType.TC_MAG` | Prop 4 grid-based m_tail > M_center | Least restrictive |

```python
qf = QFlex(x_data, y_data, terms=5, constraint_type=ConstraintType.TC_MAG)

# TC with linear reformulation (requires cvxpy)
qf = QFlex(x_data, y_data, terms=5,
           constraint_type=ConstraintType.TC, tc_method='linear')
```

---

## API Reference

All three classes (`QFlex`, `LogQFlex`, `LogitQFlex`) share the following interface.

### Constructors

| Class | Signature |
|---|---|
| `QFlex` | `QFlex(x_data, y_data, terms=5, constraint_type=..., tc_method='nonlinear')` |
| `LogQFlex` | `LogQFlex(x_data, y_data, lower_bound, terms=5, ...)` |
| `LogitQFlex` | `LogitQFlex(x_data, y_data, lower_bound, upper_bound, terms=5, ...)` |

### Instance Methods

| Method | Input | Output | Notes |
|---|---|---|---|
| `quantile(y)` | `y ∈ (0,1)` | x values | Core quantile function Q(p) |
| `pdf(y, method='numerical')` | `y ∈ (0,1)` | density values | Use `method='analytical'` for closed-form (QFlex only) |
| `cdf(x)` | x values | `p ∈ (0,1)` | Inverts Q(p) numerically |
| `sample(size=1)` | int | `np.ndarray` | Inverse transform sampling |
| `moments(order=4)` | int | `dict` | Keys: `mean`, `variance`, `std`, `skewness`, `kurtosis`, `raw_k`, `central_k` |
| `summary()` | — | printed table | Terms, γ, feasibility, moments, P10/P50/P90, coefficients |
| `plot(p_grid, show_data, ax)` | optional | `(fig, axes)` | PDF panel + quantile function panel |
| `check_proposition4()` | — | `dict` | Keys: `satisfied`, `m_tail`, `M_center`, `margin`, `q_flex_min`, `q_flex_positive` |

### Class Method

| Method | Description |
|---|---|
| `fit_from_data(data, terms=5, constraint_type=..., **kwargs)` | Fit from raw observations using Weibull plotting positions `y_i = i/(n+1)` |

### Utility

```python
from qflex.utils import compute_w1

w1, w1_norm = compute_w1(qf.quantile, x_data, y_data)
# w1       → Wasserstein-1 distance between fitted and target quantile functions
# w1_norm  → W1 normalised by (P90 - P10) of the data
```

---

## Theory and Basis

### The Quantile Function

QFlex represents a probability distribution through its **quantile function** Q(p) rather than through a PDF or CDF. Q(p) is expressed as a linear combination of three families of basis functions:

```
Q(p) = Σ_{j=1}^{m} [ a_j · R_j(p)  +  b_j · L_j(p)  +  c_j · C_j(p) ]
```

| Family | Formula | Role |
|---|---|---|
| **Right tail** R_j(p) | `[-ln(1-p)]^j` | Controls right (upper) tail behaviour |
| **Left tail** L_j(p) | `(-1)^(j+1) · [ln(p)]^j` | Controls left (lower) tail behaviour |
| **Center** C_j(p) | `(p - γ)^(2j-1)` | Controls centre/body behaviour |

The `terms` parameter sets the depth j = 1, …, m of each basis family. Coefficients are fitted by least squares (exact when `len(data) == terms`, overdetermined otherwise).

### The Gamma (γ) Parameter

γ is a location parameter for the centre basis, estimated automatically from the empirical P10, P50, and P90 of the input data to adapt to skewness:

```
γ = (P50 - P10) / (P90 - P10)
```

A symmetric distribution gives γ ≈ 0.5. Right-skewed data gives γ < 0.5 and left-skewed gives γ > 0.5.

### PDF and Feasibility

The PDF is recovered from the quantile function via:

```
f(Q(p)) = 1 / q(p)     where     q(p) = dQ/dp
```

A valid distribution requires q(p) > 0 for all p ∈ (0,1) — i.e. Q(p) must be strictly increasing. This is not guaranteed by unconstrained fitting, which motivates the constraint hierarchy.

### Feasibility Constraints (Propositions 3–5)

The paper establishes three sufficient conditions for a strictly positive PDF:

- **Proposition 3 (A)**: All non-intercept coefficients non-negative → guarantees q(p) > 0 everywhere. Most conservative.
- **Proposition 4 (TC_MAG)**: Grid-based check that the minimum tail derivative contribution exceeds the maximum centre contribution: `m_tail > M_center`. Less conservative, verifiable numerically.
- **Proposition 5 (TC)**: Sharper analytical condition on the tail-to-centre basis ratio, enforced via SLSQP or a linear auxiliary-variable reformulation. Closest to the true feasibility boundary.

### Bounded and Semibounded Variants

For restricted domains, QFlex is applied in a transformed space:

| Variant | Transform applied | Inverse (output) |
|---|---|---|
| `LogQFlex` | `z = ln(x − L)` | `x = exp(z) + L` |
| `LogitQFlex` | `z = ln((x−L)/(U−x))` | `x = L + (U−L) / (1 + exp(−z))` |

Fitting, PDF, CDF, and sampling all happen on z; every output is mapped back to the original x scale transparently.

---

## Reference

Bickel, J. Eric, Colombe, Connor, and Leibowicz, Benjamin D. **"The QFlex Distribution."** December 16, 2025.
Available at SSRN: [https://ssrn.com/abstract=5930859](https://ssrn.com/abstract=5930859)
DOI: [10.2139/ssrn.5930859](http://dx.doi.org/10.2139/ssrn.5930859)

This implementation follows the basis functions, gamma estimation, and Propositions 3–5 as described in the paper above.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
