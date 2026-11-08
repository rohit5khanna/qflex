# QFlex

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Version 1.2.0. See [CHANGELOG.md](CHANGELOG.md) for what changed.

A Python library implementing **QFlex quantile-parameterized distributions** — a flexible family of probability distributions fit directly from quantile data, with support for unbounded, semibounded, and bounded domains.

QFlex is especially well suited for **expert elicitation** (fitting from P10/P50/P90 assessments), **flexible tail modelling**, and any setting where a distribution is most naturally expressed through its quantile function rather than a PDF or CDF.

---

## Installation

Install from the GitHub repository, or from a local copy of it:

```bash
pip install git+https://github.com/rohitkhanna/qflex.git
# or, from a downloaded copy
pip install /path/to/qflex-main
```

PyPI currently has only the older release 1.0.0, so `pip install qflex` does not
give you this version.

The only required dependencies are NumPy and SciPy.

[CVXPY](https://www.cvxpy.org) is **optional** and only used by the `TC_MAG`
(Proposition 3) fit: if it is installed, CVXPY (Clarabel) solves that fit's quadratic
program; otherwise SciPy solves the same problem, giving fits within about 0.2% in
squared error in our tests. Every other constraint type, including `TC`, uses SciPy alone.

```bash
pip install cvxpy   # optional
```

---

## Quick Start

### From quantile pairs (expert elicitation)

Provide cumulative probabilities and their corresponding quantile values — exactly as you would collect them from an expert or an empirical summary.

```python
import numpy as np
from qflex import QFlex, LogQFlex, LogitQFlex, ConstraintType

# Minimal 3-point elicitation (P10, P50, P90) — default terms=3
y_elicit = [0.10, 0.50, 0.90]
x_elicit = [12.0, 25.0, 45.0]
qf = QFlex(x_elicit, y_elicit)  # terms=3 by default

# Richer fit when more quantiles are available
y_data = [0.10, 0.25, 0.50, 0.75, 0.90]
x_data = [12.0, 18.0, 25.0, 34.0, 45.0]
qf5 = QFlex(x_data, y_data, terms=5)

print(qf.quantile([0.1, 0.5, 0.9]))        # → quantile values at given probabilities
print(qf.pdf([0.1, 0.5, 0.9]))             # → density at those cumulative probabilities
print(qf.cdf([20.0, 25.0, 30.0]))          # → cumulative probabilities at given x values

samples = qf.sample(size=1000, random_state=0)   # reproducible draws

m = qf.moments(order=4)
print(m['mean'], m['std'], m['skewness'], m['kurtosis'])
```

### From raw data (Weibull plotting positions)

If you have raw observations rather than pre-computed quantiles, use `fit_from_data`. It sorts the data and assigns cumulative probabilities using Weibull plotting positions `y_i = i / (n + 1)`.

```python
data = np.random.lognormal(mean=3, sigma=0.5, size=200)

qf       = QFlex.fit_from_data(data)
log_qf   = LogQFlex.fit_from_data(data, lower_bound=0)
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
qf = QFlex(x_data, y_data)
```

### Semibounded: `LogQFlex`

For data with a lower bound (e.g. income, asset prices, durations).
Internally fits QFlex to `ln(x - lower_bound)` and maps all outputs back to the original scale.

```python
qf = LogQFlex(x_data, y_data, lower_bound=0)
```

### Bounded: `LogitQFlex`

For data bounded on both sides (e.g. proportions, test scores, rates).
Internally fits QFlex to `logit((x - L) / (U - L))`.

```python
shares = [0.12, 0.25, 0.45]                       # e.g. elicited market shares
qf = LogitQFlex(shares, [0.10, 0.50, 0.90], lower_bound=0, upper_bound=1)
```

---

## Constraint Types

Unconstrained least-squares fitting does not guarantee a valid (increasing) quantile
function. Three constraints guarantee validity; two only impose conditions that are
necessary for it. Theorem and proposition numbers follow the paper.

| Constraint | Condition | Guarantees a valid fit? |
|---|---|---|
| `ConstraintType.NONE` | Unconstrained least squares (default) | No |
| `ConstraintType.TL` | Leading (highest-order) tail coefficients > 0 (Theorem 5) | No — necessary only |
| `ConstraintType.TA` | All tail coefficients ≥ 0 | No |
| `ConstraintType.A` | All coefficients except the constant a₁ ≥ 0 (Theorem 1) | Yes; most restrictive |
| `ConstraintType.TC` | All tails ≥ 0 and (a₂ + a₃) − Σ Mⱼ\|centre coefficients\| > 0 (Proposition 4) | Yes; centre terms may be negative |
| `ConstraintType.TC_MAG` | m_tail > M_center (Proposition 3) with leading tails > 0 | Yes; least restrictive of the three |

In a test sweep of 96 fits per constraint (normal, lognormal, t3, bimodal and beta
samples of 150 points; `QFlex`, `LogQFlex` and `LogitQFlex`; K = 3, 5, 8, 11), 23 of the
TL fits and 2 of the TA fits were invalid somewhere in (0, 1), against none for A, TC
and TC_MAG. Check `is_feasible` (and `check_proposition3()`) when you use TL or TA.

```python
qf = QFlex(x_data, y_data, constraint_type=ConstraintType.TC)
qf = QFlex(x_data, y_data, constraint_type="A+")     # strings work too: "TL+", "TA+", "TC", "TC_mag"
```

`TC_MAG` is enforced on the 0.001 grid plus tail points down to p = 1e-12, and refined
until it holds on a grid ten times finer, so the returned fit is valid on the whole
range the library evaluates. Its fit is never worse than the `TC` fit, apart from the
tiny floor that keeps the leading tail coefficients above zero (at most about 1e-5
relative in squared error in our tests).

---

## API Reference

All three classes (`QFlex`, `LogQFlex`, `LogitQFlex`) share the following interface.

### Constructors

| Class | Signature |
|---|---|
| `QFlex` | `QFlex(x_data, y_data, terms=3, constraint_type=ConstraintType.NONE)` |
| `LogQFlex` | `LogQFlex(x_data, y_data, lower_bound=0, terms=3, constraint_type=...)` |
| `LogitQFlex` | `LogitQFlex(x_data, y_data, lower_bound=0, upper_bound=1, terms=3, constraint_type=...)` |

`constraint_type` takes a `ConstraintType` or its name as a string (`"A+"`, `"TL+"`,
`"TA+"`, `"TC"`, `"TC_mag"`, `"none"`).

### Instance Methods

| Method | Input | Output | Notes |
|---|---|---|---|
| `quantile(y)` | `y ∈ [0,1]` | x values | Q(p); p inside (0, 1) is clipped to [1e-12, 1 − 1e-12]; Q(0) and Q(1) are the limits (±inf, or the bounds) |
| `pdf(y, method='analytical')` | `y ∈ (0,1)` | density at Q(y) | Exact derivative by default, for all three classes; `method='numerical'` uses central differences |
| `cdf(x)` | x values | `p ∈ [0,1]` | Inverts Q(p) (vectorised; 10,000 points in about 0.02 s) |
| `sample(size=1, random_state=None)` | int or shape | `np.ndarray` | Inverse transform sampling; pass a seed or `np.random.Generator` |
| `moments(order=4, clipped=False)` | int | `dict` | `mean`, `variance`, `std`, `skewness`, `kurtosis` (not excess), `raw_k`, `central_k`; moments that do not exist are `inf` |
| `summary()` | — | printed table | Terms, γ, feasibility, constraint, moments, P10/P50/P90, coefficients |
| `plot(p_grid, show_data, ax)` | optional | `(fig, axes)` | PDF panel + quantile function panel |
| `check_proposition3(p_grid=None, num_points=None)` | — | `dict` | Proposition 3 grid condition plus Theorem 5: `satisfied`, `grid_condition`, `theorem5_satisfied`, `m_tail`, `M_center`, `margin`, `q_flex_min`, `q_flex_positive`, `leading_right`, `leading_left`. `check_proposition4()` is the old name and still works |

`is_feasible` is the paper's check: Q increasing, and a positive density, on the grid
p = 0.001, …, 0.999. It does not look beyond the grid; `check_proposition3()['theorem5_satisfied']`
tells you whether the tails eventually turn down.

**Moments of heavy tails.** For `LogQFlex`, X = L + exp(z) and the right tail decides
which moments exist: if the leading right-tail term is positive and of order ≥ 2, none
do; if it is R₁ with coefficient a₂ > 0, E[Xᵏ] exists only for k·a₂ < 1. Moments that do not exist are reported as
`inf` (with a warning) rather than as a number. `moments(clipped=True)` gives the always
finite moments of the distribution as `quantile()` and `sample()` evaluate it.

### Class Method

| Method | Description |
|---|---|
| `fit_from_data(data, terms=3, constraint_type=..., **kwargs)` | Fit from raw observations using Weibull plotting positions `y_i = i/(n+1)` |

### Utility

```python
from qflex.utils import compute_w1

w1, w1_norm = compute_w1(qf.quantile, x_data, y_data)
# w1       → ∫ |Q_fit(p) − Q_target(p)| dp over [0.01, 0.99], Q_target = linear interpolation of the data
# w1_norm  → w1 / (P90 − P10) of the data: the paper's normalised W1 (Eq. 25), on [0.01, 0.99]
```

---

## Theory and Basis

### The Quantile Function

QFlex represents a probability distribution through its **quantile function** Q(p) rather than through a PDF or CDF. Q(p) is a constant plus a linear combination of three families of basis functions, with coefficients numbered a₁, …, a_K as in the paper (`coefficients[k-1]` in code):

```
Q(p) = a₁ + a₂·R₁(p) + a₃·L₁(p) + a₄·C₁(p) + a₅·R₂(p) + a₆·L₂(p) + a₇·C₂(p) + …
```

| Family | Formula | Role |
|---|---|---|
| **Right tail** R_j(p) | `[-ln(1-p)]^j` | Controls right (upper) tail behaviour |
| **Left tail** L_j(p) | `(-1)^(j+1) · [ln(p)]^j` = `-[-ln(p)]^j` | Controls left (lower) tail behaviour |
| **Center** C_j(p) | `(p - γ)^(2j-1)` | Controls centre/body behaviour |

The `terms` parameter (K) controls how many basis functions are included. Terms cycle through constant → R₁ → L₁ → C₁ → R₂ → L₂ → C₂ → R₃ → … as `terms` increases (the paper's Table 5):

| `terms` | Basis terms | Shape capability |
|---|---|---|
| 3 | constant, R₁, L₁ | Minimal tails; no centre term |
| 4 | + C₁ | First centre term |
| 5 | + R₂ | Richer right tail |
| 6 | + L₂ | Richer left tail |
| 7 | + C₂ | Second centre term |
| 8, 9, 10 | + R₃, + L₃, + C₃ | Third-order terms |

Increase `terms` when you have more quantile data and need additional flexibility. You need at least as many data points as terms.

Coefficients are fitted by least squares (exact when `len(data) == terms`, overdetermined otherwise).

### The Gamma (γ) Parameter

γ is a location parameter for the centre basis, estimated automatically from P10, P50
and P90 of the input data (linearly interpolated from the given pairs, or extrapolated if
they do not cover 0.1 or 0.9) to adapt to skewness (the paper's Eqs. 10–11):

```
γ = 0.5 × (1 − SI),   SI = (P90 + P10 − 2·P50) / (P90 − P10)   (Kelly skewness index; γ clipped to [0, 1])
```

Right-skewed data gives γ < 0.5 and left-skewed gives γ > 0.5. An exactly symmetric
assessment (SI = 0, or P90 = P10) gets γ = 0.45, as in the paper. γ = 0.5 can make the
design matrix singular: the paper's example is K = 7 on the symmetric grid
(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95).

### PDF and Feasibility

The PDF is recovered from the quantile function via:

```
f(Q(p)) = 1 / q(p)     where     q(p) = dQ/dp
```

A valid distribution requires q(p) > 0 for all p ∈ (0,1) — i.e. Q(p) must be strictly increasing. This is not guaranteed by unconstrained fitting, which motivates the constraint hierarchy.

### Validity conditions (numbering as in the paper)

- **Theorem 1 (A)**: all coefficients except the constant a₁ non-negative (a₂, …, a_K ≥ 0) → Q is increasing. Most conservative.
- **Theorem 5 (TL)**: the leading (highest-order) right and left tail coefficients must be positive, or Q eventually decreases as p → 0 or 1. Necessary, not sufficient.
- **Proposition 3 (TC_MAG)**: write q(p) = q_tail(p) + q_center(p). If the smallest tail part exceeds the largest centre magnitude, `m_tail = inf q_tail > M_center = sup |q_center|`, then Q is strictly increasing. Checked on a grid it certifies the grid points only, so it must be paired with Theorem 5 (Remark 7). The least conservative of the guarantees.
- **Proposition 4 (TC)**: a coefficient bound, (a₂ + a₃) − Σ Mⱼ·|a_k| > 0 over the centre coefficients a_k (on C_j), with all tails ≥ 0, where Mⱼ = sup|C_j′| = (2j−1)·max(γ, 1−γ)^(2j−2). Valid everywhere, but conservative: it assumes every centre term is at its steepest where the tails are weakest.

Releases before 1.2.0 called TC "Proposition 5" and TC_MAG "Proposition 4".

### Bounded and Semibounded Variants

For restricted domains, QFlex is applied in a transformed space:

| Variant | Transform applied | Inverse (output) |
|---|---|---|
| `LogQFlex` | `z = ln(x − L)` | `x = exp(z) + L` |
| `LogitQFlex` | `z = ln((x−L)/(U−x))` | `x = L + (U−L) / (1 + exp(−z))` |

Fitting, PDF, CDF, and sampling all happen on z; every output is mapped back to the original x scale transparently.

---

## Examples

`scripts/` holds runnable end-to-end examples; the data they read is in `data/`.
`scripts/_metalog.py` is a minimal Metalog fitter used only by the examples (not part of the `qflex` API).

| Script | What it does |
|---|---|
| `scripts/fit_traction_case4_qpds.py` | Reproduces Caban et al. (2024) Case 4: fits Metalog and QFlex to vehicle cell capacities at several truncation orders, reports feasibility and mode count for each, and writes CDF/PDF comparison figures. |

```bash
pip install -e ".[examples]"   # pandas + matplotlib
python scripts/fit_traction_case4_qpds.py
```

It writes figures to `figures/` and a fit summary to `data/traction_case4_fit_summary.csv`.
Neither is tracked — rerun the script to regenerate them.

---

## Reference

Bickel, J. Eric, Colombe, Connor, and Leibowicz, Benjamin D. **"The QFlex Distribution."** December 16, 2025.
Available at SSRN: [https://ssrn.com/abstract=5930859](https://ssrn.com/abstract=5930859)
DOI: [10.2139/ssrn.5930859](http://dx.doi.org/10.2139/ssrn.5930859)

This implementation follows the basis functions, gamma estimation, Theorems 1 and 5 and Propositions 3 and 4 as described in the paper above.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
