# Changelog

## 1.2.0 (2026-11-08)

Fixes every open issue from the November 2026 code review. Issue IDs refer to that review log.

### Correctness

- **TC_MAG (Proposition 3) fits are valid and better (H1).** Solved as a convex quadratic program, started from the TC fit, with the leading tail coefficients kept strictly positive (Theorem 5). The grid condition is refined until it holds on a fine grid that includes the tails down to p = 1e-12. In a 48-fit sweep: 0 invalid fits and 0 errors (previously 4 invalid and 10 errors), and 2 s instead of 39 s. The fit is never worse than TC, apart from the tiny floor on the leading tails (at most about 1e-5 relative in squared error).
- **`check_proposition3()` also checks Theorem 5 (H2).** A negative leading tail coefficient now fails the check. New keys: `grid_condition`, `theorem5_satisfied`, `leading_right` and `leading_left`.
- **Tail densities are correct (H3).** `pdf()` now defaults to the analytical derivative. The numerical option uses a step proportional to min(p, 1 − p); the old fixed step of 0.001 was 90% off at p = 0.001.
- **Moments that do not exist are no longer reported as numbers (H4).** For a `LogQFlex` fit with a heavy right tail, those moments are `inf` and a warning is raised; previously the variance could come out negative. Moments are now integrated in log-odds without the 1e-12 clip, and match exact Beta-function values to 1e-8 or better. `moments(clipped=True)` gives the always-finite moments of the clipped distribution.
- **γ = 0.45 for symmetric assessments (H5),** as in the paper. On the paper's K = 7 grid this restores a full-rank design matrix.
- **`LogitQFlex` no longer returns NaN for large |z| (H6, L11).** It uses `expit` and a log-space density.
- **TC (Proposition 4) is solved exactly with SciPy alone (M1).** It no longer depends on CVXPY/OSQP, which failed or returned slightly infeasible results. On 144 test fits its objective was never higher than that of a convex solver (CVXPY/Clarabel).
- **TL+ keeps the leading tail coefficients strictly positive (new).** They stay at least 1e-6 × the data range above zero, since Theorem 5 and Remark 6 need them strictly positive. Previously they could be exactly 0, which handed the tail to a lower-order negative term.
- **TL+, TA+ and A+ use the exact BVLS solver (L10).** The iterative solver could stop at its iteration cap at high K.
- **The quantile derivative uses the same 1e-12 clip as the quantile (L8).**
- **NaN differences count as violations in `check_delta_p_monotonicity` (new).**

### API

- **Numbering follows the paper (M3).** `check_proposition3()` is new at module level and on every class. `check_proposition4()` remains as an alias, so existing code still runs.
- **`sample(size, random_state=None)` (L5)** accepts a seed or a `numpy.random.Generator`.
- **`cdf()` is vectorised (L6):** 10,000 points take about 0.02 s, previously 2.6 s.
- **`quantile(0)` and `quantile(1)` return the limits (L7):** ±inf, or the bounds for the log and logit classes.
- **`constraint_type` accepts strings** such as `"A+"`, `"TL+"`, `"TA+"`, `"TC"` and `"TC_mag"`.
- **`moments(order=1)` works (M5).**
- **`check_proposition3(num_points=...)` uses `num_points` (L1).**

### Input checks

- **Non-finite inputs raise `QFlexError` (M6):** NaN or inf in `x_data` or `y_data`, for all classes.
- **Clearer errors for 2-D input, non-integer `terms` and `terms=0` (L9).** Integer-valued floats such as `terms=5.0` are accepted.
- **One warning per infeasible `LogQFlex`/`LogitQFlex` fit (L4),** instead of two.

### Documentation

- **README (M2, M3, M7, L2):**
  - says which constraints guarantee validity;
  - uses the paper's theorem and proposition numbers;
  - explains γ = 0.45;
  - documents the analytical PDF for all classes;
  - gives install instructions from GitHub, since PyPI has only 1.0.0.
- **`compute_w1` (M4)** uses the trapezoid rule on the actual grid (correct on uneven grids), and its docstring states its range and that it normalises by the interdecile range.
- **pyproject comment fixed (L3).**
- **Docstrings, comments and both READMEs checked against the paper:** theorem, proposition, remark and equation numbers; coefficients numbered a₁ … a_K as in the paper; the corrected term table (K = 8, 9 add R₃, L₃); γ = 0.5(1 − SI) with 0.45 for symmetric assessments; `compute_w1` is the paper's normalised W1 (Eq. 25) on [0.01, 0.99].

### Not changed

- **`is_feasible` keeps the paper's definition:** a check on the 0.001 grid. Use `check_proposition3()` for the Theorem 5 tail condition.
- **The example script `scripts/fit_traction_case4_qpds.py`** produces identical output.
- **Publishing 1.2.0 to PyPI (M7)** has to be done by the maintainer.
