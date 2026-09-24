#!/usr/bin/env python3
"""
Reproduce Caban et al. (2024) Case 4 Metalog fits and compare QFlex.

The paper reports a bimodal PDF for Metalog with polynomial order k=4 on
Vehicle 4 cell capacities — even though a valid Metalog cannot be bimodal
at that order. This script:

1. Loads digitized cell capacities (Figure 15) and GeNIe quantile markers
2. Fits Metalog and QFlex at several truncation orders
3. Reports feasibility, mode count, and saves CDF/PDF comparison figures
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from qflex import ConstraintType, QFlex  # noqa: E402
from _metalog import Metalog  # noqa: E402

DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Paper-style GeNIe quantile scheme (Case 1 Table 3 / Case 4 yellow markers)
GENIE_P = np.array([0.05, 0.25, 0.50, 0.75, 0.95])


def load_capacities() -> np.ndarray:
    df = pd.read_csv(DATA / "traction_batteries_case4_capacities.csv", comment="#")
    return df["capacity_ah"].to_numpy(dtype=float)


def load_genie_quantiles() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "traction_batteries_case4_genie_quantiles.csv", comment="#")
    return df["capacity_ah"].to_numpy(dtype=float), df["probability"].to_numpy(dtype=float)


def empirical_quantiles(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.quantile(x, p, method="linear")


def quantile_density(qf, p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Signed Q'(p) via central differences (no abs — preserves invalidity)."""
    p = np.asarray(p, dtype=float)
    lo = np.clip(p - eps, 1e-10, 1 - 1e-10)
    hi = np.clip(p + eps, 1e-10, 1 - 1e-10)
    return (qf.quantile(hi) - qf.quantile(lo)) / (hi - lo)


def pdf_on_p(qf, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x(p), signed density f(x)=1/Q'(p), and Q'(p)."""
    x = qf.quantile(p)
    qp = quantile_density(qf, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.abs(qp) > 1e-14, 1.0 / qp, np.nan)
    return x, f, qp


def count_modes(x: np.ndarray, f: np.ndarray, qp: np.ndarray) -> int:
    """Count local maxima of positive density on the valid (Q'>0) support."""
    valid = np.isfinite(f) & (qp > 0) & (f > 0)
    if valid.sum() < 5:
        return 0
    xv, fv = x[valid], f[valid]
    # Sort by x in case Q folds
    order = np.argsort(xv)
    xv, fv = xv[order], fv[order]
    # Smooth lightly for robust peak finding on dense grids
    kernel = np.ones(7) / 7.0
    fs = np.convolve(fv, kernel, mode="same")
    peaks = []
    for i in range(1, len(fs) - 1):
        if fs[i] > fs[i - 1] and fs[i] >= fs[i + 1] and fs[i] > 0.05 * np.nanmax(fs):
            # enforce separation in x
            if not peaks or (xv[i] - xv[peaks[-1]]) > 0.25:
                peaks.append(i)
            elif fs[i] > fs[peaks[-1]]:
                peaks[-1] = i
    return len(peaks)


def delta_p_monotone(qf, delta: float = 0.001) -> bool:
    p = np.arange(delta, 1.0, delta)
    q = qf.quantile(p)
    return bool(np.all(np.diff(q) > 0))


def fit_pair(x: np.ndarray, p: np.ndarray, terms: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml = Metalog(x, p, terms=terms)
        qf = QFlex(x, p, terms=terms, constraint_type=ConstraintType.NONE)
        qf_a = QFlex(x, p, terms=terms, constraint_type=ConstraintType.A)
    return ml, qf, qf_a


def summarize(name: str, model, p_grid: np.ndarray) -> dict:
    x, f, qp = pdf_on_p(model, p_grid)
    feas = bool(getattr(model, "is_feasible", True))
    # QFlex stores feasibility differently — use delta-p and signed density
    mono = delta_p_monotone(model)
    modes = count_modes(x, f, qp)
    neg_frac = float(np.mean(qp <= 0))
    return {
        "model": name,
        "feasible_flag": feas,
        "delta_p_monotone": mono,
        "n_modes": modes,
        "frac_Qprime_nonpos": neg_frac,
        "mode_xs": _mode_locations(x, f, qp),
    }


def _mode_locations(x, f, qp, max_modes=5):
    valid = np.isfinite(f) & (qp > 0) & (f > 0)
    if valid.sum() < 5:
        return []
    xv, fv = x[valid], f[valid]
    order = np.argsort(xv)
    xv, fv = xv[order], fv[order]
    fs = np.convolve(fv, np.ones(7) / 7.0, mode="same")
    peaks = []
    for i in range(1, len(fs) - 1):
        if fs[i] > fs[i - 1] and fs[i] >= fs[i + 1] and fs[i] > 0.05 * np.nanmax(fs):
            if not peaks or (xv[i] - peaks[-1][0]) > 0.25:
                peaks.append((float(xv[i]), float(fs[i])))
            elif fs[i] > peaks[-1][1]:
                peaks[-1] = (float(xv[i]), float(fs[i]))
    peaks = sorted(peaks, key=lambda t: -t[1])[:max_modes]
    return [round(t[0], 3) for t in sorted(peaks)]


def plot_panel(x_fit, p_fit, sample, out_path: Path, title_suffix: str):
    terms_list = [3, 4, 5]
    p_grid = np.linspace(0.001, 0.999, 2000)
    fig, axes = plt.subplots(2, len(terms_list), figsize=(12.5, 7.2), sharex="row")

    colors = {"Metalog": "#c0392b", "QFlex": "#1f6f4a", "QFlex+A": "#2e86ab"}

    for j, K in enumerate(terms_list):
        ml, qf, qf_a = fit_pair(x_fit, p_fit, K)
        ax_cdf, ax_pdf = axes[0, j], axes[1, j]

        # Empirical / assessment markers
        ax_cdf.scatter(x_fit, p_fit, c="goldenrod", s=45, zorder=5, edgecolors="k", linewidths=0.5, label="assessments")
        if sample is not None:
            xs = np.sort(sample)
            # Weibull ECDF for reference
            pe = np.arange(1, len(xs) + 1) / (len(xs) + 1)
            ax_cdf.step(xs, pe, where="post", color="0.65", lw=1.0, label="sample ECDF")

        for name, model, ls in [
            ("Metalog", ml, "-"),
            ("QFlex", qf, "-"),
            ("QFlex+A", qf_a, "--"),
        ]:
            x, f, qp = pdf_on_p(model, p_grid)
            # CDF as p vs Q(p)
            ax_cdf.plot(model.quantile(p_grid), p_grid, color=colors[name], lw=1.8, ls=ls, label=name)

            # PDF vs x: only plot where Q' > 0; mark invalid regions lightly
            valid = qp > 0
            ax_pdf.plot(x[valid], f[valid], color=colors[name], lw=1.8, ls=ls, label=name)
            if np.any(~valid):
                ax_pdf.plot(x[~valid], np.abs(f[~valid]), color=colors[name], lw=1.0, ls=":", alpha=0.45)

        summ_ml = summarize(f"Metalog K={K}", ml, p_grid)
        summ_qf = summarize(f"QFlex K={K}", qf, p_grid)
        ax_cdf.set_title(
            f"K = {K}\n"
            f"Metalog modes={summ_ml['n_modes']}, mono={summ_ml['delta_p_monotone']}\n"
            f"QFlex modes={summ_qf['n_modes']}, mono={summ_qf['delta_p_monotone']}",
            fontsize=9,
        )
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_ylabel("CDF")
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_xlabel("Cell capacity [Ah]")
        ax_pdf.set_xlim(0, 4.5)
        ax_cdf.set_xlim(0, 4.5)
        if sample is not None:
            ax_pdf.hist(sample, bins=10, density=True, color="0.85", edgecolor="0.6", alpha=0.8, zorder=0)

        if j == 0:
            ax_cdf.legend(loc="lower right", fontsize=8)
            ax_pdf.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Vehicle 4 (Case 4) — Metalog vs QFlex {title_suffix}\n"
        "Paper claims Metalog K=4 is bimodal (modes ≈ 1.5 & 3.0 Ah)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    sample = load_capacities()
    x_genie, p_genie = load_genie_quantiles()
    x_emp = empirical_quantiles(sample, GENIE_P)

    print("Sample n =", len(sample))
    print("Sample mean/std =", sample.mean().round(4), sample.std(ddof=1).round(4))
    print("Digitized empirical quantiles @ GeNIe p:", np.round(x_emp, 3))
    print("GeNIe figure quantiles:                 ", np.round(x_genie, 3))

    rows = []
    p_grid = np.linspace(0.001, 0.999, 3000)

    for label, x_fit, p_fit in [
        ("genie_markers", x_genie, p_genie),
        ("empirical_quantiles", x_emp, GENIE_P),
        ("full_sample_weibull", np.sort(sample), np.arange(1, len(sample) + 1) / (len(sample) + 1)),
    ]:
        print(f"\n=== Fit source: {label} ===")
        for K in [3, 4, 5, 7]:
            if len(x_fit) < K:
                print(f"  skip K={K}: need {K} points, have {len(x_fit)}")
                continue
            ml, qf, qf_a = fit_pair(x_fit, p_fit, K)
            for name, model in [("Metalog", ml), ("QFlex", qf), ("QFlex+A", qf_a)]:
                s = summarize(name, model, p_grid)
                s.update({"source": label, "K": K})
                rows.append(s)
                print(
                    f"  {name:8s} K={K}: modes={s['n_modes']} @ {s['mode_xs']}, "
                    f"mono={s['delta_p_monotone']}, "
                    f"frac(Q'≤0)={s['frac_Qprime_nonpos']:.3f}"
                )

    plot_panel(
        x_genie,
        p_genie,
        sample,
        FIG / "traction_case4_genie_quantiles_metalog_qflex.png",
        "(GeNIe quantile markers from Figs. 16/18)",
    )
    plot_panel(
        x_emp,
        GENIE_P,
        sample,
        FIG / "traction_case4_empirical_quantiles_metalog_qflex.png",
        "(empirical quantiles from digitized Figure 15)",
    )

    # Focused K=3 vs K=4 PDF overlay matching the paper's claim
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    p_grid = np.linspace(0.001, 0.999, 3000)
    for ax, K in zip(axes, [3, 4]):
        ml, qf, _ = fit_pair(x_genie, p_genie, K)
        for name, model, color in [("Metalog", ml, "#c0392b"), ("QFlex", qf, "#1f6f4a")]:
            x, f, qp = pdf_on_p(model, p_grid)
            valid = qp > 0
            ax.plot(x[valid], f[valid], color=color, lw=2.0, label=name)
            if np.any(~valid):
                ax.plot(x[~valid], np.clip(f[~valid], -1, 5), color=color, lw=1.2, ls=":", alpha=0.5)
        ax.hist(sample, bins=10, density=True, color="0.88", edgecolor="0.55", alpha=0.9, zorder=0)
        ax.axvline(1.5, color="0.4", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(3.0, color="0.4", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlim(0, 4.5)
        ax.set_xlabel("Cell capacity [Ah]")
        ax.set_title(f"K = {K}")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("PDF")
    fig.suptitle(
        "Case 4 — paper reports Metalog K=4 bimodal at ≈1.5 and ≈3.0 Ah (dashed)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG / "traction_case4_k3_k4_pdf_focus.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")

    summary = pd.DataFrame(rows)
    out_csv = DATA / "traction_case4_fit_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
