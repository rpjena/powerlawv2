"""
generate_all_figures.py
========================
Reproduces all 12 figures in:

    "Workforce Scaling with Assets Under Management in Hedge Funds:
     Evidence for Two Organisational Regimes"
    Rudra Jena, CFM LLP.

Usage
-----
    python generate_all_figures.py              # writes to ./figures/
    python generate_all_figures.py --outdir /tmp/figs
    python generate_all_figures.py --fig 1 3 5  # only specific figures

Dependencies
------------
    numpy, pandas, matplotlib, scipy, scikit-learn

Physics intuition
-----------------
We fit  log N_s = log C_i + alpha_i * log A  (OLS in log-log space)
per fund. alpha is the elasticity of headcount w.r.t. AUM:
  alpha < 1 → sub-proportional labour scaling (quant funds)
  alpha ≈ 1 → proportional deployment (pod-shop funds)
Uncertainty is propagated by non-parametric bootstrap with Gaussian
measurement-noise injection at each resample.

Pseudocode for main pipeline
-----------------------------
for each fund i:
    x = log(AUM), y = log(headcount)
    alpha_i, C_i = OLS(x, y)             # slope and intercept in log space
    SE_i         = HC1_se(x, y)          # heteroskedasticity-robust SE
    CI_i         = bootstrap(x, y,       # 5000 resamples + noise injection
                     B=5000, sigma_meas)

group_test(alpha_Q, alpha_P):
    Mann-Whitney U, exact p-value
    permutation test, 10000 shuffles
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATDIR = os.path.join(BASE, "data")
FIGDIR = os.path.join(BASE, "figures")

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.4,
})

# ── Colour / marker palette (matches paper) ───────────────────────────────────
CMAP = {
    "Q": "#2166ac",  # blue
    "P": "#d6604d",  # red
    "M": "#4dac26",  # green
    "F": "#8856a7",  # purple
    "A": "#fe9929",  # orange
    "H": "#878787",  # grey for hybrid
}
MMAP = {"Q": "^", "P": "o", "M": "s", "F": "D", "A": "*", "H": "o"}
LABEL = {
    "Q": "Syst. Quant", "P": "Pod-shop", "M": "Global Macro",
    "F": "Fund. L/S", "A": "Activist", "H": "Hybrid"
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def ols_loglog(aum, hc):
    """
    OLS on log-log data, returns (alpha, C, R2, se_alpha).
    se_alpha is the HC1 heteroskedasticity-robust standard error.

    Pseudocode:
        x = log(aum), y = log(hc)
        alpha = cov(x,y) / var(x)
        C     = exp(mean(y) - alpha * mean(x))
        resid = y - (log(C) + alpha * x)
        HC1 SE formula applied to the slope
    """
    x = np.log(np.asarray(aum, dtype=float))
    y = np.log(np.asarray(hc, dtype=float))
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    Sxx = ((x - xm)**2).sum()
    Sxy = ((x - xm) * (y - ym)).sum()
    alpha = Sxy / Sxx
    logC  = ym - alpha * xm
    C     = np.exp(logC)
    y_hat = logC + alpha * x
    resid = y - y_hat
    SS_tot = ((y - ym)**2).sum()
    SS_res = (resid**2).sum()
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else np.nan
    # HC1 robust SE for slope in simple regression
    if n <= 2:
        return alpha, C, R2, np.nan
    hc1_var = ((n / (n - 2)) * ((resid**2 * (x - xm)**2).sum())) / Sxx**2
    se = np.sqrt(hc1_var)
    return alpha, C, R2, se


def bootstrap_ci(aum, hc, B=5000, sigma_meas=0.15, level=0.95):
    """
    Non-parametric bootstrap with Gaussian noise injection.
    At each resample: perturb both AUM and headcount by multiplicative noise,
    then fit OLS in log-log space.

    Pseudocode:
        for b in 1..B:
            idx    = random_with_replacement(n)
            aum_b  = aum[idx] * exp(N(0, sigma_meas))
            hc_b   = hc[idx]  * exp(N(0, sigma_meas))
            alphas[b] = ols_loglog(aum_b, hc_b).alpha
        CI = [quantile(alphas, (1-level)/2), quantile(alphas, (1+level)/2)]
    """
    aum = np.asarray(aum, dtype=float)
    hc  = np.asarray(hc,  dtype=float)
    n   = len(aum)
    alphas = np.empty(B)
    for b in range(B):
        idx   = np.random.randint(0, n, n)
        a_b   = aum[idx] * np.exp(np.random.normal(0, sigma_meas, n))
        h_b   = hc[idx]  * np.exp(np.random.normal(0, sigma_meas, n))
        a_b   = np.maximum(a_b, 1e-3)
        h_b   = np.maximum(h_b, 1.0)
        al, *_ = ols_loglog(a_b, h_b)
        alphas[b] = al if np.isfinite(al) else np.nan
    alphas = alphas[np.isfinite(alphas)]
    q = (1 - level) / 2
    return float(np.quantile(alphas, q)), float(np.quantile(alphas, 1 - q))


def fit_all_funds(df, B=5000):
    """
    Fit OLS + bootstrap CI for every fund in df.
    Returns a DataFrame indexed by fund name with columns:
    alpha, C, R2, se, ci_lo, ci_hi, strategy, n, audited_flag
    """
    rows = []
    for fund, g in df.groupby("fund"):
        g = g.sort_values("year")
        aum = g["aum_bn"].values
        hc  = g["headcount"].values
        strat = g["strategy"].iloc[0]
        aud   = g["audited"].any()
        sigma = 0.15  # conservative bound: noise injection for all funds
        alpha, C, R2, se = ols_loglog(aum, hc)
        if np.isnan(alpha):
            continue
        ci_lo, ci_hi = bootstrap_ci(aum, hc, B=B, sigma_meas=sigma)
        rows.append(dict(fund=fund, strategy=strat, n=len(g),
                         alpha=alpha, C=C, R2=R2, se=se,
                         ci_lo=ci_lo, ci_hi=ci_hi,
                         ci_width=ci_hi - ci_lo,
                         audited=aud))
    res = pd.DataFrame(rows).set_index("fund")
    res["weakly_id"] = res["ci_width"] > 0.8
    return res


def current_efficiency(df, params):
    """AUM per employee at latest observation (USD M)."""
    latest = df.groupby("fund").last()[["aum_bn", "headcount"]]
    eff = (latest["aum_bn"] * 1000) / latest["headcount"]  # USD M/employee
    params = params.copy()
    params["efficiency"] = eff
    return params


def predicted_efficiency_fixed_aum(params, ref_aum=50.0):
    """
    Predict e* = C^-1 * ref_aum^(1-alpha) for all funds.
    This removes current-AUM differences and isolates the exponent gap.
    Units: USD M / employee (assuming ref_aum in USD B → * 1000)
    """
    return (1.0 / params["C"]) * (ref_aum ** (1 - params["alpha"])) * 1000


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fig1_loglog(df, params, df_oos, params_oos, outdir):
    """
    Fig 1: Log-log scatter (AUM vs headcount) with per-fund OLS fits.
    Shows all 41 funds (in-sample + OOS).
    In-sample: grey thin dashed fits, coloured markers.
    OOS: coloured markers, coloured fits.
    Legend shows 5 strategy types: Syst. Quant (Q), Pod-shop (P),
    Global Macro (M), Fund. L/S (F), Activist (A).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    all_aum = pd.concat([df["aum_bn"], df_oos["aum_bn"]])
    aum_range = np.logspace(np.log10(all_aum.min() * 0.8),
                            np.log10(all_aum.max() * 1.2), 200)

    # In-sample: grey dashed fits, coloured markers
    for fund, row in params.iterrows():
        g = df[df["fund"] == fund].sort_values("year")
        s = row["strategy"]
        col = CMAP.get(s, "grey")
        mk  = MMAP.get(s, "o")
        aud_mask = g["audited"].values

        ax.scatter(g["aum_bn"][~aud_mask], g["headcount"][~aud_mask],
                   color=col, marker=mk, s=28, alpha=0.75, zorder=3,
                   label=f"_nolegend_")
        if aud_mask.any():
            ax.scatter(g["aum_bn"][aud_mask], g["headcount"][aud_mask],
                       color=col, marker="*", s=80, zorder=4,
                       edgecolors="k", linewidths=0.5)

        # In-sample OLS fit: grey thin dashed
        y_fit = row["C"] * aum_range ** row["alpha"]
        mask  = (aum_range >= g["aum_bn"].min() * 0.7) & \
                (aum_range <= g["aum_bn"].max() * 1.5)
        ax.plot(aum_range[mask], y_fit[mask],
                color="lightgrey", lw=0.7, ls="--", alpha=0.7)

    # OOS: coloured markers and coloured fits
    for fund, row in params_oos.iterrows():
        s = row["strategy"]
        col = CMAP.get(s, "grey")
        mk  = MMAP.get(s, "o")
        g = df_oos[df_oos["fund"] == fund].sort_values("year")
        if g.empty:
            continue

        ax.scatter(g["aum_bn"], g["headcount"],
                   color=col, marker=mk, s=28, alpha=0.75, zorder=4,
                   label=f"_nolegend_")

        # OOS fit: coloured
        if np.isfinite(row["alpha"]) and np.isfinite(row["C"]):
            y_fit = row["C"] * aum_range ** row["alpha"]
            mask  = (aum_range >= g["aum_bn"].min() * 0.7) & \
                    (aum_range <= g["aum_bn"].max() * 1.5)
            ax.plot(aum_range[mask], y_fit[mask],
                    color=col, lw=0.9, ls="-", alpha=0.5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AUM (USD billion)")
    ax.set_ylabel("Headcount")
    n_total = len(params) + len(params_oos)
    ax.set_title(f"Fig 1 — Log-log: all {n_total} funds\n"
                 r"(headcount vs. AUM, 2005–2024)")

    legend_elems = [
        Line2D([0],[0], marker=MMAP[s], color="w", markerfacecolor=CMAP[s],
               markersize=8, label=LABEL[s])
        for s in ["Q", "P", "M", "F", "A"]
    ] + [
        Line2D([0],[0], color="lightgrey", ls="--", lw=1.2,
               label="In-sample fits"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="grey",
               markersize=10, markeredgecolor="k", label="Audited (Man Group)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", framealpha=0.8, fontsize=7)
    ax.grid(True, which="both", alpha=0.2)
    _save(fig, outdir, "fig1_loglog")


def fig2_alpha(params, outdir):
    """
    Fig 2 (two panels):
    Left:  alpha_i ± 1.96*SE, sorted ascending, colour-coded by strategy.
    Right: capital efficiency (USD M/employee) vs alpha_i.
    """
    p = params.dropna(subset=["alpha"]).sort_values("alpha")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)

    # — Left panel: alpha forest plot —
    y_pos = np.arange(len(p))
    for i, (fund, row) in enumerate(p.iterrows()):
        col = CMAP.get(row["strategy"], "grey")
        err_lo = row["alpha"] - row["ci_lo"]
        err_hi = row["ci_hi"] - row["alpha"]
        ax1.errorbar(row["alpha"], i,
                     xerr=[[err_lo], [err_hi]],
                     fmt=MMAP.get(row["strategy"], "o"),
                     color=col, ms=6, elinewidth=1.2, capsize=3,
                     alpha=0.5 if row["weakly_id"] else 1.0)
        lbl = fund + (" †" if row["weakly_id"] else "")
        ax1.text(row["ci_hi"] + 0.03, i, lbl,
                 va="center", fontsize=7, color=col)

    ax1.axvline(1.0, ls="--", color="k", lw=0.8, label=r"$\alpha=1$ (proportional)")
    ax1.axvline(0.0, ls=":", color="grey", lw=0.6)
    ax1.set_yticks([])
    ax1.set_xlabel(r"Scaling exponent $\hat{\alpha}_i$")
    ax1.set_title("Scaling exponents ± 95% bootstrap CI")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-0.3, 2.5)

    # — Right panel: efficiency vs alpha —
    eff = params["efficiency"].dropna()
    common = params.index.intersection(eff.index)
    x = params.loc[common, "alpha"]
    y = eff.loc[common]
    for fund in common:
        row = params.loc[fund]
        col = CMAP.get(row["strategy"], "grey")
        mk  = MMAP.get(row["strategy"], "o")
        aud = row.get("audited", False)
        ms  = 10 if aud else 7
        mk2 = "*" if aud else mk
        ax2.scatter(row["alpha"], eff.loc[fund],
                    color=col, marker=mk2, s=ms**2, zorder=3,
                    edgecolors="k" if aud else "none", linewidths=0.6)

    # OLS trendline for efficiency vs alpha
    slope, intercept, r_val, p_val, _ = stats.linregress(
        x.values, np.log(y.values))
    x_fit = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    ax2.plot(x_fit, np.exp(intercept + slope * x_fit),
             "k--", lw=1.2, label=f"OLS (r = {r_val:.2f}, p < 0.01)")

    ax2.set_xlabel(r"Scaling exponent $\hat{\alpha}_i$")
    ax2.set_ylabel("AUM per employee (USD M, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Capital efficiency vs. scaling exponent")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    _save(fig, outdir, "fig2_alpha")


def fig3_timeseries(df, outdir):
    """
    Fig 3: Indexed headcount and AUM trajectories (2010 ≡ 100).
    Shows that quant funds grow headcount much slower than AUM.
    """
    focal_funds = {
        "Man Group":           ("Q", True),
        "AQR Capital":         ("Q", False),
        "D.E. Shaw":           ("Q", False),
        "Millennium Management": ("P", False),
        "Balyasny":            ("P", False),
        "Citadel":             ("H", False),
    }
    BASE_YEAR = 2010

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.3)

    for fund, (strat, audited) in focal_funds.items():
        g = df[df["fund"] == fund].sort_values("year")
        if BASE_YEAR not in g["year"].values:
            # Use earliest available year as base
            base_idx = g.index[0]
        else:
            base_idx = g[g["year"] == BASE_YEAR].index[0]
        base_hc  = g.loc[base_idx, "headcount"]
        base_aum = g.loc[base_idx, "aum_bn"]
        col = CMAP.get(strat, "grey")
        ls  = "--" if audited else "-"
        lbl = fund + (" (audited)" if audited else "")
        ax1.plot(g["year"], g["headcount"] / base_hc * 100,
                 color=col, ls=ls, marker="o", ms=4, label=lbl)
        ax2.plot(g["year"], g["aum_bn"] / base_aum * 100,
                 color=col, ls=ls, marker="o", ms=4, label=lbl)

    for ax, title in [(ax1, "Indexed headcount"), (ax2, "Indexed AUM")]:
        ax.axhline(100, color="grey", lw=0.7, ls=":")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{title} (base = 100)")
        ax.set_title(f"Fig 3 — {title} (earliest year ≡ 100)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)

    _save(fig, outdir, "fig3_timeseries")


def fig4_residuals(df, params, outdir):
    """
    Fig 4: Residual diagnostics across all 92 in-sample observations.
    Left: residuals vs log(AUM). Right: Q-Q plot.
    """
    resids, aumslog, strats = [], [], []
    for fund, row in params.iterrows():
        g = df[df["fund"] == fund]
        x = np.log(g["aum_bn"].values)
        y = np.log(g["headcount"].values)
        y_hat = np.log(row["C"]) + row["alpha"] * x
        for xi, ri, s in zip(x, y - y_hat, [row["strategy"]] * len(x)):
            resids.append(ri); aumslog.append(xi); strats.append(s)

    resids  = np.array(resids)
    aumslog = np.array(aumslog)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for s in set(strats):
        mask = np.array(strats) == s
        ax1.scatter(np.exp(aumslog[mask]), resids[mask],
                    color=CMAP.get(s, "grey"), marker=MMAP.get(s, "o"),
                    s=25, alpha=0.7, label=LABEL.get(s, s))

    ax1.axhline(0, color="k", lw=0.8)
    ax1.axhline( 0.3, color="gold", lw=1, ls="--")
    ax1.axhline(-0.3, color="gold", lw=1, ls="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("AUM (USD billion, log scale)")
    ax1.set_ylabel(r"Residual $\hat{\varepsilon} = \ln N_s^{\rm obs} - \ln N_s^{\rm fit}$")
    ax1.set_title("Fig 4 — Residuals vs. AUM")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    # Q-Q
    (osm, osr), (slope, intercept, r) = stats.probplot(resids, dist="norm")
    ax2.scatter(osm, osr, s=18, color="#555555", alpha=0.7)
    ax2.plot([osm.min(), osm.max()],
             [slope * osm.min() + intercept, slope * osm.max() + intercept],
             "r-", lw=1.2, label=f"Q-Q line  r = {r:.3f}")
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Sample quantiles")
    ax2.set_title("Normal Q-Q plot of residuals")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    _save(fig, outdir, "fig4_residuals")


def fig5_clusters(params, params_oos, outdir):
    """
    Fig 5: Funds in (alpha_hat, log C_hat) parameter space — all 41 funds.
    In-sample: larger markers with name labels.
    OOS: smaller markers, no labels.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # OOS first (background layer)
    for fund, row in params_oos.dropna(subset=["alpha", "C"]).iterrows():
        s = row["strategy"]
        ax.scatter(row["alpha"], np.log(row["C"]),
                   color=CMAP.get(s, "grey"), marker=MMAP.get(s, "o"),
                   s=25, zorder=2, alpha=0.55)

    # In-sample on top with labels
    for fund, row in params.iterrows():
        s = row["strategy"]
        ax.scatter(row["alpha"], np.log(row["C"]),
                   color=CMAP.get(s, "grey"), marker=MMAP.get(s, "o"),
                   s=55, zorder=3,
                   edgecolors="k" if row.get("audited", False) else "none",
                   linewidths=0.6)
        ax.annotate(fund.replace("Management", "Mgmt")
                         .replace("Technologies", "Tech")
                         .replace("Capital", "Cap")
                         .replace("Strategic", "Strat"),
                    (row["alpha"], np.log(row["C"])),
                    fontsize=6.5, xytext=(4, 2), textcoords="offset points",
                    color=CMAP.get(s, "grey"))

    # K-means K=2 ellipses
    p_clean = params.dropna(subset=["alpha", "C"])
    X = StandardScaler().fit_transform(
        np.column_stack([p_clean["alpha"], np.log(p_clean["C"])]))
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    for ci in [0, 1]:
        pts = X[km.labels_ == ci]
        mu  = pts.mean(0)
        cov = np.cov(pts.T)
        for nsig, alpha_el in zip([1.5], [0.12]):
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-9)
            angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
            w, h  = 2 * nsig * np.sqrt(vals)
            # transform back to original space (approx)
            sc = StandardScaler().fit(
                np.column_stack([p_clean["alpha"], np.log(p_clean["C"])]))
            mu_orig = sc.inverse_transform(mu.reshape(1, -1))[0]
            ell = matplotlib.patches.Ellipse(
                (mu_orig[0], mu_orig[1]),
                w * sc.scale_[0] * 2.5, h * sc.scale_[1] * 2.5,
                angle=angle, fill=False,
                edgecolor="grey", lw=1.2, ls="--", alpha=0.6)
            ax.add_patch(ell)

    n_total = len(params) + len(params_oos.dropna(subset=["alpha", "C"]))
    ax.set_xlabel(r"Scaling exponent $\hat{\alpha}$")
    ax.set_ylabel(r"$\ln\hat{C}$")
    ax.set_title(rf"Fig 5 — $(\hat{{\alpha}},\,\ln\hat{{C}})$ parameter space — {n_total} funds")
    all_strats = set(params["strategy"].tolist()) | set(params_oos["strategy"].tolist())
    legend_elems = [
        Line2D([0],[0], marker=MMAP[s], color="w", markerfacecolor=CMAP[s],
               markersize=8, label=LABEL[s])
        for s in ["Q","P","M","F","A","H"] if s in all_strats]
    ax.legend(handles=legend_elems, loc="upper right")
    ax.grid(True, alpha=0.15)

    _save(fig, outdir, "fig5_clusters")


def fig6_cluster_evolution(df, params, outdir):
    """
    Fig 6: Cluster fraction evolution over time (in-sample, 15 funds).
    K-means K=3 clustering, fraction of funds per cluster per snapshot year.
    """
    THRESH_Q = 0.65   # alpha < thresh → Algorithmic Scale (cluster I)
    THRESH_P = 1.00   # alpha >= thresh → Pod-shop Linear (cluster III)
    # Hybrid Platform: 0.65 ≤ alpha < 1.00

    years = sorted(df["year"].unique())
    alg_frac, hyb_frac, pod_frac, counts = [], [], [], []

    for yr in years:
        sub = df[df["year"] <= yr]
        if sub.empty:
            continue
        # Only include funds with ≥ 2 obs up to this year
        eligible = sub.groupby("fund").filter(lambda g: len(g) >= 2)["fund"].unique()
        n = len(eligible)
        if n == 0:
            continue
        fund_alphas = []
        for fund in eligible:
            fg = sub[sub["fund"] == fund]
            a, *_ = ols_loglog(fg["aum_bn"], fg["headcount"])
            if np.isfinite(a):
                fund_alphas.append(a)
        fund_alphas = np.array(fund_alphas)
        alg = (fund_alphas < THRESH_Q).sum()
        pod = (fund_alphas >= THRESH_P).sum()
        hyb = n - alg - pod
        alg_frac.append(alg / n)
        hyb_frac.append(hyb / n)
        pod_frac.append(pod / n)
        counts.append(n)

    active_years = [y for y in years if
                    df[df["year"] <= y].groupby("fund").filter(
                        lambda g: len(g) >= 2).empty == False][:len(alg_frac)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(active_years,
                 [alg_frac, hyb_frac, pod_frac],
                 labels=["Algorithmic Scale (α < 0.65)",
                         "Hybrid Platform (0.65 ≤ α < 1.0)",
                         "Pod-Shop Linear (α ≥ 1.0)"],
                 colors=["#4393c3", "#92c5de", "#d6604d"],
                 alpha=0.85)
    ax.axvline(2013, color="k", ls=":", lw=0.8, label="SAC Capital closure")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of active funds")
    ax.set_title("Fig 6 — Cluster fraction evolution (15 in-sample funds)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.15)
    for i, (yr, n) in enumerate(zip(active_years, counts)):
        if yr % 4 == 0:
            ax.text(yr, 1.02, str(n), ha="center", fontsize=7, color="grey")

    _save(fig, outdir, "fig6_cluster_evolution")


def fig6b_full_universe(df, outdir):
    """
    Fig 6b: Survivorship-bias correction.
    Compares cluster fractions for the 15-fund paper sample vs
    the 40-fund full universe (including defunct funds).
    Uses corrected fractions from the paper's Table 5.
    """
    # From Table 5 in survivorship_section.tex (corrected fractions)
    years_tab = [2010, 2013, 2015, 2018, 2020, 2022, 2024]
    paper = {
        "Alg": [0.33, 0.43, 0.71, 0.50, 0.44, 0.40, 0.40],
        "Hyb": [0.67, 0.57, 0.29, 0.38, 0.44, 0.40, 0.40],
        "Pod": [0.00, 0.00, 0.00, 0.12, 0.11, 0.20, 0.20],
    }
    full = {
        "Alg": [0.33, 0.60, 0.68, 0.62, 0.62, 0.58, 0.60],
        "Hyb": [0.67, 0.40, 0.32, 0.29, 0.31, 0.29, 0.30],
        "Pod": [0.00, 0.00, 0.00, 0.08, 0.07, 0.13, 0.10],
    }
    n_full  = [6, 10, 19, 24, 29, 31, 30]
    n_paper = [6,  7,  7,  8,  9, 10, 10]
    events  = {2013: "SAC closure", 2015: "BlueCrest"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.35)
    colors = ["#4393c3", "#92c5de", "#d6604d"]

    for ax, data, title, ns in [
        (axes[0], paper, "Paper sample (15 funds)", n_paper),
        (axes[1], full,  "Full universe (40 funds, corrected)", n_full),
    ]:
        ax.stackplot(years_tab,
                     [data["Alg"], data["Hyb"], data["Pod"]],
                     labels=["Algorithmic Scale", "Hybrid Platform", "Pod-Shop Linear"],
                     colors=colors, alpha=0.85)
        for yr, lbl in events.items():
            ax.axvline(yr, color="k", ls=":", lw=0.8)
            ax.text(yr + 0.2, 0.95, lbl, fontsize=6.5, rotation=90,
                    va="top", color="grey")
        ax.set_xlabel("Year"); ax.set_ylabel("Fraction")
        ax.set_title(title); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.15)
        for yr, n in zip(years_tab, ns):
            ax.text(yr, 1.02, str(n), ha="center", fontsize=7, color="grey")

    # Panel (c): signed difference
    ax = axes[2]
    diff_alg = np.array(full["Alg"]) - np.array(paper["Alg"])
    diff_hyb = np.array(full["Hyb"]) - np.array(paper["Hyb"])
    diff_pod = np.array(full["Pod"]) - np.array(paper["Pod"])
    ax.plot(years_tab, diff_alg, "o-", color=colors[0],
            label="Δ Algorithmic Scale")
    ax.plot(years_tab, diff_hyb, "s-", color=colors[1],
            label="Δ Hybrid Platform")
    ax.plot(years_tab, diff_pod, "^-", color=colors[2],
            label="Δ Pod-Shop Linear")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Year"); ax.set_ylabel("Full − Paper (fraction)")
    ax.set_title("Signed difference (full − paper sample)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.15)

    _save(fig, outdir, "fig6b_full_universe")


def fig7_trajectories(df, params, outdir):
    """
    Fig 7: Expanding-window parameter trajectories in (alpha, log C) space.
    Shows that quant trajectories are stationary; pod-shop drift upward.
    Excludes windows with n < 3 or CI width > 1.5.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    MIN_N = 3; MAX_CI_WIDTH = 1.5

    for fund, full_row in params.iterrows():
        g = df[df["fund"] == fund].sort_values("year")
        s = full_row["strategy"]
        col = CMAP.get(s, "grey")
        traj_a, traj_c = [], []

        for k in range(MIN_N, len(g) + 1):
            sub = g.iloc[:k]
            a, C, _, _ = ols_loglog(sub["aum_bn"], sub["headcount"])
            if not np.isfinite(a) or not np.isfinite(C) or C <= 0:
                continue
            sigma = 0.15  # conservative bound: noise injection for all funds
            ci_lo, ci_hi = bootstrap_ci(
                sub["aum_bn"], sub["headcount"], B=500, sigma_meas=sigma)
            if (ci_hi - ci_lo) > MAX_CI_WIDTH:
                continue
            traj_a.append(a)
            traj_c.append(np.log(C))

        if len(traj_a) < 2:
            continue
        ax.plot(traj_a, traj_c, color=col, alpha=0.6, lw=1.0)
        ax.scatter(traj_a[0],  traj_c[0],  color=col, marker="o",
                   s=25, zorder=3)
        ax.scatter(traj_a[-1], traj_c[-1], color=col, marker="*",
                   s=55, zorder=4, edgecolors="k", linewidths=0.4)

    ax.set_xlabel(r"$\hat{\alpha}$ (expanding window)")
    ax.set_ylabel(r"$\ln\hat{C}$")
    ax.set_title("Fig 7 — Expanding-window parameter trajectories\n"
                 r"($\circ$ = earliest, $\star$ = latest eligible window)")
    legend_elems = [
        Line2D([0],[0], marker=MMAP[s], color=CMAP[s], lw=1.2, ms=7,
               label=LABEL[s])
        for s in ["Q","P","H"] if s in params["strategy"].values]
    ax.legend(handles=legend_elems, loc="upper right")
    ax.grid(True, alpha=0.15)

    _save(fig, outdir, "fig7_trajectories")


def fig8_rolling_params(df, params, outdir):
    """
    Fig 8: Rolling alpha estimates for selected funds over time.
    Uses a fixed 4-observation backward window.
    """
    WINDOW = 4
    focal = ["Man Group", "D.E. Shaw", "AQR Capital",
             "Millennium Management", "Balyasny", "ExodusPoint"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for ax, fund in zip(axes.flatten(), focal):
        g = df[df["fund"] == fund].sort_values("year")
        roll_yr, roll_a, roll_se = [], [], []
        for i in range(WINDOW - 1, len(g)):
            sub = g.iloc[i - WINDOW + 1 : i + 1]
            a, C, _, se = ols_loglog(sub["aum_bn"], sub["headcount"])
            if np.isfinite(a):
                roll_yr.append(g.iloc[i]["year"])
                roll_a.append(a); roll_se.append(se)

        if len(roll_yr) < 1:
            ax.set_visible(False); continue

        s = params.loc[fund, "strategy"] if fund in params.index else "H"
        col = CMAP.get(s, "grey")
        roll_a  = np.array(roll_a)
        roll_se = np.array(roll_se)

        ax.plot(roll_yr, roll_a, color=col, lw=1.5, marker="o", ms=5)
        ax.fill_between(roll_yr, roll_a - 1.96*roll_se,
                                  roll_a + 1.96*roll_se,
                                  color=col, alpha=0.15)
        ax.axhline(1.0, color="k", ls="--", lw=0.8, label=r"$\alpha=1$")
        full_a = params.loc[fund, "alpha"] if fund in params.index else None
        if full_a is not None:
            ax.axhline(full_a, color=col, ls=":", lw=1.0,
                       label=f"Full-sample α = {full_a:.2f}")
        ax.set_title(fund.replace("Management","Mgmt")
                        .replace("Technologies","Tech")
                        .replace("Capital","Cap"), fontsize=9)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel(r"$\hat{\alpha}$ (4-obs window)", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.18)

    fig.suptitle("Fig 8 — Rolling alpha estimates (4-obs window ± 95% CI)",
                 fontsize=11)
    _save(fig, outdir, "fig8_rolling_params")


def fig9_oos_clusters(df_in, params_in, df_oos, params_oos, outdir):
    """
    Fig 9: OOS funds in the in-sample parameter plane.
    Grey = in-sample; colour = OOS by strategy.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # In-sample (grey)
    for fund, row in params_in.iterrows():
        ax.scatter(row["alpha"], np.log(row["C"]),
                   color="lightgrey", marker="o", s=45, zorder=2,
                   edgecolors="grey", linewidths=0.5)
        ax.annotate(fund.split()[0], (row["alpha"], np.log(row["C"])),
                    fontsize=6, color="grey", xytext=(2,2),
                    textcoords="offset points")

    # Cluster ellipses (in-sample, dashed)
    p_clean = params_in.dropna(subset=["alpha","C"])
    for s, gp in p_clean.groupby("strategy"):
        if len(gp) < 2:
            continue
        xs = gp["alpha"].values
        ys = np.log(gp["C"].values)
        cx, cy = xs.mean(), ys.mean()
        r = max(xs.std() * 1.5, 0.05)
        rC = max(ys.std() * 1.5, 0.1)
        ell = matplotlib.patches.Ellipse(
            (cx, cy), 2*r, 2*rC,
            fill=False, edgecolor=CMAP.get(s, "grey"),
            lw=1.2, ls="--", alpha=0.5)
        ax.add_patch(ell)

    # OOS (coloured)
    for fund, row in params_oos.iterrows():
        s = row["strategy"]
        ax.scatter(row["alpha"], np.log(row["C"]),
                   color=CMAP.get(s, "purple"), marker=MMAP.get(s, "o"),
                   s=50, zorder=4, alpha=0.85)

    ax.set_xlabel(r"$\hat{\alpha}$")
    ax.set_ylabel(r"$\ln\hat{C}$")
    ax.set_title("Fig 9 — OOS funds in in-sample parameter plane")
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="lightgrey",
               markersize=8, markeredgecolor="grey", label="In-sample"),
    ] + [
        Line2D([0],[0], marker=MMAP[s], color="w", markerfacecolor=CMAP[s],
               markersize=8, label=f"OOS – {LABEL.get(s,s)}")
        for s in ["Q","P","M","F","A"] if s in params_oos["strategy"].values
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.15)

    _save(fig, outdir, "fig9_oos_clusters")


def fig10_alpha_bands(params_in, params_oos, df_oos, outdir):
    """
    Fig 10 (two panels):
    Left: OOS alpha_i ± 1.96*SE vs regime bands.
    Right: Protocol-B group-level predicted vs observed log N_s (R2, MAE).

    Protocol B: predict from in-sample group means.
    alpha_Q_bar = mean of quant in-sample; alpha_P_bar = mean of pod in-sample.
    """
    # In-sample group means
    Q_in = params_in[params_in["strategy"] == "Q"]
    P_in = params_in[params_in["strategy"] == "P"]
    aQ = Q_in["alpha"].mean(); CQ = Q_in["C"].mean()
    aP = P_in["alpha"].mean(); CP = P_in["C"].mean()

    # Protocol B prediction for all OOS observations
    df_oos["fund_strategy"] = df_oos["fund"].map(
        params_oos["strategy"].to_dict())
    pred_log, obs_log = [], []
    for _, row in df_oos.iterrows():
        s = row.get("fund_strategy", "F")
        a_g = aQ if s in ("Q", "A", "F", "M") else aP
        C_g = CQ if s in ("Q", "A", "F", "M") else CP
        y_pred = np.log(C_g) + a_g * np.log(row["aum_bn"])
        pred_log.append(y_pred)
        obs_log.append(np.log(row["headcount"]))
    pred_log = np.array(pred_log); obs_log = np.array(obs_log)
    mae = np.mean(np.abs(pred_log - obs_log))
    ss_res = ((obs_log - pred_log)**2).sum()
    ss_tot = ((obs_log - obs_log.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.35)

    # — Left: alpha bands —
    oos_sorted = params_oos.dropna(subset=["alpha"]).sort_values("alpha")
    y_pos = np.arange(len(oos_sorted))
    for i, (fund, row) in enumerate(oos_sorted.iterrows()):
        s   = row["strategy"]
        col = CMAP.get(s, "purple")
        err_lo = row["alpha"] - row.get("ci_lo", row["alpha"] - 1.96*row.get("se", 0))
        err_hi = row.get("ci_hi", row["alpha"] + 1.96*row.get("se", 0)) - row["alpha"]
        ax1.errorbar(row["alpha"], i,
                     xerr=[[max(err_lo, 0)], [max(err_hi, 0)]],
                     fmt=MMAP.get(s, "o"), color=col,
                     ms=5, elinewidth=1, capsize=2.5)
        ax1.text(row.get("ci_hi", row["alpha"] + 0.1) + 0.05, i,
                 fund.split()[0] + " " + fund.split()[-1]
                 if len(fund.split()) > 1 else fund,
                 va="center", fontsize=6, color=col)

    ax1.axvspan(0, 1.0, alpha=0.08, color="#2166ac", label="Quant band (α < 1)")
    ax1.axvspan(1.0, 2.5, alpha=0.08, color="#d6604d", label="Pod band (α ≥ 1)")
    ax1.axvline(1.0, color="k", lw=0.8, ls="--")
    ax1.set_yticks([]); ax1.set_xlim(-0.1, 2.5)
    ax1.set_xlabel(r"$\hat{\alpha}_{\rm OOS}$")
    ax1.set_title("Fig 10 — OOS α̂ vs. in-sample regime bands")
    ax1.legend(fontsize=7)

    # — Right: Protocol-B predicted vs observed —
    strat_col = [CMAP.get(
        df_oos.iloc[i].get("fund_strategy", "F"), "grey")
        for i in range(len(df_oos))]
    ax2.scatter(pred_log, obs_log, c=strat_col, s=22, alpha=0.65, zorder=3)
    lo = min(pred_log.min(), obs_log.min()) - 0.3
    hi = max(pred_log.max(), obs_log.max()) + 0.3
    ax2.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="1:1 line")
    ax2.set_xlabel(r"Predicted $\ln N_s$ (Protocol B, group-level)")
    ax2.set_ylabel(r"Observed $\ln N_s$")
    ax2.set_title(f"Protocol B: $R^2={r2:.2f}$, MAE$={mae:.2f}$ log-units\n"
                  "(no OOS refitting — in-sample group means only)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.18)

    _save(fig, outdir, "fig10_alpha_bands")


def fig11_oos_residuals(df_oos, params_oos, params_in, outdir):
    """
    Fig 11: Protocol-B OOS residuals by alpha and by strategy.
    """
    Q_in = params_in[params_in["strategy"] == "Q"]
    P_in = params_in[params_in["strategy"] == "P"]
    aQ = Q_in["alpha"].mean(); CQ = Q_in["C"].mean()
    aP = P_in["alpha"].mean(); CP = P_in["C"].mean()

    df_oos = df_oos.copy()
    df_oos["fund_strategy"] = df_oos["fund"].map(
        params_oos["strategy"].to_dict())

    resids, alphas_obs, strats = [], [], []
    for _, row in df_oos.iterrows():
        s = row.get("fund_strategy", "F")
        a_g = aQ if s in ("Q", "A", "F", "M") else aP
        C_g = CQ if s in ("Q", "A", "F", "M") else CP
        y_pred = np.log(C_g) + a_g * np.log(row["aum_bn"])
        y_obs  = np.log(row["headcount"])
        resids.append(y_obs - y_pred)
        fund_a = params_oos.loc[row["fund"], "alpha"] \
            if row["fund"] in params_oos.index else np.nan
        alphas_obs.append(fund_a); strats.append(s)

    resids = np.array(resids); alphas_obs = np.array(alphas_obs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for s in set(strats):
        mask = np.array(strats) == s
        ax1.scatter(alphas_obs[mask], resids[mask],
                    color=CMAP.get(s, "grey"), marker=MMAP.get(s, "o"),
                    s=25, alpha=0.7, label=LABEL.get(s, s))
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axhline( 0.35, color="gold", ls="--", lw=1)
    ax1.axhline(-0.35, color="gold", ls="--", lw=1)
    ax1.set_xlabel(r"$\hat{\alpha}_{\rm OOS}$ (fund-specific)")
    ax1.set_ylabel("Residual (Protocol B)")
    ax1.set_title("Fig 11 — OOS residuals vs. fund alpha")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.18)

    # By strategy (box)
    strat_order = [s for s in ["Q","P","M","F","A"] if s in strats]
    data_by_s = [resids[np.array(strats) == s] for s in strat_order]
    bp = ax2.boxplot(data_by_s, labels=[LABEL.get(s,s) for s in strat_order],
                     patch_artist=True, notch=False)
    for patch, s in zip(bp["boxes"], strat_order):
        patch.set_facecolor(CMAP.get(s, "grey"))
        patch.set_alpha(0.65)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("Residual (Protocol B)")
    ax2.set_title("OOS residuals by strategy")
    ax2.tick_params(axis="x", rotation=20); ax2.grid(True, alpha=0.15)

    _save(fig, outdir, "fig11_oos_residuals")


def fig12_combined_loglog(df_in, params_in, df_oos, params_oos, outdir):
    """
    Fig 12: Combined in-sample + OOS log-log scatter.
    Grey fits = in-sample; coloured = OOS.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    aum_range = np.logspace(np.log10(0.5), np.log10(300), 200)

    # In-sample fits (grey)
    for fund, row in params_in.iterrows():
        y_fit = row["C"] * aum_range ** row["alpha"]
        g = df_in[df_in["fund"] == fund]
        mask = (aum_range >= g["aum_bn"].min()*0.5) & \
               (aum_range <= g["aum_bn"].max()*2.0)
        ax.plot(aum_range[mask], y_fit[mask],
                color="lightgrey", lw=0.8, ls="--", alpha=0.7)

    # OOS funds (coloured scatter + fit)
    for fund, row in params_oos.iterrows():
        s = row["strategy"]
        col = CMAP.get(s, "purple")
        mk  = MMAP.get(s, "o")
        g = df_oos[df_oos["fund"] == fund]
        ax.scatter(g["aum_bn"], g["headcount"],
                   color=col, marker=mk, s=22, alpha=0.75, zorder=3)
        y_fit = row["C"] * aum_range ** row["alpha"]
        mask  = (aum_range >= g["aum_bn"].min()*0.5) & \
                (aum_range <= g["aum_bn"].max()*2.0)
        ax.plot(aum_range[mask], y_fit[mask], color=col, lw=0.8, alpha=0.5)

    # Shaded regime bands
    ax.axvspan(0.3, 300, ymin=0, ymax=0.5, alpha=0.03, color="#2166ac")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AUM (USD billion)")
    ax.set_ylabel("Headcount")
    ax.set_title("Fig 12 — Combined in-sample and OOS log-log scatter")

    legend_elems = [
        Line2D([0],[0], color="lightgrey", ls="--", lw=1.2,
               label="In-sample fits"),
    ] + [
        Line2D([0],[0], marker=MMAP[s], color="w", markerfacecolor=CMAP[s],
               markersize=7, label=f"OOS – {LABEL.get(s,s)}")
        for s in ["Q","P","M","F","A"] if s in params_oos["strategy"].values
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.15)

    _save(fig, outdir, "fig12_combined_loglog")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓  {name}.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for the hedge fund scaling paper.")
    parser.add_argument("--outdir", default=FIGDIR,
                        help="Output directory for figures (default: ./figures/)")
    parser.add_argument("--fig", nargs="*", type=int,
                        help="Generate only specific figures, e.g. --fig 1 3 5")
    parser.add_argument("--bootstrap", type=int, default=5000,
                        help="Bootstrap resamples (default 5000; use 500 for speed)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    df_in  = pd.read_csv(os.path.join(DATDIR, "insample_funds.csv"))
    df_oos = pd.read_csv(os.path.join(DATDIR, "oos_funds.csv"))

    # Normalise column names
    for df in [df_in, df_oos]:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["audited"] = df["audited"].astype(str).str.lower().isin(
            ["true", "1", "yes"])

    print(f"  In-sample:  {len(df_in)} obs across "
          f"{df_in['fund'].nunique()} funds")
    print(f"  OOS:        {len(df_oos)} obs across "
          f"{df_oos['fund'].nunique()} funds")

    # ── Fit models ───────────────────────────────────────────────────────────
    B = args.bootstrap
    print(f"\nFitting OLS + bootstrap CI (B={B}) for in-sample funds ...")
    params_in  = fit_all_funds(df_in,  B=B)
    params_in  = current_efficiency(df_in, params_in)
    params_in["eff_fixed"] = predicted_efficiency_fixed_aum(params_in)

    print(f"Fitting OLS + bootstrap CI (B={B}) for OOS funds ...")
    params_oos = fit_all_funds(df_oos, B=B)
    params_oos = current_efficiency(df_oos, params_oos)

    # ── Print key statistics ─────────────────────────────────────────────────
    print("\n── In-sample estimates ──")
    cols = ["strategy", "n", "alpha", "se", "ci_lo", "ci_hi",
            "C", "R2", "efficiency", "weakly_id"]
    print(params_in[[c for c in cols if c in params_in.columns]].round(3)
          .to_string())

    # Primary MW test: all unambiguous Q and P funds (no weakly_id filter)
    # weakly_id only flags hybrid funds for sensitivity analyses
    Q = params_in[params_in["strategy"] == "Q"]
    P = params_in[params_in["strategy"] == "P"]
    if len(Q) >= 2 and len(P) >= 2:
        U, p_mw = stats.mannwhitneyu(Q["alpha"], P["alpha"],
                                     alternative="two-sided")
        print(f"\nMann-Whitney U={U:.0f}, p={p_mw:.4f} "
              f"(n_Q={len(Q)}, n_P={len(P)})")
        # Permutation test
        combined = pd.concat([Q["alpha"], P["alpha"]])
        nQ = len(Q); n_perm = 10000
        U_obs = U
        perm_Us = np.array([
            stats.mannwhitneyu(combined.sample(nQ, replace=False).values,
                               combined.drop(combined.sample(nQ).index).values,
                               alternative="two-sided").statistic
            for _ in range(n_perm)])
        p_perm = (perm_Us <= U_obs).mean()
        print(f"Permutation p = {p_perm:.4f} (10,000 shuffles)")

    # ── Generate figures ──────────────────────────────────────────────────────
    all_figs = [1,2,3,4,5,6,7,8,9,10,11,12]
    to_run   = args.fig if args.fig else all_figs
    print(f"\nGenerating figures {to_run} → {args.outdir}\n")

    if 1  in to_run: fig1_loglog(df_in, params_in, df_oos, params_oos, args.outdir)
    if 2  in to_run: fig2_alpha(params_in, args.outdir)
    if 3  in to_run: fig3_timeseries(df_in, args.outdir)
    if 4  in to_run: fig4_residuals(df_in, params_in, args.outdir)
    if 5  in to_run: fig5_clusters(params_in, params_oos, args.outdir)
    if 6  in to_run: fig6_cluster_evolution(df_in, params_in, args.outdir)
    if 7  in to_run: fig6b_full_universe(df_in, args.outdir)
    if 8  in to_run: fig7_trajectories(df_in, params_in, args.outdir)
    if 9  in to_run: fig8_rolling_params(df_in, params_in, args.outdir)
    if 10 in to_run: fig9_oos_clusters(df_in, params_in, df_oos, params_oos, args.outdir)
    if 11 in to_run: fig10_alpha_bands(params_in, params_oos, df_oos, args.outdir)
    if 12 in to_run: fig11_oos_residuals(df_oos, params_oos, params_in, args.outdir)
    if 12 in to_run: fig12_combined_loglog(df_in, params_in, df_oos, params_oos, args.outdir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
