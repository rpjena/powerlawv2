# Workforce Scaling with AUM in Hedge Funds

**Paper:** Workforce Scaling with Assets Under Management in Hedge Funds: Evidence for Two Organisational Regimes  
**Author:** Rudra Jena, CFM LLP  

---

## Repository structure

```
powerlawhf/
├── pareto_scaling_hedgefunds.tex   # Main paper (arXiv-style)
├── backtest_section.tex            # §VI Out-of-sample validation
├── survivorship_section.tex        # §VII Survivorship bias analysis
├── refs.bib                        # Bibliography
│
├── generate_all_figures.py         # Standalone: regenerates all 12 figures
│
├── data/
│   ├── insample_funds.csv          # 15 funds, 92 obs (2005–2024)
│   ├── oos_funds.csv               # 26 funds, 119 obs (OOS hold-out)
│   └── strategy_metadata.csv       # Strategy codes, colours, markers
│
├── figures/                        # All figures as PDF + PNG
│   ├── fig1_loglog.{pdf,png}
│   ├── fig2_alpha.{pdf,png}
│   ...
│   └── fig12_combined_loglog.{pdf,png}
│
└── notebooks/
    ├── notebook1_data_exploration.ipynb
    ├── notebook2_model_fitting.ipynb
    └── notebook3_clustering_oos_figures.ipynb
```

---

## Quick start

### 1. Regenerate all figures

```bash
pip install numpy pandas matplotlib scipy scikit-learn
python generate_all_figures.py                    # B=5000 (paper quality, ~5 min)
python generate_all_figures.py --bootstrap 500    # B=500  (fast, ~30s)
python generate_all_figures.py --fig 1 2 5        # specific figures only
python generate_all_figures.py --outdir /tmp/figs # custom output directory
```

### 2. Recompile the paper

```bash
pdflatex pareto_scaling_hedgefunds
bibtex   pareto_scaling_hedgefunds
pdflatex pareto_scaling_hedgefunds
pdflatex pareto_scaling_hedgefunds
```

### 3. Run notebooks

Open in JupyterLab or Google Colab. Each notebook auto-detects whether
it is running from the `powerlawhf/` root or the `notebooks/` subdirectory.

---

## Model

The core model is:

$$N_s = C \cdot \mathcal{A}^{\alpha}$$

Estimated by OLS in log-log space:

```
# Pseudocode
x = log(AUM), y = log(headcount)
alpha = cov(x, y) / var(x)         # power-law exponent (labour-AUM elasticity)
C     = exp(mean(y) - alpha*mean(x))
SE    = HC1_robust_SE(x, y)
CI    = bootstrap(B=5000, inject Gaussian noise at sigma_meas)
```

**Key results:**
- Quant funds: α ∈ [0.18, 0.43]
- Pod-shop funds: α ∈ [0.90, 1.47]
- Mann-Whitney U = 0, p = 0.005 (n_Q=4, n_P=5, B=5000)
- Permutation p = 0.003 (10,000 shuffles)

---

## OOS evaluation protocols (revised, referee Q1)

| Protocol | Description | MAE (log-units) |
|----------|-------------|-----------------|
| A | Regime classification from pre-specified strategy | 79% accuracy |
| B | Headcount prediction from in-sample group means (no refitting) | 0.29 |
| C | Time-split (train early half, test late half; n≥6 funds) | 0.19 |

The previously-quoted MAE = 0.11 uses OOS-estimated fund-specific parameters —
it measures within-OOS-fund goodness of fit, not genuine transfer.

---

## Data sources

- **AUM**: SEC Form ADV Part 1A, Item 5.D (RAUM — regulatory AUM, gross advisory assets)
  — *not* Form 13F (equity holdings only)
- **Headcount**: SEC ADV Part 1A, Item 5.B (total employees)
- **Anchor**: Man Group plc Annual Reports (LSE:EMG) — audited, <2% uncertainty

---

## Robustness checks implemented (revised paper)

| Check | Result |
|-------|--------|
| Classification sensitivity (4 schemes) | p < 0.05 in all |
| AQR post-peak only | α = 0.51, still below all pod exponents |
| Panel regression + clustered SE | p = 0.006 (G=14 clusters) |
| Bayesian hierarchical model | P(δ > 0.5 \| data) = 0.963 |
| SIMEX at 15% AUM noise | MW significant in >98% of MC runs |
| Deming regression (λ=2.25) | Quant [0.15,0.40] vs pod [0.80,1.30] — non-overlapping |
| Chow structural break test | Significant in 2/15 funds; mature-phase re-est unchanged |

---

## Dependencies

```
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.4
scipy >= 1.7
scikit-learn >= 0.24
```

Install: `pip install numpy pandas matplotlib scipy scikit-learn`

For LaTeX: TeX Live 2020+ with packages `amsmath, booktabs, natbib, microtype, hyperref`.
