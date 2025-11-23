# Term-Structure Analysis (Nelson–Siegel & Svensson)

Project focus
- This project implements daily estimation and diagnostic analysis of the term structure of interest rates using two popular parametric families: Nelson–Siegel (NS) and Svensson (SV). The goal is to (1) fit parsimonious yield-curve models to market cross-sections of yields, (2) compute time series of fitted yields and model parameters, and (3) analyze spreads and structural changes versus reference series (the FED) and market data (TRACE).

Research questions implemented in `Codefinal.py`
- Question 1 — Fit NS and Svensson models day-by-day over a date window; produce fitted yield curves and store the time series of parameters and fitted yields.
- Question 2 — Compare fitted yields (sample) to FED yield data: compute spreads, produce summary statistics and distribution plots, and save comparison outputs.
- Question 3 — Identify and visualize major changes in the term structure across short-, medium- and long-term buckets (time series of averaged fitted yields by bucket).
- Question 4 — Analyze evolution of credit spreads (TRACE): compare high-yield vs investment-grade groups for the 2007–2010 period and visualize yearly trends.

Data expectations and layout
- `data/data.csv`: primary sample. The script expects a `Time` column and yield columns named like `SVENY01`, `SVENY02`, ..., `SVENY30` (1Y, 2Y, ..., 30Y). The code selects a date window and converts last-observed yields to per-maturity cross-sections.
- `data/feds200628.csv`: FED yield panel used in comparisons. Code selects maturity columns in a block (the script currently slices columns 68:98) — confirm these indices match the file you have.
- `data/trace.csv`: TRACE trade-level dataset used for credit spread analysis (Q4). The script expects columns `TRD_EXCTN_DT`, `YLD_SPREAD`, `RATING_1`, `YLD_PT`.

Modeling approach — brief
- Parametric families:
  - Nelson–Siegel (NS): y(τ) = β0 + β1 * ((1 − e^{−τ/λ})/(τ/λ)) + β2 * ( ((1 − e^{−τ/λ})/(τ/λ)) − e^{−τ/λ} )
  - Svensson (SV): extends NS with an extra hump term controlled by β3 and parameter `k`.
- Fitting procedure: for each date the script builds the cross-section (maturity, yield) and minimizes squared errors (sum of squared residuals) using `scipy.optimize.fmin` (Nelder–Mead) to find parameter vectors that minimize MSE.
- Output of fitting: time series of β-parameters (β0,β1,β2[,β3], λ[,k]) and fitted yields for the maturities used.

Outputs and how to read them
- `output/data_treated_ns.xlsx`: per-date NS fitted yields for each maturity and the estimated parameters columns (`β0, β1, β2, λ`). Use these to reconstruct fitted curves or analyze parameter dynamics (level / slope / curvature).
- `output/data_treated_svenson.xlsx`: per-date SV fitted yields and parameter columns (`β0, β1, β2, β3, λ, k`). The extra β3/k captures an additional hump in medium maturities.
- `output/data_spread.xlsx`: contains FED vs sample fitted yields, daily spreads, and parameters used to produce spread statistics and visualizations.

Interpretation notes (project-centric)
- β0 (level): long-run, roughly the long-term rate component; increases shift curve up uniformly.
- β1 (slope): mostly affects short end — positive β1 steepens the front end.
- β2 (curvature): creates hump/curvature at intermediate maturities.
- λ (decay): controls where the hump/slope transitions occur (scale for τ).
- For Svensson, β3 and k add a second hump/decay allowing more flexible mid-term shapes.

Assumptions & limitations
- Per-date cross-sections must be complete or the code must handle NA values — some parts of the script drop or implicitly assume no missing data.
- Optimization with `fmin` (Nelder–Mead) is simple and robust but not constrained: parameter sign restrictions or bounds are not enforced. This can produce non-intuitive parameter values; consider `scipy.optimize.minimize` with bounds for robustness.
- The script currently prints optimization progress (which slows batch runs) and uses `DataFrame.append()` inside loops — both are fine for small samples but slow for large time series.

Known issues (project-focused)
- Spurious token: an isolated `jo` appears in `Codefinal.py` and prevents execution — remove it.
- Some plotting blocks reference variables (`extended_maturities`, `predicted_yields`) that are not always defined. These plot calls should be guarded or removed for batch runs.
- Column index slicing for `feds200628.csv` (e.g., `iloc[:,68:98]`) depends on file layout; verify indices match your file version.

Project next steps (analysis ideas)
- Replace per-loop `append()` with list accumulation + `pd.concat()` for speed and memory efficiency.
- Add bounded optimization or parameter regularization to avoid pathological fits.
- Store fitted curve objects (or lambda closures) to allow easy interpolation/extrapolation between maturities.
- Add automated model diagnostics (residual plots, parameter stability statistics, AIC/BIC comparisons between NS and SV).

If you want me to proceed
- I can (A) remove the stray `jo` and add small guards to the plotting calls, or (B) produce a concise one-page French summary for inclusion in `Rapport/`. Tell me which.

---
This README focuses on the project itself: goals, data expectations, methods, outputs, interpretation and suggested next steps. If you want a shorter abstract or a French/LaTeX-ready version for the report, I can produce that next.
