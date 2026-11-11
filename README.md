# Media coverage and search interest during French health awareness campaigns

Code and data for the report *Media Coverage and Online Search Interest during French
Health Awareness Campaigns: Granger Causality and Transfer Functions beyond the Campaign
Calendar*, by
Timothée Roina, supervised by Marie-Laure Charpignon.

The project studies four recurring French awareness campaigns (Octobre Rose, Movember,
Mars Bleu and Dry January) over eleven seasons (2015–2025). Google Trends (GT) serves as
a proxy for public information-seeking and MediaCloud (MC) article volume as a measure of
media salience. The analysis:

* compares the timing of media and search peaks;
* tests linear and nonlinear (kernel) Granger causality in both directions on daily
  campaign windows, with lags kept within each season;
* tests every link against season-swap surrogates, because the campaign calendar drives
  both series;
* estimates transfer functions on calendar-adjusted series (delay, gain, persistence of
  the response of one series to the other), separating the same-day association from
  lagged responses;
* validates every procedure on simulated systems with a known truth.

## Notebooks

| Notebook | Content | Report | Runs offline |
|---|---|---|---|
| [`01_query_building_and_data_collection`](notebooks/01_query_building_and_data_collection.ipynb) | GT snowball query discovery, MC article texts and k-grams, monthly series, daily campaign windows | §2 | no (APIs) |
| [`02_visualisation_and_peak_timing`](notebooks/02_visualisation_and_peak_timing.ipynb) | Daily windows, seasons aligned on the GT peak, lag between MC and GT peaks | §3.1, §4.1 | yes |
| [`03_granger_analysis`](notebooks/03_granger_analysis.ipynb) | Linear Granger causality with within-season lags: AIC lag selection, F-tests, LOYO, coefficients, jackknife intervals, impulse responses with bootstrap bands, sensitivity to the specification, season fixed effects | §4.3–4.4, App. C–D | yes |
| [`04_nonlinear_kernel_granger`](notebooks/04_nonlinear_kernel_granger.ipynb) | Nonlinear Granger causality (KGC, lsNGC), season-swap surrogates, validation on 100 simulated datasets per scenario, jackknife intervals, robustness to the k-means initialisation, five calendar adjustments (weekly cycle, season level, campaign profile) | §3.4, §4.5–4.6, App. F–G | yes* |
| [`05_report_figures`](notebooks/05_report_figures.ipynb) | Monthly GT vs MC series, snowball rounds, top k-grams, drawn from your data files | Figs. 1, 12, 13 | yes*** |
| [`06_transfer_functions`](notebooks/06_transfer_functions.ipynb) | Calendar removal, pre-whitened cross-correlations, transfer functions (distributed lag, delay + decay), jackknife intervals, season-swap tests, validation on simulated systems, coherence and phase | §3.5, §4.7, App. E | yes** |

\* A full run takes about 6 hours on two cores (simulations, 999-surrogate tests under five
calendar adjustments). Results are cached in `results/cache/kernel_granger.joblib`, so the
notebook reopens in seconds; set `RECOMPUTE = True` to recompute everything.

\*\* A full run takes about 1.5 hours on two cores (100 simulated datasets × 4 scenarios, 999
surrogates per real test); results are cached in `results/cache/transfer.joblib`.

\*\*\* Needs the monthly series, snowball rounds and k-grams in `data/raw/figures_input/`.
These files are produced by notebook 01 or by the original collection notebooks, and
are not included here. The notebook accepts both formats; see its first cell.

The shared code is in `src/phc/`:

* `config.py`: paths, campaigns, seasonal windows and queries;
* `collect.py`: Google Trends and MediaCloud collection, k-grams;
* `granger.py`: lag matrices, AIC grid, F-test, LOYO, coefficients, stationarity,
  season bootstrap, impulse responses;
* `kernel_granger.py`: KGC and lsNGC (used in notebook 04), KPCR and GPSIC (implemented,
  not used), within-season lag design, season-swap surrogates, jackknife;
* `transfer.py`: calendar removal, pre-whitening and cross-correlation, transfer-function
  fits (distributed lag, delay + decay), residual checks, jackknife, season-swap tests,
  cross-spectrum;
* `simulate.py`: simulated seasonal systems with a known causal structure or transfer
  function.

## Getting started

```bash
git clone https://github.com/timothee-roi/publichealth-campaigns-datascience
cd publichealth-campaigns-datascience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Notebooks 02, 03 and 04 run as is on the included data. For notebook 01, get a
MediaCloud API key at <https://search.mediacloud.org> and set it before starting Jupyter
(`export MEDIACLOUD_API_KEY="..."`), then turn on the sections you need in the `RUN`
dictionary. Google Trends is accessed with `pytrends`, an unofficial client that Google
rate-limits heavily, so collection is slow.

## Data

`data/processed/daily/` contains the daily series used in the report: one window per
campaign and year, 2015–2025, 11 seasons per campaign.

| Campaign | Window | Days per season |
|---|---|---|
| Octobre Rose | 1 Sep – 30 Nov | 91 |
| Movember | 1 Oct – 31 Dec | 92 |
| Mars Bleu | 2 Feb – 30 Apr | 88–89 |
| Dry January | 26 Dec – 6 Feb | 43 |

* `GT_<code>_Granger.csv`: Google Trends daily interest (`campaign_index`, 0–100,
  100 = busiest day of the season). The query is stored in the `query` column.
* `MC_<code>_Granger_normalized.csv`: MediaCloud daily article counts divided by their
  seasonal maximum, × 100 (`n_articles`). These files do not record the query used.

Codes: `OR` Octobre Rose, `Mov` Movember, `MB` Mars Bleu, `DJ` Dry January.

## Main results

Numbers refer to the report (`report/main.tex`); all tables are in `results/tables/`.

* **Standard Granger tests find links almost everywhere** (7 of 8 campaign × direction
  pairs), but simulations show that a shared campaign calendar alone produces this: with a
  common driver of fixed timing the standard tests report spurious links in 39–100% of
  100 datasets, season-swap surrogates in 4–7%. A common driver whose timing changes from
  year to year still fools every test (32–94% with surrogates).
* **Linear Granger** (within-season lags): past searches predict press coverage for Mars
  Bleu, Movember and Octobre Rose; past coverage is significant for searches in three
  campaigns with tiny gains. Season fixed effects change nothing.
* **Nonlinear, against surrogates**: one link is stable to the weekly cycle and to season
  levels, **Movember, coverage → searches** (lsNGC, Holm-adjusted p = 0.008 with 999
  surrogates; 2.7% of the variance left by past searches, 1.8% once the weekly cycle is
  removed, against 0.6% for surrogates). It **vanishes when the average campaign profile is
  subtracted**, and the transfer function of the adjusted series shows no lagged response,
  so it is not a day-to-day influence. Mars Bleu searches → coverage, significant in
  some variants, is not robust. Octobre Rose shows no link in any variant.
* **Transfer functions** (calendar, weekday and season level removed): searches and
  coverage are associated **on the same day** for Movember, Mars Bleu and Dry January
  (positive, delay 0, half-life under 0.6 day), not at all for Octobre Rose. A delayed
  response is not established for any campaign: Movember searches → coverage is
  borderline, Dry January coverage → searches (0.60 search point per coverage point over 14
  days, half-life 2.6 days) is not confirmed by the second test and rests on 43 days per
  season. The procedure is validated on simulated systems (known transfer function
  recovered without bias) and has the same blind spot as the surrogate tests: it cannot
  exclude a common driver whose date changes from year to year.
* These are predictive associations beyond a fixed calendar, not evidence of causal effects.

Full results: `results/tables/` and `results/figures/`.

## Reproducing the report

`report/main.tex` and `report/figures/` compile with `pdflatex`. Figures 1, 12 and 13 of
the report are still crops of the original slides; notebook 05 redraws them from your
data files.
