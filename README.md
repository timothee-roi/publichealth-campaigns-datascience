# Media coverage and search interest during French health awareness campaigns

Code and data for the report *Media Coverage and Online Search Interest during French
Health Awareness Campaigns: A Granger Analysis Robust to the Campaign Calendar*, by
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
* validates every test on simulated systems with a known causal structure.

## Notebooks

| Notebook | Content | Report | Runs offline |
|---|---|---|---|
| [`01_query_building_and_data_collection`](notebooks/01_query_building_and_data_collection.ipynb) | GT snowball query discovery, MC article texts and k-grams, monthly series, daily campaign windows | §2 | no (APIs) |
| [`02_visualisation_and_peak_timing`](notebooks/02_visualisation_and_peak_timing.ipynb) | Daily windows, seasons aligned on the GT peak, lag between MC and GT peaks | Figs. 2–4, Table 1 | yes |
| [`03_granger_analysis`](notebooks/03_granger_analysis.ipynb) | Linear Granger causality with within-season lags: AIC lag selection, F-tests, LOYO, coefficients, jackknife intervals, impulse responses with bootstrap bands, sensitivity to the specification | Table 2, Figs. 5–10, 14–15, App. B–C | yes |
| [`04_nonlinear_kernel_granger`](notebooks/04_nonlinear_kernel_granger.ipynb) | Nonlinear Granger causality (KGC, lsNGC), season-swap surrogates, validation on 100 simulated datasets per scenario, jackknife intervals, robustness to the k-means initialisation | Table 3, Fig. 11, Figs. 16–22 (App. E) | yes* |
| [`05_report_figures`](notebooks/05_report_figures.ipynb) | Monthly GT vs MC series, snowball rounds, top k-grams, drawn from your data files | Figs. 1, 12, 13 | yes** |

\* A full run takes about 2–3 hours on two cores. Results are cached in
`results/cache/kernel_granger.joblib`, so the notebook reopens in seconds; set
`RECOMPUTE = True` to recompute everything.

\*\* Needs the monthly series, snowball rounds and k-grams in `data/raw/figures_input/`.
These files are produced by notebook 01 or by the original collection notebooks, and
are not included here. The notebook accepts both formats; see its first cell.

The shared code is in `src/phc/`:

* `config.py`: paths, campaigns, seasonal windows and queries;
* `collect.py`: Google Trends and MediaCloud collection, k-grams;
* `granger.py`: lag matrices, AIC grid, F-test, LOYO, coefficients, stationarity,
  season bootstrap, impulse responses;
* `kernel_granger.py`: KGC and lsNGC (used in notebook 04), KPCR and GPSIC (implemented,
  not used), within-season lag design, season-swap surrogates, jackknife;
* `simulate.py`: simulated seasonal systems with a known causal structure.

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

* **Linear Granger causality** (within-season lags): past searches help predict press
  coverage for Mars Bleu, Movember and Octobre Rose (jackknife intervals clearly above
  zero). Past coverage is significant for searches in three campaigns, but the gains are
  tiny, their intervals include zero for Dry January and Octobre Rose, and the impulse
  responses show that higher coverage is *not* followed by higher searches, except for
  Dry January.
* **Simulations** (100 datasets per scenario): a common driver with fixed timing makes the
  standard tests report spurious links in 39–100% of datasets; season-swap surrogates bring
  this down to 4–7%. A common driver whose timing changes from year to year still fools
  every test (32–94% with surrogates).
* **Against surrogates**, one link is robust: **Movember, MC → GT** with lsNGC
  (Holm-adjusted p = 0.040 within its family of eight tests, significant for all 10 k-means
  initialisations). Its effect is small: 2.7% of the variance left unexplained by past
  searches (jackknife 95% interval 0.9–4.4%), against 0.6% for surrogates.
  **Mars Bleu, GT → MC** is borderline (significant for 5 of 10 initialisations).
  **Octobre Rose** shows no link beyond its calendar.
* These are predictive links beyond a fixed calendar, not proof of a causal effect.

Full results: `results/tables/` and `results/figures/`.
