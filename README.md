# Public health campaigns, media coverage and search interest

Code for the project *Measuring the Impact of Public Health Campaigns on Public
Attention: Google Trends and MediaCloud*, by Timothée Roina, supervised by
Marie-Laure Charpignon.

The project studies four recurring French awareness campaigns (Octobre Rose,
Movember, Mars Bleu and Dry January). It uses Google Trends (GT) as a proxy for
public interest and MediaCloud (MC) as a measure of press coverage. It compares the
timing of media and search peaks, then tests linear and nonlinear (kernel) Granger
causality between the two series on daily windows around each yearly campaign.

## Notebooks

| Notebook | Content | Runs offline |
|---|---|---|
| [`01_query_building_and_data_collection`](notebooks/01_query_building_and_data_collection.ipynb) | GT snowball query discovery, MC article texts and k-grams, monthly series 2015–2025, daily seasonal windows | no (APIs) |
| [`02_visualisation_and_peak_timing`](notebooks/02_visualisation_and_peak_timing.ipynb) | Figures 1–3, 9–10 and Table 1 (lag between MC and GT peaks) | yes* |
| [`03_granger_analysis`](notebooks/03_granger_analysis.ipynb) | Stationarity tests, AIC lag selection, Granger F-tests, LOYO robustness, coefficients, sensitivity analyses (Table 2, Figures 4–8, 11–12) | yes |
| [`04_nonlinear_kernel_granger`](notebooks/04_nonlinear_kernel_granger.ipynb) | Nonlinear Granger causality (KGC with kernels, lsNGC), season-swap surrogate tests, validation on simulated data | yes** |

\* Figures 1, 9 and 10 need the outputs of notebook 01; the other outputs use the
daily data included in `data/processed/daily/`.

\*\* A full run takes about 30 minutes on two cores. Results are cached in
`results/cache/kernel_granger.joblib`, so the notebook reopens in seconds; set
`RECOMPUTE = True` to recompute everything.

The shared code is in `src/phc/`:

* `config.py`: paths, campaigns, seasonal windows and queries;
* `collect.py`: Google Trends and MediaCloud collection, k-grams;
* `granger.py`: lag matrices, AIC grid, F-test, LOYO, coefficients, stationarity;
* `kernel_granger.py`: KGC and lsNGC (used in notebook 04), KPCR and GPSIC (implemented,
  not used), within-season lag design, season-swap surrogates;
* `simulate.py`: simulated seasonal systems with a known causal structure.

## Getting started

```bash
git clone https://github.com/timothee-roi/publichealth-campaigns-datascience
cd publichealth-campaigns-datascience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Notebooks 02, 03 and 04 run as is. For notebook 01, get a MediaCloud API key at
<https://search.mediacloud.org> and set it before starting Jupyter
(`export MEDIACLOUD_API_KEY="..."`). Then turn on the sections you need in the `RUN`
dictionary. Google Trends is accessed with `pytrends`, an unofficial client that Google
rate-limits heavily, so collection is slow.

## Data

`data/processed/daily/` contains the daily series used in the paper: one window per
campaign and year, 2015–2025, 11 seasons per campaign.

| Campaign | Window | Days per season |
|---|---|---|
| Octobre Rose | 1 Sep – 30 Nov | 91 |
| Movember | 1 Oct – 31 Dec | 92 |
| Mars Bleu | 2 Feb – 30 Apr | 88–89 |
| Dry January | 26 Dec – 6 Feb | 43 |

* `GT_<code>_Granger.csv`: Google Trends daily interest (`campaign_index`, 0–100,
  100 = maximum day of the season). The query used is stored in the `query` column.
* `MC_<code>_Granger_normalized.csv`: MediaCloud daily article counts divided by their
  seasonal maximum, × 100 (`n_articles`).

Codes: `OR` Octobre Rose, `Mov` Movember, `MB` Mars Bleu, `DJ` Dry January.

## What the re-analysis confirms and corrects

Notebook 03 recomputes the paper's results from the included data.

* **Confirmed:** the selected lags, the adjusted R² of reduced and full models, and the
  LOYO counts of Table 2. Table 1 (peak lags) is also confirmed exactly.
* **To correct in the paper:** two p-values of Table 2 are much smaller than printed.
  Movember GT → MC is 3.6 × 10⁻⁸, not 1.0 × 10⁻⁴. Octobre Rose GT → MC is
  4.9 × 10⁻¹¹, not 3.0 × 10⁻⁴. `statsmodels` gives the same values. Both tests were
  already significant, so no conclusion changes.
* **Queries:** the Google Trends queries stored in the daily files (and in `config.py`)
  differ from those listed in Appendix A of the paper for the four campaigns. For
  example, Octobre Rose uses "Octobre rose + cancer du sein + mammographie + dépistage
  cancer du sein + cancer du sein symptôme". Appendix A should match the files. The
  MediaCloud files do not record their query; `config.py` uses the MC terms of
  Appendix A.
* **Sensitivity analyses (new):**
  * When every lag pair is fitted on the same days, no selected lag order stays at the
    bound of 20, and all tests remain significant.
  * When lags are also kept within a single season, Dry January GT → MC (p = 0.27) and
    Mars Bleu MC → GT (p = 0.42) are no longer significant. Octobre Rose and Movember
    remain significant in both directions under every variant.

## Nonlinear Granger causality (notebook 04)

The methods are the two constraint-based approaches reviewed and unified by Murphy &
Benavoli (2026), *Constraint- and Score-Based Nonlinear Granger Causality Discovery with
Kernels*, arXiv:2601.09579:

* **KGC**, kernel Granger causality (Marinazzo, Pellicoro & Stramaglia, 2008), with a
  squared-exponential kernel, and with a linear kernel as a reference;
* **lsNGC**, large-scale nonlinear Granger causality (Wismüller et al., 2021).

These tests assume that no unobserved variable drives both series, but the campaign
calendar does. The notebook therefore adds season-swap surrogates (the cause series of
each season is replaced by that of another season) and keeps lags within a season. It
validates every test on simulated data with a known truth before applying it.

**Results.**

* On simulated data, KGC and lsNGC detect a purely nonlinear link that linear Granger
  causality misses (12/12 vs 1–2/12 datasets).
* A common driver with fixed timing makes the paper tests report false links in 50–100%
  of datasets; the surrogate tests remove them. A common driver whose timing varies by year
  fools every test.
* On the campaign data, the paper tests find a link for 7 of 8 campaign × direction
  pairs. Against the surrogates, only **Movember, MC → GT** survives correction for
  multiple testing (lsNGC, Holm-adjusted p = 0.040, stable across lag orders). The effect
  is small: 2.7% of the remaining variance, against 0.6% for surrogates.
* **Octobre Rose** shows no link beyond the calendar, although its paper tests are the most
  significant.
* These are predictive links beyond a fixed calendar, not proof of a causal effect.
