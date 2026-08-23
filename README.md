# Forecasting oil demand with XGBoost

A reproduction of the XGBoost half of *Forecasting Global Oil Demand: Application of
Machine Learning Techniques* (Aldabbagh, Economou & Christou, Oxford Institute for Energy
Studies, November 2024), built on free public data.

## The question

Can a gradient-boosted tree, fed lagged demand and macro predictors, forecast monthly oil
demand twelve months ahead better than a simple time-series benchmark?

Seven refined products across seven regions, forecast from December 2018 over 2019 — the
paper's own out-of-sample window, chosen so that COVID touches neither the training sample
nor the score.

| 2019, demand-weighted MAPE | US | China |
|---|---|---|
| the paper's XGBoost (its Table 3) | 4.95 | 10.83 |
| seasonal naive, no model at all | 3.15 | 12.87 |
| this model | **3.07** | **8.36** |
| this model, post-processed | **2.95** | **8.27** |

The asymmetry is the result worth reporting. Where demand is a stable seasonal series with
a flat trend, last year's same month is already close to unbeatable and the model adds
almost nothing; where the level is moving, as in China, it adds a third.

## Method

```
JODI, FRED, OECD, World Bank, CPB        download.py  ->  data/<source>/*.csv
        |
        |  sql/demand.sql   filter, join to regions, aggregate, in place
        |  prepare.py       stamp each observation with its publication date,
        |                   merge_asof(direction="backward") onto each month end
        v
   demand, predictors, panel              data.py      ->  data/oil.duckdb
        |
        |  X = demand(t-2) ... demand(t-13) + 19 drivers as of t
        |  y = demand(t + h),  one model per horizon h
        |  expanding-window CV, purged, inside the training sample
        v
   XGBoost forecast, evaluated on 2019     model.py
```

### Data

The paper runs on IEA MODS for demand and Oxford Economics for the predictors, both paid.
Every source here is the closest free public substitute; none needs an API key.

| Paper variable | Source | Frequency |
|---|---|---|
| demand, 7 products | JODI-Oil, ~110 countries | monthly |
| rpo, wci | Brent and the IMF non-fuel index, via FRED | monthly |
| ind, cpi, cbui, pcars | OECD Key Economic Indicators | monthly |
| gdp, inv, cons, out, inc | OECD Quarterly National Accounts | quarterly |
| ind, cpi, ppi, chem, inc, pop | FRED, for the US | monthly |
| pop, gdp, inv, airp, egen | World Bank WDI | annual |
| wtr | CPB World Trade Monitor | monthly |
| rpk, airf | US revenue passenger miles and air fares, via FRED | monthly |

Where two sources carry the same variable for a region they disagree on units and
definitions, so exactly one wins: FRED where it reaches, then the OECD panel, then the
World Bank annual, which is the only source that covers non-OECD countries at all.

### Point-in-time alignment

The part that decides whether the experiment means anything. Every predictor observation
carries two dates — the end of the period it describes, and the day it was published.
A national account for 2019-Q1 describes January to March but does not exist until late
April, so it must not appear in a February row.

`prepare.py` stamps each observation with `available_at` and joins the drivers onto each
month end with `merge_asof(direction="backward")`: as of the end of month *t*, take the
most recent value already released. A two-year `tolerance` goes with it, so a series that
has stopped publishing goes missing rather than being carried forward forever.

The same rule applies to the target's own lags. JODI publishes a month about two months
after it closes, so the newest demand reading available at *t* is demand(t-2), and the lag
block starts there.

**This is a deliberate deviation from the paper**, which interpolates its quarterly
predictors to monthly. Linear interpolation reads the *next* quarter's number to fill the
months in between — a number nobody had. Notebook 01 plots the two through 2008: the
interpolated series starts falling months before the fall was reported.

### Why XGBoost

The paper's argument, and it holds: the predictors are tabular, the relationships between
macro aggregates and fuel demand are non-linear and interacting, the sample is small, and
gradient-boosted trees handle missing values natively — which matters here, because China
has no free monthly production index and several columns are simply absent for some regions.

*XGBoost is the algorithm.* `XGBRegressor` is XGBoost's own scikit-learn-compatible
estimator, so scikit-learn is an interface dependency rather than the model — it is in
`requirements.txt` because `from xgboost import XGBRegressor` fails without it, and there
is deliberately no `import sklearn` anywhere in `src/`. Nothing from scikit-learn is used
beyond the estimator API that XGBoost implements: no `Pipeline`, no `GridSearchCV`, no
scaler, and above all none of its cross-validation objects, for the reason below.

### Why hand-written validation

Every scikit-learn cross-validation object shuffles or splits without regard to time.
Putting 2016 in a training fold and 2012 in the matching validation fold is not a
forecasting experiment. The protocol here is an explicit expanding window:

```
fold 1   train [========]             validate [====]
fold 2   train [==============]       validate [====]
fold 3   train [====================] validate [====]
```

with a purge: for an *h*-step target, the last *h* training rows of each fold have targets
inside the validation block, so they are dropped. The test year is never involved in
selecting hyperparameters.

### Baseline

Seasonal naive — next year's January is this year's January. It costs nothing and it
captures the seasonality that dominates monthly fuel demand. Reporting a model's MAPE
without it says nothing about whether the model is any good.

### Forecasting strategy

Direct, one model per horizon, as in the paper: seven products times twelve steps is 84
models per region. A recursive forecast would feed each prediction back in as an input and
compound its own error across the horizon.

## Layout

```
data/<source>/    raw csv, one folder per source, as downloaded
data/oil.duckdb   demand, predictors, panel, plus three views  (built, gitignored)
sql/demand.sql    the JODI aggregation, run against the 650 MB csv in place
sql/schema.sql    typed, keyed declarations of demand and predictors
sql/views.sql     coverage, product mix and publication lag, for the notebooks
src/config.py     paths, the paper's geography and products, every timing assumption
src/download.py   fetch the raw inputs
src/prepare.py    publication dates, regional aggregation, the as-of join
src/data.py       build the store and read it back
src/model.py      features, expanding-window CV, XGBoost, metrics, post-processing
notebooks/01_data.ipynb    the data story and its controls
notebooks/02_model.ipynb   the forecasting experiment
```

Run it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download.py
python src/data.py
jupyter lab notebooks/
```

### The SQL

`sql/demand.sql` is where DuckDB earns its place: the JODI download is a single 650 MB csv
of every flow, unit and product it collects, and the query filters, casts, joins the region
and product maps and aggregates without the file ever entering memory. `sql/schema.sql`
declares the two tables everything rests on with types and primary keys; `sql/views.sql`
holds the three analytical views the notebooks read — regional coverage, product mix, and
the median publication lag per series.

On macOS, XGBoost needs the OpenMP runtime, which its wheel does not ship:
`brew install libomp`. That is the only system dependency.

## What the controls found

Four checks in notebook 01 that changed what the project does:

1. **JODI's `JETKERO` is the jet subset of `KEROSENE`, not a sibling.** The seven mapped
   products reconstruct JODI's own total to 0.03%; adding `JETKERO` breaks it by 5%.
   Reading it wrong would have double-counted jet fuel.
2. **US 2019 demand matches the paper within 1% on five of seven products.** The two that
   miss are a definition difference: JODI excludes ethane from LPG and the paper includes
   it, so ~1.5 mb/d sits in `d7_other` here instead of `d1_lpg`.
3. **JODI's coverage erodes.** Other non-OECD falls from 22 mb/d in 2015 to 12 in 2025 as
   members stop reporting, while its actual demand grew. That region is a data artefact
   over a long span and is not treated as a demand view.
4. **Two products only start in 2009.** The paper backcasts them with Prophet; here they
   are left missing, because fabricating seven years of history from a trend fitted to the
   later data is a larger liability than a shorter sample, and XGBoost splits on missing
   values natively.

And two in notebook 02:

5. **The macro block is what beats the benchmark.** Stripping the nineteen drivers and
   forecasting from the demand lags alone costs the US 0.6 points and China 3.4, and the
   lags-only model is *worse than the seasonal naive* in both regions. Without the macro
   predictors this is an expensive way to reproduce last year — which is also the
   justification for the point-in-time work: those series only help if they are read at
   dates a forecaster could have read them.
6. **The paper's post-processing helps, modestly.** Clipping each step into the range of
   month-on-month changes that calendar month has historically shown improves both regions.
   It binds rarely, on steps the model had no business making; it is a guard rail, not a
   source of accuracy.

## Limitations

- **Sample size.** 216 monthly origins in the training window, and the twelve-month target
  means overlapping windows, so the effective independent sample is far smaller than the
  row count. This is why the hyperparameter grid is eight points and not eight hundred.
- **Publication timing is modelled, not observed.** The lags in `prepare.LAG` are the
  publication conventions of each source, applied uniformly. Real release calendars vary.
- **No vintages.** FRED and the OECD serve the *current* estimate of a past quarter, not
  what was first printed. GDP is revised for years afterwards, so a 2012 row carries a
  better number than a forecaster had in 2012. Fixing this needs ALFRED-style vintage data
  and would be the single largest improvement to the experiment.
- **Structural breaks.** COVID is excluded rather than modelled, and the sample straddles
  the 2008 crisis, the shale build-out and the 2022 energy shock.
- **Long-horizon exogenous variables.** The medium-term application in the paper needs
  *projected* predictors to 2028. This project forecasts twelve months and reads the
  drivers as of the origin, which sidesteps the problem rather than solving it.
- **One test year.** 2019 is twelve months and seven products, and the gap between the
  model and the benchmark for the US is a tenth of a point. That is not a sample from
  which to conclude much about the US.
- **Definitions.** Three of the seven products are not like-for-like with the paper's, so
  the comparison holds for gasoline, jet/kero, gasoil and fuel oil and is indicative for
  the rest.
