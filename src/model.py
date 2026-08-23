"""Section 3 of the paper: the supervised formulation, the fit, the evaluation.

**Direct multi-horizon forecasting**, one model per step, as in the paper. To forecast
twelve months from an origin, twelve models are trained per product:

    model_1 -> demand(t+1)    model_2 -> demand(t+2)   ...   model_12 -> demand(t+12)

Seven products makes 84 models per region. The alternative, recursive forecasting, feeds
each prediction back in as an input and compounds its own error across the horizon.

**What a row knows.** Features at origin t are the twelve most recent *published* demand
readings plus the panel's drivers as of t. JODI publishes a month about two months after it
closes, so the newest demand a forecaster holds at t is d(t-2), not d(t). The drivers were
already aligned point-in-time in prepare.py.

**No feature scaling.** A tree splits on thresholds, so a monotone rescaling leaves the
split structure untouched. The paper scales and reports no difference; that is a property
of the model rather than a finding, and the step is simply left out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
# XGBRegressor is XGBoost's own scikit-learn-compatible estimator: same algorithm as the
# native Booster, less ceremony than DMatrix for a plain tabular fit. It imports parts of
# scikit-learn internally, which is why scikit-learn is in requirements.txt even though
# nothing here imports it directly. Its cross-validation objects are deliberately not used:
# every one of them either shuffles or ignores the purge, and cv_score below does neither.
from xgboost import XGBRegressor

import config as c

LAGS = 12       # d(t-2) back to d(t-13), the paper's twelve target lags
HORIZON = 12    # section 4.1 forecasts twelve months out of sample

PARAMS = dict(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
              random_state=42, n_jobs=4, verbosity=0)
# The paper tunes learning rate and number of trees; depth is added because it is the knob
# that decides how much interaction a tree can express. Eight fits per product, on purpose:
# the effective sample is 192 monthly origins and a large search on 192 rows selects noise.
GRID = [{"learning_rate": lr, "max_depth": d, "n_estimators": n}
        for lr in (0.03, 0.1) for d in (3, 5) for n in (200, 600)]


def features(df: pd.DataFrame, product: str, step: int) -> tuple[pd.DataFrame, pd.Series]:
    """The supervised problem for one product and one horizon.

    X is stamped at origin t and holds demand(t-2) ... demand(t-13) plus every driver as of
    t. y is that product's demand at t + step.
    """
    first = c.LAG["demand"]
    lags = {f"lag{k}": df[product].shift(k) for k in range(first, first + LAGS)}
    return pd.concat([pd.DataFrame(lags), df.drop(columns=c.PRODUCTS)], axis=1), \
        df[product].shift(-step)


def trainable(X: pd.DataFrame, y: pd.Series, step: int, origin: pd.Timestamp) -> pd.Series:
    """Rows whose target month has already happened by the origin, and is observed."""
    return (X.index + pd.DateOffset(months=step) <= origin) & y.notna()


def cv_score(X: pd.DataFrame, y: pd.Series, step: int, params: dict, folds: int = 3) -> float:
    """Expanding-window validation: train on the past, score the block that follows.

        fold 1   train [========]             validate [====]
        fold 2   train [==============]       validate [====]
        fold 3   train [====================] validate [====]

    A shuffled KFold would put later months in the training fold and earlier ones in the
    validation fold, which is not a forecasting experiment.
    """
    block, errors = len(X) // (folds + 1), []
    for k in range(1, folds + 1):
        cut = block * k
        # Remove training rows whose future target overlaps the validation window.
        fit_X, fit_y = X.iloc[:cut - step], y.iloc[:cut - step]
        keep = fit_y.notna()
        model = XGBRegressor(**PARAMS, **params).fit(fit_X[keep], fit_y[keep])
        valid = y.iloc[cut:cut + block]
        errors.append(np.nanmean((valid - model.predict(X.iloc[cut:cut + block])) ** 2))
    return float(np.mean(errors))


def tune(X: pd.DataFrame, y: pd.Series, step: int) -> dict:
    """Small explicit grid, scored by the same chronological protocol used everywhere else.
    Tuned once per product at the longest step and reused for the shorter ones."""
    return min(GRID, key=lambda params: cv_score(X, y, step, params))


def forecast(df: pd.DataFrame, origin: str = c.ORIGIN, horizon: int = HORIZON,
             chosen: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Forecast every product from `origin`, one model per product per step."""
    origin = pd.Timestamp(origin)
    path, chosen = {}, dict(chosen or {})
    for product in c.PRODUCTS:
        X, y = features(df, product, horizon)
        if product not in chosen:
            keep = trainable(X, y, horizon, origin)
            chosen[product] = tune(X[keep], y[keep], horizon)

        values = []
        for step in range(1, horizon + 1):
            X, y = features(df, product, step)
            keep = trainable(X, y, step, origin)
            model = XGBRegressor(**PARAMS, **chosen[product]).fit(X[keep], y[keep])
            values.append(model.predict(X.loc[[origin]])[0])
        path[product] = pd.Series(values, index=pd.date_range(
            origin + pd.DateOffset(months=1), periods=horizon, freq="MS"))
    return pd.DataFrame(path), chosen


def seasonal_naive(df: pd.DataFrame, origin: str = c.ORIGIN, horizon: int = HORIZON) -> pd.DataFrame:
    """The benchmark: next year's month is this year's same month. Free, and the number
    XGBoost has to beat before any of the machinery has earned its place."""
    index = pd.date_range(pd.Timestamp(origin) + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return df.loc[index - pd.DateOffset(years=1), c.PRODUCTS].set_axis(index)


# --- 3.3 post-processing ----------------------------------------------------------
def bounds(history: pd.Series, low: float = 5, high: float = 95) -> pd.DataFrame:
    """How much each calendar month has historically been allowed to move, in percent."""
    change = history.pct_change() * 100
    return (change.groupby(change.index.month).quantile([low / 100, high / 100])
            .unstack().set_axis(["low", "high"], axis=1))


def apply_bounds(path: pd.Series, last: float, band: pd.DataFrame) -> pd.Series:
    """Walk the forecast forward, clipping each step's percent change into its band. Unlike
    a moving average this leaves a plausible step alone and only touches the implausible."""
    out, previous = [], last
    for month, value in path.items():
        previous *= 1 + np.clip((value / previous - 1) * 100, *band.loc[month.month]) / 100
        out.append(previous)
    return pd.Series(out, index=path.index)


# --- 2.5 evaluation ---------------------------------------------------------------
def mse(actual, predicted) -> float:
    return float(np.mean((np.asarray(actual, float) - np.asarray(predicted, float)) ** 2))


def mape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(100 * np.mean(np.abs((actual - predicted) / actual)))


def score(df: pd.DataFrame, path: pd.DataFrame) -> pd.Series:
    """MAPE per product plus the paper's demand-weighted average, which stops 200 kb/d of
    naphtha counting as much as 9,000 kb/d of gasoline."""
    actual = df.loc[path.index, c.PRODUCTS]
    by_product = pd.Series({p: mape(actual[p], path[p]) for p in c.PRODUCTS})
    weight = actual.mean()
    return pd.concat([by_product,
                      pd.Series({"weighted": (by_product * weight).sum() / weight.sum()})])
