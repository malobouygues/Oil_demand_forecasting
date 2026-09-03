"""The supervised problem, the fit, the score.

Direct multi-horizon forecasting, one model per step, as in the paper:

    model_1 -> demand(t+1)    model_2 -> demand(t+2)   ...   model_24 -> demand(t+24)

Recursive forecasting would feed each prediction back in as an input and compound its own
error across the horizon.

Features at origin t are the twelve most recent published demand readings plus the panel's
drivers as of t. JODI publishes a month about two months after it closes, so the newest
demand a forecaster holds at t is d(t-2), not d(t); the drivers were aligned point-in-time
in timeseries.py. Nothing is scaled: a tree splits on thresholds, so a monotone rescaling
leaves the split structure untouched.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config as c

LAGS = 12       # d(t-2) back to d(t-13), the paper's twelve target lags
HORIZON = 24    # the 2022-23 test window is twenty-four months

PARAMS = dict(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
              random_state=42, n_jobs=4, verbosity=0)
# The paper tunes learning rate and number of trees; depth decides how much interaction a
# tree can express. Eight fits on purpose: 192 monthly origins, and a large search on 192
# rows selects noise.
GRID = [{"learning_rate": lr, "max_depth": d, "n_estimators": n}
        for lr in (0.03, 0.1) for d in (3, 5) for n in (200, 600)]


def features(df, product, step):
    """X is stamped at origin t and holds demand(t-2) ... demand(t-13) plus every driver as
    of t; y is demand at t + step."""
    first = c.LAG["demand"]
    lags = {f"lag{k}": df[product].shift(k) for k in range(first, first + LAGS)}
    X = pd.concat([pd.DataFrame(lags), df.drop(columns=c.PRODUCTS)], axis=1)
    return X, df[product].shift(-step)


def trainable(X, y, step, origin):
    """Rows whose target month has already happened by the origin, and is observed. Capped
    at TRAIN_END as well: the origin sits two years past the training window and everything
    between them is COVID, which stays out of the fit."""
    target = X.index + pd.DateOffset(months=step)
    return (target <= min(pd.Timestamp(origin), pd.Timestamp(c.TRAIN_END))) & y.notna()


def cv_score(X, y, step, params, folds=3):
    """Expanding window: train on the past, score the block that follows.

        fold 1   train [========]             validate [====]
        fold 2   train [==============]       validate [====]
        fold 3   train [====================] validate [====]

    A shuffled KFold would train on months it is then scored on.
    """
    block, errors = len(X) // (folds + 1), []
    for k in range(1, folds + 1):
        cut = block * k
        # purge: drop the training rows whose target lands inside the validation window
        fit_X, fit_y = X.iloc[:cut - step], y.iloc[:cut - step]
        keep = fit_y.notna()
        model = XGBRegressor(**PARAMS, **params).fit(fit_X[keep], fit_y[keep])
        valid = y.iloc[cut:cut + block]
        errors.append(np.nanmean((valid - model.predict(X.iloc[cut:cut + block])) ** 2))
    return float(np.mean(errors))


def forecast(df, origin=c.ORIGIN, horizon=HORIZON, chosen=None):
    """Forecast every product from `origin`, one model per product per step. Parameters are
    tuned once per product at the longest step and reused for the shorter ones."""
    origin = pd.Timestamp(origin)
    path, chosen = {}, dict(chosen or {})
    for product in c.PRODUCTS:
        X, y = features(df, product, horizon)
        if product not in chosen:
            keep = trainable(X, y, horizon, origin)
            chosen[product] = min(GRID, key=lambda p: cv_score(X[keep], y[keep], horizon, p))

        values = []
        for step in range(1, horizon + 1):
            X, y = features(df, product, step)
            keep = trainable(X, y, step, origin)
            model = XGBRegressor(**PARAMS, **chosen[product]).fit(X[keep], y[keep])
            values.append(model.predict(X.loc[[origin]])[0])
        path[product] = pd.Series(values, index=pd.date_range(
            origin + pd.DateOffset(months=1), periods=horizon, freq="MS"))
    return pd.DataFrame(path), chosen


def seasonal_naive(df, origin=c.ORIGIN, horizon=HORIZON):
    """Repeat the same calendar month of the origin's own year: free, and the number
    XGBoost has to beat. Over a 24-month horizon both years repeat it, because the year in
    between has not happened when the forecast is made."""
    origin = pd.Timestamp(origin)
    index = pd.date_range(origin + pd.DateOffset(months=1), periods=horizon, freq="MS")
    reference = pd.to_datetime({"year": origin.year, "month": index.month, "day": 1})
    return df.loc[reference, c.PRODUCTS].set_axis(index)


# --- 3.3 post-processing ----------------------------------------------------------
def bounds(history, low=5, high=95):
    """How much each calendar month has historically been allowed to move, in percent."""
    change = history.pct_change() * 100
    return (change.groupby(change.index.month).quantile([low / 100, high / 100])
            .unstack().set_axis(["low", "high"], axis=1))


def apply_bounds(path, last, band):
    """Walk the forecast forward, clipping each step's percent change into its band."""
    out, previous = [], last
    for month, value in path.items():
        previous *= 1 + np.clip((value / previous - 1) * 100, *band.loc[month.month]) / 100
        out.append(previous)
    return pd.Series(out, index=path.index)


# --- 2.5 evaluation ---------------------------------------------------------------
def mse(actual, predicted):
    return float(np.mean((np.asarray(actual, float) - np.asarray(predicted, float)) ** 2))


def mape(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(100 * np.mean(np.abs((actual - predicted) / actual)))


def score(df, path):
    """MAPE per product plus the paper's demand-weighted average, which stops 200 kb/d of
    naphtha counting as much as 9,000 kb/d of gasoline."""
    actual = df.loc[path.index, c.PRODUCTS]
    by_product = pd.Series({p: mape(actual[p], path[p]) for p in c.PRODUCTS})
    weight = actual.mean()
    return pd.concat([by_product,
                      pd.Series({"weighted": (by_product * weight).sum() / weight.sum()})])
