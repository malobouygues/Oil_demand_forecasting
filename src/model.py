"""Direct multi-horizon XGBoost: from origin t, one model per step forecasts demand(t + step),
step = 1 ... 12, on demand(t-2) ... demand(t-13) and the drivers as of t.

JODI publishes a month about two months after it closes, so the newest demand a forecaster
holds at t is d(t-2); the drivers were aligned point-in-time in data.py. Nothing is scaled: a
tree splits on thresholds, and a monotone rescaling leaves every split where it is.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config as c

LAGS = range(c.LAG["demand"], c.LAG["demand"] + 12)  # d(t-2) ... d(t-13)
HORIZON = 12

PARAMS = dict(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
              random_state=42, n_jobs=4, verbosity=0)
# eight fits on purpose: some 200 monthly origins, and a large search on 200 rows selects noise
GRID = [{"learning_rate": lr, "max_depth": d, "n_estimators": n}
        for lr in (0.03, 0.1) for d in (3, 5) for n in (200, 600)]


def features(df, product, step):
    """X at origin t: the demand lags and every driver as of t. y: demand at t + step."""
    lags = pd.DataFrame({f"lag{k}": df[product].shift(k) for k in LAGS})
    return pd.concat([lags, df.drop(columns=c.PRODUCTS)], axis=1), df[product].shift(-step)


def trainable(X, y, step, origin):
    """Origins whose target had been published by the origin, outside the excluded years."""
    target = X.index + pd.DateOffset(months=step)
    published = target + pd.DateOffset(months=c.LAG["demand"]) <= pd.Timestamp(origin)
    covid = X.index.year.isin(c.EXCLUDE_YEARS) | target.year.isin(c.EXCLUDE_YEARS)
    return published & y.notna() & ~covid


def cv_score(X, y, step, params, folds=3):
    """Expanding window: fit on the past, score the block after it. The last `step` rows before
    each block are purged, their targets fall inside it."""
    block, errors = len(X) // (folds + 1), []
    for k in range(1, folds + 1):
        cut = block * k
        fit = XGBRegressor(**PARAMS, **params).fit(X.iloc[:cut - step], y.iloc[:cut - step])
        valid = y.iloc[cut:cut + block]
        errors.append(np.mean((valid - fit.predict(X.iloc[cut:cut + block])) ** 2))
    return np.mean(errors)


def forecast(df, origin=c.ORIGIN, chosen=None):
    """Every product from `origin`, one model per step. Parameters are tuned once per product at
    the longest step and reused for the shorter ones."""
    origin, path, chosen = pd.Timestamp(origin), {}, dict(chosen or {})
    for product in c.PRODUCTS:
        X, y = features(df, product, HORIZON)
        rows = trainable(X, y, HORIZON, origin)
        if product not in chosen:
            chosen[product] = min(GRID, key=lambda p: cv_score(X[rows], y[rows], HORIZON, p))
        path[product] = []
        for step in range(1, HORIZON + 1):
            X, y = features(df, product, step)
            rows = trainable(X, y, step, origin)
            fit = XGBRegressor(**PARAMS, **chosen[product]).fit(X[rows], y[rows])
            path[product].append(fit.predict(X.loc[[origin]])[0])
    index = pd.date_range(origin + pd.DateOffset(months=1), periods=HORIZON, freq="MS")
    return pd.DataFrame(path, index=index), chosen


def seasonal_naive(df, origin=c.ORIGIN):
    """Same calendar month, latest year published by the origin: the free benchmark."""
    origin = pd.Timestamp(origin)
    known = df.loc[:origin - pd.DateOffset(months=c.LAG["demand"]), c.PRODUCTS]
    index = pd.date_range(origin + pd.DateOffset(months=1), periods=HORIZON, freq="MS")
    return pd.DataFrame([known[known.index.month == m].iloc[-1] for m in index.month], index=index)


def score(df, path):
    """MAPE per product and the demand-weighted average, so 200 kb/d of naphtha does not weigh
    as much as 9,000 kb/d of gasoline."""
    actual = df.loc[path.index, c.PRODUCTS]
    by_product = (100 * (actual - path).abs() / actual).mean()
    weight = actual.mean()
    weighted = (by_product * weight).sum() / weight.sum()
    return pd.concat([by_product, pd.Series({"weighted": weighted})])
