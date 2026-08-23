"""Point-in-time alignment: raw source files in, one modelling table out.

    demand()      the target, straight out of sql/demand.sql
    predictors()  the paper's Table 2 drivers, long, each stamped with its publication date
    panel()       the two joined as of each month end, backward

The timing rule drives everything here. A national account for 2019-Q1 describes January
to March but does not exist until late April, so it must not appear in a February row.
Each observation carries `available_at` and `pd.merge_asof(direction="backward")` takes the
most recent value already released. That makes the drivers step functions rather than
smooth lines, which costs a tree nothing.

The paper instead interpolates its quarterly predictors to monthly. Linear interpolation
reads the *next* quarter's number to fill the months in between - a number nobody had.
"""

from __future__ import annotations

import duckdb
import pandas as pd

import config as c


# --- target -------------------------------------------------------------------
def demand(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run sql/demand.sql against the raw JODI csv, which stays on disk."""
    con.execute(f"""CREATE OR REPLACE TEMP VIEW jodi_raw AS
                    SELECT * FROM read_csv_auto('{c.JODI_DIR / "jodi_secondary.csv"}',
                                                all_varchar = true)""")
    con.register("countries", pd.DataFrame(c.ISO2.items(), columns=["country", "region"]))
    con.register("products", pd.DataFrame(c.JODI_PRODUCTS.items(), columns=["code", "product"]))
    return con.sql((c.SQL_DIR / "demand.sql").read_text()).df()


# --- regional aggregation ------------------------------------------------------
def weights() -> pd.DataFrame:
    """Each country's share of its region's GDP, per year, for averaging indices."""
    wb = pd.read_csv(c.WORLDBANK_DIR / "worldbank_annual.csv")
    gdp = wb[wb.variable == "gdp"].assign(region=lambda d: d.iso3.map(c.ISO3))
    return gdp.dropna(subset=["region"])[["iso3", "year", "region", "value"]] \
              .rename(columns={"value": "gdp"})


def by_region(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Country rows to region rows.

    Levels already comparable across countries (USD PPP, people, passengers) are summed.
    Indices are not addable, so each country is first put on the base of its own first 24
    months - a rescaling that reads only the start of the series, never the end - then
    averaged with GDP weights renormalised over whoever reported that month.
    """
    key = ["period_end", "region", "variable"]
    if how == "sum":
        return df.groupby(key, as_index=False).value.sum(min_count=1)
    first = df.sort_values("period_end").groupby(["iso3", "variable"]).value.head(24)
    base = df.loc[first.index].groupby(["iso3", "variable"]).value.mean().rename("base")
    df = df.merge(base, on=["iso3", "variable"]).assign(year=lambda d: d.period_end.dt.year)
    df = df.merge(weights(), on=["iso3", "year", "region"])
    df["value"] = df.value / df.base * 100 * df.gdp / df.groupby(key).gdp.transform("sum")
    return df.groupby(key, as_index=False).value.sum()


def stamp(df: pd.DataFrame, lag: str, source: str) -> pd.DataFrame:
    """Attach the publication date: end of the reference period plus the source's lag."""
    return df.assign(available_at=df.period_end + pd.DateOffset(months=c.LAG[lag]), source=source)


def everywhere(df: pd.DataFrame) -> pd.DataFrame:
    """A world price or a world trade index is the same number for every region."""
    return df.drop(columns="region", errors="ignore").merge(
        pd.DataFrame({"region": c.ALL_REGIONS}), how="cross")


# --- predictors ----------------------------------------------------------------
def read_fred(name: str, freq: str, lag: str, region: str | None) -> pd.DataFrame:
    """FRED files are wide and stamped on the first day of the reference period."""
    df = pd.read_csv(c.FRED_DIR / f"fred_{name}.csv", parse_dates=["date"])
    df = df.melt("date", var_name="variable", value_name="value").dropna(subset=["value"])
    df["period_end"] = df.date.dt.to_period(freq).dt.end_time.dt.normalize()
    df["variable"] = df.variable.str.removesuffix("_us")
    df = df.drop(columns="date").assign(region=region)
    return stamp(everywhere(df) if region is None else df, lag, "fred")


def read_oecd_kei() -> pd.DataFrame:
    """Production volume by activity, consumer prices, car registrations - all indices."""
    names = {("PRVM", "C"): "ind", ("PRVM", "F"): "cbui",
             ("CP", "_Z"): "cpi", ("TOCAPA", "G45"): "pcars"}
    df = pd.read_csv(c.OECD_DIR / "oecd_kei.csv")
    df["variable"] = pd.Series(list(zip(df.MEASURE, df.ACTIVITY)), index=df.index).map(names)
    df = df.dropna(subset=["variable"]).rename(columns={"REF_AREA": "iso3", "OBS_VALUE": "value"})
    df["region"] = df.iso3.map(c.ISO3)
    df["period_end"] = pd.PeriodIndex(df.TIME_PERIOD, freq="M").end_time.normalize()
    df = df.groupby(["iso3", "region", "variable", "period_end"], as_index=False).value.mean()
    return stamp(by_region(df, "mean"), "oecd", "oecd")


def read_oecd_qna() -> pd.DataFrame:
    """gdp, inv and cons are USD PPP levels and add across countries; value added and
    disposable income are national currency, so they are treated as indices."""
    out = []
    for name, keep, rename, how in [
        ("expenditure", lambda d: d.PRICE_BASE == "LR",
         {"B1GQ": "gdp", "P51G": "inv", "P3": "cons"}, "sum"),
        ("value_added", lambda d: (d.ACTIVITY == "_T") & (d.PRICE_BASE == "L"),
         {"B1G": "out"}, "mean"),
        ("income", lambda d: d.TRANSACTION == "B6N", {"B6N": "inc"}, "mean"),
    ]:
        df = pd.read_csv(c.OECD_DIR / f"oecd_qna_{name}.csv", low_memory=False)
        df = df[keep(df)].rename(columns={"REF_AREA": "iso3", "OBS_VALUE": "value"})
        df["variable"] = df.TRANSACTION.map(rename)
        df["region"] = df.iso3.map(c.ISO3)
        df["period_end"] = pd.PeriodIndex(df.TIME_PERIOD, freq="Q").end_time.normalize()
        df = df.dropna(subset=["variable", "region", "value"])
        df = df.groupby(["iso3", "region", "variable", "period_end"], as_index=False).value.mean()
        out.append(by_region(df, how))
    return stamp(pd.concat(out), "oecd", "oecd")


def read_worldbank() -> pd.DataFrame:
    """The only source that reaches other non-OECD, taken as the world aggregate minus the
    six named regions. Annual, so it holds flat for a year and steps once."""
    wb = pd.read_csv(c.WORLDBANK_DIR / "worldbank_annual.csv")
    wb["period_end"] = pd.to_datetime(wb.year.astype(str) + "-12-31")
    named = wb.assign(region=wb.iso3.map(c.ISO3)).dropna(subset=["region"])
    levels = by_region(named[named.variable != "egen"], "sum")
    world = wb[(wb.iso3 == "WLD") & (wb.variable != "egen")] \
        .set_index(["period_end", "variable"]).value
    rest = (world - levels.set_index(["period_end", "variable"]).value.groupby(level=[0, 1]).sum())
    rest = rest.dropna().reset_index().assign(region=c.RESIDUAL)
    shares = by_region(named[named.variable == "egen"], "mean")
    return stamp(pd.concat([levels, rest, shares]), "worldbank", "worldbank")


def read_cpb() -> pd.DataFrame:
    """World merchandise trade volume, the stand-in for the paper's wtr."""
    df = pd.read_csv(c.CPB_DIR / "cpb_world_trade.csv")
    df["period_end"] = pd.PeriodIndex(df.month, freq="M").end_time.normalize()
    df = df.melt(["period_end"], ["wtr"], var_name="variable", value_name="value").dropna()
    return stamp(everywhere(df), "trade", "fred")


def predictors() -> pd.DataFrame:
    """Every Table 2 driver we can source free, long, one source per region and variable."""
    everything = pd.concat([
        read_fred("prices", "M", "price", None),            # rpo, wci: world prices
        read_fred("macro_monthly", "M", "fred_monthly", "us"),
        read_fred("macro_quarterly", "Q", "fred_quarterly", "us"),
        read_fred("aviation", "M", "aviation", None),       # rpk, airf: world air travel proxy
        read_oecd_kei(), read_oecd_qna(), read_worldbank(), read_cpb(),
    ], ignore_index=True)
    rank = everything.source.map({name: i for i, name in enumerate(c.PRIORITY)})
    best = everything[rank == rank.groupby([everything.region, everything.variable]).transform("min")]
    return best[["region", "variable", "source", "period_end", "available_at", "value"]]


# --- the join ------------------------------------------------------------------
def panel(demand: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Join the drivers onto each region-month as of that month end, backward."""
    grid = demand.pivot_table(index=["region", "month"], columns="product", values="kbd")
    grid = grid.reset_index()
    # one dtype on both sides of the join; pandas mixes ns and us resolution otherwise
    grid["as_of"] = (pd.to_datetime(grid.month) + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")

    for name, series in drivers.groupby("variable"):
        series = series[["region", "available_at", "value"]].rename(columns={"value": name})
        series["available_at"] = series.available_at.astype("datetime64[ns]")
        grid = pd.merge_asof(grid.sort_values("as_of"), series.sort_values("available_at"),
                             left_on="as_of", right_on="available_at", by="region",
                             direction="backward",
                             tolerance=pd.Timedelta(days=c.STALE_AFTER_DAYS))
        grid = grid.drop(columns="available_at")
    return grid.drop(columns="as_of").sort_values(["region", "month"]).reset_index(drop=True)
