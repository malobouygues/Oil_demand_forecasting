"""Paths, the two regions and seven products, and every publication-lag assumption."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "oil.duckdb"

# JODI reports by ISO-2 country. The paper covers seven regions; this project forecasts the two
# it reports in its Table 3.
REGIONS = {"us": "US", "china": "CN"}

# Paper Table 1. JETKERO is the jet subset of KEROSENE, not a sibling: these seven codes
# reconstruct JODI's own TOTPRODS within 0.04%, adding JETKERO on top breaks it by 5%.
JODI_PRODUCTS = {"LPG": "d1_lpg", "NAPHTHA": "d2_naphtha", "GASOLINE": "d3_gasoline",
                 "KEROSENE": "d4_jetkero", "GASDIES": "d5_gasoil", "RESFUEL": "d6_fueloil",
                 "ONONSPEC": "d7_other"}
PRODUCTS = list(JODI_PRODUCTS.values())

# --- when a number becomes readable -------------------------------------------
# Months from the end of the period a number describes to the day it is published, from each
# source's release calendar: available_at = period_end + LAG. "demand" is JODI's own lag, so
# the newest demand a forecaster holds at t is d(t-2).
LAG = {
    "demand": 2,
    "inc": 1,              # BEA personal income, end of the following month
    "vmt": 3,              # FHWA traffic volume trends
    "ind": 1,              # Fed G.17, mid following month
    "rpk": 3,              # BTS T-100
    "rpo_real": 1,         # Brent is known at month end, the CPI it is deflated by a month later
    "lkq_electricity": 5,  # Ember's yearly review comes out in April-May
    "lkq_credit": 6,       # BIS total credit waits for the national financial accounts
    "inv": 12,             # World Bank annual, about a year after the year
    "wtr": 2,              # CPB, around the 25th of the second month after
}

# Past this a value is stale rather than late: it goes missing, which XGBoost handles and which
# is honest about a series that has stopped publishing.
STALE_DAYS = 730

# --- the sample and the experiment ---------------------------------------------
SAMPLE = ("2000-01-01", "2025-12-31")
# Forecast 2024 from December 2023, with what was published by its last day. 2020 is dropped
# from the training sample: the collapse and its rebound are a shock, not a pattern to learn.
ORIGIN = "2023-12-01"
EXCLUDE_YEARS = [2020]
