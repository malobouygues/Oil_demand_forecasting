"""Paths, the paper's regions and products, and every timing assumption."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQL_DIR = ROOT / "sql"
DB_PATH = DATA / "oil.duckdb"

JODI_DIR = DATA / "jodi"
FRED_DIR = DATA / "fred"
OECD_DIR = DATA / "oecd"
WORLDBANK_DIR = DATA / "worldbank"
CPB_DIR = DATA / "cpb"

# --- geography, paper Table 1 ------------------------------------------------
# Israel is in the paper's OECD APAC but does not report to JODI, so it carries macro only.
# Anything JODI reports that is not named here falls into the residual region.
REGIONS = {
    "us": ["US"], "other_americas": ["CA", "CL", "MX"],
    "europe": ["AT", "CZ", "DK", "EE", "FR", "DE", "GR", "HU", "IT", "NL",
               "NO", "PL", "PT", "SK", "SI", "ES", "SE", "TR", "GB"],
    "apac": ["AU", "IL", "JP", "KR", "NZ"], "china": ["CN"], "india": ["IN"],
}
RESIDUAL = "other_non_oecd"
ALL_REGIONS = list(REGIONS) + [RESIDUAL]

ISO2 = {country: region for region, members in REGIONS.items() for country in members}
ISO3 = {"USA": "us", "CAN": "other_americas", "CHL": "other_americas", "MEX": "other_americas",
        "AUT": "europe", "CZE": "europe", "DNK": "europe", "EST": "europe", "FRA": "europe",
        "DEU": "europe", "GRC": "europe", "HUN": "europe", "ITA": "europe", "NLD": "europe",
        "NOR": "europe", "POL": "europe", "PRT": "europe", "SVK": "europe", "SVN": "europe",
        "ESP": "europe", "SWE": "europe", "TUR": "europe", "GBR": "europe",
        "AUS": "apac", "ISR": "apac", "JPN": "apac", "KOR": "apac", "NZL": "apac",
        "CHN": "china", "IND": "india"}

# --- oil products, paper Table 1 ---------------------------------------------
# JETKERO is the jet subset of KEROSENE, not a sibling: these seven codes reconstruct JODI's
# own TOTPRODS within 0.04%, adding JETKERO on top breaks it by 5%. Notebook 01 checks it.
JODI_PRODUCTS = {"LPG": "d1_lpg", "NAPHTHA": "d2_naphtha", "GASOLINE": "d3_gasoline",
                 "KEROSENE": "d4_jetkero", "GASDIES": "d5_gasoil", "RESFUEL": "d6_fueloil",
                 "ONONSPEC": "d7_other"}
PRODUCTS = list(JODI_PRODUCTS.values())

# --- when a number becomes readable ------------------------------------------
# Months from the end of the reference period to publication, from each source's own
# calendar: national accounts print about two months after the quarter closes, the World
# Bank annual file about a year after the year, a Brent monthly average is known the day the
# month ends. "demand" is JODI's lag, so the newest demand available at t is d(t-2).
LAG = {"demand": 2, "price": 0, "fred_monthly": 1, "fred_quarterly": 1, "aviation": 2,
       "oecd": 2, "worldbank": 12, "trade": 2}

# Several sources carry the same variable for the same region on different definitions and
# units, so exactly one has to win. Most timely first.
PRIORITY = ["fred", "oecd", "worldbank"]

# Past this a value is stale rather than late: it goes missing, which XGBoost handles and
# which is honest about a series that has stopped publishing.
STALE_AFTER_DAYS = 730

# --- the experiment, paper 2.4 ------------------------------------------------
# Train and select through 2018, test on 2019. 2020-21 are in neither: COVID distorts the
# fit and the score alike.
TRAIN_END = "2018-12-31"
TEST = ("2019-01-01", "2019-12-31")
ORIGIN = "2018-12-01"  # the month the forecast is made from
