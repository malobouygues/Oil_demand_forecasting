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

# --- geography ----------------------------------------------------------------
# The paper covers seven regions; this project forecasts the two it reports in its Table 3.
REGIONS = {"us": ["US"], "china": ["CN"]}
ALL_REGIONS = list(REGIONS)

ISO2 = {country: region for region, members in REGIONS.items() for country in members}
ISO3 = {"USA": "us", "CHN": "china"}

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
PRIORITY = ["fred", "cpb", "oecd", "worldbank"]

# Past this a value is stale rather than late: it goes missing, which XGBoost handles and
# which is honest about a series that has stopped publishing.
STALE_AFTER_DAYS = 730

# --- paper Table 2, and the free series that stands in for each line ----------
# One row per predictor the paper names, in its own order: group, what the paper specifies,
# what this project actually feeds the model. None means no free equivalent was found and
# the variable is absent. Notebook 01 renders this against the store and checks the two
# agree, so the table cannot drift away from the code.
TABLE2 = {
    # Global prices
    "rpo": ("prices", "Brent price, period average",
            "FRED POILBREUSDM, Brent monthly average"),
    "wci": ("prices", "World commodity index: food, beverages, agricultural raw materials, metals",
            "FRED PNFUELINDEXM, IMF index of all non-fuel commodities"),
    # Global economics
    "pop": ("economics", "Total population",
            "World Bank SP.POP.TOTL summed over the region; FRED POPTHM for the US"),
    "gdp": ("economics", "GDP, in USD terms",
            "OECD QNA B1GQ in USD PPP summed over the region; FRED GDP for the US; "
            "World Bank NY.GDP.MKTP.CD for China and other non-OECD"),
    "cpi": ("economics", "Consumer price index, all items",
            "OECD KEI CP, GDP-weighted over the region; FRED CPIAUCSL for the US"),
    "inc": ("economics", "Personal disposable income",
            "FRED DSPIC96, US only - the OECD publishes B6N for neither the US nor China"),
    "wtr": ("economics", "Oxford Economics global trade index",
            "CPB World Trade Monitor, world merchandise trade volume"),
    "lkq": ("economics", "Li Keqiang index: rail freight, electricity output, bank loans",
            None),  # dropped: no free monthly rail freight or bank credit series
    # Global industry
    "ind": ("industry", "Manufacturing production index",
            "OECD KEI PRVM manufacturing; FRED INDPRO for the US"),
    "ppi": ("industry", "Producer price index",
            "FRED PPIACO, US only - no free cross-country PPI panel"),
    "inv": ("industry", "Total fixed investment",
            "OECD QNA P51G gross fixed capital formation; FRED GPDI for the US; "
            "World Bank NE.GDI.TOTL.CD for China and other non-OECD"),
    "out": ("industry", "Gross output, total value added",
            None),  # dropped: unused
    "pcars": ("industry", "Stock of light vehicles for personal use",
              "OECD KEI TOCAPA car registrations; FRED TOTALSA sales for the US - "
              "a flow, not the paper's stock"),
    "ccars": ("industry", "Stock of light and heavy commercial vehicles",
              None),  # no free commercial-fleet stock series
    "cbui": ("industry", "Industrial production in construction of buildings",
             "OECD KEI PRVM construction; FRED TTLCONS construction spending for the US"),
    "chem": ("industry", "Gross output of chemicals excluding pharmaceuticals",
             "FRED IPG325S, industrial production of chemicals, US only"),
    "egen": ("industry", "Electricity produced from oil, % of total",
             "World Bank EG.ELC.PETR.ZS"),
    # Air passenger forecasts
    "airf": ("aviation", "Average air fares index",
             "FRED CUSR0000SETG01, US CPI airline fares, read for every region"),
    "airp": ("aviation", "Total air passengers",
             "World Bank IS.AIR.PSGR, passengers carried"),
    "rpk": ("aviation", "Revenue passenger kilometres",
            "FRED AIRRPMTSI, US revenue passenger miles, read for every region"),
    # Not in Table 2. Carried because it is the closest regional consumer-demand level the
    # OECD publishes quarterly, and it arrives in the same request as gdp and inv.
    "cons": ("extra", "-",
             "OECD QNA P3, household final consumption in USD PPP"),
}

# --- the sample ---------------------------------------------------------------
SAMPLE = ("2000-01-01", "2025-12-31")

# --- the experiment -----------------------------------------------------------
# Train and select through 2019, forecast 2022-23. 2020-21 fall in neither: COVID distorts
# the fit and the score alike, and here they sit in the gap between the two windows.
TRAIN_END = "2019-12-31"
TEST = ("2022-01-01", "2023-12-31")
ORIGIN = "2021-12-01"  # the month the forecast is made from
