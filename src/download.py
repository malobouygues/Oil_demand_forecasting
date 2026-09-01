"""Download the raw inputs into data/, one folder per source.

    python -m src.download            # everything
    python -m src.download oecd fred  # only the sources whose name matches

The paper runs on IEA MODS for demand and Oxford Economics for the predictors, both paid.
Every source below is the closest free public substitute, and none needs a key.

Requests go through curl rather than urllib: FRED sits behind a CDN that hangs up on
Python's TLS handshake, reproducibly, while curl on the same URL answers in half a second.
"""

import io
import json
import re
import subprocess
import sys
import zipfile

import pandas as pd

from src import config as c

CURL = ["curl", "-sS", "-L", "--retry", "3", "--max-time", "900"]

# FRED series -> the paper's Table 2 name. Monthly and quarterly are fetched separately: a
# mixed-frequency request comes back as a zip of one file per series instead of a table.
FRED = {
    "prices": {"POILBREUSDM": "rpo", "PNFUELINDEXM": "wci"},
    "macro_monthly": {"INDPRO": "ind_us", "IPG325S": "chem_us", "PPIACO": "ppi_us",
                      "CPIAUCSL": "cpi_us", "DSPIC96": "inc_us", "POPTHM": "pop_us",
                      "TTLCONS": "cbui_us", "TOTALSA": "pcars_us"},
    "macro_quarterly": {"GDP": "gdp_us", "GPDI": "inv_us"},
    "aviation": {"AIRRPMTSI": "rpk_us", "CUSR0000SETG01": "airf_us"},
}

# World Bank indicators -> Table 2 name. Annual, but the only source that reaches every
# non-OECD country on one definition.
WORLDBANK = {"SP.POP.TOTL": "pop", "NY.GDP.MKTP.CD": "gdp", "NE.GDI.TOTL.CD": "inv",
             "IS.AIR.PSGR": "airp", "EG.ELC.PETR.ZS": "egen"}

# OECD SDMX keys are positional: a key with the wrong number of dots is rejected outright,
# so each one is built by joining the dataflow's dimensions in order.
OECD = "https://sdmx.oecd.org/public/rest/data/{}/{}?startPeriod={}&format=csvfile"


def get(url):
    return subprocess.run(CURL + [url], capture_output=True, check=True).stdout


def save(payload, folder, name):
    path = folder / name
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(path, index=False)
    elif payload:  # jodi writes itself, straight out of the zip
        path.write_bytes(payload)
    print(f"  {name:32} {path.stat().st_size / 1e6:8.1f} MB")


def jodi():
    """Monthly demand by product and country, the stand-in for IEA MODS. Kept as the raw
    650 MB csv: sql/demand.sql queries it in place rather than loading it."""
    archive = c.JODI_DIR / "jodi_secondary.zip"
    if not archive.exists():
        subprocess.run(CURL + ["-o", str(archive), "https://www.jodidata.org/_resources/"
                       "files/downloads/oil-data/world_Secondary_CSV.zip"], check=True)
    with zipfile.ZipFile(archive) as z:
        (c.JODI_DIR / "jodi_secondary.csv").write_bytes(z.read(z.namelist()[0]))
    save(b"", c.JODI_DIR, "jodi_secondary.csv")


def fred(group):
    series = FRED[group]
    df = pd.read_csv(io.BytesIO(get("https://fred.stlouisfed.org/graph/fredgraph.csv?id="
                                    + ",".join(series))), na_values=".")
    save(df.set_axis(["date"] + [series[s] for s in df.columns[1:]], axis=1),
         c.FRED_DIR, f"fred_{group}.csv")


def oecd_kei():
    """Monthly indices: production volume (ind, cbui), consumer prices (cpi), car
    registrations (pcars). Key is REF_AREA.FREQ.MEASURE.UNIT..."""
    key = ".".join(["+".join(sorted(c.ISO3)), "M", "PRVM+CP+TOCAPA", "IX", "", "", ""])
    save(get(OECD.format("OECD.SDD.STES,DSD_KEI@DF_KEI", key, "2000-01")),
         c.OECD_DIR, "oecd_kei.csv")


def oecd_qna():
    """Quarterly national accounts: gdp, inv and cons in USD PPP, value added (out) and
    disposable income (inc) in national currency."""
    areas = "+".join(sorted(c.ISO3))
    # DSD_NAMAIN1 has 13 dimensions: FREQ, ADJUSTMENT, REF_AREA, SECTOR, COUNTERPART_SECTOR,
    # TRANSACTION, INSTR_ASSET, ACTIVITY, EXPENDITURE, UNIT_MEASURE, PRICE_BASE,
    # TRANSFORMATION, TABLE_IDENTIFIER. Filtering ACTIVITY in the request rather than after
    # takes value added from 17 MB to 1.4 MB.
    for name, flow, key in [
        ("expenditure", "DSD_NAMAIN1@DF_QNA_EXPENDITURE_USD,1.1",
         ["Q", "", areas, "", "", "B1GQ+P51G+P3"] + [""] * 7),
        ("value_added", "DSD_NAMAIN1@DF_QNA_BY_ACTIVITY_OUTPUT,1.1",
         ["Q", "", areas, "", "", "B1G", "", "_T"] + [""] * 5),
        ("income", "DSD_NAMAIN1@DF_QNA_INC_SAV,1.1",
         ["Q", "", areas, "", "", "B6N+B6G"] + [""] * 7),
    ]:
        save(get(OECD.format("OECD.SDD.NAD," + flow, ".".join(key), "2000-Q1")),
             c.OECD_DIR, f"oecd_qna_{name}.csv")


def worldbank():
    frames = []
    for code, name in WORLDBANK.items():
        page = json.loads(get(f"https://api.worldbank.org/v2/country/all/indicator/{code}"
                              f"?format=json&per_page=20000&date=1990:2026"))[1]
        frames.append(pd.DataFrame([
            {"iso3": r["countryiso3code"], "year": int(r["date"]),
             "variable": name, "value": r["value"]}
            for r in page if r["countryiso3code"]]))
    save(pd.concat(frames).dropna(subset=["value"]), c.WORLDBANK_DIR, "worldbank_annual.csv")


def cpb():
    """World trade volume index, standing in for Oxford Economics' wtr. The workbook name
    carries its vintage, so read the link off the page; converted to csv on the way in."""
    page = get("https://www.cpb.nl/en/worldtrademonitor/latest").decode("utf-8", "ignore")
    href = re.search(r'href="([^"]*World-trade-monitor[^"]*\.xlsx)"', page, re.I).group(1)
    book = pd.read_excel(io.BytesIO(get("https://www.cpb.nl" + href)),
                         sheet_name="trade_out", header=None)
    periods = pd.PeriodIndex(book.iloc[3, 5:].astype(str).str.replace("m", "-"), freq="M")
    trade = book[book[1] == "World trade"].iloc[0, 5:].astype(float)
    save(pd.DataFrame({"month": periods.astype(str), "wtr": trade.to_numpy()}),
         c.CPB_DIR, "cpb_world_trade.csv")


SOURCES = [("jodi", jodi), ("fred prices", lambda: fred("prices")),
           ("fred macro monthly", lambda: fred("macro_monthly")),
           ("fred macro quarterly", lambda: fred("macro_quarterly")),
           ("fred aviation", lambda: fred("aviation")), ("oecd kei", oecd_kei),
           ("oecd qna", oecd_qna), ("world bank", worldbank), ("cpb world trade", cpb)]


def main(only):
    for folder in (c.JODI_DIR, c.FRED_DIR, c.OECD_DIR, c.WORLDBANK_DIR, c.CPB_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES:
        if only and not any(word in name for word in only):
            continue
        print(name)
        source()


if __name__ == "__main__":
    main(sys.argv[1:])
