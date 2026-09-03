"""Build data/oil.duckdb from the raw files, and read it back.

    python -m src.sql

Two tables. demand is the JODI extract filtered and mapped in SQL, so the 650 MB csv never
enters memory. predictors holds every raw driver in long form, one row per region, variable and
period_end: the last day of the period the number describes, not the day it was published.
"""

import duckdb
import pandas as pd

from src import config as c


def query(sql):
    with duckdb.connect(c.DB, read_only=True) as con:
        return con.sql(sql).df()


def build():
    c.DB.unlink(missing_ok=True)
    with duckdb.connect(c.DB) as con:
        con.register("regions", pd.DataFrame(c.REGIONS.items(), columns=["region", "country"]))
        con.register("products", pd.DataFrame(c.JODI_PRODUCTS.items(), columns=["code", "product"]))
        # JODI: the demand flow in thousand barrels per day, the seven products, the two countries
        con.execute(f"""
            CREATE TABLE demand AS
            SELECT regions.region, products.product,
                   CAST(TIME_PERIOD || '-01' AS DATE) AS month,
                   TRY_CAST(OBS_VALUE AS DOUBLE)      AS kbd   -- '-' and 'N/A' mark missing
            FROM read_csv('{c.DATA}/jodi_secondary.csv', all_varchar = true) AS jodi
            JOIN regions  ON regions.country = jodi.REF_AREA
            JOIN products ON products.code   = jodi.ENERGY_PRODUCT
            WHERE FLOW_BREAKDOWN = 'TOTDEMO' AND UNIT_MEASURE = 'KBD' AND kbd IS NOT NULL
            ORDER BY 1, 2, 3""")

        con.execute("CREATE TABLE predictors (region VARCHAR, variable VARCHAR, period_end DATE, "
                    "value DOUBLE)")
        # FRED files are wide and dated on the first day of the period: months for the US file,
        # quarters for the BIS credit series
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'us', variable, last_day(date), value
            FROM (UNPIVOT read_csv('{c.DATA}/fred_us.csv') ON COLUMNS(* EXCLUDE (date))
                  INTO NAME variable VALUE value)""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'china', variable, last_day(date + INTERVAL 2 MONTH), value
            FROM (UNPIVOT read_csv('{c.DATA}/fred_china.csv') ON COLUMNS(* EXCLUDE (date))
                  INTO NAME variable VALUE value)""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'china', 'lkq_electricity', make_date(Year, 12, 31), Value  -- consumption, TWh
            FROM read_csv('{c.DATA}/ember_yearly.csv')
            WHERE Area = 'China' AND Category = 'Electricity demand' AND Unit = 'TWh'""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'china', 'inv', make_date(year, 12, 31), inv
            FROM read_csv('{c.DATA}/worldbank_china.csv')""")
        con.execute(f"""
            INSERT INTO predictors
            SELECT 'china', 'wtr', last_day(CAST(month || '-01' AS DATE)), wtr
            FROM read_csv('{c.DATA}/cpb_world_trade.csv')""")

        print(con.sql("SELECT region, product, MIN(month), MAX(month), COUNT(*) "
                      "FROM demand GROUP BY 1, 2 ORDER BY 1, 2"))
        print(con.sql("SELECT region, variable, MIN(period_end), MAX(period_end), COUNT(*) "
                      "FROM predictors GROUP BY 1, 2 ORDER BY 1, 2"))


if __name__ == "__main__":
    build()
