"""Build data/oil.duckdb from the raw files, and read it back.

    python -m src.data

`demand` and `predictors` are declared in sql/schema.sql because they are the contract
everything rests on; `panel` is created from the frame, since its columns follow whatever
predictors exist. The notebooks read through here rather than touching a raw file.
"""

import duckdb
import pandas as pd

from src import config as c, timeseries


def build():
    c.DB_PATH.unlink(missing_ok=True)
    with duckdb.connect(c.DB_PATH) as con:
        demand = timeseries.demand(con)
        drivers = timeseries.predictors()
        table = timeseries.panel(demand, drivers)

        con.execute((c.SQL_DIR / "schema.sql").read_text())
        for name, frame in [("demand", demand), ("predictors", drivers)]:
            con.register("frame", frame)
            con.execute(f"INSERT INTO {name} SELECT * FROM frame")
        con.register("frame", table)
        con.execute("CREATE TABLE panel AS SELECT * FROM frame")
        con.execute((c.SQL_DIR / "views.sql").read_text())

        for name in ["demand", "predictors", "panel"]:
            rows = con.sql(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  {name:12} {rows:>7,} rows  {len(con.table(name).columns):>2} columns")


def query(sql):
    with duckdb.connect(c.DB_PATH, read_only=True) as con:
        return con.sql(sql).df()


def load_panel():
    return query("SELECT * FROM panel ORDER BY region, month").astype({"month": "datetime64[ns]"})


def region_frame(region):
    """One region as a monthly frame, reindexed on a complete monthly range so a gap in the
    source shows up as a gap rather than as a missing row."""
    df = load_panel()
    df = df[df["region"] == region].drop(columns="region").set_index("month")
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS"))


if __name__ == "__main__":
    build()
