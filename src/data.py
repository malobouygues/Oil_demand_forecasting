"""The store: build data/oil.duckdb from the raw files, and read it back.

    python src/data.py

Three tables. `demand` and `predictors` are declared in sql/schema.sql with types and keys
because they are the contract everything else rests on; `panel` is created from the frame,
since its columns follow whatever predictors exist. sql/views.sql then adds the three views
the notebooks query for coverage and timing checks.

Everything downstream reads through `load_panel()`, so a notebook never touches a raw file.
"""

from __future__ import annotations

import duckdb
import pandas as pd

import config as c
import prepare


def build() -> None:
    c.DB_PATH.unlink(missing_ok=True)
    with duckdb.connect(c.DB_PATH) as con:
        demand = prepare.demand(con)
        drivers = prepare.predictors()
        table = prepare.panel(demand, drivers)

        con.execute((c.SQL_DIR / "schema.sql").read_text())
        for name, frame in [("demand", demand), ("predictors", drivers)]:
            con.register("frame", frame)
            con.execute(f"INSERT INTO {name} SELECT * FROM frame")
        con.register("frame", table)
        con.execute("CREATE TABLE panel AS SELECT * FROM frame")
        con.execute((c.SQL_DIR / "views.sql").read_text())

        for name in ["demand", "predictors", "panel"]:
            rows, cols = con.sql(f"SELECT COUNT(*) FROM {name}").fetchone()[0], \
                         len(con.table(name).columns)
            print(f"  {name:12} {rows:>7,} rows  {cols:>2} columns")
        span = con.sql("SELECT MIN(month), MAX(month), COUNT(DISTINCT region) FROM panel").fetchone()
        print(f"\npanel spans {span[0]:%Y-%m} to {span[1]:%Y-%m}, {span[2]} regions")


def query(sql: str) -> pd.DataFrame:
    """Read-only SQL against the store, for the notebooks."""
    with duckdb.connect(c.DB_PATH, read_only=True) as con:
        return con.sql(sql).df()


def load_panel() -> pd.DataFrame:
    return query("SELECT * FROM panel ORDER BY region, month").astype({"month": "datetime64[ns]"})


def region_frame(region: str) -> pd.DataFrame:
    """One region as a monthly frame: rows are months, columns are products and drivers.
    Reindexed on a complete monthly range so a gap in the source shows up as a gap."""
    df = load_panel()
    df = df[df.region == region].drop(columns="region").set_index("month")
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS"))


if __name__ == "__main__":
    build()
