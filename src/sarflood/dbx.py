"""Spark/Delta helpers for the Databricks notebooks. Each function takes the `spark`
session as an argument, so this module imports nothing from pyspark at load time and
the wheel stays installable (and the rest of sarflood testable) without Spark.

Silver write paths, compared by the write benchmark (Phase 0 measured the first at
~10k rows/s for 200k rows, too slow for ~20M):
    arrow    spark.createDataFrame(pandas) in chunks: data streams from the driver
             through Spark Connect.
    parquet  pyarrow writes the frame to a Volume staging folder, then Spark reads the
             Parquet files: the bulk bytes go through the file system, not the session.
Both replace one AOI's partition (replaceWhere), so a re-run never duplicates rows.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def table_exists(spark, table: str) -> bool:
    return spark.catalog.tableExists(table)


def replace_aoi(sdf, spark, table: str, aoi: str, append: bool = False) -> None:
    """Write a Spark frame as the rows of `aoi` (partition column 'aoi')."""
    if not table_exists(spark, table):
        sdf.write.format("delta").partitionBy("aoi").saveAsTable(table)
    elif append:
        sdf.write.format("delta").mode("append").saveAsTable(table)
    else:
        (sdf.write.format("delta").mode("overwrite")
         .option("replaceWhere", f"aoi = '{aoi}'").saveAsTable(table))


def write_arrow(spark, pdf, table: str, aoi: str, chunk_rows: int = 1_000_000) -> float:
    t0 = time.perf_counter()
    for i, start in enumerate(range(0, len(pdf), chunk_rows)):
        part = pdf.iloc[start:start + chunk_rows].copy()
        for c in part.columns[part.dtypes == "category"]:   # plain strings per chunk:
            part[c] = part[c].astype(str)                    # safest for Spark's converter
        replace_aoi(spark.createDataFrame(part), spark, table, aoi, append=i > 0)
    return time.perf_counter() - t0


def write_parquet(spark, pdf, table: str, aoi: str, staging: Path,
                  row_group_rows: int = 1_000_000) -> float:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    stage = Path(staging) / aoi
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pdf, preserve_index=False), stage / "part-0.parquet",
                   row_group_size=row_group_rows)
    replace_aoi(spark.read.parquet(str(stage)), spark, table, aoi)
    shutil.rmtree(stage, ignore_errors=True)
    return time.perf_counter() - t0


WRITERS = {"arrow": write_arrow, "parquet": write_parquet}


def loaded_keys(spark, table: str, aoi: str) -> list[tuple[str, str, str]]:
    """(stem, change_sha256, labels_sha256) already in silver for this AOI."""
    if not table_exists(spark, table):
        return []
    rows = spark.sql(f"SELECT DISTINCT stem, change_sha256, labels_sha256 FROM {table} "
                     f"WHERE aoi = '{aoi}'").collect()
    return [(r[0], r[1], r[2]) for r in rows]


def silver_aoi(spark, table: str, aoi: str, columns: list[str]):
    """One AOI's silver rows as pandas (only the columns training/scoring need)."""
    return spark.table(table).where(f"aoi = '{aoi}'").select(*columns).toPandas()


def current_user(spark) -> str:
    return spark.sql("SELECT current_user()").first()[0]
