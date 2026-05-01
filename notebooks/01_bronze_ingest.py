# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze Ingest
# MAGIC
# MAGIC Reads the raw `electronics_dataset.xlsx` from a Databricks Volume and persists
# MAGIC two **Bronze** Delta tables, one per sheet, with full schema evolution and
# MAGIC ingestion-time metadata.
# MAGIC
# MAGIC **Idempotency**: `MERGE` by `ID` so the notebook can be re-run safely.

# COMMAND ----------
# MAGIC %pip install -q openpyxl pandas
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters and module path setup

# COMMAND ----------
import os
import sys

dbutils.widgets.text("source_path", "/Volumes/nimble_challenge/raw/files/electronics_dataset.xlsx")
dbutils.widgets.text("catalog", "nimble_challenge")
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/amonterove@gmail.com/NimbleGravityChallenge",
)

repo_root = dbutils.widgets.get("repo_root")
sys.path.insert(0, f"{repo_root}/src")

source_path = dbutils.widgets.get("source_path")
os.environ["NIMBLE_CATALOG"] = dbutils.widgets.get("catalog")

# COMMAND ----------
import datetime as dt

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import functions as F

from nimble_pipeline.config import default_config

cfg = default_config()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Ensure catalog & schemas exist

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.bronze_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.silver_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.gold_schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Read both sheets from the Excel file
# MAGIC
# MAGIC We read with pandas first because Spark has no native xlsx reader on Free
# MAGIC Edition. The dataset is tiny (30 rows), so this is fine.

# COMMAND ----------
products_pdf = pd.read_excel(source_path, sheet_name="Products", dtype=str)
vendors_pdf = pd.read_excel(source_path, sheet_name="Vendors", dtype=str)

# Normalize column names to snake_case so downstream Spark code is consistent.
products_pdf.columns = [c.strip().lower().replace(" ", "_") for c in products_pdf.columns]
vendors_pdf.columns = [c.strip().lower().replace(" ", "_") for c in vendors_pdf.columns]

print(f"Products: {len(products_pdf)} rows | columns: {list(products_pdf.columns)}")
print(f"Vendors:  {len(vendors_pdf)} rows | columns: {list(vendors_pdf.columns)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Convert to Spark and stamp ingestion metadata

# COMMAND ----------
ingested_at = dt.datetime.now(dt.timezone.utc).isoformat()

products_df = (
    spark.createDataFrame(products_pdf)
    .withColumn("_ingested_at", F.lit(ingested_at).cast("timestamp"))
    .withColumn("_source_file", F.lit(source_path))
)
vendors_df = (
    spark.createDataFrame(vendors_pdf)
    .withColumn("_ingested_at", F.lit(ingested_at).cast("timestamp"))
    .withColumn("_source_file", F.lit(source_path))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Idempotent MERGE into Bronze
# MAGIC
# MAGIC Re-runs update existing rows (and pick up any new columns via schema
# MAGIC evolution) without creating duplicates.

# COMMAND ----------
def upsert_bronze(df, table_fqn: str, key_col: str) -> None:
    if not spark.catalog.tableExists(table_fqn):
        df.write.format("delta").option("mergeSchema", "true").saveAsTable(table_fqn)
        print(f"Created {table_fqn} ({df.count()} rows)")
        return

    target = DeltaTable.forName(spark, table_fqn)
    (
        target.alias("t")
        .merge(df.alias("s"), f"t.{key_col} = s.{key_col}")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"Merged into {table_fqn}")


upsert_bronze(products_df, cfg.bronze_products, key_col="id")
upsert_bronze(vendors_df, cfg.bronze_vendors, key_col="product_id")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Smoke-check the bronze tables

# COMMAND ----------
display(spark.table(cfg.bronze_products).limit(5))
display(spark.table(cfg.bronze_vendors).limit(5))

# COMMAND ----------
dbutils.notebook.exit("bronze_ok")
