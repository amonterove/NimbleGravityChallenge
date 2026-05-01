# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Silver Taxonomy (enriched)
# MAGIC
# MAGIC Joins the Judge-validated extraction with the canonical vendor and the
# MAGIC standardized weight/price. Only rows that **passed** the Judge land here;
# MAGIC failures stay in `products_manual_review` for human triage.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters and module path setup

# COMMAND ----------
import os
import sys

dbutils.widgets.text("catalog", "nimble_challenge")
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/amonterove@gmail.com/NimbleGravityChallenge",
)

repo_root = dbutils.widgets.get("repo_root")
sys.path.insert(0, f"{repo_root}/src")

os.environ["NIMBLE_CATALOG"] = dbutils.widgets.get("catalog")

# COMMAND ----------
from delta.tables import DeltaTable
from pyspark.sql import functions as F

from nimble_pipeline.config import default_config

cfg = default_config()

# COMMAND ----------
clean = spark.table(cfg.silver_products_clean)
judged = spark.table(cfg.silver_products_judged).filter(F.col("judge_verdict") == "pass")
vendors = spark.table(cfg.silver_vendors_canonical)

enriched = (
    clean.alias("c")
    .join(judged.alias("j"), on="product_id", how="inner")
    .join(vendors.alias("v"), on="product_id", how="left")
    .select(
        F.col("c.product_id"),
        F.col("j.extracted_name").alias("name"),
        F.col("j.extracted_sub_category").alias("sub_category"),
        F.col("j.extracted_brand").alias("brand"),
        F.col("c.weight_kg"),
        F.col("c.unit_price_usd"),
        F.col("v.vendor_id"),
        F.col("v.vendor_canonical_name"),
        F.col("c.description"),
    )
)

# COMMAND ----------
def upsert(df, table_fqn: str, key_col: str) -> None:
    if not spark.catalog.tableExists(table_fqn):
        df.write.format("delta").option("mergeSchema", "true").saveAsTable(table_fqn)
        return
    target = DeltaTable.forName(spark, table_fqn)
    (
        target.alias("t")
        .merge(df.alias("s"), f"t.{key_col} = s.{key_col}")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


upsert(enriched, cfg.silver_products_enriched, key_col="product_id")

display(spark.table(cfg.silver_products_enriched).limit(10))

# COMMAND ----------
dbutils.notebook.exit("silver_taxonomy_ok")
