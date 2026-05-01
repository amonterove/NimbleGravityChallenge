# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Gold Aggregation
# MAGIC
# MAGIC Builds the final analytical table:
# MAGIC `Category | Sub-Category | num_products | avg_price_usd | min_price_usd | max_price_usd`.
# MAGIC
# MAGIC The `Category` rollup is a documented design choice (the source doc
# MAGIC mentions both columns but only Sub-Category is extracted by the LLM, so
# MAGIC we map deterministically via `SUB_CATEGORY_TO_CATEGORY`).

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
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from nimble_pipeline.config import SUB_CATEGORY_TO_CATEGORY, default_config

cfg = default_config()

# COMMAND ----------
def map_category(sub_cat: str) -> str:
    return SUB_CATEGORY_TO_CATEGORY.get(sub_cat, "Other")


map_category_udf = F.udf(map_category, StringType())

# COMMAND ----------
enriched = spark.table(cfg.silver_products_enriched)

gold = (
    enriched
    .withColumn("category", map_category_udf(F.col("sub_category")))
    .groupBy("category", F.col("sub_category"))
    .agg(
        F.count(F.lit(1)).alias("num_products"),
        F.round(F.avg("unit_price_usd"), 2).alias("avg_price_usd"),
        F.min("unit_price_usd").alias("min_price_usd"),
        F.max("unit_price_usd").alias("max_price_usd"),
    )
    .orderBy("category", "sub_category")
)

# Idempotent overwrite: aggregations are deterministic over the Silver state.
(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(cfg.gold_product_aggregates)
)

display(spark.table(cfg.gold_product_aggregates))

# COMMAND ----------
dbutils.notebook.exit("gold_aggregate_ok")
