# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Standardize
# MAGIC
# MAGIC Cleans Bronze data into well-typed Silver tables:
# MAGIC
# MAGIC * `weight` (free text, mixed units) → `weight_kg: double`
# MAGIC * `unit_price` (mixed currencies, OCR errors) → `unit_price_usd: double`
# MAGIC * `vendor` raw → canonical `vendor_id` + `vendor_canonical_name` (fuzzy clustering)
# MAGIC
# MAGIC All transforms are idempotent (overwrite-with-key semantics via MERGE).

# COMMAND ----------
# MAGIC %pip install -q rapidfuzz
# MAGIC %restart_python

# COMMAND ----------
import hashlib

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

from nimble_pipeline.config import default_config
from nimble_pipeline.parsers import (
    canonicalize_vendor,
    parse_price_usd,
    parse_weight_kg,
)

# COMMAND ----------
dbutils.widgets.text("catalog", "nimble_challenge")
import os
os.environ["NIMBLE_CATALOG"] = dbutils.widgets.get("catalog")
cfg = default_config()

# COMMAND ----------
# MAGIC %md
# MAGIC ## UDFs around the unit-tested pure-Python parsers
# MAGIC
# MAGIC We register the parsers as UDFs rather than re-implementing them in
# MAGIC Spark — keeping a single source of truth that's covered by `pytest`.

# COMMAND ----------
parse_weight_udf = F.udf(parse_weight_kg, DoubleType())
parse_price_udf = F.udf(parse_price_usd, DoubleType())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Standardize products

# COMMAND ----------
bronze_products = spark.table(cfg.bronze_products)

silver_products = (
    bronze_products
    .withColumn("weight_kg", parse_weight_udf(F.col("weight")))
    .withColumn("unit_price_usd", parse_price_udf(F.col("unit_price")))
    .withColumn("description", F.col("product_description"))
    .select(
        F.col("id").cast("int").alias("product_id"),
        "description",
        F.col("weight").alias("weight_raw"),
        "weight_kg",
        F.col("unit_price").alias("unit_price_raw"),
        "unit_price_usd",
        "_ingested_at",
        "_source_file",
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Canonicalize vendors with deterministic fuzzy clustering
# MAGIC
# MAGIC The clustering is done on the driver (30 rows is trivial). The result
# MAGIC is broadcast back as a small DataFrame and joined in.

# COMMAND ----------
bronze_vendors = spark.table(cfg.bronze_vendors)
raw_vendor_rows = bronze_vendors.select("product_id", "vendor").collect()

raw_vendor_names = [r["vendor"] for r in raw_vendor_rows if r["vendor"]]
raw_to_canonical = canonicalize_vendor(raw_vendor_names, threshold=85)


def stable_vendor_id(canonical_name: str) -> str:
    """Hash the canonical name into a short stable ID."""
    return "v_" + hashlib.sha1(canonical_name.lower().encode("utf-8")).hexdigest()[:10]


vendor_rows = []
for r in raw_vendor_rows:
    canonical = raw_to_canonical.get(r["vendor"], r["vendor"]) if r["vendor"] else None
    vendor_rows.append(
        {
            "product_id": int(r["product_id"]),
            "vendor_raw": r["vendor"],
            "vendor_canonical_name": canonical,
            "vendor_id": stable_vendor_id(canonical) if canonical else None,
        }
    )

silver_vendors_canonical = spark.createDataFrame(vendor_rows)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data quality checks (Databricks-native expectations)
# MAGIC
# MAGIC We don't fail the run on violations — we surface counts so the demo can
# MAGIC discuss them. In a production pipeline we'd quarantine bad rows.

# COMMAND ----------
total = silver_products.count()
bad_weight = silver_products.filter(F.col("weight_kg").isNull()).count()
bad_price = silver_products.filter(F.col("unit_price_usd").isNull()).count()
out_of_range_weight = silver_products.filter(
    (F.col("weight_kg") <= 0) | (F.col("weight_kg") > 100)
).count()
out_of_range_price = silver_products.filter(F.col("unit_price_usd") <= 0).count()

print(f"DQ: total={total} | null_weight={bad_weight} | null_price={bad_price}")
print(f"DQ: out_of_range_weight={out_of_range_weight} | out_of_range_price={out_of_range_price}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Persist Silver tables (idempotent MERGE)

# COMMAND ----------
def upsert_silver(df, table_fqn: str, key_col: str) -> None:
    if not spark.catalog.tableExists(table_fqn):
        df.write.format("delta").option("mergeSchema", "true").saveAsTable(table_fqn)
        print(f"Created {table_fqn}")
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


upsert_silver(silver_products, cfg.silver_products_clean, key_col="product_id")
upsert_silver(silver_vendors_canonical, cfg.silver_vendors_canonical, key_col="product_id")

# COMMAND ----------
display(spark.table(cfg.silver_products_clean).limit(10))
display(spark.table(cfg.silver_vendors_canonical).limit(10))

# COMMAND ----------
dbutils.notebook.exit("silver_standardize_ok")
