# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver LLM Extract
# MAGIC
# MAGIC Calls an LLM to extract `name`, `sub_category`, `brand` from each
# MAGIC product description. The prompt restricts `sub_category` to the **5-item**
# MAGIC list — narrower than the Judge's 9-item taxonomy, by design.
# MAGIC
# MAGIC **Observability**: every call is logged to MLflow with prompt fingerprint,
# MAGIC latency, model, and the parsed result.
# MAGIC
# MAGIC **Idempotency / cost control**: a content-hash cache table avoids re-calling
# MAGIC the LLM for descriptions that haven't changed.

# COMMAND ----------
# MAGIC %pip install -q openai pyyaml mlflow
# MAGIC %restart_python

# COMMAND ----------
import hashlib
import json
import os

import mlflow
from delta.tables import DeltaTable
from pyspark.sql import Row
from pyspark.sql import functions as F

from nimble_pipeline.config import default_config
from nimble_pipeline.llm import LLMClient, extract_product_attributes, load_prompt

# COMMAND ----------
dbutils.widgets.text("catalog", "nimble_challenge")
dbutils.widgets.text("prompt_path", "/Workspace/Repos/nimble/NimbleGravityChallenge/prompts/extract_v1.yaml")
dbutils.widgets.text("mlflow_experiment", "/Shared/nimble_challenge/llm_extract")

os.environ["NIMBLE_CATALOG"] = dbutils.widgets.get("catalog")
cfg = default_config()
prompt = load_prompt(dbutils.widgets.get("prompt_path"))

mlflow.set_experiment(dbutils.widgets.get("mlflow_experiment"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the LLM client
# MAGIC
# MAGIC On Databricks Free Edition we use the workspace's Foundation Model APIs
# MAGIC via the OpenAI-compatible endpoint. No external secret needed — the
# MAGIC notebook context provides `DATABRICKS_HOST` / `DATABRICKS_TOKEN`.

# COMMAND ----------
host = spark.conf.get("spark.databricks.workspaceUrl", None) or os.environ.get("DATABRICKS_HOST", "")
if host and not host.startswith("http"):
    host = f"https://{host}"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = LLMClient(
    base_url=f"{host}/serving-endpoints",
    api_key=token,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Determine which descriptions need a fresh LLM call

# COMMAND ----------
products = spark.table(cfg.silver_products_clean).select("product_id", "description")


def description_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


hash_udf = F.udf(description_hash)
products = products.withColumn("description_hash", hash_udf(F.col("description")))

# Pull the existing cache (if any) and skip re-calling for unchanged hashes.
if spark.catalog.tableExists(cfg.silver_llm_cache):
    cache = spark.table(cfg.silver_llm_cache).select(
        "description_hash", "prompt_fingerprint"
    ).filter(F.col("prompt_fingerprint") == prompt.fingerprint)
    to_call = products.join(cache, on="description_hash", how="left_anti")
else:
    to_call = products

rows_to_call = to_call.collect()
print(f"Descriptions to call LLM for: {len(rows_to_call)} (prompt={prompt.fingerprint})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Call the LLM (driver-side loop is fine for 30 rows)

# COMMAND ----------
extraction_records = []
with mlflow.start_run(run_name=f"extract_{prompt.fingerprint}") as run:
    mlflow.log_param("prompt_name", prompt.name)
    mlflow.log_param("prompt_version", prompt.version)
    mlflow.log_param("model", client.model)
    mlflow.log_dict(
        {"system": prompt.system, "user_template": prompt.user_template},
        "prompt.json",
    )

    for row in rows_to_call:
        result = extract_product_attributes(row["description"], client, prompt)
        extraction_records.append(
            Row(
                product_id=int(row["product_id"]),
                description_hash=row["description_hash"],
                description=row["description"],
                extracted_name=result.name,
                extracted_sub_category=result.sub_category,
                extracted_brand=result.brand,
                prompt_fingerprint=result.prompt_fingerprint,
                model=result.model,
                latency_ms=float(result.latency_ms),
                parse_ok=result.parse_ok,
                raw_response=result.raw_response,
            )
        )
        mlflow.log_metric("latency_ms", result.latency_ms, step=int(row["product_id"]))

    mlflow.log_metric("rows_called", len(rows_to_call))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Persist extracted rows + LLM cache

# COMMAND ----------
if extraction_records:
    new_df = spark.createDataFrame(extraction_records)

    # Write to the extraction table (MERGE on product_id).
    if not spark.catalog.tableExists(cfg.silver_products_extracted):
        (
            new_df.write.format("delta")
            .option("mergeSchema", "true")
            .saveAsTable(cfg.silver_products_extracted)
        )
    else:
        target = DeltaTable.forName(spark, cfg.silver_products_extracted)
        (
            target.alias("t")
            .merge(new_df.alias("s"), "t.product_id = s.product_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    # Write to the LLM cache so re-runs skip these descriptions.
    cache_df = new_df.select(
        "description_hash",
        "prompt_fingerprint",
        F.col("extracted_name").alias("name"),
        F.col("extracted_sub_category").alias("sub_category"),
        F.col("extracted_brand").alias("brand"),
        F.col("model"),
    )
    if not spark.catalog.tableExists(cfg.silver_llm_cache):
        cache_df.write.format("delta").option("mergeSchema", "true").saveAsTable(
            cfg.silver_llm_cache
        )
    else:
        target = DeltaTable.forName(spark, cfg.silver_llm_cache)
        (
            target.alias("t")
            .merge(
                cache_df.alias("s"),
                "t.description_hash = s.description_hash AND t.prompt_fingerprint = s.prompt_fingerprint",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
else:
    print("No new descriptions to call — cache covers everything.")

# COMMAND ----------
display(spark.table(cfg.silver_products_extracted).limit(10))

# COMMAND ----------
dbutils.notebook.exit("llm_extract_ok")
