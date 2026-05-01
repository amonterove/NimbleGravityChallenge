# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Silver LLM Judge
# MAGIC
# MAGIC Validates the extractor's output against the **full** approved taxonomy
# MAGIC (9 sub-categories vs. the extractor's 5). Records that fail are stored
# MAGIC in a `manual_review` table; the rest become `products_judged`.
# MAGIC
# MAGIC This is the heart of the LLM-as-Judge pattern: by giving the Judge a
# MAGIC wider taxonomy, it catches forced misclassifications coming out of the
# MAGIC narrow extractor (e.g. an HP printer wrongly tagged as "Accessories").

# COMMAND ----------
# MAGIC %pip install -q openai pyyaml mlflow
# MAGIC %restart_python

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
dbutils.widgets.text("prompt_filename", "judge_v1.yaml")
dbutils.widgets.text("mlflow_experiment", "/Shared/nimble_challenge/llm_judge")

repo_root = dbutils.widgets.get("repo_root")
sys.path.insert(0, f"{repo_root}/src")

os.environ["NIMBLE_CATALOG"] = dbutils.widgets.get("catalog")

# COMMAND ----------
import mlflow
from delta.tables import DeltaTable
from pyspark.sql import Row
from pyspark.sql import functions as F

from nimble_pipeline.config import default_config
from nimble_pipeline.llm import LLMClient, judge_extraction, load_prompt

cfg = default_config()
prompt_path = f"{repo_root}/prompts/{dbutils.widgets.get('prompt_filename')}"
prompt = load_prompt(prompt_path)

mlflow.set_experiment(dbutils.widgets.get("mlflow_experiment"))

# COMMAND ----------
host = spark.conf.get("spark.databricks.workspaceUrl", None) or os.environ.get("DATABRICKS_HOST", "")
if host and not host.startswith("http"):
    host = f"https://{host}"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = LLMClient(base_url=f"{host}/serving-endpoints", api_key=token)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load extracted rows and run the Judge

# COMMAND ----------
extracted = spark.table(cfg.silver_products_extracted).select(
    "product_id", "description", "extracted_name", "extracted_sub_category", "extracted_brand"
)
rows = extracted.collect()
print(f"Rows to judge: {len(rows)}")

# COMMAND ----------
judged_records = []
with mlflow.start_run(run_name=f"judge_{prompt.fingerprint}") as run:
    mlflow.log_param("prompt_name", prompt.name)
    mlflow.log_param("prompt_version", prompt.version)
    mlflow.log_param("model", client.model)
    mlflow.log_dict(
        {"system": prompt.system, "user_template": prompt.user_template},
        "prompt.json",
    )

    pass_count = 0
    fail_count = 0
    for row in rows:
        result = judge_extraction(
            description=row["description"],
            candidate_sub_category=row["extracted_sub_category"],
            client=client,
            prompt=prompt,
        )
        if result.verdict == "pass":
            pass_count += 1
        else:
            fail_count += 1
        judged_records.append(
            Row(
                product_id=int(row["product_id"]),
                description=row["description"],
                extracted_name=row["extracted_name"],
                extracted_sub_category=row["extracted_sub_category"],
                extracted_brand=row["extracted_brand"],
                judge_verdict=result.verdict,
                judge_suggested_sub_category=result.suggested_sub_category,
                judge_reason=result.reason,
                judge_prompt_fingerprint=result.prompt_fingerprint,
                judge_model=result.model,
                judge_latency_ms=float(result.latency_ms),
                judge_parse_ok=result.parse_ok,
            )
        )
        mlflow.log_metric("latency_ms", result.latency_ms, step=int(row["product_id"]))

    mlflow.log_metric("verdict_pass", pass_count)
    mlflow.log_metric("verdict_fail", fail_count)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Persist judged + manual-review tables

# COMMAND ----------
judged_df = spark.createDataFrame(judged_records)


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


upsert(judged_df, cfg.silver_products_judged, key_col="product_id")
manual_review = judged_df.filter(F.col("judge_verdict") == "fail")
upsert(manual_review, cfg.silver_products_manual_review, key_col="product_id")

print(f"pass={pass_count} | fail={fail_count}")
display(spark.table(cfg.silver_products_judged).limit(10))
display(spark.table(cfg.silver_products_manual_review).limit(10))

# COMMAND ----------
dbutils.notebook.exit("llm_judge_ok")
