# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Pipeline Orchestrator
# MAGIC
# MAGIC Runs the six Medallion stages end-to-end. Each step is its own notebook
# MAGIC and can be re-run independently; this orchestrator just sequences them
# MAGIC and short-circuits on failure.
# MAGIC
# MAGIC On Databricks Workflows / Asset Bundles each stage would be its own task
# MAGIC node — this orchestrator is for ad-hoc runs and the live demo.

# COMMAND ----------
dbutils.widgets.text("catalog", "nimble_challenge")
dbutils.widgets.text(
    "source_path",
    "/Volumes/nimble_challenge/raw/files/electronics_dataset.xlsx",
)
dbutils.widgets.text(
    "repo_root",
    "/Workspace/Users/amonterove@gmail.com/NimbleGravityChallenge",
)

catalog = dbutils.widgets.get("catalog")
source_path = dbutils.widgets.get("source_path")
repo_root = dbutils.widgets.get("repo_root")

# COMMAND ----------
COMMON = {"catalog": catalog, "repo_root": repo_root}

STAGES = [
    ("01_bronze_ingest", {**COMMON, "source_path": source_path}),
    ("02_silver_standardize", {**COMMON}),
    ("03_silver_llm_extract", {**COMMON, "prompt_filename": "extract_v1.yaml"}),
    ("04_silver_llm_judge", {**COMMON, "prompt_filename": "judge_v1.yaml"}),
    ("05_silver_taxonomy", {**COMMON}),
    ("06_gold_aggregate", {**COMMON}),
]

for name, params in STAGES:
    print(f"\n=== running {name} ===")
    result = dbutils.notebook.run(name, timeout_seconds=900, arguments=params)
    print(f"=== {name} -> {result} ===")

print("\n✅ pipeline finished")
