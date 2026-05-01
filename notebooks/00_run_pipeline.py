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
    "extract_prompt_path",
    "/Workspace/Repos/nimble/NimbleGravityChallenge/prompts/extract_v1.yaml",
)
dbutils.widgets.text(
    "judge_prompt_path",
    "/Workspace/Repos/nimble/NimbleGravityChallenge/prompts/judge_v1.yaml",
)

catalog = dbutils.widgets.get("catalog")
source_path = dbutils.widgets.get("source_path")
extract_prompt_path = dbutils.widgets.get("extract_prompt_path")
judge_prompt_path = dbutils.widgets.get("judge_prompt_path")

# COMMAND ----------
STAGES = [
    ("01_bronze_ingest", {"catalog": catalog, "source_path": source_path}),
    ("02_silver_standardize", {"catalog": catalog}),
    ("03_silver_llm_extract", {"catalog": catalog, "prompt_path": extract_prompt_path}),
    ("04_silver_llm_judge", {"catalog": catalog, "prompt_path": judge_prompt_path}),
    ("05_silver_taxonomy", {"catalog": catalog}),
    ("06_gold_aggregate", {"catalog": catalog}),
]

for name, params in STAGES:
    print(f"\n=== running {name} ===")
    result = dbutils.notebook.run(name, timeout_seconds=900, arguments=params)
    print(f"=== {name} -> {result} ===")

print("\n✅ pipeline finished")
