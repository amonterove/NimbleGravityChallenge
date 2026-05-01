"""Centralized pipeline configuration.

Catalog and schema names live here so notebooks stay declarative. Override per
environment by setting ``NIMBLE_CATALOG`` before running the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"

    @property
    def bronze_products(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.products_raw"

    @property
    def bronze_vendors(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.vendors_raw"

    @property
    def silver_products_clean(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.products_clean"

    @property
    def silver_vendors_canonical(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.vendors_canonical"

    @property
    def silver_products_extracted(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.products_extracted"

    @property
    def silver_products_judged(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.products_judged"

    @property
    def silver_products_manual_review(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.products_manual_review"

    @property
    def silver_products_enriched(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.products_enriched"

    @property
    def silver_llm_cache(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.llm_cache"

    @property
    def gold_product_aggregates(self) -> str:
        return f"{self.catalog}.{self.gold_schema}.product_aggregates"


# Sub-Category -> Category rollup. Documented design choice: the source doc
# asks for both Category and Sub-Category in Gold but only the Sub-Category is
# extracted by the LLM, so we map deterministically here.
SUB_CATEGORY_TO_CATEGORY: dict[str, str] = {
    "Televisions": "Consumer Electronics",
    "Phones": "Consumer Electronics",
    "Smartwatches": "Consumer Electronics",
    "Cameras": "Consumer Electronics",
    "Consoles": "Consumer Electronics",
    "Computers": "Computing",
    "Printers": "Computing",
    "Hardware": "Computing",
    "Accessories": "Accessories",
}


def default_config() -> PipelineConfig:
    """Return the default config, honoring the ``NIMBLE_CATALOG`` env var."""
    return PipelineConfig(catalog=os.environ.get("NIMBLE_CATALOG", "nimble_challenge"))
