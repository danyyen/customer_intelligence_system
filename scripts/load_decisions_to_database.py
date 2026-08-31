"""Load an existing verified decision CSV into the serving database.

This is a fast local bootstrap. Normal scheduled operation should run
``scripts.score_latest_customers`` so features, scores and publication happen
as one workflow.
"""

from pathlib import Path

import pandas as pd

from customer_intelligence.api.config import get_settings
from customer_intelligence.api.storage import DecisionRepository


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "latest_customer_decisions.csv"


def main() -> None:
    decisions = pd.read_csv(INPUT_PATH, parse_dates=["SnapshotDate"])
    repository = DecisionRepository(get_settings().database_url)
    repository.initialize()
    run_id = repository.publish_snapshot(decisions)
    print(f"Published {len(decisions):,} decisions")
    print(f"Run ID: {run_id}")
    print(f"Database: {get_settings().database_url}")


if __name__ == "__main__":
    main()
