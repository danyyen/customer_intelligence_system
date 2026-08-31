"""Train and save the selected production churn-model bundle.

Run from the repository root:
    python -m scripts.package_churn_model
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from customer_intelligence.churn import (
    ChurnModelBundle,
    DEFAULT_ACTION_POLICY,
    build_logistic_pipeline,
    make_logistic_feature_contract,
    save_churn_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "data" / "processed" / "churn_training_table.csv"
ARTIFACT_PATH = ROOT / "models" / "churn_logistic_v1.joblib"


def main() -> None:
    training = pd.read_csv(TRAINING_PATH, parse_dates=["SnapshotDate"])
    contract = make_logistic_feature_contract(training)
    model = build_logistic_pipeline(contract, regularization_c=0.01)

    # Final production training happens only after model choice and holdout
    # evaluation are complete. We can now use every fully labelled snapshot.
    model.fit(training[contract.all_features], training["Churn"])
    bundle = ChurnModelBundle(
        model=model,
        feature_contract=contract,
        probability_threshold=0.40,
        model_version="churn-logistic-1.0.0",
        action_policy=DEFAULT_ACTION_POLICY.copy(),
        metadata={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(training),
            "training_customers": int(training["Customer ID"].nunique()),
            "training_snapshot_start": str(training["SnapshotDate"].min().date()),
            "training_snapshot_end": str(training["SnapshotDate"].max().date()),
            "observation_window_days": 180,
            "prediction_horizon_days": 90,
            "selected_c": 0.01,
            "threshold": 0.40,
            "sklearn_version": sklearn.__version__,
            "feature_columns": contract.all_features,
        },
    )
    save_churn_bundle(bundle, ARTIFACT_PATH)
    print(f"Saved {ARTIFACT_PATH.relative_to(ROOT)}")
    print(f"Model version: {bundle.model_version}")
    print(f"Training rows: {len(training):,}")
    print(f"Required features: {len(contract.all_features)}")


if __name__ == "__main__":
    main()
