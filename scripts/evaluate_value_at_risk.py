"""Compare probability and value-at-risk targeting on the final holdout."""

from pathlib import Path

import numpy as np
import pandas as pd

from customer_intelligence.churn import (
    add_value_at_risk_ranking,
    build_logistic_pipeline,
    final_temporal_holdout,
    make_logistic_feature_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "churn_training_table.csv"


def main() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["SnapshotDate"])
    contract = make_logistic_feature_contract(data)
    development_idx, test_idx, _ = final_temporal_holdout(
        data, pd.Timestamp("2011-09-01"), horizon_days=90
    )
    development = data.loc[development_idx]
    test = data.loc[test_idx].copy()
    model = build_logistic_pipeline(contract, regularization_c=0.01)
    model.fit(development[contract.all_features], development["Churn"])
    test["ChurnProbability"] = model.predict_proba(
        test[contract.all_features]
    )[:, 1]
    test["CustomerValue"] = test["LifetimeMonetary"]

    # Learn the cap from the latest safe development snapshot, not September.
    latest_development_date = development["SnapshotDate"].max()
    value_cap = float(
        development.loc[
            development["SnapshotDate"].eq(latest_development_date),
            "LifetimeMonetary",
        ].quantile(0.99)
    )
    ranked = add_value_at_risk_ranking(test, value_cap=value_cap)
    capacity = 0.20
    contacts = int(np.ceil(len(ranked) * capacity))
    total_churners = ranked["Churn"].sum()
    overall_churn = ranked["Churn"].mean()
    total_churn_value = ranked.loc[ranked["Churn"].eq(1), "CustomerValue"].sum()
    strategies = {
        "Probability ranking": ranked.nsmallest(contacts, "ProbabilityRank"),
        "Value-at-risk ranking": ranked.nsmallest(contacts, "ValueAtRiskRank"),
    }
    threshold_eligible = ranked.loc[ranked["ChurnProbability"].ge(0.40)]
    if len(threshold_eligible) >= contacts:
        strategies["Threshold-gated value ranking"] = (
            threshold_eligible.nsmallest(contacts, "ValueAtRiskRank")
        )
    rows = []
    for name, targeted in strategies.items():
        churners = targeted["Churn"].sum()
        rows.append(
            {
                "Strategy": name,
                "Customers": len(targeted),
                "ChurnPrecision": churners / len(targeted),
                "ChurnRecall": churners / total_churners,
                "Lift": (churners / len(targeted)) / overall_churn,
                "ChurnedValueCoverage": (
                    targeted.loc[targeted["Churn"].eq(1), "CustomerValue"].sum()
                    / total_churn_value
                ),
                "AverageCustomerValue": targeted["CustomerValue"].mean(),
            }
        )
    probability_ids = set(strategies["Probability ranking"]["Customer ID"])
    value_ids = set(strategies["Value-at-risk ranking"]["Customer ID"])
    print(f"Value cap learned from {latest_development_date.date()}: £{value_cap:,.2f}")
    print(f"Top-20% overlap: {len(probability_ids & value_ids)}/{contacts} ")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
