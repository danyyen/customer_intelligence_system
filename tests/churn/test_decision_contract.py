"""Regression checks for the stakeholder-facing decision contract."""

import pandas as pd

from customer_intelligence.churn import (
    DEFAULT_ACTION_POLICY,
    VALUE_PROTECTION_ACTION_POLICY,
)


def test_latest_decisions_use_unambiguous_campaign_contract():
    decisions = pd.read_csv("data/processed/latest_customer_decisions.csv")

    assert "PredictedChurn" not in decisions.columns
    assert "RiskBand" not in decisions.columns
    assert "AboveModelThreshold" in decisions.columns
    assert "ProbabilityThreshold" in decisions.columns
    assert "CampaignPriorityBand" in decisions.columns
    assert "ValueAtRiskScore" in decisions.columns
    assert "ValueAtRiskRank" in decisions.columns
    assert "ValueProtectionPriorityBand" in decisions.columns
    expected_flag = (
        decisions["ChurnProbability"] >= decisions["ProbabilityThreshold"]
    ).astype("int8")
    pd.testing.assert_series_equal(
        decisions["AboveModelThreshold"], expected_flag,
        check_names=False,
        check_dtype=False,
    )
    assert "RecommendedAction" not in decisions.columns
    expected_actions = decisions["CampaignPriorityBand"].map(DEFAULT_ACTION_POLICY)
    pd.testing.assert_series_equal(
        decisions["CampaignRecommendedAction"], expected_actions,
        check_names=False,
    )
    expected_value_actions = decisions["ValueProtectionPriorityBand"].map(
        VALUE_PROTECTION_ACTION_POLICY
    )
    pd.testing.assert_series_equal(
        decisions["ValueProtectionRecommendedAction"], expected_value_actions,
        check_names=False,
    )
