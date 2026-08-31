"""Tests for churn campaign decision utilities."""

import pandas as pd
import pytest

from customer_intelligence.churn import (
    add_value_at_risk_ranking,
    assign_campaign_priority_bands,
    assign_risk_bands,
    capacity_table,
    learn_risk_band_cutoffs,
)


def test_capacity_table_counts_contacts_and_churners():
    result = capacity_table(
        actual=[1, 0, 1, 0, 1],
        probability=[0.9, 0.8, 0.7, 0.2, 0.1],
        fractions=[0.4],
    ).iloc[0]
    assert result["CustomersContacted"] == 2
    assert result["ChurnersReached"] == 1
    assert result["Precision"] == 0.5


def test_cutoffs_are_learned_from_requested_quantiles():
    cutoffs = learn_risk_band_cutoffs([0.1, 0.2, 0.3, 0.4, 0.5])
    assert cutoffs["Medium"] == pytest.approx(0.3)
    assert cutoffs["High"] == pytest.approx(0.42)
    assert cutoffs["Very High"] == pytest.approx(0.46)


def test_risk_bands_use_fixed_cutoffs():
    bands = assign_risk_bands(
        [0.1, 0.4, 0.7, 0.9],
        {"Medium": 0.3, "High": 0.6, "Very High": 0.8},
    )
    assert list(bands) == ["Low", "Medium", "High", "Very High"]
    assert bands.ordered


def test_invalid_cutoffs_are_rejected():
    with pytest.raises(ValueError, match="ordered"):
        assign_risk_bands(
            [0.5], {"Medium": 0.7, "High": 0.6, "Very High": 0.8}
        )


def test_capacity_bands_reserve_top_twenty_percent():
    bands = pd.Series(assign_campaign_priority_bands([0.9, 0.8, 0.7, 0.6, 0.5,
                                                  0.4, 0.3, 0.2, 0.1, 0.0]))
    assert (bands == "Priority 1 - Immediate").sum() == 1
    assert (bands == "Priority 2 - Targeted").sum() == 1
    assert bands.str.startswith("Priority 1").sum() == 1


def test_value_at_risk_ranking_uses_capped_value_and_preserves_original():
    decisions = pd.DataFrame(
        {"CustomerValue": [100.0, 200.0, 10000.0],
         "ChurnProbability": [0.9, 0.8, 0.2]}
    )
    ranked = add_value_at_risk_ranking(decisions, cap_quantile=0.5)
    assert ranked["CustomerValue"].tolist() == [100.0, 200.0, 10000.0]
    assert set(ranked["CustomerValueCap"]) == {200.0}
    assert ranked.loc[0, "ValueAtRiskScore"] == 90.0
    assert ranked.loc[2, "ValueAtRiskScore"] == 40.0
    assert ranked.loc[1, "ValueAtRiskRank"] == 1


def test_value_at_risk_can_use_a_cap_learned_from_earlier_data():
    decisions = pd.DataFrame(
        {"CustomerValue": [100.0, 1000.0], "ChurnProbability": [0.5, 0.5]}
    )
    ranked = add_value_at_risk_ranking(decisions, value_cap=200.0)
    assert set(ranked["CustomerValueCap"]) == {200.0}
    assert ranked["ValueAtRiskScore"].tolist() == [50.0, 100.0]
