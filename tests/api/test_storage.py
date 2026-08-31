"""Serving-store tests, including atomic snapshot replacement behavior."""

import pandas as pd
import pytest

from customer_intelligence.api.storage import DecisionRepository, create_database_engine


def decisions():
    return pd.DataFrame(
        {
            "Customer ID": ["1", "2"],
            "SnapshotDate": pd.to_datetime(["2021-01-01", "2021-01-01"]),
            "Segment": ["Champions", "Lapsed low-value"],
            "CustomerValue": [1000.0, 100.0],
            "SegmentFrequency": [8, 1],
            "ChurnProbability": [0.2, 0.9],
            "AboveModelThreshold": [0, 1],
            "ProbabilityThreshold": [0.4, 0.4],
            "CampaignPriorityBand": [
                "Standard monitoring", "Priority 1 - Immediate"
            ],
            "CampaignRecommendedAction": ["Normal", "Call now"],
            "CustomerValueCap": [2000.0, 2000.0],
            "ValueAtRiskScore": [200.0, 90.0],
            "ProbabilityRank": [2, 1],
            "ValueAtRiskRank": [1, 2],
            "ValueProtectionPriorityBand": [
                "Priority 1 - Immediate", "Standard monitoring"
            ],
            "ValueProtectionRecommendedAction": [
                "Protect value now", "Monitor value"
            ],
            "ModelVersion": ["test-1", "test-1"],
        }
    )


@pytest.fixture()
def repository(tmp_path):
    repo = DecisionRepository(f"sqlite:///{tmp_path / 'serving.db'}")
    repo.initialize()
    return repo


def test_publish_and_get_latest_customer(repository):
    run_id = repository.publish_snapshot(decisions())
    customer = repository.get_customer("2")
    assert run_id
    assert customer["campaign_priority_band"] == "Priority 1 - Immediate"
    assert customer["run_id"] == run_id


def test_list_filters_and_orders_by_risk(repository):
    repository.publish_snapshot(decisions())
    rows = repository.list_customers(limit=2)
    assert [row["customer_id"] for row in rows] == ["2", "1"]
    assert len(repository.list_customers(
        campaign_priority_band="Standard monitoring"
    )) == 1
    value_ordered = repository.list_customers(limit=2, order_by="value_at_risk")
    assert [row["customer_id"] for row in value_ordered] == ["1", "2"]


def test_republishing_snapshot_replaces_instead_of_duplicates(repository):
    repository.publish_snapshot(decisions())
    replacement = decisions().iloc[[0]].copy()
    repository.publish_snapshot(replacement)
    assert repository.get_customer("2") is None
    assert len(repository.list_customers()) == 1


def test_multiple_snapshot_dates_are_rejected(repository):
    invalid = decisions()
    invalid.loc[1, "SnapshotDate"] = pd.Timestamp("2021-02-01")
    with pytest.raises(ValueError, match="one snapshot"):
        repository.publish_snapshot(invalid)


def test_generic_postgresql_url_selects_psycopg3_driver():
    engine = create_database_engine(
        "postgresql://user:password@localhost:5432/customer_intelligence"
    )
    assert engine.url.drivername == "postgresql+psycopg"
