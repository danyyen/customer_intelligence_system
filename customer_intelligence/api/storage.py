"""Serving-store abstraction for scored customer decisions.

SQLAlchemy keeps business queries independent of the database engine. Local
development uses SQLite; production can use PostgreSQL by changing DATABASE_URL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    select,
)
from sqlalchemy.engine import Engine


metadata = MetaData()

scoring_runs = Table(
    "scoring_runs",
    metadata,
    Column("run_id", String(36), primary_key=True),
    Column("snapshot_date", Date, nullable=False, index=True),
    Column("model_version", String(100), nullable=False),
    Column("status", String(20), nullable=False),
    Column("row_count", Integer, nullable=False, default=0),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("error_message", Text),
)

customer_scores = Table(
    "customer_scores",
    metadata,
    Column("snapshot_date", Date, primary_key=True),
    Column("customer_id", String(100), primary_key=True),
    Column("segment", String(100), nullable=False, index=True),
    Column("customer_value", Float, nullable=False),
    Column("segment_frequency", Float, nullable=False),
    Column("churn_probability", Float, nullable=False, index=True),
    Column("above_model_threshold", Integer, nullable=False),
    Column("probability_threshold", Float, nullable=False),
    Column("campaign_priority_band", String(40), nullable=False, index=True),
    Column("campaign_recommended_action", Text, nullable=False),
    Column("customer_value_cap", Float, nullable=False),
    Column("value_at_risk_score", Float, nullable=False, index=True),
    Column("probability_rank", Integer, nullable=False),
    Column("value_at_risk_rank", Integer, nullable=False),
    Column("value_protection_priority_band", String(40), nullable=False, index=True),
    Column("value_protection_recommended_action", Text, nullable=False),
    Column("model_version", String(100), nullable=False),
    Column("run_id", String(36), nullable=False, index=True),
    Column("scored_at", DateTime(timezone=True), nullable=False),
)


def create_database_engine(database_url: str) -> Engine:
    """Create an engine and ensure a local SQLite parent directory exists."""
    if database_url.startswith("sqlite:///"):
        database_path = Path(database_url.removeprefix("sqlite:///"))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            database_url,
            future=True,
            connect_args={"check_same_thread": False},
        )
    # Render and similar platforms commonly expose a generic postgresql:// URL.
    # Be explicit that this project uses the installed psycopg 3 driver.
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    return create_engine(database_url, future=True, pool_pre_ping=True)


class DecisionRepository:
    """Write complete scoring snapshots and query the latest successful one."""

    def __init__(self, database_url: str):
        self.engine = create_database_engine(database_url)

    def initialize(self) -> None:
        # This pre-deployment migration preserves local data created under the
        # earlier misleading RiskBand/PredictedChurn names. In a mature system,
        # versioned Alembic migrations would own schema evolution.
        existing_tables = inspect(self.engine).get_table_names()
        if "customer_scores" in existing_tables:
            columns = {
                column["name"]
                for column in inspect(self.engine).get_columns("customer_scores")
            }
            with self.engine.begin() as connection:
                if "risk_band" in columns and "campaign_priority_band" not in columns:
                    connection.exec_driver_sql(
                        "ALTER TABLE customer_scores RENAME COLUMN risk_band "
                        "TO campaign_priority_band"
                    )
                if "predicted_churn" in columns and "above_model_threshold" not in columns:
                    connection.exec_driver_sql(
                        "ALTER TABLE customer_scores RENAME COLUMN predicted_churn "
                        "TO above_model_threshold"
                    )
            columns = {
                column["name"]
                for column in inspect(self.engine).get_columns("customer_scores")
            }
            if "probability_threshold" not in columns:
                with self.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "ALTER TABLE customer_scores ADD COLUMN "
                        "probability_threshold FLOAT NOT NULL DEFAULT 0.40"
                    )
            columns = {
                column["name"]
                for column in inspect(self.engine).get_columns("customer_scores")
            }
            if "recommended_action" in columns and "campaign_recommended_action" not in columns:
                with self.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "ALTER TABLE customer_scores RENAME COLUMN recommended_action "
                        "TO campaign_recommended_action"
                    )
            columns = {
                column["name"]
                for column in inspect(self.engine).get_columns("customer_scores")
            }
            additions = {
                "customer_value_cap": "FLOAT NOT NULL DEFAULT 0",
                "value_at_risk_score": "FLOAT NOT NULL DEFAULT 0",
                "probability_rank": "INTEGER NOT NULL DEFAULT 0",
                "value_at_risk_rank": "INTEGER NOT NULL DEFAULT 0",
                "value_protection_priority_band": (
                    "VARCHAR(40) NOT NULL DEFAULT 'Standard monitoring'"
                ),
                "value_protection_recommended_action": (
                    "TEXT NOT NULL DEFAULT 'Monitor customer value and churn risk'"
                ),
            }
            with self.engine.begin() as connection:
                for name, definition in additions.items():
                    if name not in columns:
                        connection.exec_driver_sql(
                            f"ALTER TABLE customer_scores ADD COLUMN {name} {definition}"
                        )
        metadata.create_all(self.engine)

    def ping(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1)).scalar_one()

    def publish_snapshot(self, decisions: pd.DataFrame) -> str:
        """Replace one snapshot inside a single all-or-nothing transaction."""
        required = {
            "Customer ID", "SnapshotDate", "Segment", "CustomerValue",
            "SegmentFrequency", "ChurnProbability", "AboveModelThreshold",
            "ProbabilityThreshold", "CampaignPriorityBand",
            "CampaignRecommendedAction", "CustomerValueCap", "ValueAtRiskScore",
            "ProbabilityRank", "ValueAtRiskRank", "ValueProtectionPriorityBand",
            "ValueProtectionRecommendedAction", "ModelVersion",
        }
        missing = required.difference(decisions.columns)
        if missing:
            raise ValueError(f"Missing decision columns: {sorted(missing)}")
        if decisions.empty:
            raise ValueError("Cannot publish an empty decision snapshot")
        if decisions["Customer ID"].duplicated().any():
            raise ValueError("Customer IDs must be unique within a snapshot")

        snapshot_days = pd.to_datetime(decisions["SnapshotDate"]).dt.date
        if snapshot_days.nunique() != 1:
            raise ValueError("Publish exactly one snapshot at a time")
        snapshot_date = snapshot_days.iloc[0]
        model_versions = decisions["ModelVersion"].unique()
        if len(model_versions) != 1:
            raise ValueError("A snapshot must use exactly one model version")

        run_id = str(uuid4())
        now = datetime.now(timezone.utc)
        records = []
        for (_, row), snapshot_day in zip(decisions.iterrows(), snapshot_days):
            records.append(
                {
                    "snapshot_date": snapshot_day,
                    "customer_id": str(row["Customer ID"]),
                    "segment": str(row["Segment"]),
                    "customer_value": float(row["CustomerValue"]),
                    "segment_frequency": float(row["SegmentFrequency"]),
                    "churn_probability": float(row["ChurnProbability"]),
                    "above_model_threshold": int(row["AboveModelThreshold"]),
                    "probability_threshold": float(row["ProbabilityThreshold"]),
                    "campaign_priority_band": str(row["CampaignPriorityBand"]),
                    "campaign_recommended_action": str(
                        row["CampaignRecommendedAction"]
                    ),
                    "customer_value_cap": float(row["CustomerValueCap"]),
                    "value_at_risk_score": float(row["ValueAtRiskScore"]),
                    "probability_rank": int(row["ProbabilityRank"]),
                    "value_at_risk_rank": int(row["ValueAtRiskRank"]),
                    "value_protection_priority_band": str(
                        row["ValueProtectionPriorityBand"]
                    ),
                    "value_protection_recommended_action": str(
                        row["ValueProtectionRecommendedAction"]
                    ),
                    "model_version": str(row["ModelVersion"]),
                    "run_id": run_id,
                    "scored_at": now,
                }
            )

        # If any insert fails, SQLAlchemy rolls back the delete and all inserts.
        # API users therefore see either the previous complete snapshot or the
        # new complete snapshot, never a half-published campaign list.
        with self.engine.begin() as connection:
            connection.execute(
                insert(scoring_runs),
                {
                    "run_id": run_id,
                    "snapshot_date": snapshot_date,
                    "model_version": str(model_versions[0]),
                    "status": "running",
                    "row_count": 0,
                    "started_at": now,
                },
            )
            connection.execute(
                delete(customer_scores).where(
                    customer_scores.c.snapshot_date == snapshot_date
                )
            )
            connection.execute(insert(customer_scores), records)
            connection.execute(
                scoring_runs.update()
                .where(scoring_runs.c.run_id == run_id)
                .values(status="completed", row_count=len(records), completed_at=now)
            )
        return run_id

    def latest_snapshot_date(self):
        with self.engine.connect() as connection:
            return connection.execute(
                select(func.max(customer_scores.c.snapshot_date))
            ).scalar_one()

    def get_customer(self, customer_id: str) -> dict | None:
        latest = self.latest_snapshot_date()
        if latest is None:
            return None
        statement = select(customer_scores).where(
            and_(
                customer_scores.c.snapshot_date == latest,
                customer_scores.c.customer_id == str(customer_id),
            )
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        return dict(row) if row else None

    def list_customers(
        self,
        campaign_priority_band: str | None = None,
        value_protection_priority_band: str | None = None,
        segment: str | None = None,
        order_by: str = "churn_probability",
        limit: int = 50,
    ) -> list[dict]:
        latest = self.latest_snapshot_date()
        if latest is None:
            return []
        conditions = [customer_scores.c.snapshot_date == latest]
        if campaign_priority_band:
            conditions.append(
                customer_scores.c.campaign_priority_band == campaign_priority_band
            )
        if segment:
            conditions.append(customer_scores.c.segment == segment)
        if value_protection_priority_band:
            conditions.append(
                customer_scores.c.value_protection_priority_band
                == value_protection_priority_band
            )
        ordering = {
            "churn_probability": customer_scores.c.churn_probability,
            "value_at_risk": customer_scores.c.value_at_risk_score,
        }
        if order_by not in ordering:
            raise ValueError("order_by must be churn_probability or value_at_risk")
        statement = (
            select(customer_scores)
            .where(and_(*conditions))
            .order_by(ordering[order_by].desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]
