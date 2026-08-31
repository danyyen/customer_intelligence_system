"""Environment-based API configuration.

Environment variables let the same container run locally, in CI and in the
cloud without editing source code for each environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class APISettings:
    model_path: Path
    environment: str
    database_url: str


def get_settings() -> APISettings:
    return APISettings(
        model_path=Path(
            os.getenv("CHURN_MODEL_PATH", "models/churn_logistic_v1.joblib")
        ),
        environment=os.getenv("APP_ENV", "development"),
        database_url=os.getenv(
            "DATABASE_URL",
            "sqlite:///data/serving/customer_intelligence.db",
        ),
    )
