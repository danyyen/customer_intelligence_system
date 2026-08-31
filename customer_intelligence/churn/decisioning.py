"""Business decision utilities for converting churn scores into actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def capacity_table(
    actual: Sequence[int],
    probability: Sequence[float],
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    """Measure campaign value when only a fraction can be contacted."""
    frame = pd.DataFrame({"Actual": actual, "Probability": probability})
    if frame.empty or frame.isna().any().any():
        raise ValueError("actual and probability must be non-empty and complete")
    if len(frame["Actual"].unique()) < 2:
        raise ValueError("actual must contain both outcome classes")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must be between 0 and 1")

    ranked = frame.sort_values("Probability", ascending=False).reset_index(drop=True)
    total_churners = ranked["Actual"].sum()
    overall_rate = ranked["Actual"].mean()
    rows = []
    for fraction in fractions:
        contacts = max(1, int(np.ceil(len(ranked) * fraction)))
        targeted = ranked.head(contacts)
        churners_reached = int(targeted["Actual"].sum())
        rows.append(
            {
                "Capacity": fraction,
                "CustomersContacted": contacts,
                "ChurnersReached": churners_reached,
                "Precision": churners_reached / contacts,
                "Recall": churners_reached / total_churners,
                "Lift": (churners_reached / contacts) / overall_rate,
            }
        )
    return pd.DataFrame(rows)


def learn_risk_band_cutoffs(
    validation_probability: Sequence[float],
    quantiles: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Learn fixed probability cutoffs from historical validation scores.

    Defaults create Low (bottom 50%), Medium (next 30%), High (next 10%) and
    Very High (top 10%) bands on validation data.
    """
    quantiles = quantiles or {"Medium": 0.50, "High": 0.80, "Very High": 0.90}
    probability = pd.Series(validation_probability, dtype="float64")
    if probability.empty or probability.isna().any():
        raise ValueError("validation_probability must be non-empty and complete")
    if not probability.between(0, 1).all():
        raise ValueError("probabilities must be between 0 and 1")
    if any(not 0 < value < 1 for value in quantiles.values()):
        raise ValueError("quantiles must be strictly between 0 and 1")
    return {
        band: float(probability.quantile(quantile))
        for band, quantile in quantiles.items()
    }


def assign_risk_bands(
    probability: Sequence[float], cutoffs: Mapping[str, float]
) -> pd.Categorical:
    """Apply fixed validation-derived cutoffs to a later scoring population."""
    required = {"Medium", "High", "Very High"}
    if set(cutoffs) != required:
        raise ValueError(f"cutoffs must contain exactly {sorted(required)}")
    medium, high, very_high = (
        cutoffs["Medium"], cutoffs["High"], cutoffs["Very High"]
    )
    if not 0 <= medium <= high <= very_high <= 1:
        raise ValueError("cutoffs must be ordered between 0 and 1")
    values = np.asarray(probability, dtype=float)
    if np.isnan(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be complete and between 0 and 1")
    labels = np.select(
        [values >= very_high, values >= high, values >= medium],
        ["Very High", "High", "Medium"],
        default="Low",
    )
    return pd.Categorical(
        labels, categories=["Low", "Medium", "High", "Very High"], ordered=True
    )


def assign_campaign_priority_bands(score: Sequence[float]) -> pd.Categorical:
    """Assign campaign-priority bands that reserve 20% for outreach.

    Priority 1 is the top 10%, Priority 2 the next 10%, Priority 3 the next
    30%, and Standard monitoring the remaining 50%. Input order breaks exact
    probability ties, so callers should use a stable customer ordering.
    """
    values = pd.Series(score, dtype="float64")
    if values.empty or values.isna().any() or (values < 0).any():
        raise ValueError("ranking scores must be non-empty and non-negative")
    percentile = values.rank(method="first", ascending=False) / len(values)
    labels = np.select(
        [percentile <= 0.10, percentile <= 0.20, percentile <= 0.50],
        ["Priority 1 - Immediate", "Priority 2 - Targeted", "Priority 3 - Nurture"],
        default="Standard monitoring",
    )
    return pd.Categorical(
        labels,
        categories=[
            "Standard monitoring",
            "Priority 3 - Nurture",
            "Priority 2 - Targeted",
            "Priority 1 - Immediate",
        ],
        ordered=True,
    )


def add_value_at_risk_ranking(
    decisions: pd.DataFrame,
    probability_column: str = "ChurnProbability",
    value_column: str = "CustomerValue",
    cap_quantile: float = 0.99,
    value_cap: float | None = None,
) -> pd.DataFrame:
    """Add a capped historical-value-at-risk ranking to a decision snapshot.

    ``ValueAtRiskScore`` is a prioritization proxy, not expected profit:

        churn probability × min(historical customer value, value p99)

    The original uncapped customer value remains unchanged for reporting.
    """
    if not 0 < cap_quantile <= 1:
        raise ValueError("cap_quantile must be between 0 and 1")
    missing = {probability_column, value_column}.difference(decisions.columns)
    if missing:
        raise ValueError(f"Missing value-at-risk columns: {sorted(missing)}")
    result = decisions.copy()
    probability = pd.to_numeric(result[probability_column], errors="coerce")
    customer_value = pd.to_numeric(result[value_column], errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise ValueError("Churn probabilities must be complete and between 0 and 1")
    if customer_value.isna().any() or (customer_value < 0).any():
        raise ValueError("Customer values must be complete and non-negative")

    if value_cap is None:
        value_cap = float(customer_value.quantile(cap_quantile))
    elif value_cap < 0:
        raise ValueError("value_cap must be non-negative")
    result["CustomerValueCap"] = value_cap
    result["ValueAtRiskScore"] = probability * customer_value.clip(upper=value_cap)
    # method='first' makes ties deterministic after callers sort by customer ID.
    result["ProbabilityRank"] = probability.rank(
        method="first", ascending=False
    ).astype(int)
    result["ValueAtRiskRank"] = result["ValueAtRiskScore"].rank(
        method="first", ascending=False
    ).astype(int)
    result["ValueProtectionPriorityBand"] = assign_campaign_priority_bands(
        result["ValueAtRiskScore"]
    )
    return result
