# Value-at-risk targeting policy

## Purpose

Probability ranking answers **who is most likely to churn**. Value-at-risk
ranking answers **where the most historical customer value is exposed**. They
are different campaign objectives and must not be presented as interchangeable.

The implemented proxy is:

```text
ValueAtRiskScore = ChurnProbability × min(CustomerValue, CustomerValue p99)
```

`CustomerValue` remains uncapped in reports. Only ranking influence is capped,
preventing a few wholesale-scale accounts from dominating the entire list.
This is a historical-value proxy, not expected profit or causal treatment value.

## Untouched September comparison at 20% capacity

| Strategy | Churn precision | Churn recall | Lift | Churned-value coverage |
|---|---:|---:|---:|---:|
| Probability ranking | 70.9% | 36.1% | 1.80× | 12.1% |
| Value-at-risk ranking | 39.5% | 20.1% | 1.01× | 46.7% |
| Threshold-gated value ranking | 57.8% | 29.4% | 1.47× | 21.0% |

The value cap (£33,176.47) was learned from the latest safe development
snapshot, not the September holdout.

## Current decision

- Keep probability ranking as the default `CampaignPriorityBand` because it
  best reaches actual churners.
- Publish `ValueAtRiskScore`, `ValueAtRiskRank`, and
  `ValueProtectionPriorityBand` as a selectable business strategy.
- If the business explicitly prioritizes value protection, prefer the
  threshold-gated strategy over pure value ranking.
- Do not call this expected profit. True expected incremental profit requires
  future margin, intervention cost, and treatment-effect data.

Reproduce the comparison with:

```powershell
python -m scripts.evaluate_value_at_risk
```

## The compromise works as follows:
1. Require ChurnProbability ≥ 0.40
2. Rank eligible customers by ValueAtRiskScore
3. Select the top customers within campaign capacity

It offers:
- Better churn targeting than pure value ranking
- Better value coverage than probability ranking
- Lower churn performance than pure probability ranking