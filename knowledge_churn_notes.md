Recommended sequence
1. Analyse interpurchase gaps.
2. Select a provisional churn horizon.
3. Define customer eligibility.
4. Create monthly historical snapshots.
5. Create forward-looking churn labels.
6. Write leakage and boundary tests.
7. Build features as of each snapshot.
8. Inspect label balance and cohort stability.
9. Establish a recency-rule baseline.
10. Train logistic regression.
11. Add tree-based models.
12. Validate chronologically.
13. Compare models using lift, calibration and business capacity.
14. Train the selected pipeline.
15. Score the latest customer snapshot.
16. Export segment, churn probability, risk band and recommended action.

## Snapshot, observation window, and horizon

### What does "snapshot" mean?

A snapshot represents the customer information available on a particular historical date.

Imagine the business stopped on 1 June 2011 and asked:

> Based only on what we know today, which customers are unlikely to purchase again soon?

The customer's information on 1 June is the snapshot.

```text
◄────── Previous 180 days ──────►│◄────── Following 90 days ──────►
      Customer history            Snapshot         Future outcome
      used by the model           1 June           used for the label
```

The model can use purchases before 1 June. It cannot use purchases after 1 June as input, because those had not happened yet.

Historical snapshots let us simulate how a churn model would have worked at different points in the past.

### What does "horizon" mean?

The prediction horizon is how far into the future we look to determine whether the customer returned.

For a snapshot on 1 June:

| Horizon | Future period examined |
|---|---|
| 30 days | 1 June – 30 June |
| 60 days | 1 June – 30 July |
| 90 days | 1 June – 29 August |
| 120 days | 1 June – 28 September |
| 180 days | 1 June – 27 November |

If the horizon is 90 days:

| Outcome | Label |
|---|---|
| Customer purchases during the following 90 days | Churn = 0 |
| No purchase during the following 90 days | Churn = 1 |

A horizon is therefore not how much history the model sees — it is how much future time is used to judge the outcome.

### Snapshot vs. observation window vs. horizon

| Term | Meaning |
|---|---|
| Observation window | Historical period used to understand the customer |
| Snapshot date | Date on which the prediction would be made |
| Prediction horizon | Future period used to determine whether the customer churned |

Example:

- **Observation window:** 1 December – 31 May
- **Snapshot date:** 1 June
- **Prediction horizon:** 1 June – 29 August

### Current configuration in code

- Observation window: 180 days
- Candidate horizons: 30, 60, 90, 120, 180 days
- Snapshot cadence: monthly


## I first converted the cleaned transaction history into customer purchase days. I then recreated the customer base at the start of each historical month. For every snapshot, I looked back 180 days to identify active customers and summarize only the information available at that time. I then looked forward by 30, 60, 90, 120 and 180 days to determine whether each customer returned. Customers who did not return were labelled as churned. I compared the resulting churn rates across horizons, months, customer lifecycle stages and Recency levels. This allowed me to select a churn definition based on customer behaviour, label stability and business usefulness before training any model. 

- Past builds the features
- Snapshot represents prediction time
- Future creates the label

#### Recency at snapshot - RecencyAtSnapshot is just "how many days had it been since this customer's last purchase, measured as of the pretend-today date" — nothing more exotic than that. As customers remain inactive for longer, their probability of remaining inactive during the following 90 days increases.

Concretely, it's this subtraction: RecencyAtSnapshot = SnapshotDate − LastPurchaseBeforeSnapshot



### I selected a 180-day observation window as a practical starting point because it covered approximately 93% of observed repeat-purchase intervals and gave enough history to calculate customer trends, including recent 90 days versus the previous 90 days. A shorter window risked excluding legitimate occasional customers, while a 365-day window would have substantially reduced the number of usable historical snapshots in a dataset containing only two years. I treated 180 days as an eligibility and recent-behaviour window, not as a claim that older history has no value. Longer-term and seasonal features can still use earlier information available before each snapshot.