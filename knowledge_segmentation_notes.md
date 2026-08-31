# Retail Customer Segmentation — Interview and Project Guide

This document is a personal reference for explaining the project confidently in interviews. It covers what I did, why I did it, what I learned, the trade-offs I accepted, and how the work could help a business.

## 1. The project in one sentence

I transformed two years of retail transactions into five actionable customer segments using carefully reconciled RFM features, K-Means, and DBSCAN, then validated that the segments were stable and not merely artifacts of geography or customer tenure.

## 2. A 30-second interview explanation

> I used the UCI Online Retail II dataset, containing about 1.07 million transaction lines from a UK online retailer. My goal was to help the business decide which customers to retain, nurture, reactivate, or manage individually. I cleaned the transaction history, reconciled cancellations chronologically, engineered Recency, Frequency and Monetary features for 5,840 identified customers, transformed and scaled the features, and evaluated K-Means and DBSCAN. The final hybrid segmentation produced five groups with different business actions. The most important finding was that the largest inactive segment was mostly never-repeat buyers, not formerly loyal customers, so onboarding and second-purchase activation would be more appropriate than a traditional win-back campaign.

## 3. A two-minute explanation

The retailer had transaction data but no practical customer strategy. Treating every customer the same would waste campaign and service resources.

I began with 1,067,371 invoice lines covering December 2009 through December 2011. I separated ordinary sales, formal cancellations and accounting adjustments; audited unusual product codes; removed non-merchandise activity; and excluded missing customer IDs only when moving to customer-level modelling.

The most difficult part was cancellation handling. A simple positive-quantity filter would count cancelled orders as genuine spending. One customer placed an £168,469.60 order and reversed it 12 minutes later. I therefore created a chronological FIFO lot-matching process. Each sale creates an available quantity lot, and a later cancellation consumes earlier visible lots without reaching into future purchases. Partial cancellations reduce the retained quantity rather than deleting or retaining an entire row. I validated this using seven synthetic cases with known answers and real edge cases.

I then calculated RFM:

- Recency: days since the latest retained purchase.
- Frequency: number of distinct invoices with retained purchase quantity.
- Monetary: retained quantity multiplied by the original sale price.

Frequency and Monetary were extremely skewed, so I applied `log1p` and standard scaling while preserving the original values for reporting. I evaluated K-Means across several values of `k`. Although two clusters had the strongest silhouette score, four clusters were balanced and much more actionable. The four-cluster structure was highly stable across initializations, with mean pairwise ARI of 0.991 and a minimum of 0.983.

DBSCAN then provided a density-based second opinion and consistently identified a small population of unusual, high-value customers. The final exploratory hybrid kept 82 persistent DBSCAN outliers as an exceptional segment and fitted four K-Means groups to the remaining customers.

Finally, I profiled the segments by geography and dataset year. That step changed the business interpretation: most low-value inactive customers were early one-time buyers, not lapsed loyalists. The technically derived segments therefore became specific customer actions rather than generic cluster labels.

## 4. Business question and objective

### Business question

Who are the retailer's customers, how do their purchasing behaviours differ, and what action should the business take for each group?

### Analytical objective

Create customer-level behavioural segments that are:

- statistically defensible;
- understandable to non-technical stakeholders;
- sufficiently large to act on;
- connected to different marketing or account-management strategies;
- reproducible from the transaction history.

### What the project does not claim

This project describes observed customer behaviour. It does not prove why customers behave this way, measure causal campaign impact, or predict future churn.

## 5. Dataset and scope

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The dataset contains transactions for a UK-based online gift retailer from December 2009 to December 2011.

Key scope figures:

- 1,067,371 raw transaction lines
- 820,647 customer-attributed merchandise rows after cleaning
- 5,840 identified customers in the final RFM population
- approximately 241,000 rows without a Customer ID excluded from customer modelling

Rows without Customer ID were not automatically treated as invalid. They can still support overall product, revenue and operational analysis, but they cannot be reliably assigned to a customer history.

## 6. End-to-end methodology

```text
Raw transaction sheets
        ↓
Data profiling and quality checks
        ↓
Invoice and StockCode classification
        ↓
Customer-level modelling population
        ↓
Chronological cancellation reconciliation
        ↓
RFM feature engineering
        ↓
log1p transformation and scaling
        ↓
K-Means model selection and stability testing
        ↓
DBSCAN outlier analysis
        ↓
Hybrid customer segments
        ↓
Lifecycle and geography validation
        ↓
Business actions and limitations
```

## 7. Data-cleaning decisions

### Invoice types

The invoice prefix was used to distinguish transaction types:

- Numeric invoice: ordinary sale
- `C` prefix: formal cancellation
- `A` prefix: accounting adjustment, such as bad-debt adjustment

The `A` rows were excluded from customer purchasing behaviour because they are ledger-level entries rather than customer baskets.

Formal cancellations were preserved for reconciliation instead of being dismissed as bad data.

### Negative quantities without `C`

The raw data contained 3,457 negative-quantity rows without a cancellation prefix. Inspection showed that these were predominantly zero-price, customer-unattributed inventory corrections such as damaged, lost, short or discarded stock.

They were excluded from customer RFM because they describe internal operations, not customer returns.

### Non-merchandise StockCodes

Postage, bank charges, manual adjustments, test entries and similar codes were removed from the customer-product modelling population. Real alternative product codes and gift vouchers were retained.

The reason was consistency: every retained row should represent behaviour that can reasonably contribute to a customer's purchase profile.

### Missing Customer ID

The master transaction history can preserve anonymous rows, but customer-level RFM cannot use them. Assigning every missing ID to one value such as `Unknown` would create a fictitious customer containing unrelated transactions.

### Zero-price promotions

The remaining positive-quantity, customer-attributed zero-price rows were retained because they represented genuine promotional products. They contribute £0 to Monetary but can still belong to a legitimate purchasing occasion.

## 8. Cancellation reconciliation

### Why positive-only filtering was insufficient

If all negative rows were removed before RFM, a fully cancelled order would remain as positive spending. This was not hypothetical: customer 16446 placed an 80,995-unit order worth £168,469.60 and cancelled it 12 minutes later. Their genuine retained value was approximately £2.90.

### Primary FIFO rule

FIFO means First In, First Out. Within each customer and product history:

1. A positive sale opens a quantity lot.
2. A later cancellation consumes the oldest available earlier quantity first.
3. Cancellation quantity is capped at the visible available purchase quantity.
4. Excess cancellation quantity remains unmatched rather than consuming a future sale.
5. Each sale ends with a `remaining_quantity`.
6. Net revenue equals `remaining_quantity × original sale price`.

### Why chronology matters

A cancellation in January cannot reverse a purchase made in February. This protects later genuine purchases from early cancellations that may refer to activity before the dataset began.

### The partial-cancellation bug I caught

An earlier version calculated the correct final customer-product balance but used it as a yes/no switch. If a customer purchased 10 units and cancelled 4, the code retained the original 10-unit row because the final balance was positive.

The corrected version retains six units. This was an important lesson: tests must verify the business result, not merely confirm that two implementations produce the same output.

### How it was tested

Seven synthetic cases covered:

- partial cancellation;
- full cancellation followed by a later purchase;
- cancellation before any visible sale;
- one cancellation spanning several earlier invoices;
- the same product purchased at different prices;
- an invoice containing both retained and cancelled products;
- identical timestamps and deterministic ordering.

Known-answer synthetic tests were more valuable than simply rerunning the same logic in a different cell.

### FIFO versus price-compatible closest-prior

An alternative method matched cancellations to the most recent earlier lot with the same price.

Results:

- 536 of 5,840 customers, or 9.18%, differed on at least one RFM measure.
- Aggregate Monetary differed by £41,565, or 0.249% of approximately £16.7 million.
- Individual differences could be large; one customer differed by about £24,185.

FIFO remained the primary method because it is chronological, predictable and does not require cancellation prices to be perfectly reliable. The accepted trade-off is that it may assign a cancellation to the wrong historical price lot.

This means FIFO Monetary is a reproducible analytical estimate, not audited invoice-level revenue.

## 9. RFM explained simply

### Recency

How long has it been since the customer's last retained purchase?

- Lower Recency means the customer purchased more recently.
- A cancellation does not make a customer look newly active.

### Frequency

How many distinct purchase invoices did the customer retain?

- Frequency counts purchasing occasions, not product rows.
- An invoice containing 20 products is still one purchasing occasion.
- A fully reversed invoice does not count.

### Monetary

How much retained purchase value did the customer generate?

- It uses reconciled remaining quantity.
- It represents net retained product value under the FIFO assumption.
- Operational fees and accounting adjustments do not contribute.

### Snapshot date

All Recency values were measured against the same snapshot date: one day after the latest transaction in the dataset. A common snapshot makes customers comparable.

## 10. Transformation, scaling and outliers

### Why transform

Raw Frequency and Monetary were extremely right-skewed:

- Frequency skewness: approximately 12
- Monetary skewness: approximately 27

A few wholesale-scale customers would dominate Euclidean distance and make ordinary customer differences look negligible.

### Why `log1p`

`log1p(x)` computes `log(1 + x)`:

- compresses extreme upper values;
- preserves customer ordering;
- safely supports Monetary equal to zero;
- keeps every genuine customer in the analysis.

Original values remained available for stakeholder interpretation.

### Why scale

Recency, Frequency and Monetary use different units. Standard scaling places them on comparable scales so Monetary does not dominate only because it is measured in pounds.

### Why capping was not automatically selected

After `log1p`, Monetary skewness was already approximately 0.22 and Frequency approximately 1.00. Upper-tail capping improved symmetry but removed distinctions above the 99th percentile.

The uncapped log-transformed data remained the primary path because genuine exceptional customers are commercially important. Capping should be selected only if it materially improves cluster stability and interpretability.

## 11. Why K-Means and how `k=4` was chosen

### K-Means meaning

K-Means groups customers so members of a cluster are close to their cluster centre in transformed and scaled RFM space.

### Evidence used

- Elbow/inertia
- Silhouette score
- Cluster sizes
- Real-unit segment profiles
- Business interpretability
- Stability across random initialization

### Why not `k=2`

Two clusters had the highest silhouette score, but produced only a broad split between active/high-value and inactive/lower-value customers.

### Why `k=4`

Four clusters were balanced, distinct and actionable. The smallest contained about 20% of the customer base, so the solution did not create a tiny cluster around a handful of extremes.

This was a trade-off: slightly weaker mathematical separation in exchange for materially greater business usefulness.

### Stability result

Across eight random seeds:

- Mean pairwise ARI: 0.991
- Minimum pairwise ARI: 0.983

Adjusted Rand Index compares whether two runs group the same customers together while ignoring arbitrary cluster label numbers. These values indicate highly stable structure. They are not literal percentages of identical customer assignments.

## 12. Why DBSCAN and the hybrid approach

### DBSCAN meaning

DBSCAN identifies dense regions and labels points outside those regions as noise. It does not require every customer to belong to an ordinary cluster.

In this project, DBSCAN consistently found:

- a large established customer population;
- a lapsed one-time population;
- a small group of unusual, high-value profiles.

Noise does not mean bad data. Here it means behaviour unlike the dense majority.

### Hybrid approach

The exploratory final approach:

1. Identified 82 customers labelled as noise across the tested DBSCAN settings.
2. Treated them as an Exceptional high-value segment.
3. Re-fitted K-Means with four clusters on the other 5,758 customers.

The worst segment Monetary mean-to-median ratio fell from 2.08 to 1.47, showing reduced within-segment right-skew. This does not by itself prove universal superiority or complete homogeneity.

Before production use, the remainder-only K-Means and the full hybrid solution should receive additional resampling, time stability and future-outcome validation.

## 13. Final segments and actions

### Champions — 1,186 customers, 20.3%

Typical profile:

- Recency: 17 days
- Frequency: 12 orders
- Monetary: £4,557

Meaning: sustained, recent and valuable purchasing behaviour.

Action: protect the relationship through VIP service, early access, relevant loyalty benefits and avoidance of unnecessary discounts.

### Exceptional high-value — 82 customers, 1.4%

Typical profile:

- Recency: 28 days
- Frequency: approximately 23 orders
- Monetary: £18,164

Meaning: unusual, often wholesale-scale behaviour requiring more attention than an automated campaign segment.

Action: named-account management, individual review, service-level monitoring and country/logistics analysis.

### At-risk established — 1,427 customers, 24.4%

Typical profile:

- Recency: 185 days
- Frequency: 4 orders
- Monetary: £1,437

Meaning: customers demonstrated repeat purchasing but engagement is fading.

Action: targeted win-back based on prior categories, purchase timing and value.

### Recent developing — 1,208 customers, 20.7%

Typical profile:

- Recency: 24 days
- Frequency: 3 orders
- Monetary: £707

Meaning: recent customers with moderate development, but lifecycle profiling revealed two different audiences.

Action:

- genuinely new customers: onboarding and next-purchase encouragement;
- longer-tenured casual customers: lower-cost engagement rather than aggressive growth investment.

### Lapsed low-value — 1,937 customers, 33.2%

Typical profile:

- Recency: 402 days
- Frequency: 1 order
- Monetary: £273

Meaning: mostly early one-time buyers rather than formerly loyal customers.

Action: second-purchase activation or carefully tested re-entry campaigns, not loyalty-oriented “we miss you” messaging.

## 14. Main business impact

### Better allocation of marketing spend

The business can avoid sending the same offer to customers with very different histories.

### Correct treatment of never-repeat buyers

The largest segment is primarily a conversion problem, not a loyalty-retention problem. That changes the message, timing and budget owner.

### Protection of important accounts

Exceptional and Champion customers can be prioritized for service and account management rather than treated as ordinary campaign recipients.

### Earlier intervention

At-risk established customers have demonstrated real repeat value but are becoming inactive. They are stronger win-back candidates than one-time customers.

### Clear measurement framework

The segment table provides sizes, shares and median customer profiles that stakeholders can use to budget and design experiments.

## 15. How this helps different stakeholders

### Marketing

- Tailor onboarding, nurture, loyalty and win-back campaigns.
- Avoid expensive offers to customers unlikely to develop.
- Measure campaign performance by segment.

### CRM and customer success

- Prioritize Champions and exceptional accounts.
- Detect established customers whose activity is declining.
- Apply differentiated contact strategies.

### Sales and account management

- Review wholesale-scale accounts individually.
- Identify new high-value international relationships.
- Protect concentrated customer value.

### Finance

- Understand the distinction between gross orders and reconciled customer value.
- See where cancellation assumptions affect individual accounts.
- Avoid treating accounting adjustments as customer behaviour.

### Data and analytics teams

- Reuse a documented customer feature definition.
- Monitor segment movement and feature drift.
- Extend RFM into churn, customer lifetime value or response prediction.

### Executives

- Understand customer-base composition.
- Direct resources toward commercially distinct groups.
- See both the opportunity and the limitations of the available customer identity data.

## 16. Key findings I should remember

1. The largest group was not primarily churned loyalists; it was largely never-repeat customers.
2. Recent developing customers contained both genuinely new and established casual customers.
3. At-risk established customers showed real repeat behaviour, making them credible win-back targets.
4. Exceptional customers were more internationally represented than the overall base, although the absolute count was small.
5. Cancellation reconciliation materially affected individual customers even when aggregate revenue impact was small.
6. A mathematically cleaner two-cluster solution was less useful than four balanced, actionable clusters.
7. Stable clusters do not automatically prove future business value; outcome validation remains necessary.

## 17. Important trade-offs

| Decision | Benefit | Trade-off |
|---|---|---|
| Exclude missing IDs from RFM | Prevents a fictitious “unknown customer” | Segmentation excludes substantial unidentified activity |
| FIFO cancellation netting | Robust, chronological and explainable | May assign cancellation to the wrong price lot |
| Retain zero-price promotions | Preserves genuine purchasing occasions | Some standalone free-item invoices may be operational rather than behavioural |
| `log1p` transformation | Prevents extreme accounts dominating distance | Model-space values are less intuitive to stakeholders |
| No automatic capping | Preserves genuine high-value distinctions | Some upper-tail influence remains |
| Choose `k=4` instead of `k=2` | More actionable segmentation | Lower silhouette score |
| DBSCAN outliers as a separate segment | Prevents extreme accounts distorting ordinary groups | “Noise” is algorithm-dependent and the segment remains heterogeneous |
| Use medians for profiles | Represents the typical customer robustly | Does not communicate total financial contribution alone |

## 18. What I learned

### Technical lessons

- Data cleaning requires domain interpretation, not just null removal.
- Negative quantities can represent different business processes.
- Cancellation matching is a temporal reconciliation problem.
- A passing pipeline is not enough; synthetic tests need known expected outputs.
- Scaling cannot fix severe skew by itself.
- Cluster selection needs both statistical and business evidence.
- ARI measures similarity of partitions, not percentage agreement.
- DBSCAN noise is context-dependent, not automatically anomalous data.

### Business lessons

- A segment label can lead to the wrong campaign even when the clustering is mathematically correct.
- Customer tenure can explain why two people have similar RFM values.
- Aggregate accuracy can hide important customer-level errors.
- The typical customer is often better described by the median than the mean.
- A technically sophisticated model is only useful when connected to a decision.

### Project-delivery lessons

- Preserve raw and excluded populations for audit.
- Keep modelling variables separate from stakeholder reporting values.
- Record assumptions explicitly.
- Put the executive summary first.
- A notebook should tell a story, while reusable logic ultimately belongs in modules and tests.

## 19. Limitations and honest interview answers

### “Are these segments proven to increase revenue?”

No. They are behaviourally and statistically defensible segments with plausible actions. A controlled campaign or future-period evaluation is needed to measure incremental business impact.

### “Is Monetary exact net revenue?”

No. It is FIFO-reconciled retained product value. The source does not explicitly link cancellations to original invoices, and excluded fees mean it is not a complete audited profit measure.

### “Why exclude anonymous customers?”

They cannot be connected into reliable customer histories. I would retain them for transaction and product reporting but not fabricate a shared customer identity.

### “Why did you not use demographics?”

The dataset does not provide reliable demographic attributes. I limited the analysis to observable purchasing behaviour rather than inventing unsupported customer characteristics.

### “Why is DBSCAN noise called high-value?”

It was not labelled high-value merely because DBSCAN called it noise. I profiled the noise population in original RFM units and found exceptionally high Frequency and Monetary. The business label comes from that profile.

### “Why not deploy immediately?”

Production requires pipeline modularization, time-based validation, monitoring, clear refresh rules, assignment logic for new customers, and ideally validation against future behaviour or campaign outcomes.

## 20. Likely interview questions and concise answers

### Why did you choose RFM?

RFM uses fields the retailer actually has and converts transaction history into three interpretable dimensions: how recently, how often and how much a customer purchased. It is a strong baseline for behavioural segmentation.

### Why was cancellation handling so important?

Without it, a reversed £168,000 order would make a customer appear to be one of the retailer's biggest spenders. That would distort both the individual profile and distance-based clustering.

### Why FIFO?

The dataset lacks an explicit original-invoice reference. FIFO is chronological, deterministic, quantity-safe and less dependent on an unaudited cancellation-price field. I also built a price-compatible alternative to quantify the assumption's impact.

### Why use `log1p` before scaling?

Scaling equalizes units but does not remove extreme skew. `log1p` first compresses the long Frequency and Monetary tails, and scaling then gives the three features comparable influence.

### Why four K-Means clusters?

Two clusters separated the population best statistically but were too broad for different business actions. Four produced balanced, stable and interpretable groups, supported by the elbow, a secondary silhouette peak and real-unit profiles.

### How did you validate stability?

I refitted K-Means across eight random seeds and compared partitions with Adjusted Rand Index. Mean pairwise ARI was 0.991 and the minimum was 0.983.

### Why use DBSCAN as well?

K-Means must assign every customer to a cluster. DBSCAN can identify customers outside dense behavioural regions, which helped isolate exceptional accounts instead of letting them distort the general high-value cluster.

### What was the most important business finding?

The largest inactive segment was mainly composed of one-time early buyers, not former loyal customers. That changes the intervention from loyalty win-back to second-purchase conversion.

### What would you do next?

I would modularize the pipeline, validate segments over time, define a churn outcome using historical purchase gaps, create leakage-free features at a cutoff date, and test whether segments or churn scores improve campaign decisions.

## 21. STAR story for behavioural interviews

### Situation

The raw retail dataset contained cancellations, adjustments, extreme wholesale orders and missing customer identities. A standard positive-only RFM tutorial would have produced misleading customer values.

### Task

Build customer segments that were technically defensible and useful for business action.

### Action

I profiled invoice and product-code behaviour, created a time-aware FIFO cancellation reconciliation algorithm, tested it on known-answer synthetic cases, compared it with a price-aware alternative, transformed and scaled RFM, evaluated K-Means and DBSCAN, tested stability with ARI, and profiled the final segments by lifecycle and geography.

### Result

The project produced five actionable segments for 5,840 identified customers. It corrected a £168,000 false-spend edge case, found highly stable four-cluster structure, isolated 82 exceptional accounts, and revealed that the largest inactive segment was mainly a first-to-second-purchase conversion problem rather than loyalty churn.

## 22. What I would improve in a production version

1. Move cleaning, reconciliation, feature engineering and segmentation into tested Python modules.
2. Add unit tests for every synthetic reconciliation case.
3. Pin dependency versions and automate notebook execution.
4. Replace static cluster IDs with profile-based segment naming rules.
5. Add time-based segment stability and migration monitoring.
6. Compare FIFO and price-aware cluster assignments directly.
7. Measure future repeat purchase, revenue and churn by segment.
8. Define how new customers are assigned between retraining cycles.
9. Track data-quality and unmatched-cancellation rates.
10. Run controlled campaigns to estimate incremental lift by segment.

## 23. Next project stage: churn prediction

Segmentation answers:

> What behavioural state is each customer currently in?

Churn prediction would answer:

> Which active customers are likely to become inactive soon?

Before modelling churn, I would:

1. Study inter-purchase gaps and define churn using the retailer's purchasing cycle.
2. Choose an observation cutoff and future outcome window.
3. Build all features using information available before the cutoff.
4. Use a time-based train/validation/test split rather than random splitting.
5. Evaluate precision, recall, PR-AUC and expected campaign value rather than accuracy alone.
6. Compare a simple baseline with interpretable and higher-capacity models.
7. Select an intervention threshold using retention cost and expected saved value.

## 24. Final interview closing statement

> The strongest part of this project was not choosing a clustering algorithm. It was turning messy transaction semantics into defensible customer features, testing the assumptions, and then checking whether the mathematical groups told the right business story. The lifecycle analysis ultimately changed the recommended action for the largest segment, which is exactly why technical validation and stakeholder interpretation need to happen together.

