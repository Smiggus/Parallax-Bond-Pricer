# Parallax Bond Pricer

Machine-learning fair-value model for US corporate bonds. Predicts credit spreads from bond terms, trading activity,
issuer equity and balance-sheet data, and market conditions, benchmarked against dealer-style matrix pricing on TRACE
data.

## The problem

Most corporate bonds do not trade on a given day, yet every holder has to mark them daily. The industry fallback is
**matrix pricing**: apply the median spread of comparable bonds by rating, maturity and sector. Parallax tests whether
machine learning can do better, and where.

The target is the bond's **spread**: its yield minus the Treasury par yield interpolated at the same remaining
maturity, in basis points.

## Data

One row per bond per month-end, January 2024 to February 2026, built by `build_bond_panel.py` from a private data
platform (read-only):

| Source | Used for |
|---|---|
| TRACE daily trade aggregates (FINRA, via WRDS) | Yield (par-volume-weighted over the month's last week), trading activity |
| TRACE bond master | Coupon, maturity, grade flag, seniority, 144A |
| US Treasury par curve | The risk-free yield at each bond's maturity |
| Daily stock prices, balance sheets, company profiles | Issuer volatility, return, momentum, beta, book leverage, size, sector |
| ICE BofA credit indices, VIX, SOFR | Market conditions |

About 170,000 bond-months, 13,000 bonds, 684 issuers after cleaning: plain corporate bonds only (no equity-linked or
church bonds), fixed coupon, non-convertible, 1 to 30 years to maturity, spreads between -50 and 2,500 bp, and an
issuer-to-stock name check so every bond carries its own issuer's equity data. Balance sheets enter only once filed.

The data are licensed: the panel, the cache and executed notebooks are not in this repository.

## Validation

Strictly chronological, with one empty month between partitions so a one-month-ahead target cannot leak:

| Partition | Months |
|---|---|
| Train | 2024-01 to 2025-05 |
| Valid | 2025-07 to 2025-10 |
| Test (held out) | 2025-12 to 2026-02 |

The benchmark is matrix pricing re-marked on each date: the median spread of **other issuers'** bonds in the same
grade x maturity bucket x sector cell, falling back to coarser cells when fewer than 20 bonds remain.

## Models

Each notebook applies one family of methods and explains the concept behind it.

| Notebook | Method |
|---|---|
| R1 | Decision trees: depth, cost-complexity pruning, a classifier for 25 bp widenings |
| R2 | Bagging, random forest, gradient boosting: OOB error on panel data, feature importance |
| R3 | k-nearest neighbours: "comparable bonds" learned from the data |
| R4 | Naive Bayes for spread widening: Bayes' rule by hand, priors, calibration |
| R5 | Neural network (MLP) |
| R6 | Summary across models |

## Results so far (validation months)

Mean absolute error in basis points.

| Model | All | Investment grade | High yield | R squared |
|---|---|---|---|---|
| Matrix pricing (same date) | 31.9 | 23.1 | 105.4 | 0.23 |
| Decision tree, depth 3 | 38.7 | 32.2 | 93.5 | 0.26 |
| Decision tree, pruned (3,802 leaves) | 29.5 | 24.9 | 68.2 | 0.43 |

A single pruned tree already cuts high-yield pricing error by about a third against matrix pricing, but is slightly
worse on investment grade, where same-day peer spreads are hard to beat. Ensembles are next.

## Caveats

- The investment-grade / high-yield flag is TRACE's latest flag, not the rating at each date, and "high yield" also
  covers some unrated bonds.
- TRACE yields of callable bonds are yields to worst.
- Hyperparameters are chosen on the validation months, so validation scores are slightly optimistic; the test months
  are scored once, at the end.

## Run it

```bash
conda env create -f environment.yml
cp .env.example .env              # then fill in LDP_DB_URL
python build_bond_panel.py        # writes cache/bond_panel.parquet
python build_project_notebooks.py R1 --execute
```
