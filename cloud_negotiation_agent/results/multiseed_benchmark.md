# Multi-Seed Benchmarking Results

Evaluated on held-out test split (899 scenarios across seeds: [42, 0, 1, 7, 99]).

## Overall Aggregate Metrics (Mean ± Std Dev)

| Algorithm | Success % | Feasible Success % | Avg Reward | Avg Rounds | Avg Savings % | Tier Kept % | Auto Success % | WA Precision % | WA Recall % |
|---|---|---|---|---|---|---|---|---|---|
| Q-Learning | 88.76 ± 1.42% | 90.60 ± 1.51% | +4.34 ± 0.20 | 2.30 ± 0.11 | 14.94 ± 0.87% | 66.80 ± 2.74% | 97.18 ± 0.97% | 12.62 ± 6.90% | 16.68 ± 10.37% |

## Provider Success Rate (Mean ± Std Dev)

| Algorithm | AWS | GCP | Azure |
|---|---|---|---|
| Q-Learning | 92.64 ± 2.95% | 88.00 ± 3.15% | 85.52 ± 4.36% |

## Difficulty Success Rate (Mean ± Std Dev)

| Algorithm | Easy | Medium | Hard | Expert |
|---|---|---|---|---|
| Q-Learning | 100.00 ± 0.00% | 89.26 ± 2.09% | 86.90 ± 6.12% | 67.70 ± 3.32% |

## Per-Seed Breakdown

| Algorithm | Seed | Success % | Feasible Success % | Avg Reward | Avg Savings % | WA Prec % | WA Rec % | Train Time |
|---|---|---|---|---|---|---|---|---|
| Q-Learning | 42 | 90.3% | 92.2% | +4.57 | 15.1% | 7.7% | 5.6% | 0.7s |
| Q-Learning | 0 | 89.4% | 91.3% | +4.30 | 16.3% | 6.2% | 5.6% | 0.7s |
| Q-Learning | 1 | 89.5% | 91.4% | +4.50 | 14.9% | 19.0% | 22.2% | 0.7s |
| Q-Learning | 7 | 87.8% | 89.6% | +4.26 | 14.4% | 21.1% | 22.2% | 0.8s |
| Q-Learning | 99 | 86.8% | 88.5% | +4.08 | 14.0% | 9.1% | 27.8% | 0.6s |
