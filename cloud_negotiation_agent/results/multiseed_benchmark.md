# Multi-Seed Benchmarking Results

Evaluated on held-out test split (899 scenarios across seeds: [42, 0, 1, 7, 99]).

## Overall Aggregate Metrics (Mean ± Std Dev)

| Algorithm | Success % | Feasible Success % | Avg Reward | Avg Rounds | Avg Savings % | Tier Kept % | Auto Success % | WA Precision % | WA Recall % |
|---|---|---|---|---|---|---|---|---|---|
| Q-Learning | 93.76 ± 1.81% | 95.68 ± 1.84% | +5.00 ± 0.33 | 2.17 ± 0.12 | 14.30 ± 0.63% | 65.94 ± 1.65% | 97.84 ± 0.68% | 38.34 ± 17.83% | 81.08 ± 27.39% |

## Provider Success Rate (Mean ± Std Dev)

| Algorithm | AWS | GCP | Azure |
|---|---|---|---|
| Q-Learning | 96.90 ± 0.54% | 92.26 ± 3.02% | 92.08 ± 3.08% |

## Difficulty Success Rate (Mean ± Std Dev)

| Algorithm | Easy | Medium | Hard | Expert |
|---|---|---|---|---|
| Q-Learning | 100.00 ± 0.00% | 98.86 ± 1.37% | 92.66 ± 3.92% | 72.00 ± 5.91% |

## Per-Seed Breakdown

| Algorithm | Seed | Success % | Feasible Success % | Avg Reward | Avg Savings % | WA Prec % | WA Rec % | Train Time |
|---|---|---|---|---|---|---|---|---|
| Q-Learning | 42 | 91.5% | 93.4% | +4.85 | 13.6% | 23.6% | 94.4% | 0.7s |
| Q-Learning | 0 | 92.1% | 94.0% | +4.48 | 15.3% | 15.8% | 33.3% | 0.7s |
| Q-Learning | 1 | 95.4% | 97.4% | +5.26 | 14.0% | 56.7% | 94.4% | 0.7s |
| Q-Learning | 7 | 94.8% | 96.7% | +5.17 | 14.3% | 43.9% | 100.0% | 0.8s |
| Q-Learning | 99 | 95.0% | 96.9% | +5.22 | 14.3% | 51.7% | 83.3% | 0.6s |
