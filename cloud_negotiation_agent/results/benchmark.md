| Algorithm | Success % | Feasible Ceiling % | Feasible Success % | Avg reward | Avg rounds | Avg savings % | Tier kept % | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| Q-Learning | 91.5% | 98.0% | 93.4% | +4.85 | 2.04 | 13.6% | 64.8% | 0.7s |

Success rate by provider:

| Algorithm | AWS | GCP | Azure |
|---|---|---|---|
| Q-Learning | 97.0% | 89.5% | 87.9% |

Auto provider selection (best of AWS / GCP / Azure):

| Algorithm | Success % (auto) | Success % (single provider) | Tier kept % | AWS wins | GCP wins | Azure wins |
|---|---|---|---|---|---|---|
| Q-Learning | 97.7% | 91.5% | 82.2% | 40 | 676 | 162 |
