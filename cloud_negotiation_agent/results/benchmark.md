| Algorithm | Success % | Feasible Ceiling % | Feasible Success % | Avg reward | Avg rounds | Avg savings % | Tier kept % | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| Q-Learning | 91.2% | 98.0% | 93.1% | +4.60 | 2.16 | 13.5% | 65.1% | 0.6s |
| DQN | 92.2% | 98.0% | 94.1% | +4.59 | 3.04 | 20.6% | 74.8% | 11.1s |

Success rate by provider:

| Algorithm | AWS | GCP | Azure |
|---|---|---|---|
| Q-Learning | 92.7% | 92.4% | 88.6% |
| DQN | 93.2% | 93.6% | 89.8% |

Auto provider selection (best of AWS / GCP / Azure):

| Algorithm | Success % (auto) | Success % (single provider) | Tier kept % | AWS wins | GCP wins | Azure wins |
|---|---|---|---|---|---|---|
| Q-Learning | 98.4% | 91.2% | 85.8% | 23 | 369 | 100 |
| DQN | 99.2% | 92.2% | 96.2% | 2 | 303 | 191 |
