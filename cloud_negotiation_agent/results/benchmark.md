| Algorithm | Success % | Feasible Ceiling % | Feasible Success % | Avg reward | Avg rounds | Avg savings % | Tier kept % | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| DQN | 98.0% | 98.0% | 100.0% | +5.82 | 2.43 | 17.4% | 80.1% | 11.0s |

Success rate by provider:

| Algorithm | AWS | GCP | Azure |
|---|---|---|---|
| DQN | 97.7% | 98.4% | 97.9% |

Auto provider selection (best of AWS / GCP / Azure):

| Algorithm | Success % (auto) | Success % (single provider) | Tier kept % | AWS wins | GCP wins | Azure wins |
|---|---|---|---|---|---|---|
| DQN | 99.4% | 98.0% | 95.6% | 0 | 531 | 363 |
