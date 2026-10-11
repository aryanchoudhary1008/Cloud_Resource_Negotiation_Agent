| Algorithm | Success % | Avg reward | Avg rounds | Avg savings % | Tier kept % | Train time (s) |
|---|---|---|---|---|---|---|
| Q-Learning | 87.0 | 3.983 | 2.96 | 16.9 | 57.5 | 0.9 |
| DQN | 87.6 | 4.859 | 3.09 | 20.3 | 76.3 | 49.2 |

Success rate by provider:

| Algorithm | AWS | GCP | Azure |
|---|---|---|---|
| Q-Learning | 80.5 | 87.0 | 93.6 |
| DQN | 74.1 | 91.6 | 97.7 |

Auto provider selection (best of AWS / GCP / Azure):

| Algorithm | Success % (auto) | Success % (single provider) | Tier kept % | AWS wins | GCP wins | Azure wins |
|---|---|---|---|---|---|---|
| Q-Learning | 96.8 | 87.0 | 87.6 | 7 | 297 | 180 |
| DQN | 97.8 | 87.6 | 98.4 | 0 | 296 | 193 |
