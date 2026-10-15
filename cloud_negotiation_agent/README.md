# Cloud Resource Negotiation Agent using Reinforcement Learning

A B.Tech CS project in which a **reinforcement-learning customer agent**
negotiates cloud pricing with rule-based **AWS, GCP and Azure** provider
simulations. Two learners are implemented and benchmarked against each other:
**tabular Q-learning** and a **Deep Q-Network (DQN)**.

## Architecture

```
Browser (HTML/CSS/JS)
      │ fetch
      ▼
Flask API (app.py) ───────────────► SQLite history (database/db.py)
      │                               provider · algorithm · tier · qty
      ▼
UserAgent (agents/user_agent.py)
  ├─ learner: QLearningAgent  (rl/q_learning.py)   ← encode_tabular()
  │           DQNAgent        (rl/dqn_agent.py)    ← encode_vector()
  └─ NegotiationEnv (rl/environment.py)  rules + rewards
            └─ ProviderAgent (agents/provider_agent.py) ← providers.json
```

**Auto provider selection.** Leave the provider on *Auto* and the agent negotiates
with AWS, GCP and Azure using the same requirements, then keeps the best deal:
a closed deal beats a failed one → fewer tier downgrades → lower price → fewer rounds.
The UI shows every provider's quote with the winner starred; all three negotiations
are saved in history under one `request_id`. Pick AWS/GCP/Azure explicitly to force a
single provider (used for the per-provider comparison).

**Workflow of one negotiation:** the env opens with the provider's list price →
each round the agent observes the state, picks one of 4 actions
(raise bid · downgrade tier · accept · walk away) → the env applies it, returns a
reward → the provider concedes a little or the deal closes / patience runs out.

## Project structure

```
cloud_negotiation_agent/
├── app.py              Flask app + API
├── train.py            Train + benchmark Q-learning vs DQN
├── config.py           All file paths (one DB, one model, no duplicates)
├── providers.json      Pricing + negotiation personality for AWS / GCP / Azure
├── requirements.txt
├── agents/             provider_agent.py · user_agent.py
├── rl/                 environment.py · state_space.py · q_learning.py
│                       dqn_agent.py · trainer.py
├── models/ database/   session dataclass · SQLite (auto-migrating)
├── static/ templates/  dashboard
├── tests/              pytest suite
└── results/            benchmark.json / .md / learning_curves.png (from train.py)
```

## Setup

```bash
pip install -r requirements.txt
python train.py              # trains both agents + writes results/
python app.py                # http://127.0.0.1:5000
python -m pytest tests -q    # run tests
```
Q-learning works without PyTorch; DQN needs `torch`.
`python train.py --algo dqn --episodes 5000 --seed 7` trains one algorithm.
Old `q_table.pkl` files are detected as stale and ignored — just retrain.

## Providers (`providers.json`)

| | Concession / round | Patience | Price floor | Accept margin |
|---|---|---|---|---|
| AWS | 3 % (slow) | 8–10 rounds | 78 % of list | 2 % |
| GCP | 6.5 % | 5–7 rounds | 60 % | 4 % |
| Azure | 10 % (aggressive) | 4–6 rounds | 55 % | 6 % |

Tiers: **Small** 2 vCPU/4 GB · **Medium** 2/8 · **Large** 4/16. Prices are hourly
per instance; the simulator uses `hourly × 730 × quantity` = USD per month.
Volume discounts: AWS from the report; GCP (sustained-use) and Azure
(enterprise agreement) are *modelled* as quantity tiers — a simulation assumption.

## RL design

**State.** Q-learning: 6-tuple `(provider, tier, price_bin, budget_bin, gap_bin,
patience_bin)` → 5,400 states. DQN: 13-dim normalised vector (provider one-hot,
tier one-hot, log-quantity, round fraction, remaining patience, offer/bid/budget
ratios, offer–bid gap).

**Reward.** −0.3 per round · on a deal: +5 + 10 × savings vs list + 2 × unused-patience
fraction − 1.5 per downgrade · −5 on failure (walk-away, patience exhausted,
accepting over budget) · −1 for bidding below the provider's floor · −3 extra for downgrading a tier that was still affordable. The price used
in the reward is clipped at the provider's floor.

**DQN.** MLP 13→64→64→4, experience replay (20k), target network (hard sync every
200 steps), Huber loss, Adam 1e-3, γ = 0.95, ε-greedy 0.5 → 0.05.

## Results

The DQN agent was trained for 2,000 episodes using seed 42 and evaluated on 899 test scenarios.

| Metric | DQN |
|---|---:|
| Test success rate | 98.0% |
| Feasible success rate | 100.0% |
| Feasible ceiling | 98.0% |
| Average reward | +5.82 |
| Average negotiation rounds | 2.43 |
| Average savings | 17.4% |
| Tier kept | 80.1% |
| Training time | 11 seconds |

### Success rate by provider

| Provider | Success rate |
|---|---:|
| AWS | 97.7% |
| GCP | 98.4% |
| Azure | 97.9% |

### Automatic provider selection

The agent can negotiate with AWS, GCP, and Azure and select the best deal.

| Metric | Result |
|---|---:|
| Auto-selection success rate | 99.4% |
| Single-provider success rate | 98.0% |
| Tier kept | 95.6% |
| AWS wins | 0 |
| GCP wins | 531 |
| Azure wins | 363 |

These results are from a simulated negotiation environment, not live cloud procurement. The reported results are specific to the experiment configuration and scenario set. See `results/benchmark.md` and `results/learning_curves.png` for the benchmark and learning curve.

Reproduce the experiment using:

```bash
python train.py --algo dqn --episodes 2000 --eval-episodes 899 --eval-split test --no-curriculum --seed 42
```


## Data corrections (vs. the data report)
GCP on-demand for Small ($0.1120 → $0.0335) and Large ($0.2280 → $0.1942),
and GCP spot (Medium $0.0617 → $0.0201), were inconsistent with the report's own
committed-use prices and were corrected. **Re-verify all prices against the official
pricing pages before quoting them.**

## Future scope
Curriculum learning (Easy→Expert; `reset(max_rounds=…)` is already the hook),
mixed-provider multi-resource negotiations, Chart.js dashboard, Double/Dueling DQN.
