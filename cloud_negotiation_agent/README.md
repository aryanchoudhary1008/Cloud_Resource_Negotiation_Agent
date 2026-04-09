# Cloud Resource Negotiation Agent using Reinforcement Learning

A B.Tech CS project that demonstrates a **Q-learning** agent negotiating
cloud resource prices against a rule-based provider agent, with a
Flask web interface and SQLite persistence.

---

## Project Description

Cloud providers (AWS, GCP, Azure) offer dynamic pricing for compute
resources. This project simulates a **multi-round price negotiation**
between:

- **User Agent (RL):** Uses Q-learning to learn optimal negotiation
  strategies — when to push for a lower price, when to accept, and
  when to reduce resource requirements to fit the budget.

- **Provider Agent (Rule-based):** Simulates a cloud provider with a
  fixed pricing model that reduces prices 5% per round to close deals.

The Q-learning agent improves over many training episodes by exploring
different actions and learning which choices maximize the reward.

---

## Architecture Diagram

```
  Browser (HTML/CSS/JS)
       |  Fetch API calls
       v
  Flask REST API  (app.py)
       |
       +---> UserAgent (RL)
       |          |
       |          +---> QLearningAgent
       |          |         |-- Q-table (q_table.pkl)
       |          |         |-- encode_state() → state tuple
       |          |         |-- choose_action() → epsilon-greedy
       |          |         \-- update() → Bellman equation
       |          |
       |          \---> ProviderAgent (Rule-based)
       |                    |-- calculate_base_price()
       |                    |-- make_offer()  (5% discount/round)
       |                    \-- evaluate_user_offer()  (60% floor)
       |
       +---> SQLite DB  (negotiation_history.db)
                 \-- NegotiationSession records
```

---

## Project Structure

```
cloud_negotiation_agent/
├── app.py                   Flask app + all API routes
├── train.py                 Standalone Q-table pre-training script
├── requirements.txt
├── README.md
├── q_table.pkl              (generated after training)
├── negotiation_history.db   (generated at runtime)
│
├── agents/
│   ├── __init__.py
│   ├── user_agent.py        RL-based negotiator (uses Q-learning)
│   └── provider_agent.py    Rule-based cloud provider
│
├── rl/
│   ├── __init__.py
│   ├── q_learning.py        QLearningAgent class (core RL logic)
│   └── state_space.py       State encoding / discretization
│
├── models/
│   ├── __init__.py
│   └── negotiation.py       NegotiationSession dataclass
│
├── database/
│   └── db.py                SQLite init, CRUD, stats
│
├── static/
│   ├── style.css            Dark-theme CSS
│   └── script.js            Frontend JS (Fetch API)
│
└── templates/
    └── index.html           Single-page UI
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pre-train the Q-agent (recommended)

```bash
python train.py
```

This runs 1000 training episodes and saves `q_table.pkl`.
You will see progress printed every 100 episodes:

```
  Episode   Success%   Batch Avg Reward   Epsilon   States
  ----------------------------------------------------------
      100     45.0%             -2.150    0.2952      38
      200     52.0%             -0.840    0.2735      62
      ...
     1000     78.0%             +5.200    0.0500      94
```

### 3. Start the web server

```bash
python app.py
```

### 4. Open the browser

```
http://127.0.0.1:5000
```

---

## API Reference

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/`              | Serve the single-page UI                 |
| POST   | `/api/negotiate` | Run a negotiation session                |
| GET    | `/api/history`   | Return all past sessions (newest first)  |
| GET    | `/api/stats`     | Return aggregate statistics              |
| POST   | `/api/train`     | Run N training episodes (default: 500)   |
| GET    | `/api/graph`     | Return charts as base64 PNG              |

### POST /api/negotiate — Request body

```json
{
  "cpu":           4,
  "ram":           8,
  "storage":       100,
  "budget":        40.0,
  "training_mode": false
}
```

### POST /api/negotiate — Response

```json
{
  "success":      true,
  "rounds":       2,
  "final_price":  27.20,
  "final_cpu":    4,
  "final_ram":    8,
  "storage":      100,
  "budget":       40.0,
  "total_reward": 8.0,
  "log": [
    "Round 1: Provider offers $32.00 for 4 CPU, 8GB RAM, 100GB storage",
    "Round 1: Agent action → Accept provider's offer",
    "  Deal accepted at $32.00!"
  ]
}
```

---

## RL Algorithm Details

### State Space (5 dimensions → tuple key)

| Dimension       | Values                     | Bins |
|-----------------|----------------------------|------|
| cpu_bin         | [2, 4, 8, 16] cores        | 0-3  |
| ram_bin         | [4, 8, 16, 32] GB          | 0-3  |
| budget_bin      | [10, 20, 50, 100] USD      | 0-3  |
| price_bin       | 0=cheap … 4=expensive      | 0-4  |
| round_bin       | rounds 1-5 → 0-4           | 0-4  |

Total distinct states: 4 × 4 × 4 × 5 × 5 = **1600**

### Action Space (4 actions)

| ID | Action                          | Reward effect         |
|----|---------------------------------|-----------------------|
| 0  | Increase offered price by 10%   | +10 if provider accepts|
| 1  | Decrease resource tier (CPU/RAM)| -1 per round          |
| 2  | Accept provider's current offer | +10 (within budget)   |
| 3  | Reject — end negotiation        | -10                   |

### Reward Function

```
+10  deal accepted AND final_price <= budget
 -5  deal accepted BUT final_price  > budget
-10  negotiation failed (rejected or max rounds reached)
 -1  per round (encourages faster resolution)
```

### Bellman Equation (Q-table update)

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') − Q(s, a)]

Where:
  α = 0.1   (learning rate)
  γ = 0.9   (discount factor)
  ε = 0.3   (initial exploration rate, decays to 0.05)
```

### Provider Pricing Formula

```
base_price = (cpu × $2.50) + (ram × $1.50) + (storage × $0.10)

Round discounts:
  Round 1: 0%  off  (full price)
  Round 2: 5%  off
  Round 3: 10% off
  Round 4: 15% off  (final)
  Round 5: 15% off  (same)

Minimum acceptable user offer: 60% of base_price
```

---

## Demo Steps for Professors

1. **Show the cold start:** Run `python app.py` without training first.
   Make a negotiation with budget=$40, CPU=4, RAM=8.
   Show that without training the agent sometimes rejects good deals.

2. **Train the agent:** Click the "Training" tab in the UI → set 500
   episodes → click Start Training. Watch states learned increase.

3. **Show improvement:** Run the same negotiation again. The agent
   should now consistently accept deals within budget in fewer rounds.

4. **History tab:** Show the table of all sessions. Point out
   `final_cpu` vs `cpu` column to demonstrate resource downgrade logic.

5. **Stats tab:** Show success rate, avg rounds, cost efficiency.
   Explain that cost efficiency = budget − final_price (bigger = better).

6. **Graph tab:** Click "Load Graph" to show the reward improvement
   curve and session outcome bar chart (requires matplotlib).

7. **Explain the code:**
   - `rl/q_learning.py` — the Bellman equation in `update()`
   - `rl/state_space.py` — why discretization is needed
   - `agents/user_agent.py` — the negotiation loop
   - `database/db.py` — SQLite persistence

---

## Requirements

- Python 3.8+
- flask
- numpy
- matplotlib (optional, for graphs)
- sqlite3 (built-in, no install needed)
