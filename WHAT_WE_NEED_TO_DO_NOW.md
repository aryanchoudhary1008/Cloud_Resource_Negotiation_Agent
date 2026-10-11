# Action Plan: Next Implementation Steps

Based on the existing codebase (which features tabular Q-Learning, a generic provider, and a basic Flask dashboard) and the requirements outlined in `Cloud_Resource_Negotiation_Agent.md` and `Multi_Provider_Cloud_Data_Report.md`, here is the complete checklist of what we need to implement next.

## Phase 1: Foundation (High Priority)

### 1. Deep Q-Network (DQN) Integration
*Current Status: Only Tabular Q-Learning (`rl/q_learning.py`) is implemented.*
- [ ] **Create DQN Agent:** Add `rl/dqn_agent.py` using PyTorch or TensorFlow.
- [ ] **Neural Network & Replay:** Implement a neural network for value approximation, along with an Experience Replay Buffer and a Target Network for stable training.
- [ ] **Training Script Update:** Modify `train.py` to support toggling between Q-Learning and DQN to facilitate benchmarking.
- [ ] **State Space Update:** Expand `rl/state_space.py` to include provider ID (one-hot), requested tiers/quantities, and remaining patience.

### 2. Multi-Provider Cloud Support
*Current Status: `agents/provider_agent.py` simulates only one generic provider.*
- [ ] **Configuration:** Create a `providers.json` file using the exact pricing and parameters from the Data Report.
- [ ] **Provider Profiles:** Update `ProviderAgent` to read `providers.json` and adjust its behavior dynamically for **AWS**, **GCP**, and **Azure**:
  - *AWS:* Slow concession, high patience (8-10 rounds), floor ~78%.
  - *GCP:* Moderate concession, medium patience (5-7 rounds), floor ~60%.
  - *Azure:* Aggressive concession, low patience (4-6 rounds), floor ~55%.
- [ ] **Reward Shaping:** Enforce per-provider price floors in the environment so the agent gets penalized if negotiations fail or if it bids below absolute limits.

## Phase 2: Advanced Extension

### 3. Curriculum Learning Pipeline
*Current Status: `train.py` trains on completely random scenarios.*
- [ ] **Progressive Training Levels:** Introduce a curriculum system in `train.py`:
  - *Easy:* 5-10% gap, high patience, 1 resource.
  - *Medium:* 10-20% gap, moderate patience, 1-2 resources.
  - *Hard:* 20-35% gap, low patience, 2-3 resources.
  - *Expert:* 35%+ gap, very low patience, mixed providers, random noise.
- [ ] **Benchmarking:** Generate performance and convergence comparisons between curriculum-based training and standard training.

## Phase 3: Polish & Presentation

### 4. UI/Dashboard Enhancements
*Current Status: The web dashboard (`app.py`, `script.js`, `index.html`) is basic and lacks provider selection or advanced charts.*
- [ ] **Provider Selection UI:** Add a dropdown menu in the frontend to let users pick AWS, GCP, or Azure.
- [ ] **Analytics & Charts:** Integrate interactive charts (e.g., using Chart.js) to visualize:
  - Q-Learning vs. DQN learning curves.
  - Negotiation success rates per provider.
- [ ] **Database Updates:** Ensure `database/db.py` saves the selected provider and algorithm (Q-Learning or DQN) for each session.

### 5. Final Deliverables & Documentation
- [ ] **Testing:** Add automated unit tests for core functionalities (reward calculation, concession logic, state serialization).
- [ ] **Project README:** Write a thorough, structured `README.md` explaining architecture, setup, algorithms, and results.
- [ ] **Reports & Presentation:** Draft the 8-10 page final report, presentation slides, and prepare the live demo script.
