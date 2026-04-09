"""
train.py
--------
Standalone pre-training script for the Q-learning negotiation agent.

Run this BEFORE starting the Flask server to build a quality Q-table.
The trained Q-table is saved to q_table.pkl and will be auto-loaded
by app.py on startup.

Usage:
    python train.py

What it does:
    1. Creates a UserAgent in training_mode=True
    2. Runs TOTAL_EPISODES negotiation episodes with randomized inputs
    3. Prints progress every PRINT_EVERY episodes (success rate, avg reward, epsilon)
    4. Decays epsilon each episode (less exploration over time)
    5. Saves the final Q-table to q_table.pkl
    6. Prints a summary table

Why pre-train?
    Without training, the agent acts randomly (epsilon=0.3 → lots of
    guessing). After 1000 episodes, it has seen hundreds of state
    combinations and learned which actions lead to good outcomes.
"""

import random
import sys
import time
from agents.user_agent  import UserAgent
from rl.state_space     import CPU_BINS, RAM_BINS

# ── Configuration ────────────────────────────────────────────────────

TOTAL_EPISODES = 1000     # total training episodes
PRINT_EVERY    = 100      # print progress every N episodes
Q_TABLE_PATH   = "q_table.pkl"

# ─────────────────────────────────────────────────────────────────────


def run_training():
    """Main training loop."""

    print()
    print("=" * 65)
    print("  Cloud Resource Negotiation Agent — Q-Learning Training")
    print("=" * 65)
    print(f"  Episodes   : {TOTAL_EPISODES}")
    print(f"  Batch size : {PRINT_EVERY}")
    print(f"  Output     : {Q_TABLE_PATH}")
    print(f"  CPU tiers  : {CPU_BINS}")
    print(f"  RAM tiers  : {RAM_BINS}")
    print("=" * 65)
    print()

    start_time = time.time()

    # Create training agent (loads existing Q-table if available)
    agent = UserAgent(training_mode=True, q_table_path=Q_TABLE_PATH)

    # ── Per-episode tracking ──────────────────────────────────────────
    successes       = 0
    total_reward    = 0.0
    episode_rewards = []   # one reward per episode
    batch_rewards   = []   # rewards within the current PRINT_EVERY window

    print(f"  {'Episode':>8}  {'Success%':>9}  {'Batch Avg Reward':>17}  "
          f"{'Epsilon':>8}  {'States':>7}")
    print("  " + "-" * 58)

    for episode in range(1, TOTAL_EPISODES + 1):

        # ── Randomize scenario ─────────────────────────────────────────
        cpu     = random.choice(CPU_BINS)
        ram     = random.choice(RAM_BINS)
        storage = random.randint(50, 500)
        budget  = random.uniform(10, 150)

        # ── Run one negotiation ────────────────────────────────────────
        result = agent.negotiate(cpu, ram, storage, budget)

        # ── Track stats ────────────────────────────────────────────────
        if result.success:
            successes += 1
        total_reward        += result.total_reward
        episode_rewards.append(result.total_reward)
        batch_rewards.append(result.total_reward)

        # ── Epsilon decay ──────────────────────────────────────────────
        # Exploration rate drops from 0.3 → 0.05 over training.
        # After ~1000 episodes with rate=0.995:
        #   0.3 * (0.995^1000) ≈ 0.0018 → clamped to min 0.05
        agent.ql_agent.decay_epsilon(decay_rate=0.995, min_epsilon=0.05)

        # ── Print batch summary ────────────────────────────────────────
        if episode % PRINT_EVERY == 0:
            batch_avg    = sum(batch_rewards) / len(batch_rewards)
            success_rate = successes / episode * 100

            print(f"  {episode:>8}  {success_rate:>8.1f}%  {batch_avg:>+17.3f}  "
                  f"{agent.ql_agent.epsilon:>8.4f}  {len(agent.ql_agent.q_table):>7}")

            # Store batch average for the /api/graph reward chart
            agent.ql_agent.training_rewards.append(round(batch_avg, 3))
            batch_rewards = []   # reset window

            sys.stdout.flush()

    # ── Save final metadata ───────────────────────────────────────────
    agent.ql_agent.episode_count = TOTAL_EPISODES
    agent.save_q_table()

    elapsed = time.time() - start_time

    # ── Final Summary ─────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Training Complete!")
    print("=" * 65)
    print(f"  Total Episodes   : {TOTAL_EPISODES}")
    print(f"  Successful Deals : {successes}  ({successes/TOTAL_EPISODES*100:.1f}%)")
    print(f"  Failed Deals     : {TOTAL_EPISODES - successes}  "
          f"({(TOTAL_EPISODES-successes)/TOTAL_EPISODES*100:.1f}%)")
    print(f"  Avg Reward       : {total_reward/TOTAL_EPISODES:+.3f} per episode")
    print(f"  Final Epsilon    : {agent.ql_agent.epsilon:.4f}")
    print(f"  States in Q-table: {len(agent.ql_agent.q_table)}")
    print(f"  Training Time    : {elapsed:.1f} seconds")
    print(f"  Q-table saved to : {Q_TABLE_PATH}")
    print("=" * 65)
    print()
    print("  Next step: python app.py")
    print("  Then open: http://127.0.0.1:5000")
    print()


if __name__ == "__main__":
    run_training()
