"""
tests/test_audit_fixes.py
-------------------------
Regression tests for the three improvements:
1. Affordability state separation (disentangling state aliasing).
2. Hard budget constraint enforcement (action masking during training/inference).
3. Justified walk-away reward logic vs unnecessary walk-away penalty.
"""

import pytest
from rl.environment import (
    NegotiationEnv,
    RAISE_BID,
    DOWNGRADE,
    ACCEPT,
    WALK_AWAY,
    STEP_COST,
    FAILURE_PENALTY,
    JUSTIFIED_WALKAWAY_REWARD,
)
from rl.state_space import encode_tabular, STATE_VERSION
from agents.user_agent import UserAgent
from agents.provider_agent import ProviderAgent


def test_state_version_bumped():
    assert STATE_VERSION == 5


def test_affordability_state_separation():
    """
    Two states where offer_ratio and budget_ratio fall in the same discrete bins,
    but in one offer <= budget and in the other offer > budget.
    They MUST produce different state tuples.
    """
    # Suppose provider_idx=0, tier_idx=0, remaining=3
    # Both offer_ratio 0.72 and 0.78 fall into price_bin=2 (since 0.5 + 0.2 = 0.7)
    # Both budget_ratio 0.75 fall into budget_bin=2 (since 0.70 <= 0.75 < 0.85)
    obs_affordable = {
        "provider_idx": 0,
        "tier_idx": 0,
        "offer": 72.0,
        "budget": 75.0,
        "offer_ratio": 0.72,
        "budget_ratio": 0.75,
        "gap": 0.25,
        "remaining": 3,
    }
    obs_unaffordable = {
        "provider_idx": 0,
        "tier_idx": 0,
        "offer": 78.0,
        "budget": 75.0,
        "offer_ratio": 0.78,
        "budget_ratio": 0.75,
        "gap": 0.25,
        "remaining": 3,
    }

    s1 = encode_tabular(obs_affordable)
    s2 = encode_tabular(obs_unaffordable)

    assert len(s1) == 7
    assert len(s2) == 7
    assert s1[6] == 1, "Affordable state must have is_affordable=1"
    assert s2[6] == 0, "Unaffordable state must have is_affordable=0"
    assert s1 != s2, "State aliasing must be resolved"


def test_hard_budget_constraint_action_masking():
    """
    UserAgent.legal_actions must exclude ACCEPT (action 2) whenever offer > budget.
    """
    agent = UserAgent("qlearning", load=False)
    obs_unaffordable = {"offer": 100.0, "budget": 80.0}
    obs_affordable = {"offer": 80.0, "budget": 100.0}

    legals_unaff = agent.legal_actions(obs_unaffordable)
    legals_aff = agent.legal_actions(obs_affordable)

    assert ACCEPT not in legals_unaff
    assert ACCEPT in legals_aff


def test_agent_never_accepts_over_budget_even_with_greedy_q():
    """
    Even if Q(state, ACCEPT) is artificially set to +9999, UserAgent with legal
    action masking will NEVER choose ACCEPT when offer > budget.
    """
    agent = UserAgent("qlearning", load=False)
    obs = {
        "provider": "AWS",
        "provider_idx": 0,
        "tier": "Small",
        "tier_idx": 0,
        "offer": 100.0,
        "budget": 50.0,
        "offer_ratio": 1.0,
        "budget_ratio": 0.5,
        "gap": 0.5,
        "remaining": 5,
    }
    state = encode_tabular(obs)
    agent.agent.q_table[state][ACCEPT] = 9999.0  # Tempting invalid action
    agent.agent.epsilon = 0.0  # Greedy

    legals = agent.legal_actions(obs)
    chosen = agent.agent.choose_action(state, legal_actions=legals)

    assert chosen != ACCEPT
    assert chosen in legals


def test_justified_walkaway_reward_on_impossible_scenario():
    """
    Walking away on an impossible scenario gives a non-punitive reward (+1.0 bonus).
    """
    env = NegotiationEnv()
    # Extremely low budget: impossible even on Small
    env.reset("AWS", "Large", 100, 1.0)
    _, reward, done, info = env.step(WALK_AWAY)

    assert done
    assert not info["success"]
    assert "justified" in info["reason"]
    # reward should be STEP_COST + JUSTIFIED_WALKAWAY_REWARD (-0.3 + 1.0 = +0.7)
    assert reward == pytest.approx(STEP_COST + JUSTIFIED_WALKAWAY_REWARD)


def test_unnecessary_walkaway_reward_on_feasible_deal():
    """
    Walking away when a deal was still feasible receives FAILURE_PENALTY (-5.0).
    """
    env = NegotiationEnv()
    # Generous budget: completely feasible
    env.reset("AWS", "Small", 1, 5000.0)
    _, reward, done, info = env.step(WALK_AWAY)

    assert done
    assert not info["success"]
    assert "feasible" in info["reason"]
    assert reward <= FAILURE_PENALTY + 0.01


def test_budget_boundaries_below_at_above():
    """
    Tests exact boundary conditions for offers:
    - Offer just below budget: ACCEPT is legal and succeeds.
    - Offer exactly at budget: ACCEPT is legal and succeeds.
    - Offer just above budget: ACCEPT is illegal and masked out; if forced on env, it fails.
    """
    env = NegotiationEnv()
    agent = UserAgent("qlearning", load=False)

    # 1. Offer exactly at budget ($100.00 vs $100.00)
    obs_at = {"offer": 100.00, "budget": 100.00}
    assert ACCEPT in agent.legal_actions(obs_at)

    env.reset("AWS", "Small", 1, 100.0)
    env.offer = 100.00
    _, reward, done, info = env.step(ACCEPT)
    assert done and info["success"], "Offer exactly at budget must close successfully"
    assert info["final_price"] == 100.00

    # 2. Offer just below budget ($99.99 vs $100.00)
    obs_below = {"offer": 99.99, "budget": 100.00}
    assert ACCEPT in agent.legal_actions(obs_below)

    env.reset("AWS", "Small", 1, 100.0)
    env.offer = 99.99
    _, reward, done, info = env.step(ACCEPT)
    assert done and info["success"], "Offer just below budget must close successfully"
    assert info["final_price"] == 99.99

    # 3. Offer just above budget ($100.01 vs $100.00)
    obs_above = {"offer": 100.01, "budget": 100.00}
    assert ACCEPT not in agent.legal_actions(obs_above), "Offer above budget must be masked out"

    env.reset("AWS", "Small", 1, 100.0)
    env.offer = 100.01
    _, reward, done, info = env.step(ACCEPT)
    assert done and not info["success"], "Offer above budget must fail if ACCEPT is attempted"
    assert "over-budget" in info["reason"]


def test_api_train_metrics_distinction(client):
    """
    Verify POST /api/train returns separate training and evaluation metrics,
    feasible success rate, and distinguishes episodes trained from cumulative episodes.
    """
    res = client.post("/api/train", json={"algorithm": "qlearning", "episodes": 10})
    assert res.status_code == 200
    d = res.get_json()

    assert "episodes_trained" in d and d["episodes_trained"] == 10
    assert "total_episodes" in d and d["total_episodes"] >= 10
    assert "training_success_rate" in d
    assert "training_avg_reward" in d
    assert "eval_success_rate" in d
    assert "eval_feasible_success_rate" in d
    assert "eval_avg_reward" in d

