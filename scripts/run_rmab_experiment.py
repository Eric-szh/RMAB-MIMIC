#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class EvalConfig:
    episodes: int = 100
    horizon_days: int = 30
    cohort_size: int = 200
    k_capacity: int = 20
    gamma: float = 0.98
    icu_stay_days: int = 7
    reward_alive: float = 1.0
    death_penalty: float = -100.0
    icu_cost: float = 0.25
    seed: int = 42


class Policy:
    name = "base"

    def select(self, states: np.ndarray, eligible: np.ndarray, k: int, **kwargs) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select(self, states: np.ndarray, eligible: np.ndarray, k: int, **kwargs) -> np.ndarray:
        idx = np.where(eligible)[0]
        if len(idx) <= k:
            return idx
        return self.rng.choice(idx, size=k, replace=False)


class SeverityPolicy(Policy):
    name = "severity"

    def __init__(self, severity_score: np.ndarray):
        self.severity_score = severity_score

    def select(self, states: np.ndarray, eligible: np.ndarray, k: int, **kwargs) -> np.ndarray:
        idx = np.where(eligible)[0]
        if len(idx) == 0:
            return idx
        score = self.severity_score[states[idx]]
        order = np.argsort(-score)
        return idx[order[:k]]


class MyopicPolicy(Policy):
    name = "myopic"

    def __init__(self, one_step_gain: np.ndarray):
        self.one_step_gain = one_step_gain

    def select(self, states: np.ndarray, eligible: np.ndarray, k: int, **kwargs) -> np.ndarray:
        idx = np.where(eligible)[0]
        if len(idx) == 0:
            return idx
        gain = self.one_step_gain[states[idx]]
        order = np.argsort(-gain)
        return idx[order[:k]]


class IndexPolicy(Policy):
    name = "index"

    def __init__(self, index: np.ndarray):
        self.index = index

    def select(self, states: np.ndarray, eligible: np.ndarray, k: int, **kwargs) -> np.ndarray:
        idx = np.where(eligible)[0]
        if len(idx) == 0:
            return idx
        ind = self.index[states[idx]]
        order = np.argsort(-ind)
        return idx[order[:k]]


def solve_value(P0: np.ndarray, P1: np.ndarray, reward_state: np.ndarray, icu_cost: float, gamma: float, iters: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Run value iteration for a two-action single-arm MDP and return state values."""
    v = np.zeros_like(reward_state, dtype=float)
    for _ in range(iters):
        q0 = reward_state + gamma * (P0 @ v)
        q1 = reward_state - icu_cost + gamma * (P1 @ v)
        v_next = np.maximum(q0, q1)
        if np.max(np.abs(v_next - v)) < tol:
            v = v_next
            break
        v = v_next
    return v


def make_rewards(P0: np.ndarray, death_state: int, reward_alive: float, death_penalty: float) -> np.ndarray:
    r = np.full(P0.shape[0], reward_alive, dtype=float)
    r[death_state] = death_penalty
    return r


def policy_bundle(P0: np.ndarray, P1: np.ndarray, death_state: int, cfg: EvalConfig, rng: np.random.Generator) -> List[Policy]:
    """Construct random, severity, myopic, and index-style policies from model primitives."""
    mortality_risk = P0[:, death_state]
    severity = mortality_risk.copy()
    r = make_rewards(P0, death_state, cfg.reward_alive, cfg.death_penalty)
    v = solve_value(P0, P1, r, cfg.icu_cost, cfg.gamma)
    q0 = r + cfg.gamma * (P0 @ v)
    q1 = r - cfg.icu_cost + cfg.gamma * (P1 @ v)
    gain = q1 - q0
    return [
        RandomPolicy(rng),
        SeverityPolicy(severity),
        MyopicPolicy(gain),
        IndexPolicy(gain),
    ]


def sample_initial_population(cohort: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(cohort) == 0:
        raise ValueError("Initial cohort is empty")
    idx = rng.choice(np.arange(len(cohort)), size=n, replace=True)
    return cohort.iloc[idx].reset_index(drop=True)


def simulate_episode(
    policy: Policy,
    P0: np.ndarray,
    P1: np.ndarray,
    init_states: np.ndarray,
    subgroup: np.ndarray,
    death_state: int,
    cfg: EvalConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Simulate one RMAB episode with fixed ICU-stay timers and daily top-K admissions."""
    n = len(init_states)
    states = init_states.copy()
    icu_remaining = np.zeros(n, dtype=int)

    total_reward = 0.0
    total_deaths = 0
    icu_days = 0
    new_admissions = 0
    subgroup_reward: Dict[str, float] = {}

    reward_state = make_rewards(P0, death_state, cfg.reward_alive, cfg.death_penalty)

    for _ in range(cfg.horizon_days):
        alive = states != death_state
        in_icu = icu_remaining > 0
        eligible = alive & (~in_icu)

        selected = policy.select(states, eligible, cfg.k_capacity)
        actions = in_icu.astype(int)
        actions[selected] = 1
        icu_remaining[selected] = cfg.icu_stay_days

        new_admissions += int(len(selected))
        icu_days += int(actions.sum())

        total_reward += float(reward_state[states].sum() - cfg.icu_cost * actions.sum())

        next_states = states.copy()
        for i in range(n):
            if states[i] == death_state:
                continue
            probs = P1[states[i]] if actions[i] == 1 else P0[states[i]]
            next_states[i] = int(rng.choice(np.arange(len(probs)), p=probs))

        newly_dead = (states != death_state) & (next_states == death_state)
        total_deaths += int(newly_dead.sum())

        for g in np.unique(subgroup):
            mask = subgroup == g
            subgroup_reward[str(g)] = subgroup_reward.get(str(g), 0.0) + float(reward_state[states[mask]].sum() - cfg.icu_cost * actions[mask].sum())

        icu_remaining = np.maximum(icu_remaining - 1, 0)
        states = next_states

    fairness_gap = 0.0
    if subgroup_reward:
        vals = np.array(list(subgroup_reward.values()), dtype=float) / cfg.horizon_days
        fairness_gap = float(vals.max() - vals.min())

    return {
        "cumulative_reward": total_reward,
        "mortality_events": float(total_deaths),
        "icu_utilization": float(icu_days) / (cfg.horizon_days * n),
        "new_icu_admissions_per_day": float(new_admissions) / cfg.horizon_days,
        "fairness_gap": fairness_gap,
    }


def evaluate_policy(policy: Policy, P0: np.ndarray, P1: np.ndarray, cohort: pd.DataFrame, death_state: int, cfg: EvalConfig, rng: np.random.Generator) -> Dict[str, float]:
    """Estimate policy performance over repeated episodes and aggregate mean/std metrics."""
    metrics = []
    for _ in range(cfg.episodes):
        pop = sample_initial_population(cohort, cfg.cohort_size, rng)
        m = simulate_episode(
            policy=policy,
            P0=P0,
            P1=P1,
            init_states=pop["state"].to_numpy(dtype=int),
            subgroup=pop["age_group"].astype(str).to_numpy(),
            death_state=death_state,
            cfg=cfg,
            rng=rng,
        )
        metrics.append(m)

    out = {"policy": policy.name}
    for k in metrics[0]:
        vals = np.array([m[k] for m in metrics], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RMAB policy simulation and offline evaluation")
    parser.add_argument("--artifact-dir", default="artifacts/rmab")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--cohort-size", type=int, default=200)
    parser.add_argument("--k-capacity", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--icu-stay-days", type=int, default=7)
    parser.add_argument("--reward-alive", type=float, default=1.0)
    parser.add_argument("--death-penalty", type=float, default=-100.0)
    parser.add_argument("--icu-cost", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = EvalConfig(
        episodes=args.episodes,
        horizon_days=args.horizon_days,
        cohort_size=args.cohort_size,
        k_capacity=args.k_capacity,
        gamma=args.gamma,
        icu_stay_days=args.icu_stay_days,
        reward_alive=args.reward_alive,
        death_penalty=args.death_penalty,
        icu_cost=args.icu_cost,
        seed=args.seed,
    )

    artifact_dir = Path(args.artifact_dir).resolve()
    P1 = np.load(artifact_dir / "matrices" / "P_icu_smoothed.npy")
    P0 = np.load(artifact_dir / "matrices" / "P_nonicu_smoothed.npy")
    cohort = pd.read_csv(artifact_dir / "processed" / "initial_cohort.csv")

    diag = json.loads((artifact_dir / "diagnostics.json").read_text(encoding="utf-8"))
    death_state = int(diag["death_state"])

    rng = np.random.default_rng(cfg.seed)
    policies = policy_bundle(P0, P1, death_state, cfg, rng)
    rows = [evaluate_policy(p, P0, P1, cohort, death_state, cfg, rng) for p in policies]
    result = pd.DataFrame(rows).sort_values("cumulative_reward_mean", ascending=False)

    out_dir = artifact_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "policy_comparison.csv", index=False)

    sensitivity = []
    for k in [max(1, cfg.k_capacity // 2), cfg.k_capacity, cfg.k_capacity * 2]:
        for icu_cost in [cfg.icu_cost * 0.5, cfg.icu_cost, cfg.icu_cost * 2]:
            cfg2 = replace(cfg, k_capacity=k, icu_cost=icu_cost)
            for p in policies:
                metrics = evaluate_policy(p, P0, P1, cohort, death_state, cfg2, rng)
                metrics["k_capacity"] = k
                metrics["icu_cost"] = icu_cost
                sensitivity.append(metrics)
    pd.DataFrame(sensitivity).to_csv(out_dir / "sensitivity.csv", index=False)

    summary = {
        "config": cfg.__dict__,
        "best_policy": result.iloc[0]["policy"],
        "outputs": {
            "policy_comparison": str((out_dir / "policy_comparison.csv").relative_to(artifact_dir)),
            "sensitivity": str((out_dir / "sensitivity.csv").relative_to(artifact_dir)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
