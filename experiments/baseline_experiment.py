"""Run baseline experiments comparing classical policies over repeated simulations.

Saves results to `results/baseline_results.csv` and returns a pandas DataFrame.
"""
from typing import Callable, Dict, List
import os
import random
import pandas as pd
from simulator.beer_game import BeerGame
from policies.base_stock import base_stock_order
from policies.moving_average import moving_average_order
from policies.random_policy import random_order


POLICIES = {
    "random": lambda env, node: random_order(env.get_state()[node]),
    "base_stock": lambda env, node: base_stock_order(env.get_state()[node], target_inventory=20),
    "moving_avg": lambda env, node: moving_average_order(env.get_state()[node], env.get_history()["demand"], window=3, target_inventory=20),
}


def run_single(seed: int, policy_name: str, max_weeks: int = 50) -> Dict:
    random.seed(seed)
    env = BeerGame(max_weeks=max_weeks, verbose=False)
    env.reset()

    done = False
    while not done:
        actions = {node.name: POLICIES[policy_name](env, node.name) for node in env.nodes}
        _, _, done, info = env.step(actions)

    history = env.get_history()
    total_cost = history["total_cost"][-1] if history["total_cost"] else None
    bull = info.get("bullwhip")

    return {
        "seed": seed,
        "policy": policy_name,
        "total_cost": total_cost,
        "bullwhip_overall": bull.get("overall") if bull else None,
        "max_backlog": max(max(vals) if vals else 0 for vals in history["backlog"].values()),
    }


def run_experiment(repeats: int = 30, max_weeks: int = 50, out_csv: str = "results/baseline_results.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows: List[Dict] = []

    for policy in POLICIES.keys():
        for i in range(repeats):
            seed = 1000 + i
            rows.append(run_single(seed, policy, max_weeks=max_weeks))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    df = run_experiment(repeats=10, max_weeks=30)
    print(df.groupby('policy')[['total_cost','bullwhip_overall']].agg(['mean','std']))
