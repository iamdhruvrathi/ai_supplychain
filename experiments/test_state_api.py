"""Verify RL-ready state API on BeerGame."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.beer_game import BeerGame

env = BeerGame(max_weeks=3, verbose=False)
env.reset()

actions = {
    "Retailer": 5,
    "Wholesaler": 5,
    "Distributor": 5,
    "Factory": 5,
}

done = False
while not done:
    _, _, done, _ = env.step(actions)

print("Testing RL-ready state API...")
retailer = env.get_state("Retailer")
required = {
    "inventory",
    "backlog",
    "incoming_shipments",
    "pipeline_inventory",
    "last_customer_demand",
    "last_order",
    "current_week",
}
assert required <= retailer.keys(), retailer
print(f"Retailer state: {retailer}")

all_states = env.get_all_states()
assert len(all_states) == 4
assert all(required <= s.keys() for s in all_states.values())

print("[OK] State API test passed!")
