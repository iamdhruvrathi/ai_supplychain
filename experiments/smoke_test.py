import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.beer_game import BeerGame
import random

env = BeerGame(max_weeks=5, verbose=False)
env.reset()

actions = {'Retailer':5,'Wholesaler':5,'Distributor':5,'Factory':5}

random.seed(0)

done = False
while not done:
    _, _, done, info = env.step(actions)
    print('week', info['week'], 'total_cost', info['total_system_cost'])
    print('bullwhip', info['bullwhip'])

print('History keys:', list(env.get_history().keys()))
