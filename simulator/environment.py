"""RL-ready environment alias (Gymnasium wrapper planned)."""

from simulator.beer_game import BeerGame

BeerGameEnvironment = BeerGame

__all__ = ["BeerGameEnvironment", "BeerGame"]
