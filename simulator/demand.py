"""Customer demand generation with reproducible paths for repeated-run analysis."""

from __future__ import annotations

import random
from typing import List, Optional

from simulator.config import SimulationConfig


class DemandGenerator:
    """Generates retailer demand; supports fixed paths across repeated runs."""

    def __init__(
        self,
        path: Optional[List[int]] = None,
        low: int = 2,
        high: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        self.low = low
        self.high = high
        self._path = list(path) if path is not None else None
        self._index = 0
        self._rng = random.Random(seed) if seed is not None else random.Random()

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "DemandGenerator":
        if config.fixed_demand_path is not None:
            return cls(path=config.fixed_demand_path, low=config.demand_low, high=config.demand_high)
        if config.demand_seed is not None:
            rng = random.Random(config.demand_seed)
            path = [
                rng.randint(config.demand_low, config.demand_high)
                for _ in range(config.max_weeks)
            ]
            return cls(path=path, low=config.demand_low, high=config.demand_high)
        return cls(low=config.demand_low, high=config.demand_high, seed=config.demand_seed)

    def reset(self) -> None:
        """Rewind to start of demand path (for repeated runs)."""
        self._index = 0

    def next_demand(self) -> int:
        if self._path is not None:
            if self._index >= len(self._path):
                return self._path[-1] if self._path else self.low
            value = int(self._path[self._index])
            self._index += 1
            return value
        return int(self._rng.randint(self.low, self.high))

    @property
    def path(self) -> Optional[List[int]]:
        return list(self._path) if self._path is not None else None
