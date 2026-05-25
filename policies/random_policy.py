import random
from typing import Dict


def random_order(node_state: Dict[str, object], low: int = 0, high: int = 10) -> int:
    return random.randint(low, high)
