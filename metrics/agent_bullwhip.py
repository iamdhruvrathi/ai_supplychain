"""Agent bullwhip metrics (Long et al. Definition 1).

Measures run-to-run decision variance amplification across echelons and time.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np

ECHELON_ORDER = ["Retailer", "Wholesaler", "Distributor", "Factory"]


def _run_variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pvariance(values))


def sigma_squared(
    orders_by_run: Dict[int, Dict[str, List[int]]],
    echelon: str,
    week: int,
) -> float:
    """σ²(k,t) = Var_r(q^{(r)}_{k,t})."""
    values = []
    for run_orders in orders_by_run.values():
        series = run_orders.get(echelon, [])
        if week < len(series):
            values.append(float(series[week]))
    return _run_variance(values)


def psi_ratio(
    orders_by_run: Dict[int, Dict[str, List[int]]],
    echelon: str,
    downstream_echelon: str,
    week: int,
) -> Optional[float]:
    """Ψ_k(t) = Var_r(q_{k,t}) / Var_r(q_{k-1,t})."""
    num = sigma_squared(orders_by_run, echelon, week)
    den = sigma_squared(orders_by_run, downstream_echelon, week)
    if den <= 0:
        return None
    return num / den


def phi_ratio(
    orders_by_run: Dict[int, Dict[str, List[int]]],
    echelon: str,
    week: int,
) -> Optional[float]:
    """Φ_k(t) = Var_r(q_{k,t+1}) / Var_r(q_{k,t})."""
    num = sigma_squared(orders_by_run, echelon, week + 1)
    den = sigma_squared(orders_by_run, echelon, week)
    if den <= 0:
        return None
    return num / den


def cumulative_psi(
    psi_values: List[Optional[float]],
) -> Optional[float]:
    """C_j(t) = ∏_{k=1}^{j} Ψ_k(t) for adjacent positive denominators."""
    product = 1.0
    used = False
    for val in psi_values:
        if val is None:
            continue
        product *= val
        used = True
    return product if used else None


def agent_bullwhip_report(
    orders_by_run: Dict[int, Dict[str, List[int]]],
    echelons: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Full agent-bullwhip report for repeated-run order tensors."""
    echelons = echelons or ECHELON_ORDER
    if not orders_by_run:
        return {"sigma_squared": {}, "psi": {}, "phi": {}, "summary": {}}

    max_weeks = max(
        len(series)
        for run in orders_by_run.values()
        for series in run.values()
    )

    sigma: Dict[str, List[float]] = {e: [] for e in echelons}
    psi: Dict[str, List[Optional[float]]] = {e: [] for e in echelons[1:]}
    phi: Dict[str, List[Optional[float]]] = {e: [] for e in echelons}

    for t in range(max_weeks):
        for e in echelons:
            sigma[e].append(sigma_squared(orders_by_run, e, t))
        for i in range(1, len(echelons)):
            psi[echelons[i]].append(
                psi_ratio(orders_by_run, echelons[i], echelons[i - 1], t)
            )
        for e in echelons:
            if t + 1 < max_weeks:
                phi[e].append(phi_ratio(orders_by_run, e, t))
            else:
                phi[e].append(None)

    psi_means = {
        e: float(np.nanmean([v for v in vals if v is not None]))
        if any(v is not None for v in vals)
        else None
        for e, vals in psi.items()
    }
    phi_means = {
        e: float(np.nanmean([v for v in vals if v is not None]))
        if any(v is not None for v in vals)
        else None
        for e, vals in phi.items()
    }

    cross_echelon_amplification = sum(
        1 for v in psi_means.values() if v is not None and v > 1.0
    )
    intertemporal_amplification = sum(
        1 for v in phi_means.values() if v is not None and v > 1.0
    )

    return {
        "sigma_squared": sigma,
        "psi": psi,
        "phi": phi,
        "psi_mean_by_echelon": psi_means,
        "phi_mean_by_echelon": phi_means,
        "summary": {
            "echelons_with_psi_gt_1": cross_echelon_amplification,
            "echelons_with_phi_gt_1": intertemporal_amplification,
        },
    }


def orders_tensor_from_runs(
    run_histories: List[Dict],
) -> Dict[int, Dict[str, List[int]]]:
    """Build {run_id: {echelon: [orders]}} from list of env history dicts."""
    tensor: Dict[int, Dict[str, List[int]]] = {}
    for r, history in enumerate(run_histories):
        tensor[r] = {
            name: list(history.get("orders", {}).get(name, []))
            for name in ECHELON_ORDER
        }
    return tensor
