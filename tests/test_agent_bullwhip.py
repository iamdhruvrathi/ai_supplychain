"""Unit tests for agent bullwhip metrics (Ψ, Φ, σ²)."""

from __future__ import annotations

import pytest

from metrics.agent_bullwhip import (
    agent_bullwhip_report,
    cumulative_psi,
    orders_tensor_from_runs,
    phi_ratio,
    psi_ratio,
    sigma_squared,
)


def _toy_orders(
    retailer: list[int],
    wholesaler: list[int],
    distributor: list[int] | None = None,
    factory: list[int] | None = None,
) -> dict[int, dict[str, list[int]]]:
    distributor = distributor or wholesaler
    factory = factory or distributor
    return {
        0: {
            "Retailer": retailer,
            "Wholesaler": wholesaler,
            "Distributor": distributor,
            "Factory": factory,
        },
        1: {
            "Retailer": [x + 2 for x in retailer],
            "Wholesaler": [x + 4 for x in wholesaler],
            "Distributor": [x + 6 for x in distributor],
            "Factory": [x + 8 for x in factory],
        },
    }


class TestSigmaSquared:
    def test_single_run_returns_zero_variance(self) -> None:
        orders = {0: {"Retailer": [10, 10, 10]}}
        assert sigma_squared(orders, "Retailer", week=0) == 0.0

    def test_two_runs_with_spread(self) -> None:
        orders = {
            0: {"Retailer": [10]},
            1: {"Retailer": [14]},
        }
        assert sigma_squared(orders, "Retailer", week=0) == 4.0

    def test_missing_week_returns_zero(self) -> None:
        orders = {0: {"Retailer": []}}
        assert sigma_squared(orders, "Retailer", week=0) == 0.0


class TestPsiRatio:
    def test_upstream_amplification(self) -> None:
        orders = _toy_orders([10, 10], [10, 30])
        psi = psi_ratio(orders, "Wholesaler", "Retailer", week=1)
        assert psi is not None
        assert psi > 1.0

    def test_division_by_zero_returns_none(self) -> None:
        orders = {
            0: {"Retailer": [5, 5], "Wholesaler": [7, 9]},
            1: {"Retailer": [5, 5], "Wholesaler": [7, 11]},
        }
        assert psi_ratio(orders, "Wholesaler", "Retailer", week=0) is None

    def test_deterministic_toy_example(self) -> None:
        orders = {
            0: {"Retailer": [4], "Wholesaler": [8]},
            1: {"Retailer": [4], "Wholesaler": [12]},
        }
        # Var(retailer)=0 -> None
        assert psi_ratio(orders, "Wholesaler", "Retailer", week=0) is None
        # Var(retailer runs still 0 at week 0; wholesaler var = 4
        assert sigma_squared(orders, "Wholesaler", week=0) == 4.0


class TestPhiRatio:
    def test_temporal_amplification(self) -> None:
        orders = {
            0: {"Retailer": [10, 25]},
            1: {"Retailer": [14, 35]},
        }
        phi = phi_ratio(orders, "Retailer", week=0)
        assert phi is not None
        assert phi > 1.0

    def test_zero_denominator_returns_none(self) -> None:
        orders = {0: {"Retailer": [5, 5]}, 1: {"Retailer": [5, 5]}}
        assert phi_ratio(orders, "Retailer", week=0) is None


class TestCumulativePsi:
    def test_multiplies_valid_values(self) -> None:
        assert cumulative_psi([2.0, 1.5, None]) == pytest.approx(3.0)

    def test_all_none_returns_none(self) -> None:
        assert cumulative_psi([None, None]) is None


class TestAgentBullwhipReport:
    def test_empty_tensor(self) -> None:
        report = agent_bullwhip_report({})
        assert report["summary"] == {}

    def test_report_contains_expected_keys(self) -> None:
        histories = [
            {"orders": {"Retailer": [4, 8], "Wholesaler": [6, 12],
                         "Distributor": [8, 16], "Factory": [10, 20]}},
            {"orders": {"Retailer": [4, 10], "Wholesaler": [6, 18],
                         "Distributor": [8, 24], "Factory": [10, 30]}},
        ]
        tensor = orders_tensor_from_runs(histories)
        report = agent_bullwhip_report(tensor)

        assert "sigma_squared" in report
        assert "psi" in report
        assert "phi" in report
        assert "psi_mean_by_echelon" in report
        assert "phi_mean_by_echelon" in report
        assert report["summary"]["echelons_with_psi_gt_1"] >= 0

    def test_orders_tensor_from_runs_shape(self) -> None:
        histories = [
            {"orders": {"Retailer": [1, 2], "Wholesaler": [3, 4],
                         "Distributor": [5, 6], "Factory": [7, 8]}},
        ]
        tensor = orders_tensor_from_runs(histories)
        assert 0 in tensor
        assert tensor[0]["Retailer"] == [1, 2]
        assert len(tensor[0]) == 4
