from collections import deque
import random
import statistics
from typing import Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from simulator.config import OrchestratorMode, RewardConfig, SimulationConfig
from simulator.demand import DemandGenerator
from simulator.node import SupplyChainNode
from simulator.orchestrator import Orchestrator
from simulator.rewards import compute_shaped_reward


class BeerGame:
    def __init__(
        self,
        max_weeks=50,
        verbose=False,
        alpha=1.0,
        beta=0.1,
        gamma=0.5,
        simulation_config: Optional[SimulationConfig] = None,
    ):

        if simulation_config is not None:
            self.config = simulation_config
            self.max_weeks = simulation_config.max_weeks
            self.verbose = simulation_config.verbose
            self.alpha = simulation_config.reward.alpha
            self.beta = simulation_config.reward.beta
            self.gamma = simulation_config.reward.gamma
        else:
            self.config = SimulationConfig(
                max_weeks=max_weeks,
                verbose=verbose,
            )
            self.config.reward = RewardConfig(alpha=alpha, beta=beta, gamma=gamma)
            self.max_weeks = max_weeks
            self.verbose = verbose
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        self.demand_generator = DemandGenerator.from_config(self.config)
        self.orchestrator = Orchestrator(
            mode=self.config.orchestrator_mode,
            demand_history_window=self.config.demand_history_window,
        )
        self._last_customer_demand = 0

        self.reset()

    def reset(self):
        """
        Reset environment
        """

        self.week = 0

        node_kw = {
            "initial_inventory": self.config.initial_inventory,
            "lead_time": self.config.lead_time,
            "holding_cost": self.config.holding_cost,
            "backlog_cost": self.config.backlog_cost,
        }

        self.retailer = SupplyChainNode("Retailer", **node_kw)
        self.wholesaler = SupplyChainNode("Wholesaler", **node_kw)
        self.distributor = SupplyChainNode("Distributor", **node_kw)
        self.factory = SupplyChainNode("Factory", **node_kw)

        self.nodes = [
            self.retailer,
            self.wholesaler,
            self.distributor,
            self.factory
        ]

        self.history = {
            "demand": [],
            "orders": {node.name: [] for node in self.nodes},
            "inventory": {node.name: [] for node in self.nodes},
            "backlog": {node.name: [] for node in self.nodes},
            "step_cost": [],
            "total_cost": [],
            "bullwhip": [],
            "reward": [],
            "tool_order": {node.name: [] for node in self.nodes},
            "llm_order": {node.name: [] for node in self.nodes},
            "difference": {node.name: [] for node in self.nodes},
            "negotiation_proposals": [],
            "consensus_gap": [],
        }

        self.trajectories = []
        self.demand_generator.reset()
        self._last_customer_demand = 0

        return self.get_state()

    def get_state(self, agent_name=None):
        """
        RL-style state representation.

        If agent_name is provided, return a structured state dict for that agent.
        Otherwise return the full multi-agent state mapping.
        """

        if agent_name is not None:
            return self.get_state_dict(agent_name)

        return {
            node.name: node.get_state()
            for node in self.nodes
        }

    def get_state_dict(self, agent_name: str) -> dict:
        """Get RL-ready structured state for a specific agent.

        Returns:
            dict with keys: inventory, backlog, incoming_shipments, pipeline_inventory,
                           last_customer_demand, last_order, current_week
        """
        node_state = next(
            (node for node in self.nodes if node.name == agent_name),
            None
        )
        if node_state is None:
            raise ValueError(f"Unknown agent: {agent_name}")

        downstream_demand = self._get_downstream_demand(agent_name)

        return {
            "inventory": int(node_state.inventory),
            "backlog": int(node_state.backlog),
            "incoming_shipments": int(sum(node_state.incoming_shipments)),
            "pipeline_inventory": int(sum(node_state.incoming_shipments)),
            "last_customer_demand": int(downstream_demand),
            "last_order": int(node_state.last_order),
            "current_week": int(self.week),
            "lead_time": int(self.config.lead_time),
        }

    def _get_downstream_demand(self, agent_name: str) -> int:
        """Demand observed from the downstream echelon (or end customer)."""
        if agent_name == "Retailer":
            return int(self.history["demand"][-1]) if self.history["demand"] else 0
        if agent_name == "Wholesaler":
            return int(self.retailer.last_order)
        if agent_name == "Distributor":
            return int(self.wholesaler.last_order)
        if agent_name == "Factory":
            return int(self.distributor.last_order)
        return 0

    def get_all_states_dict(self) -> dict:
        """Get RL-ready structured state for all agents.

        Returns:
            dict mapping agent_name -> state dict
        """
        return {
            node.name: self.get_state_dict(node.name)
            for node in self.nodes
        }

    def get_all_states(self) -> dict:
        """Alias for get_all_states_dict to support a stable RL-friendly API."""
        return self.get_all_states_dict()

    def get_agent_state(self, agent_name: str) -> dict:
        """RL observation with optional orchestrator augmentation."""
        local = self.get_state_dict(agent_name)
        return self.orchestrator.augment_state(
            agent_name,
            local,
            self._global_context(),
        )

    def get_global_state(self) -> dict:
        """Full supply-chain snapshot for analysis and centralized modes."""
        return {
            "week": self.week,
            "demand_history": list(self.history["demand"]),
            "current_demand": self._last_customer_demand,
            "total_backlog": sum(n.backlog for n in self.nodes),
            "total_inventory": sum(n.inventory for n in self.nodes),
            "echelon_snapshot": {
                n.name: self.get_state_dict(n.name) for n in self.nodes
            },
            "orchestrator_mode": self.config.orchestrator_mode.value,
        }

    def _global_context(self) -> dict:
        return {
            "demand_history": list(self.history["demand"]),
            "current_demand": self._last_customer_demand,
            "total_backlog": sum(n.backlog for n in self.nodes),
            "total_inventory": sum(n.inventory for n in self.nodes),
            "echelon_snapshot": {
                n.name: self.get_state_dict(n.name) for n in self.nodes
            },
        }

    def get_all_agent_states(self) -> dict:
        """All echelon observations with orchestrator augmentation."""
        return {n.name: self.get_agent_state(n.name) for n in self.nodes}

    def generate_customer_demand(self):
        demand = self.demand_generator.next_demand()
        self._last_customer_demand = demand
        return demand

    def compute_metrics(self) -> dict:
        """Aggregate simulator metrics for evaluation pipelines."""
        from metrics.bullwhip import bullwhip_per_agent
        from metrics.stability import stability_summary

        history = self.get_history()
        bull = bullwhip_per_agent(history)
        stability = stability_summary(history)
        return {
            "bullwhip": bull,
            "stability": stability,
            "total_cost": history["total_cost"][-1] if history["total_cost"] else 0.0,
            "mean_step_cost": (
                statistics.mean(history["step_cost"])
                if history["step_cost"]
                else 0.0
            ),
            "trajectory_count": len(self.trajectories),
            "mean_consensus_gap": (
                statistics.mean(history["consensus_gap"])
                if history.get("consensus_gap")
                else 0.0
            ),
            "max_consensus_gap": (
                max(history["consensus_gap"])
                if history.get("consensus_gap")
                else 0
            ),
        }

    def step(self, actions, action_metadata=None):
        """
        RL Environment Step

        actions = {
            "Retailer": 5,
            "Wholesaler": 4,
            "Distributor": 6,
            "Factory": 7
        }

        Returns:
            next_state,
            reward,
            done,
            info
        """

        if self.verbose:
            print(
                f"\n========== WEEK {self.week} =========="
            )

        pre_states = {
            node.name: self.get_state_dict(node.name)
            for node in self.nodes
        }
        current_week = self.week

        # ----------------------------------------
        # STEP 1 — Receive Shipments
        # ----------------------------------------

        for node in self.nodes:

            arrived = node.receive_shipment()

            if self.verbose:
                print(
                    f"{node.name} received "
                    f"{arrived}"
                )

        # ----------------------------------------
        # STEP 2 — Customer Demand
        # ----------------------------------------

        customer_demand = (
            self.generate_customer_demand()
        )

        if self.verbose:
            print(
                f"\nCustomer Demand: "
                f"{customer_demand}"
            )

        # ----------------------------------------
        # STEP 3 — Fulfill Demand
        # ----------------------------------------

        retailer_shipped = (
            self.retailer.fulfill_demand(
                customer_demand
            )
        )

        wholesaler_shipped = (
            self.wholesaler.fulfill_demand(
                self.retailer.last_order
            )
        )

        distributor_shipped = (
            self.distributor.fulfill_demand(
                self.wholesaler.last_order
            )
        )

        factory_shipped = (
            self.factory.fulfill_demand(
                self.distributor.last_order
            )
        )

        # ----------------------------------------
        # STEP 4 — Ship Downstream
        # ----------------------------------------

        self.retailer.add_incoming_shipment(
            wholesaler_shipped
        )

        self.wholesaler.add_incoming_shipment(
            distributor_shipped
        )

        self.distributor.add_incoming_shipment(
            factory_shipped
        )

        # Factory production
        factory_production = (
            self.factory.last_order
        )

        self.factory.add_incoming_shipment(
            factory_production
        )

        # ----------------------------------------
        # STEP 5 — Apply Agent Actions
        # ----------------------------------------

        if self.verbose:
            print("\nOrders Placed:")

        for node in self.nodes:

            action = actions.get(
                node.name,
                0
            )

            node.place_order(action)

            if self.verbose:
                print(
                    f"{node.name}: ordered "
                    f"{action}"
                )

        # ----------------------------------------
        # STEP 6 — Compute Costs
        # ----------------------------------------

        total_system_cost = 0

        for node in self.nodes:

            step_cost = node.compute_costs()

            total_system_cost += step_cost

        action_metadata = action_metadata or {}
        self._record_history(customer_demand, total_system_cost, action_metadata)
        bullwhip_metrics = self.compute_bullwhip()
        total_backlog = sum(node.backlog for node in self.nodes)
        bullwhip_overall = (
            bullwhip_metrics.get("overall")
            if bullwhip_metrics else None
        )
        reward, reward_components = compute_shaped_reward(
            total_system_cost,
            total_backlog,
            bullwhip_overall,
            self.config.reward,
        )

        post_states = {
            node.name: self.get_state_dict(node.name)
            for node in self.nodes
        }

        for node in self.nodes:
            metadata = action_metadata.get(node.name, {})
            self.trajectories.append({
                "week": current_week,
                "agent": node.name,
                "state": dict(pre_states[node.name]),
                "action": int(actions.get(node.name, 0)),
                "tool_order": metadata.get("tool_order"),
                "llm_order": metadata.get("llm_order"),
                "difference": metadata.get("difference"),
                "negotiation_proposals": metadata.get("negotiation_proposals"),
                "consensus_gap": self.history["consensus_gap"][-1],
                "reward": float(reward),
                "next_state": dict(post_states[node.name]),
                "cost": float(total_system_cost),
                "bullwhip": bullwhip_overall,
            })

        self.history["reward"].append(reward)

        # ----------------------------------------
        # STEP 7 — Check Termination
        # ----------------------------------------

        self.week += 1

        done = (
            self.week >= self.max_weeks
        )

        # ----------------------------------------
        # STEP 8 — Extra Info
        # ----------------------------------------

        info = {
            "week": self.week,
            "customer_demand": customer_demand,
            "total_system_cost": total_system_cost,
            "total_backlog": total_backlog,
            "bullwhip": bullwhip_metrics,
            "reward": reward,
            "reward_components": reward_components,
            "consensus_gap": self.history["consensus_gap"][-1],
        }

        next_state = self.get_state()

        # ----------------------------------------
        # Debug Output
        # ----------------------------------------

        if self.verbose:
            self.display_state()

            print(
                f"\nReward: {reward}"
            )

        return (
            next_state,
            reward,
            done,
            info
        )

    def _record_history(self, demand, total_system_cost, action_metadata=None):
        action_metadata = action_metadata or {}
        self.history["demand"].append(demand)

        order_vector = [int(node.last_order) for node in self.nodes]
        consensus_gap = max(order_vector) - min(order_vector) if order_vector else 0

        for node in self.nodes:
            metadata = action_metadata.get(node.name, {})
            self.history["orders"][node.name].append(
                node.last_order
            )
            self.history["inventory"][node.name].append(
                node.inventory
            )
            self.history["backlog"][node.name].append(
                node.backlog
            )
            self.history["tool_order"][node.name].append(
                metadata.get("tool_order")
            )
            self.history["llm_order"][node.name].append(
                metadata.get("llm_order")
            )
            self.history["difference"][node.name].append(
                metadata.get("difference")
            )

        self.history["step_cost"].append(total_system_cost)
        self.history["total_cost"].append(
            sum(node.total_cost for node in self.nodes)
        )
        self.history["bullwhip"].append(
            self.compute_bullwhip()
        )
        self.history["negotiation_proposals"].append(
            {
                node.name: action_metadata.get(node.name, {}).get(
                    "negotiation_proposals"
                )
                for node in self.nodes
            }
        )
        self.history["consensus_gap"].append(consensus_gap)

    def get_trajectories(self):
        """Return rollout trajectories for RL training and analysis."""
        return list(self.trajectories)

    def compute_bullwhip(self):
        if len(self.history["demand"]) < 2:
            return None

        demand_variance = statistics.pvariance(
            self.history["demand"]
        )

        if demand_variance == 0:
            return None

        bullwhip = {}

        for name, orders in self.history["orders"].items():
            if len(orders) < 2:
                bullwhip[name] = None
                continue

            bullwhip[name] = (
                statistics.pvariance(orders)
                / demand_variance
            )

        valid_ratios = [
            value for value in bullwhip.values()
            if value is not None
        ]

        bullwhip["overall"] = (
            statistics.mean(valid_ratios)
            if valid_ratios else None
        )

        return bullwhip

    def get_history(self):
        return self.history

    def plot_orders_vs_demand(self, save_path=None):
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with `pip install matplotlib`."
            )

        if not self.history["demand"]:
            raise ValueError(
                "No history available. Run reset() and step() first."
            )

        weeks = list(
            range(1, len(self.history["demand"]) + 1)
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            weeks,
            self.history["demand"],
            label="Customer Demand",
            color="black",
            linestyle="--",
            marker="o"
        )

        for node_name, orders in self.history["orders"].items():
            plt.plot(
                weeks,
                orders,
                label=f"{node_name} Orders",
                marker="o"
            )

        plt.title("Demand and Order History")
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_inventory_and_backlog(self, save_path=None):
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with `pip install matplotlib`."
            )

        if not self.history["demand"]:
            raise ValueError(
                "No history available. Run reset() and step() first."
            )

        weeks = list(
            range(1, len(self.history["demand"]) + 1)
        )

        plt.figure(figsize=(12, 6))

        for node_name in self.history["inventory"].keys():
            plt.plot(
                weeks,
                self.history["inventory"][node_name],
                label=f"{node_name} Inventory"
            )

        for node_name in self.history["backlog"].keys():
            plt.plot(
                weeks,
                self.history["backlog"][node_name],
                label=f"{node_name} Backlog",
                linestyle="--"
            )

        plt.title("Inventory and Backlog Over Time")
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_inventory(self, save_path=None):
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with `pip install matplotlib`."
            )

        if not self.history["demand"]:
            raise ValueError(
                "No history available. Run reset() and step() first."
            )

        weeks = list(range(1, len(self.history["demand"]) + 1))

        plt.figure(figsize=(10, 6))
        for node_name, inventory in self.history["inventory"].items():
            plt.plot(
                weeks,
                inventory,
                label=f"{node_name} Inventory",
                marker="o"
            )

        plt.title("Inventory Trajectories")
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_backlog(self, save_path=None):
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with `pip install matplotlib`."
            )

        if not self.history["demand"]:
            raise ValueError(
                "No history available. Run reset() and step() first."
            )

        weeks = list(range(1, len(self.history["demand"]) + 1))

        plt.figure(figsize=(10, 6))
        for node_name, backlog in self.history["backlog"].items():
            plt.plot(
                weeks,
                backlog,
                label=f"{node_name} Backlog",
                marker="o"
            )

        plt.title("Backlog Trajectories")
        plt.xlabel("Week")
        plt.ylabel("Units")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def display_state(self):

        print(
            "\nCurrent Supply Chain State:\n"
        )

        for node in self.nodes:
            print(node)


if __name__ == "__main__":

    env = BeerGame(
        max_weeks=10
    )

    state = env.reset()

    done = False

    while not done:

        actions = {
            "Retailer": random.randint(1, 10),
            "Wholesaler": random.randint(1, 10),
            "Distributor": random.randint(1, 10),
            "Factory": random.randint(1, 10)
        }

        next_state, reward, done, info = (
            env.step(actions)
        )

        print("\nInfo:", info)
