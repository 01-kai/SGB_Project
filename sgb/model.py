from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

import mesa
import networkx as nx
import numpy as np
import pandas as pd

from sgb.agents import OrganizationAgent
from sgb.config import EXPECTED_DIMENSION_SUBMETRICS
from sgb.dimensions import (
    ALL_DIMENSIONS,
    STANDARD_DIMENSIONS,
)
from sgb.distributions import (
    sample_beta,
    sample_categorical,
    sample_truncated_pareto_integer,
)


FAILURE_REASONS = (
    "schema",
    "security",
    "metadata",
    "regulatory",
)


class ModelStateError(ValueError):
    """Raised when the ecosystem model enters an invalid state."""


class OperationalDynamics(Protocol):
    """
    Interface for exchange-driven updates and recovery.

    A concrete neutral implementation will be added in Step 4.
    """

    def apply_exchange_outcome(
        self,
        model: SGBModel,
        sender: OrganizationAgent,
        receiver: OrganizationAgent,
        event: Mapping[str, Any],
    ) -> None:
        """Apply state changes resulting from one exchange."""

    def recover_agent(
        self,
        model: SGBModel,
        agent: OrganizationAgent,
    ) -> None:
        """Apply one recovery operation to an organization."""


class NullOperationalDynamics:
    """
    Dynamics implementation that deliberately performs no state updates.

    This provides a stable integration boundary for Step 4 without placing
    arbitrary update constants inside the model.
    """

    def apply_exchange_outcome(
        self,
        model: SGBModel,
        sender: OrganizationAgent,
        receiver: OrganizationAgent,
        event: Mapping[str, Any],
    ) -> None:
        return None

    def recover_agent(
        self,
        model: SGBModel,
        agent: OrganizationAgent,
    ) -> None:
        return None


class SGBModel(mesa.Model):
    """
    Framework-neutral synthetic governance ecosystem.

    Operational behavior is calculated from D1-D5 without selecting DBL,
    DMM, or ISF. Framework scoring is performed separately in a later
    stage so a framework cannot generate favorable outcomes for itself.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        seed: int | None = None,
        dynamics: OperationalDynamics | None = None,
    ) -> None:
        super().__init__(rng=seed)

        if not isinstance(config, Mapping):
            raise ModelStateError(
                "config must be a mapping."
            )

        self.config = deepcopy(dict(config))
        self.seed = seed

        simulation = self._require_config_mapping(
            "simulation"
        )
        network = self._require_config_mapping(
            "network"
        )

        self.population_size = self._require_positive_integer(
            simulation,
            "population_size",
        )
        self.max_steps = self._require_positive_integer(
            simulation,
            "max_steps",
        )

        attachment_m = self._require_positive_integer(
            network,
            "attachment_m",
        )

        if attachment_m >= self.population_size:
            raise ModelStateError(
                "network.attachment_m must be smaller than "
                "simulation.population_size."
            )

        if network.get("model") != "barabasi_albert":
            raise ModelStateError(
                "Only the barabasi_albert network model "
                "is currently supported."
            )

        self.graph = nx.barabasi_albert_graph(
            n=self.population_size,
            m=attachment_m,
            seed=seed,
        )

        self.dynamics = (
            dynamics
            if dynamics is not None
            else NullOperationalDynamics()
        )

        self._validate_dynamics(self.dynamics)

        self.current_step = 0
        self.records: list[dict[str, Any]] = []
        self.event_records: list[dict[str, Any]] = []
        self.agents_by_uid: dict[
            int,
            OrganizationAgent,
        ] = {}

        self._create_agents()

    def _create_agents(self) -> None:
        """Create and register the configured synthetic organizations."""

        for uid in range(self.population_size):
            agent = OrganizationAgent(
                model=self,
                uid=uid,
                org_type=self._sample_organization_type(),
                exchange_volume=self._sample_exchange_volume(),
                submetrics_by_dimension=(
                    self._sample_submetric_state()
                ),
                formula_config=self.config[
                    "dimension_formulas"
                ],
            )

            self.agents_by_uid[uid] = agent

    def _sample_organization_type(self) -> str:
        probabilities = self.config[
            "population"
        ]["organization_type_distribution"]["probabilities"]

        return sample_categorical(
            rng=self.rng,
            probabilities=probabilities,
        )

    def _sample_exchange_volume(self) -> int:
        exchange = self.config["exchange"]

        return sample_truncated_pareto_integer(
            rng=self.rng,
            minimum=int(exchange["minimum_volume"]),
            maximum=int(exchange["maximum_volume"]),
            alpha=float(exchange["pareto_alpha"]),
        )

    def _sample_submetric_state(
        self,
    ) -> dict[str, dict[str, float]]:
        distributions = self.config[
            "baseline"
        ]["initialization"]["submetric_distributions"]

        state: dict[
            str,
            dict[str, float],
        ] = {}

        for dimension in ALL_DIMENSIONS:
            state[dimension] = {}

            submetric_names = sorted(
                EXPECTED_DIMENSION_SUBMETRICS[
                    dimension
                ]
            )

            for submetric_name in submetric_names:
                parameters = distributions[
                    submetric_name
                ]

                state[dimension][submetric_name] = (
                    sample_beta(
                        rng=self.rng,
                        alpha=float(
                            parameters["alpha"]
                        ),
                        beta=float(
                            parameters["beta"]
                        ),
                    )
                )

        return state

    def get_agent(
        self,
        uid: int,
    ) -> OrganizationAgent:
        """Return an organization by its research identifier."""

        if uid not in self.agents_by_uid:
            raise ModelStateError(
                f"Unknown organization uid: {uid}."
            )

        return self.agents_by_uid[uid]

    def select_partner(
        self,
        agent: OrganizationAgent,
    ) -> OrganizationAgent | None:
        """
        Select one connected exchange partner.

        Every organization initiates one exchange attempt per simulation
        step. Its exchange-volume attribute represents the amount of data
        carried by that attempt rather than an arbitrary number of events.
        """

        neighbors = sorted(
            self.graph.neighbors(agent.uid)
        )

        if not neighbors:
            return None

        selected_uid = int(
            self.rng.choice(neighbors)
        )

        return self.get_agent(selected_uid)

    def operational_readiness(
        self,
        agent: OrganizationAgent,
    ) -> float:
        """
        Calculate framework-neutral operational readiness.

        Readiness is the unweighted arithmetic mean of D1-D5. SVC is
        intentionally excluded because it is a sovereignty-evaluation
        dimension rather than a direct technical exchange capability.
        """

        dimensions = agent.dimensions

        readiness = float(
            np.mean(
                [
                    dimensions[dimension]
                    for dimension
                    in STANDARD_DIMENSIONS
                ]
            )
        )

        return min(
            1.0,
            max(0.0, readiness),
        )

    def pair_success_probability(
        self,
        sender: OrganizationAgent,
        receiver: OrganizationAgent,
    ) -> float:
        """
        Calculate exchange success probability without framework weights.

        Both participants contribute equally through the arithmetic mean
        of their operational-readiness scores.
        """

        probability = (
            self.operational_readiness(sender)
            + self.operational_readiness(receiver)
        ) / 2.0

        return min(
            1.0,
            max(0.0, float(probability)),
        )

    def failure_reason_probabilities(
        self,
        sender: OrganizationAgent,
        receiver: OrganizationAgent,
    ) -> dict[str, float]:
        """
        Derive failure-cause probabilities from operational deficiencies.

        No fixed failure-reason probabilities are used. A lower combined
        dimension score creates a larger corresponding failure deficit.
        """

        sender_dimensions = sender.dimensions
        receiver_dimensions = receiver.dimensions

        schema_capability = float(
            np.mean(
                [
                    sender_dimensions["D1"],
                    sender_dimensions["D5"],
                    receiver_dimensions["D1"],
                    receiver_dimensions["D5"],
                ]
            )
        )

        security_capability = (
            sender_dimensions["D2"]
            + receiver_dimensions["D2"]
        ) / 2.0

        metadata_capability = (
            sender_dimensions["D3"]
            + receiver_dimensions["D3"]
        ) / 2.0

        regulatory_capability = (
            sender_dimensions["D4"]
            + receiver_dimensions["D4"]
        ) / 2.0

        deficits = {
            "schema": max(
                0.0,
                1.0 - schema_capability,
            ),
            "security": max(
                0.0,
                1.0 - security_capability,
            ),
            "metadata": max(
                0.0,
                1.0 - metadata_capability,
            ),
            "regulatory": max(
                0.0,
                1.0 - regulatory_capability,
            ),
        }

        total_deficit = sum(deficits.values())

        if math.isclose(
            total_deficit,
            0.0,
            abs_tol=1e-15,
        ):
            equal_probability = (
                1.0 / len(FAILURE_REASONS)
            )

            return {
                reason: equal_probability
                for reason in FAILURE_REASONS
            }

        return {
            reason: deficits[reason] / total_deficit
            for reason in FAILURE_REASONS
        }

    def exchange(
        self,
        sender: OrganizationAgent,
        receiver: OrganizationAgent,
    ) -> dict[str, Any]:
        """Simulate and record one framework-neutral exchange event."""

        success_probability = (
            self.pair_success_probability(
                sender,
                receiver,
            )
        )

        success = bool(
            self.rng.random()
            < success_probability
        )

        failure_probabilities: dict[
            str,
            float,
        ] | None = None

        if success:
            failure_reason = None
        else:
            failure_probabilities = (
                self.failure_reason_probabilities(
                    sender,
                    receiver,
                )
            )

            failure_reason = str(
                self.rng.choice(
                    list(FAILURE_REASONS),
                    p=[
                        failure_probabilities[reason]
                        for reason
                        in FAILURE_REASONS
                    ],
                )
            )

        event = {
            "event_id": len(self.event_records),
            "step": self.current_step,
            "sender_uid": sender.uid,
            "receiver_uid": receiver.uid,
            "volume": sender.exchange_volume,
            "sender_readiness": (
                self.operational_readiness(sender)
            ),
            "receiver_readiness": (
                self.operational_readiness(receiver)
            ),
            "success_probability": success_probability,
            "success": success,
            "failure_reason": failure_reason,
            "failure_probabilities": (
                failure_probabilities
            ),
        }

        self.event_records.append(event)

        return event

    def execute_agent_step(
        self,
        agent: OrganizationAgent,
    ) -> None:
        """
        Execute one organization action.

        Outcome and recovery hooks are delegated to the configured dynamics
        object. The default null dynamics keeps this model free of arbitrary
        update constants.
        """

        partner = self.select_partner(agent)

        if partner is not None:
            event = self.exchange(
                sender=agent,
                receiver=partner,
            )

            self.dynamics.apply_exchange_outcome(
                model=self,
                sender=agent,
                receiver=partner,
                event=event,
            )

        self.dynamics.recover_agent(
            model=self,
            agent=agent,
        )

    def step(self) -> None:
        """Advance the ecosystem by one complete simulation step."""

        if self.current_step >= self.max_steps:
            raise ModelStateError(
                "The model has already reached max_steps."
            )

        self.agents.shuffle_do("step")

        self.collect_metrics()

        self.current_step += 1

    def run(
        self,
        steps: int | None = None,
    ) -> pd.DataFrame:
        """
        Run the model for additional steps.

        When ``steps`` is omitted, execution continues until max_steps.
        """

        remaining_steps = (
            self.max_steps - self.current_step
        )

        if steps is None:
            requested_steps = remaining_steps
        else:
            requested_steps = self._validate_requested_steps(
                steps
            )

        if requested_steps > remaining_steps:
            raise ModelStateError(
                "Requested steps exceed the configured max_steps."
            )

        for _ in range(requested_steps):
            self.step()

        return self.get_results()

    def collect_metrics(self) -> dict[str, Any]:
        """Collect framework-neutral ecosystem metrics."""

        agents = self.agents.to_list()

        current_events = [
            event
            for event in self.event_records
            if event["step"] == self.current_step
        ]

        total_events = len(current_events)

        successful_events = sum(
            int(event["success"])
            for event in current_events
        )

        total_volume = sum(
            int(event["volume"])
            for event in current_events
        )

        successful_volume = sum(
            int(event["volume"])
            for event in current_events
            if event["success"]
        )

        success_rate = (
            successful_events / total_events
            if total_events
            else 0.0
        )

        volume_success_rate = (
            successful_volume / total_volume
            if total_volume
            else 0.0
        )

        row: dict[str, Any] = {
            "step": self.current_step,
            "agent_count": len(agents),
            "total_events": total_events,
            "successful_events": successful_events,
            "success_rate": success_rate,
            "total_volume": total_volume,
            "successful_volume": successful_volume,
            "volume_success_rate": (
                volume_success_rate
            ),
            "mean_operational_readiness": float(
                np.mean(
                    [
                        self.operational_readiness(
                            agent
                        )
                        for agent in agents
                    ]
                )
            ),
        }

        for dimension in ALL_DIMENSIONS:
            row[dimension] = float(
                np.mean(
                    [
                        agent.get_dimension_score(
                            dimension
                        )
                        for agent in agents
                    ]
                )
            )

        self.records.append(row)

        return dict(row)

    def get_results(self) -> pd.DataFrame:
        """Return step-level operational results."""

        return pd.DataFrame(
            deepcopy(self.records)
        )

    def get_events(self) -> pd.DataFrame:
        """Return exchange-level event results."""

        return pd.DataFrame(
            deepcopy(self.event_records)
        )

    def agent_snapshots(
        self,
    ) -> list[dict[str, Any]]:
        """Return all organization states ordered by uid."""

        return [
            self.agents_by_uid[uid].snapshot()
            for uid in sorted(
                self.agents_by_uid
            )
        ]

    def network_summary(self) -> dict[str, float | int]:
        """Return basic network metadata for reproducibility."""

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "average_degree": float(
                sum(
                    degree
                    for _, degree
                    in self.graph.degree()
                )
                / self.graph.number_of_nodes()
            ),
            "density": float(
                nx.density(self.graph)
            ),
            "connected": bool(
                nx.is_connected(self.graph)
            ),
        }

    def _require_config_mapping(
        self,
        key: str,
    ) -> dict[str, Any]:
        value = self.config.get(key)

        if not isinstance(value, dict):
            raise ModelStateError(
                f"{key} must be a mapping."
            )

        return value

    def _require_positive_integer(
        self,
        container: Mapping[str, Any],
        key: str,
    ) -> int:
        value = container.get(key)

        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise ModelStateError(
                f"{key} must be an integer."
            )

        if value <= 0:
            raise ModelStateError(
                f"{key} must be greater than zero."
            )

        return value

    def _validate_requested_steps(
        self,
        steps: Any,
    ) -> int:
        if isinstance(steps, bool) or not isinstance(
            steps,
            int,
        ):
            raise ModelStateError(
                "steps must be an integer."
            )

        if steps < 0:
            raise ModelStateError(
                "steps cannot be negative."
            )

        return steps

    def _validate_dynamics(
        self,
        dynamics: Any,
    ) -> None:
        required_methods = {
            "apply_exchange_outcome",
            "recover_agent",
        }

        for method_name in required_methods:
            if not callable(
                getattr(
                    dynamics,
                    method_name,
                    None,
                )
            ):
                raise ModelStateError(
                    "dynamics must implement "
                    f"{method_name}()."
                )