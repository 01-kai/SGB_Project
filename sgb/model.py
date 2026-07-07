import mesa
import numpy as np
import networkx as nx
import pandas as pd

from sgb.agents import OrganizationAgent
from sgb.frameworks import FRAMEWORKS
from sgb.metrics import compute_gmi, compute_egmi


class SGBModel(mesa.Model):
    def __init__(self, config, framework_name="DBL", seed=None, beta_override=None):
        super().__init__(rng=seed)

        self.config = config
        self.framework_name = framework_name
        self.framework_config = FRAMEWORKS[framework_name]
        self.N = config["simulation"]["N"]
        self.T = config["simulation"]["T"]

        self.rng = np.random.default_rng(seed)

        self.graph = nx.barabasi_albert_graph(
            self.N,
            config["simulation"]["ba_m"],
            seed=seed
        )

        self.records = []
        self.event_records = []
        self.current_step = 0
        self.beta_override = beta_override

        self.agents_by_uid = {}
        self.create_agents()

    def create_agents(self):
        self.agents_by_uid = {}

        for uid in range(self.N):
            profile = self.generate_profile(uid)
            agent = OrganizationAgent(self, uid, profile)
            self.agents_by_uid[uid] = agent

    def generate_profile(self, uid):
        org_type = self.sample_org_type()
        exchange_volume = self.sample_exchange_volume()

        beta_config = self.beta_override or self.config["population"]["base_distributions"]

        dimensions = {
            "D1": self.sample_beta(beta_config["D1"]),
            "D2": self.sample_beta(beta_config["D2"]),
            "D3": self.sample_beta(beta_config["D3"]),
            "D4": self.sample_beta(beta_config["D4"]),
            "D5": self.sample_beta(beta_config["D5"]),
        }

        if self.framework_name == "ISF":
            dimensions["SVC"] = self.sample_beta(beta_config.get("SVC", [3, 2]))

        return {
            "uid": uid,
            "org_type": org_type,
            "exchange_volume": exchange_volume,
            "dimensions": dimensions,
        }

    def sample_beta(self, params):
        alpha, beta = params
        return float(self.rng.beta(alpha, beta))

    def sample_org_type(self):
        dist = self.config["population"]["org_type_distribution"]
        labels = list(dist.keys())
        probs = list(dist.values())
        return self.rng.choice(labels, p=probs)

    def sample_exchange_volume(self):
        min_v = self.config["exchange"]["min_volume"]
        max_v = self.config["exchange"]["max_volume"]
        value = self.rng.power(2.1) * (max_v - min_v) + min_v
        return int(value)

    def select_partners(self, agent):
        neighbors = list(self.graph.neighbors(agent.uid))

        if not neighbors:
            return []

        number_of_events = max(1, int(agent.exchange_volume / 500))
        number_of_events = min(number_of_events, len(neighbors))

        selected_uids = self.rng.choice(
            neighbors,
            size=number_of_events,
            replace=False,
        )

        return [
            self.agents_by_uid[int(uid)]
            for uid in selected_uids
        ]

    def exchange(self, sender, receiver):
        sender_gmi = compute_gmi(sender.dimensions, self.framework_name)
        receiver_gmi = compute_gmi(receiver.dimensions, self.framework_name)

        success_probability = 0.5 * sender_gmi + 0.5 * receiver_gmi
        success_probability = min(max(success_probability, 0.05), 0.98)

        success = self.rng.random() < success_probability

        if success:
            failure_reason = None
        else:
            failure_reason = self.rng.choice(
                ["schema", "security", "regulatory", "metadata"],
                p=[0.30, 0.25, 0.25, 0.20]
            )

        event = {
            "step": self.current_step,
            "sender": sender.uid,
            "receiver": receiver.uid,
            "success": success,
            "failure_reason": failure_reason,
        }

        self.event_records.append(event)
        return event

    def step(self):
        self.agents.shuffle_do("step")
        self.collect_metrics()
        self.current_step += 1

    def collect_metrics(self):
        agent_gmis = [
            compute_gmi(agent.dimensions, self.framework_name)
            for agent in self.agents
        ]

        egmi = compute_egmi(agent_gmis)

        recent_events = [
            e for e in self.event_records
            if e["step"] == self.current_step
        ]

        if recent_events:
            successful_events = sum(event["success"] for event in recent_events)
            total_events = len(recent_events)
            success_rate = successful_events / total_events
        else:
            successful_events = 0
            total_events = 0
            success_rate = 0.0

        row = {
            "step": self.current_step,
            "framework": self.framework_name,
            "E_GMI": egmi,
            "success_rate": success_rate,
            "successful_events": successful_events,
            "total_events": total_events,
        }

        for d in ["D1", "D2", "D3", "D4", "D5", "SVC"]:
            vals = [
                agent.dimensions[d]
                for agent in self.agents
                if d in agent.dimensions
            ]
            row[d] = np.mean(vals) if vals else None

        self.records.append(row)

    def get_results(self):
        return pd.DataFrame(self.records)

    def get_events(self):
        return pd.DataFrame(self.event_records)