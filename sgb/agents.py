import mesa
import numpy as np
from sgb.metrics import clamp, compute_gmi


class OrganizationAgent(mesa.Agent):
    def __init__(self, model, uid, profile):
        super().__init__(model)

        self.uid = uid
        self.org_type = profile["org_type"]
        self.exchange_volume = profile["exchange_volume"]
        self.dimensions = profile["dimensions"]
        self.gmi_history = []

    def step(self):
        partners = self.model.select_partners(self)

        for partner in partners:
            event = self.model.exchange(self, partner)
            self.update_dimensions(event)

        self.recover()
        self.record_gmi()

    def update_dimensions(self, event):
        if event["success"]:
            self.dimensions["D1"] = clamp(self.dimensions["D1"] + 0.002)
            self.dimensions["D3"] = clamp(self.dimensions["D3"] + 0.001)
            self.dimensions["D5"] = clamp(self.dimensions["D5"] + 0.001)
        else:
            reason = event["failure_reason"]

            if reason == "schema":
                self.dimensions["D1"] = clamp(self.dimensions["D1"] - 0.010)
                self.dimensions["D5"] = clamp(self.dimensions["D5"] - 0.008)

            elif reason == "security":
                self.dimensions["D2"] = clamp(self.dimensions["D2"] - 0.015)

            elif reason == "regulatory":
                self.dimensions["D4"] = clamp(self.dimensions["D4"] - 0.012)

            elif reason == "metadata":
                self.dimensions["D3"] = clamp(self.dimensions["D3"] - 0.010)

    def recover(self):
        framework = self.model.framework_config

        self.dimensions["D2"] = clamp(
            self.dimensions["D2"] + framework["security_recovery_rate"] * 0.01
        )

        self.dimensions["D4"] = clamp(
            self.dimensions["D4"] + framework["regulatory_recovery_rate"] * 0.01
        )

        self.dimensions["D1"] = clamp(
            self.dimensions["D1"] + framework["dq_recovery_rate"] * 0.01
        )

        if "SVC" in self.dimensions:
            self.dimensions["SVC"] = clamp(self.dimensions["SVC"] + 0.0002)

    def record_gmi(self):
        gmi = compute_gmi(self.dimensions, self.model.framework_name)
        self.gmi_history.append(gmi)