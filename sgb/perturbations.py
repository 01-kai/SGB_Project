# sgb/perturbations.py

from sgb.metrics import clamp


def apply_security_incident(model, target_uid, impact):
    """
    Apply a Security Incident perturbation.

    Used in Experiment 2.

    The incident directly reduces the Security Posture D2
    of the target organization and slightly affects its direct neighbors.
    """

    agents_by_uid = {agent.uid: agent for agent in model.agents}

    target = agents_by_uid[target_uid]

    target.dimensions["D2"] = clamp(
        target.dimensions["D2"] - impact
    )

    for neighbor_uid in model.graph.neighbors(target_uid):
        neighbor = agents_by_uid[neighbor_uid]

        neighbor.dimensions["D2"] = clamp(
            neighbor.dimensions["D2"] - impact * 0.25
        )


def apply_regulatory_change(model, impact):
    """
    Apply a Regulatory Change perturbation.

    Used in Experiment 3.

    The regulatory change reduces D4 for all organizations.
    ISF agents are less affected because ISF includes sovereignty
    and regulatory adaptation logic.
    """

    for agent in model.agents:
        reduction = impact

        if model.framework_name == "ISF" and "SVC" in agent.dimensions:
            reduction *= 0.70

        agent.dimensions["D4"] = clamp(
            agent.dimensions["D4"] - reduction
        )

        if "SVC" in agent.dimensions:
            agent.dimensions["SVC"] = clamp(
                agent.dimensions["SVC"] - reduction * 0.50
            )


def apply_dq_shock(model, impact):
    """
    Apply a Data Quality Shock perturbation.

    Used in Experiment 4.

    The shock reduces D1 for all organizations.
    """

    for agent in model.agents:
        agent.dimensions["D1"] = clamp(
            agent.dimensions["D1"] - impact
        )