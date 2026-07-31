"""Registry and normalization for SEEKER optimization objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ObjectiveDefinition:
    name: str
    label: str
    maximize: bool = False
    description: str = ""

    def fitness_value(self, physical_value: float) -> float:
        return -float(physical_value) if self.maximize else float(physical_value)

    def display_value(self, fitness_value: float) -> float:
        return -float(fitness_value) if self.maximize else float(fitness_value)


OBJECTIVES: dict[str, ObjectiveDefinition] = {
    "energy": ObjectiveDefinition(
        "energy", "E",
        description="Electronic or force-field energy from the selected backend.",
    ),
    "hbond": ObjectiveDefinition(
        "hbond", "HB",
        description="Geometric score for conventional N/O/S hydrogen bonds.",
    ),
    "hbond_pi": ObjectiveDefinition(
        "hbond_pi", "HBpi", maximize=True,
        description="Rewards N/O/S-H donors directed toward an eligible pi-ring center.",
    ),
    "hbond_=": ObjectiveDefinition(
        "hbond_=", "HB=", maximize=True,
        description=(
            "Rewards N/O/S-H donors directed toward a detected double-bond midpoint."
        ),
    ),
    "disconnected_components_penalty": ObjectiveDefinition(
        "disconnected_components_penalty", "Disconnected",
        description=(
            "Penalizes intermolecular candidates split into disconnected interaction groups."
        ),
    ),
    "rotational_a": ObjectiveDefinition(
        "rotational_a", "A", maximize=True,
        description="Maximizes rotational constant A.",
    ),
    "rotational_b": ObjectiveDefinition(
        "rotational_b", "B", maximize=True,
        description="Maximizes rotational constant B.",
    ),
    "rotational_c": ObjectiveDefinition(
        "rotational_c", "C", maximize=True,
        description="Maximizes rotational constant C.",
    ),
    "rotor_prolate": ObjectiveDefinition(
        "rotor_prolate", "RotorPro", maximize=True,
        description="Favors a prolate symmetric-top rotor shape.",
    ),
    "rotor_oblate": ObjectiveDefinition(
        "rotor_oblate", "RotorObl", maximize=True,
        description="Favors an oblate symmetric-top rotor shape.",
    ),
    "rotor_spherical": ObjectiveDefinition(
        "rotor_spherical", "RotorSph", maximize=True,
        description="Favors a spherical-top rotor shape.",
    ),
}

ALIASES = {
    "hpi": "hbond_pi",
    "xh_pi": "hbond_pi",
    "hbond_double": "hbond_=",
    "hbond_double_bond": "hbond_=",
    "hbond_eq": "hbond_=",
    "hb_eq": "hbond_=",
    "connectivity_penalty": "disconnected_components_penalty",
    "interaction_connectivity": "disconnected_components_penalty",
    "rot_a": "rotational_a",
    "rot_b": "rotational_b",
    "rot_c": "rotational_c",
    "prolate": "rotor_prolate",
    "oblate": "rotor_oblate",
    "spherical": "rotor_spherical",
}

BASE_OBJECTIVES = ("energy", "hbond")


def normalize_objective(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return ALIASES.get(normalized, normalized)


def active_objectives(extra_csv: str = "") -> tuple[str, ...]:
    extras: list[str] = []
    for token in extra_csv.split(","):
        if not token.strip():
            continue
        name = normalize_objective(token)
        if name not in OBJECTIVES:
            allowed = ", ".join(sorted(set(OBJECTIVES) - set(BASE_OBJECTIVES)))
            raise ValueError(f"unknown optional objective '{token.strip()}'; available: {allowed}")
        if name in BASE_OBJECTIVES:
            continue
        if name not in extras:
            extras.append(name)
    return (*BASE_OBJECTIVES, *extras)


def validate_objectives(names: Sequence[str]) -> None:
    unknown = [name for name in names if name not in OBJECTIVES]
    if unknown:
        raise ValueError("unknown objectives: " + ", ".join(unknown))
    if tuple(names[:2]) != BASE_OBJECTIVES:
        raise ValueError("energy and hbond must remain the first two objectives")


def display_objective_value(name: str, fitness_value: float) -> float:
    return OBJECTIVES[name].display_value(fitness_value)
