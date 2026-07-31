"""Evidence-informed first-run search presets.

The policy deliberately classifies physical coordinate families before looking
at their scalar dimensionality.  In particular, one rigid-fragment POSE is a
correlated R+ x S2 x SO(3) block, not six unrelated torsions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import FragmentPoseGene, Gene


CoordinateSpec = Gene | FragmentPoseGene


@dataclass(frozen=True)
class SearchPreset:
    """One balanced first-run budget and the decision path that selected it."""

    profile: str
    family: str
    population: int
    offspring: int
    generations: int
    islands: int
    migration_size: int
    hybrid_max_candidates: int
    initialization_strategy: str = "lhs"
    migration_interval: int = 10
    decision_path: tuple[str, ...] = ()

    @property
    def maximum_evaluations(self) -> int:
        return self.islands * (
            self.population + self.generations * self.offspring
        )

    def launcher_settings(self) -> dict[str, int | str]:
        return {
            "population": self.population,
            "offspring": self.offspring,
            "generations": self.generations,
            "islands": self.islands,
            "migration_size": self.migration_size,
            "hybrid_max_candidates": self.hybrid_max_candidates,
            "initialization_strategy": self.initialization_strategy,
            "migration_interval": self.migration_interval,
        }


def recommend_search_preset(
    specs: Sequence[CoordinateSpec],
    *,
    objective_count: int = 2,
) -> SearchPreset:
    """Choose a balanced first-run budget from the physical search topology.

    Budget anchors come from SEEKER's paired/reference campaigns: roughly
    1,000 evaluations for screening and 3,000 for confirmation.  Larger mixed
    and multi-fragment spaces receive intermediate extensions rather than the
    old exponential scalar-dimension jump.
    """

    torsions = sum(isinstance(spec, Gene) for spec in specs)
    poses = sum(isinstance(spec, FragmentPoseGene) for spec in specs)
    internal_dimensions = torsions
    path = [
        f"topology: torsions={torsions}, poses={poses}",
    ]

    if poses:
        family = "intermolecular"
        if poses == 1 and internal_dimensions == 0:
            profile = "single-pose"
            population, generations, islands = 24, 40, 2
            path.append("one rigid POSE block without internal flexibility")
        elif poses <= 2 and internal_dimensions <= 4:
            profile = "coupled-pose"
            population, generations, islands = 32, 50, 2
            path.append("up to two POSE blocks with limited internal flexibility")
        else:
            profile = "multi-pose"
            population, generations, islands = 32, 60, 3
            path.append("three or more POSE blocks, or strongly coupled flexibility")
    else:
        family = "torsional"
        if torsions <= 3:
            profile = "torsional-small"
            population, generations, islands = 24, 30, 2
            path.append("at most three acyclic quaternion torsions")
        elif torsions <= 6:
            profile = "torsional-medium"
            population, generations, islands = 24, 60, 2
            path.append("four to six acyclic quaternion torsions")
        elif torsions <= 10:
            profile = "torsional-large"
            population, generations, islands = 32, 60, 2
            path.append("seven to ten acyclic quaternion torsions")
        else:
            profile = "torsional-very-large"
            population, generations, islands = 40, 70, 3
            path.append("more than ten acyclic quaternion torsions")

    if objective_count >= 3 and population < 32:
        population = 32
        path.append(
            "population raised to 32 because three or more Pareto objectives "
            "need additional crowding support"
        )

    offspring = population
    migration_size = min(4, max(2, population // 12))
    candidate_cap = 24 if population <= 24 else 32 if population <= 32 else 40
    return SearchPreset(
        profile=profile,
        family=family,
        population=population,
        offspring=offspring,
        generations=generations,
        islands=islands,
        migration_size=migration_size,
        hybrid_max_candidates=candidate_cap,
        decision_path=tuple(path),
    )
