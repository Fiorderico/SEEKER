"""SE(3)-aware genetic operators for native rigid-fragment poses."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .engine import GeneticConformerSearch
from .fragment_pose import FragmentPoseBlock
from .geometry import TorsionDistance
from .models import Individual, NativePoseCoordinate
from .operators import sample_periodic_angle


class PoseGeneticConformerSearch(GeneticConformerSearch):
    """GA engine with manifold-aware operators for fragment-pose blocks."""

    def __init__(
        self,
        *args: Any,
        pose_variables: Sequence[NativePoseCoordinate],
        pose_blocks: Sequence[FragmentPoseBlock] = (),
        **kwargs: Any,
    ) -> None:
        self.pose_variables = tuple(pose_variables)
        self.pose_blocks = tuple(pose_blocks)
        self.pose_variable_indices = frozenset(
            index for block in self.pose_blocks for index in block.variable_indices
        )
        super().__init__(*args, **kwargs)

    def _values_from_alleles(self, alleles: Sequence[float]) -> list[float]:
        return [
            variable.value_from_allele(allele)
            for variable, allele in zip(self.pose_variables, alleles)
        ]

    def _alleles_from_values(self, values: Sequence[float]) -> list[float]:
        return [
            variable.allele_from_value(value)
            for variable, value in zip(self.pose_variables, values)
        ]

    def _random_alleles(self) -> list[float]:
        alleles = super()._random_alleles()
        values = self._values_from_alleles(alleles)
        for block in self.pose_blocks:
            values = block.sample_values(values, self.rng)
        return self._alleles_from_values(values)

    def _mutate(self, alleles: Sequence[float]) -> list[float]:
        scalar_indices = [
            index
            for index in range(len(self.genes))
            if index not in self.pose_variable_indices
        ]
        units: list[tuple[str, object]] = [
            *(("pose", block) for block in self.pose_blocks),
            *(("scalar", index) for index in scalar_indices),
        ]
        if not units:
            raise ValueError("cannot mutate an empty pose genotype")
        kind, selected = self.rng.choice(units)
        if kind == "pose":
            values = self._values_from_alleles(alleles)
            assert isinstance(selected, FragmentPoseBlock)
            return self._alleles_from_values(selected.mutate_values(values, self.rng))
        index = int(selected)
        result = [float(value) for value in alleles]
        gene = self.genes[index]
        previous = result[index]
        for _attempt in range(12):
            candidate = (
                sample_periodic_angle(
                    gene.periodicity,
                    self.rng,
                    self.config.periodicity_grid_step_deg,
                )
                if gene.periodic and not gene.uniform_prior
                else 360.0 * self.rng.random()
            )
            if abs(candidate - previous) > 1.0e-8:
                result[index] = candidate
                break
        return result

    def _crossover(self, first: Individual, second: Individual) -> list[float]:
        alleles = super()._crossover(first, second)
        first_values = self._values_from_alleles(first.alleles)
        second_values = self._values_from_alleles(second.alleles)
        child_values = self._values_from_alleles(alleles)
        for block in self.pose_blocks:
            crossed = block.crossover_values(first_values, second_values, self.rng)
            for index in block.variable_indices:
                child_values[index] = crossed[index]
        return self._alleles_from_values(child_values)

    def _distance_components(
        self, first: Sequence[float], second: Sequence[float]
    ) -> TorsionDistance:
        first_values = self._values_from_alleles(first)
        second_values = self._values_from_alleles(second)
        normalized: list[float] = []
        for block in self.pose_blocks:
            normalized.extend(
                block.normalized_distance_components(first_values, second_values)
            )
        for index, variable in enumerate(self.pose_variables):
            if index not in self.pose_variable_indices:
                normalized.append(
                    variable.scaled_distance(first_values[index], second_values[index])
                )
        if not normalized:
            return TorsionDistance(float("inf"), float("inf"), float("inf"))
        values = [360.0 * min(1.0, max(0.0, item)) for item in normalized]
        return TorsionDistance(
            sum(values) / len(values),
            math.sqrt(sum(value * value for value in values) / len(values)),
            max(values),
        )

    def _genotype_rms(self, first: Sequence[float], second: Sequence[float]) -> float:
        return self._distance_components(first, second).rms_deg

    def _genotype_score(self, first: Sequence[float], second: Sequence[float]) -> float:
        return self._distance_components(first, second).score(
            self.config.resolved_duplicate_mean_threshold_deg,
            self.config.resolved_duplicate_max_threshold_deg,
        )

    def _genotypes_similar(
        self,
        first: Sequence[float],
        second: Sequence[float],
        mean_threshold: float,
        max_threshold: float,
    ) -> bool:
        distance = self._distance_components(first, second)
        return distance.mean_deg < mean_threshold and distance.max_deg < max_threshold
