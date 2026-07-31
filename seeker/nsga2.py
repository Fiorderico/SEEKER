"""Deterministic NSGA-II ranking and elitist environmental selection."""

from __future__ import annotations

import math
import random
from typing import Sequence

from .geometry import genotype_rms_deg
from .models import Individual


DEFAULT_OBJECTIVES = ("energy", "hbond")


def dominates(
    left: Individual,
    right: Individual,
    objective_names: Sequence[str] = DEFAULT_OBJECTIVES,
) -> bool:
    if left.valid and not right.valid:
        return True
    if not left.valid:
        return False
    if not right.valid:
        return True
    left_values = [left.objective_value(name) for name in objective_names]
    right_values = [right.objective_value(name) for name in objective_names]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def non_dominated_fronts(
    population: Sequence[Individual],
    objective_names: Sequence[str] = DEFAULT_OBJECTIVES,
) -> list[list[int]]:
    dominated_by_count = [0] * len(population)
    dominates_set: list[list[int]] = [[] for _ in population]
    first: list[int] = []

    for p, candidate in enumerate(population):
        for q, other in enumerate(population):
            if p == q:
                continue
            if dominates(candidate, other, objective_names):
                dominates_set[p].append(q)
            elif dominates(other, candidate, objective_names):
                dominated_by_count[p] += 1
        if dominated_by_count[p] == 0:
            candidate.rank = 0
            first.append(p)

    fronts = [first] if first else []
    front_index = 0
    while front_index < len(fronts):
        following: list[int] = []
        for p in fronts[front_index]:
            for q in dominates_set[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    population[q].rank = front_index + 1
                    following.append(q)
        if following:
            fronts.append(following)
        front_index += 1
    return fronts


def assign_crowding(
    population: Sequence[Individual],
    front: Sequence[int],
    objective_names: Sequence[str] = DEFAULT_OBJECTIVES,
) -> None:
    for index in front:
        population[index].crowding = 0.0
    if not front:
        return
    if len(front) <= 2:
        for index in front:
            population[index].crowding = float("inf")
        return

    for objective_name in objective_names:
        ordered = sorted(front, key=lambda idx: population[idx].objective_value(objective_name))
        population[ordered[0]].crowding = float("inf")
        population[ordered[-1]].crowding = float("inf")
        minimum = population[ordered[0]].objective_value(objective_name)
        maximum = population[ordered[-1]].objective_value(objective_name)
        if not (math.isfinite(minimum) and math.isfinite(maximum)) or maximum <= minimum:
            continue
        span = maximum - minimum
        for offset in range(1, len(ordered) - 1):
            index = ordered[offset]
            if math.isinf(population[index].crowding):
                continue
            previous_value = population[ordered[offset - 1]].objective_value(objective_name)
            next_value = population[ordered[offset + 1]].objective_value(objective_name)
            if math.isfinite(previous_value) and math.isfinite(next_value):
                population[index].crowding += (next_value - previous_value) / span


def assign_rank_and_crowding(
    population: Sequence[Individual],
    objective_names: Sequence[str] = DEFAULT_OBJECTIVES,
) -> list[list[int]]:
    for individual in population:
        individual.rank = 10**9
        individual.crowding = 0.0
    fronts = non_dominated_fronts(population, objective_names)
    for front in fronts:
        assign_crowding(population, front, objective_names)
    return fronts


def environmental_selection(
    population: Sequence[Individual],
    target_size: int,
    objective_names: Sequence[str] = DEFAULT_OBJECTIVES,
) -> list[Individual]:
    if target_size < 1:
        raise ValueError("target_size deve essere positivo")
    if len(population) < target_size:
        raise ValueError("popolazione più piccola del target")
    fronts = assign_rank_and_crowding(population, objective_names)
    selected: list[Individual] = []
    for front in fronts:
        members = [population[index] for index in front]
        if len(selected) + len(members) <= target_size:
            selected.extend(members)
            continue
        members.sort(key=lambda item: (-item.crowding, item.id))
        selected.extend(members[: target_size - len(selected)])
        break
    assign_rank_and_crowding(selected, objective_names)
    return selected


def tournament(population: Sequence[Individual], rng: random.Random) -> Individual:
    if not population:
        raise ValueError("tournament su popolazione vuota")
    if len(population) == 1:
        return population[0]
    first, second = rng.sample(list(population), 2)
    first_key = (first.rank, -first.crowding, first.id)
    second_key = (second.rank, -second.crowding, second.id)
    return first if first_key <= second_key else second


def population_diversity_deg(
    population: Sequence[Individual], periodic: Sequence[bool] | None = None
) -> float:
    if len(population) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for offset, left in enumerate(population):
        for right in population[offset + 1 :]:
            value = genotype_rms_deg(left.alleles, right.alleles, periodic)
            if math.isfinite(value):
                total += value
                pairs += 1
    return total / pairs if pairs else 0.0
