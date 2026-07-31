"""Periodicity-aware variation operators for torsional genotypes.

The probability law and oscillating schedule intentionally preserve the
behaviour of the pre-refactor SEEKER engine while keeping all randomness
inside the run-local :class:`random.Random` instance.
"""

from __future__ import annotations

import itertools
import math
import random
from functools import lru_cache
from typing import Sequence

from .geometry import circular_distance_deg
from .models import Gene, NativePoseCoordinate

GeneticCoordinate = Gene | NativePoseCoordinate


def periodic_angle_modes(periodicity: int) -> tuple[float, ...]:
    """Return the maxima of SEEKER's historical periodic torsional prior."""

    if (
        isinstance(periodicity, bool)
        or int(periodicity) != periodicity
        or periodicity < 1
    ):
        raise ValueError("periodicity deve essere un intero positivo")
    order = int(periodicity)
    phase = 180.0 / order if order % 2 else 0.0
    return tuple((phase + 360.0 * index / order) % 360.0 for index in range(order))


@lru_cache(maxsize=None)
def periodic_angle_distribution(
    periodicity: int,
    step_degrees: float = 20.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the discrete torsional prior used by the original engine.

    The support covers the full 0..360 degree domain. ``periodicity`` changes
    the number and position of the statistical modes; it does *not* declare
    angles separated by ``360 / periodicity`` physically equivalent.
    """

    if (
        isinstance(periodicity, bool)
        or int(periodicity) != periodicity
        or periodicity < 1
    ):
        raise ValueError("periodicity deve essere un intero positivo")
    if not math.isfinite(step_degrees) or not 0.0 < step_degrees < 360.0:
        raise ValueError("step_degrees deve essere compreso tra 0 e 360")

    number_of_points = max(1, int(round(360.0 / step_degrees)))
    actual_step = 360.0 / number_of_points
    angles = tuple(index * actual_step for index in range(number_of_points))

    # Same periodic density used before the architectural refactor.
    offset = 1.0 / (2.0 * math.pi) - 0.08
    order = int(periodicity)
    weights = tuple(
        max(
            1.0e-12,
            1.0 + offset + ((-1.0) ** order) * math.cos(order * math.radians(angle)),
        )
        for angle in angles
    )
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    return angles, probabilities


def sample_periodic_angle(
    periodicity: int,
    rng: random.Random,
    step_degrees: float = 20.0,
) -> float:
    """Sample one angle from the periodic torsional prior."""

    angles, probabilities = periodic_angle_distribution(periodicity, step_degrees)
    draw = rng.random()
    cumulative = 0.0
    for angle, probability in zip(angles, probabilities):
        cumulative += probability
        if draw <= cumulative:
            return angle
    return angles[-1]


def sample_periodic_genotype(
    genes: Sequence[GeneticCoordinate],
    rng: random.Random,
    step_degrees: float = 20.0,
) -> list[float]:
    return [
        (
            sample_periodic_angle(gene.periodicity, rng, step_degrees)
            if gene.periodic and not gene.uniform_prior
            else 360.0 * rng.random()
        )
        for gene in genes
    ]


def _periodic_angle_from_quantile(
    periodicity: int,
    quantile: float,
    step_degrees: float,
) -> float:
    angles, probabilities = periodic_angle_distribution(periodicity, step_degrees)
    cumulative = 0.0
    for angle, probability in zip(angles, probabilities):
        cumulative += probability
        if quantile < cumulative:
            return angle
    return angles[-1]


def sample_periodic_genotypes_latin_hypercube(
    genes: Sequence[GeneticCoordinate],
    size: int,
    rng: random.Random,
    step_degrees: float = 20.0,
) -> list[list[float]]:
    """Stratify each gene through the inverse CDF of its existing prior."""

    if size < 1:
        raise ValueError("size deve essere positivo")
    columns: list[list[float]] = []
    for gene in genes:
        quantiles = [(index + rng.random()) / size for index in range(size)]
        rng.shuffle(quantiles)
        columns.append(
            [
                (
                    _periodic_angle_from_quantile(
                        gene.periodicity, value, step_degrees
                    )
                    if gene.periodic and not gene.uniform_prior
                    else 360.0 * value
                )
                for value in quantiles
            ]
        )
    return [
        [columns[gene_index][sample_index] for gene_index in range(len(genes))]
        for sample_index in range(size)
    ]


def scan_initial_genotypes(
    genes: Sequence[GeneticCoordinate],
    reference_alleles: Sequence[float],
    layout: str = "tensor",
    grid: str = "uniform",
    points_mode: str = "fixed",
    fixed_points: int = 3,
) -> list[list[float]]:
    """Build a deterministic torsional SCAN around the reference geometry.

    Values are ordered by a deterministic nearest-neighbour traversal from the
    reference geometry. Unwrapped angles preserve the full-period branch
    convention while the returned chromosome remains in ``0..360`` degrees.
    """

    if not genes or len(genes) != len(reference_alleles):
        raise ValueError("geni e diedri di riferimento incompatibili nello SCAN")
    if layout not in {"tensor", "one-at-a-time"}:
        raise ValueError("layout SCAN non valido")
    if grid not in {"uniform", "periodicity-modes"}:
        raise ValueError("griglia SCAN non valida")
    if points_mode not in {"fixed", "periodicity"}:
        raise ValueError("modalità dei punti SCAN non valida")
    if fixed_points < 1:
        raise ValueError("fixed_points deve essere almeno 1")

    references = [
        float(value) % 360.0
        if gene.periodic
        else min(360.0, max(0.0, float(value)))
        for value, gene in zip(reference_alleles, genes)
    ]

    def unwrap_near(angle: float, reference: float) -> float:
        return reference + (float(angle) - reference + 180.0) % 360.0 - 180.0

    axes: list[list[float]] = []
    for gene, reference in zip(genes, references):
        count = (
            gene.periodicity
            if points_mode == "periodicity"
            else getattr(gene, "scan_points", fixed_points)
        )
        if not gene.periodic:
            if grid == "periodicity-modes":
                raise ValueError(
                    f"{gene.name}: periodicity-modes non è definito per coordinate bounded"
                )
            axis = (
                [reference]
                if count <= 1
                else [360.0 * (index + 0.5) / count for index in range(count)]
            )
        elif grid == "periodicity-modes":
            if count != gene.periodicity:
                raise ValueError(
                    f"{gene.name}: periodicity-modes richiede "
                    f"scan_points={gene.periodicity}, ricevuto {count}"
                )
            axis = [
                unwrap_near(mode, reference)
                for mode in periodic_angle_modes(gene.periodicity)
            ]
        elif count <= 1:
            axis = [reference]
        else:
            axis = [
                reference - 180.0 + 360.0 * (index + 0.5) / count
                for index in range(count)
            ]
        axes.append(axis)

    if layout == "tensor":
        candidates = [list(values) for values in itertools.product(*axes)]
    else:
        candidates = [list(references)]
        seen = {tuple(round(value, 12) for value in references)}
        for gene_index, axis in enumerate(axes):
            for value in axis:
                candidate = list(references)
                candidate[gene_index] = value
                key = tuple(round(item, 12) for item in candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

    remaining = candidates
    ordered: list[list[float]] = []
    current = list(references)
    while remaining:
        chosen_index = min(
            range(len(remaining)),
            key=lambda index: (
                math.sqrt(
                    sum(
                        ((left - right) / 360.0) ** 2
                        for left, right in zip(current, remaining[index])
                    )
                    / len(genes)
                ),
                math.sqrt(
                    sum(
                        (
                            (
                                circular_distance_deg(left, right)
                                if gene.periodic
                                else abs(left - right)
                            )
                            / 360.0
                        ) ** 2
                        for left, right, gene in zip(current, remaining[index], genes)
                    )
                    / len(genes)
                ),
                tuple(remaining[index]),
            ),
        )
        current = remaining.pop(chosen_index)
        ordered.append(current)
    return [
        [
            value % 360.0 if gene.periodic else min(360.0, max(0.0, value))
            for value, gene in zip(candidate, genes)
        ]
        for candidate in ordered
    ]


def mutate_one_periodic(
    alleles: Sequence[float],
    genes: Sequence[GeneticCoordinate],
    rng: random.Random,
    step_degrees: float = 20.0,
    epsilon_deg: float = 1.0e-6,
) -> tuple[list[float], int]:
    """Resample one gene, using a periodic prior or a bounded uniform law."""

    if not alleles:
        raise ValueError("impossibile mutare un genotipo vuoto")
    if len(alleles) != len(genes):
        raise ValueError("numero di alleli e geni incoerente")

    index = rng.randrange(len(alleles))
    previous = float(alleles[index]) % 360.0
    candidate = previous
    for _ in range(10):
        candidate = (
            sample_periodic_angle(
                genes[index].periodicity,
                rng,
                step_degrees,
            )
            if genes[index].periodic and not genes[index].uniform_prior
            else 360.0 * rng.random()
        )
        distance = (
            circular_distance_deg(candidate, previous)
            if genes[index].periodic
            else abs(candidate - previous)
        )
        if distance > epsilon_deg:
            break
    mutated = [
        float(value) % 360.0
        if gene.periodic
        else min(360.0, max(0.0, float(value)))
        for value, gene in zip(alleles, genes)
    ]
    mutated[index] = candidate
    return mutated, index


def mutate_one_local_gaussian(
    alleles: Sequence[float],
    genes: Sequence[GeneticCoordinate],
    rng: random.Random,
    sigma_deg: float = 20.0,
) -> tuple[list[float], int]:
    """Perturb one coordinate locally, wrapping periodic and reflecting bounded genes."""

    if not alleles:
        raise ValueError("impossibile mutare un genotipo vuoto")
    if len(alleles) != len(genes):
        raise ValueError("numero di alleli e geni incoerente")
    if not math.isfinite(sigma_deg) or sigma_deg <= 0.0:
        raise ValueError("sigma_deg deve essere positivo e finito")
    index = rng.randrange(len(alleles))
    candidate = float(alleles[index]) + rng.gauss(0.0, sigma_deg)
    if genes[index].periodic:
        candidate %= 360.0
    else:
        # Reflect instead of clipping so boundary values do not accumulate mass.
        candidate %= 720.0
        if candidate > 360.0:
            candidate = 720.0 - candidate
    mutated = [
        float(value) % 360.0 if gene.periodic else min(360.0, max(0.0, float(value)))
        for value, gene in zip(alleles, genes)
    ]
    mutated[index] = candidate
    return mutated, index


def _bounded_sbx(
    first: float,
    second: float,
    lower: float,
    upper: float,
    eta: float,
    rng: random.Random,
) -> float:
    if first > second:
        first, second = second, first
    if abs(second - first) < 1.0e-12:
        return 0.5 * (first + second)

    draw = rng.random()
    beta = 1.0 + 2.0 * (first - lower) / (second - first)
    alpha = 2.0 - beta ** (-(eta + 1.0))
    if draw <= 1.0 / alpha:
        beta_q = (draw * alpha) ** (1.0 / (eta + 1.0))
    else:
        beta_q = (1.0 / (2.0 - draw * alpha)) ** (1.0 / (eta + 1.0))
    child_one = 0.5 * ((first + second) - beta_q * (second - first))

    beta = 1.0 + 2.0 * (upper - second) / (second - first)
    alpha = 2.0 - beta ** (-(eta + 1.0))
    if draw <= 1.0 / alpha:
        beta_q = (draw * alpha) ** (1.0 / (eta + 1.0))
    else:
        beta_q = (1.0 / (2.0 - draw * alpha)) ** (1.0 / (eta + 1.0))
    child_two = 0.5 * ((first + second) + beta_q * (second - first))
    return child_one if rng.random() < 0.5 else child_two


def circular_sbx_angle(
    first: float,
    second: float,
    rng: random.Random,
    eta: float = 15.0,
) -> float:
    """Apply bounded simulated-binary crossover on the shortest arc."""

    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta SBX deve essere positivo e finito")
    delta = (second - first + 540.0) % 360.0 - 180.0
    relative = _bounded_sbx(0.0, delta, -180.0, 180.0, eta, rng)
    return (first + relative) % 360.0


def circular_sbx_genotype(
    first: Sequence[float],
    second: Sequence[float],
    rng: random.Random,
    eta: float = 15.0,
) -> list[float]:
    if len(first) != len(second):
        raise ValueError("genotipi di lunghezza diversa nel crossover")
    return [
        circular_sbx_angle(left, right, rng, eta)
        for left, right in zip(first, second)
    ]


def mixed_sbx_genotype(
    first: Sequence[float],
    second: Sequence[float],
    genes: Sequence[GeneticCoordinate],
    rng: random.Random,
    eta: float = 15.0,
) -> list[float]:
    """Use circular SBX for torsions and linear bounded SBX for ring genes."""

    if len(first) != len(second) or len(first) != len(genes):
        raise ValueError("genotipi e geni di lunghezza diversa nel crossover")
    children: list[float] = []
    for left, right, gene in zip(first, second, genes):
        if gene.periodic:
            children.append(circular_sbx_angle(left, right, rng, eta))
        else:
            children.append(_bounded_sbx(left, right, 0.0, 360.0, eta, rng))
    return children


def uniform_gene_crossover(
    first: Sequence[float],
    second: Sequence[float],
    genes: Sequence[GeneticCoordinate],
    rng: random.Random,
) -> list[float]:
    """Choose each complete genetic coordinate independently from either parent."""

    if len(first) != len(second) or len(first) != len(genes):
        raise ValueError("genotipi e geni devono avere la stessa lunghezza")
    return [
        (float(left) if rng.random() < 0.5 else float(right)) % 360.0
        if gene.periodic
        else min(360.0, max(0.0, float(left) if rng.random() < 0.5 else float(right)))
        for left, right, gene in zip(first, second, genes)
    ]


def oscillating_operator_probabilities(
    generation: int,
    generations: int,
    base_mutation_weight: float = 0.45,
    base_crossover_weight: float = 0.80,
    mutation_amplitude: float = 0.10,
    crossover_amplitude: float = 0.10,
    oscillations: int = 2,
) -> tuple[float, float]:
    """Return exclusive mutation/crossover probabilities for a generation.

    This is the original counter-phase sinusoidal schedule. ``oscillations``
    retains its historical meaning: the multiplier of pi across the complete
    run (therefore ``2`` describes one complete sine cycle).
    """

    if generation < 0:
        raise ValueError("generation non può essere negativa")
    if generations < 0:
        raise ValueError("generations non può essere negativo")
    if (
        isinstance(oscillations, bool)
        or int(oscillations) != oscillations
        or oscillations < 0
    ):
        raise ValueError("oscillations deve essere un intero non negativo")
    values = (
        base_mutation_weight,
        base_crossover_weight,
        mutation_amplitude,
        crossover_amplitude,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("pesi e ampiezze degli operatori devono essere finiti e non negativi")

    phase = math.pi * int(oscillations) * generation / max(1, generations - 1)
    sine = math.sin(phase)
    mutation_weight = max(0.0, base_mutation_weight + mutation_amplitude * sine)
    crossover_weight = max(0.0, base_crossover_weight - crossover_amplitude * sine)
    total = mutation_weight + crossover_weight
    if total <= 1.0e-12:
        raise ValueError("la schedule produce peso totale nullo per gli operatori")
    return mutation_weight / total, crossover_weight / total


def choose_operator(
    mutation_probability: float,
    crossover_probability: float,
    rng: random.Random,
) -> str:
    if not (
        0.0 <= mutation_probability <= 1.0
        and 0.0 <= crossover_probability <= 1.0
    ):
        raise ValueError("le probabilità degli operatori devono essere comprese tra zero e uno")
    if not math.isclose(
        mutation_probability + crossover_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("le probabilità degli operatori devono sommare a uno")
    return "mutation" if rng.random() < mutation_probability else "crossover"
