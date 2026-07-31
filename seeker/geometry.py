"""Molecular graph construction and rigid quaternion torsion moves."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Gene, Molecule

Vector = tuple[float, float, float]
BondGraph = tuple[frozenset[int], ...]


COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "SI": 1.11,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def canonical_element(value: str) -> str:
    token = value.strip()
    if not token:
        raise ValueError("empty atomic symbol")
    return token[0].upper() + token[1:].lower()


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vector, factor: float) -> Vector:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vector) -> Vector:
    length = norm(a)
    if length < 1.0e-14:
        raise ValueError("degenerate rotation axis")
    return _scale(a, 1.0 / length)


def distance(a: Vector, b: Vector) -> float:
    return norm(_sub(a, b))


def angle_deg(a: Vector, vertex: Vector, c: Vector) -> float:
    left = _sub(a, vertex)
    right = _sub(c, vertex)
    denominator = norm(left) * norm(right)
    if denominator < 1.0e-14:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(left, right) / denominator))
    return math.degrees(math.acos(cosine))


def circular_delta_deg(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def circular_distance_deg(a: float, b: float) -> float:
    return abs(circular_delta_deg(a, b))


def genotype_rms_deg(
    left: Sequence[float],
    right: Sequence[float],
    periodic: Sequence[bool] | None = None,
) -> float:
    return torsion_distance(left, right, periodic).rms_deg


@dataclass(frozen=True)
class TorsionDistance:
    """Circular torsional distance components in degrees.

    ``score`` normalizes mean and maximum displacement by explicit scales and
    takes their maximum.  This keeps a large local displacement visible even
    for genotypes with many genes.
    """

    mean_deg: float
    rms_deg: float
    max_deg: float

    def score(self, mean_scale_deg: float, max_scale_deg: float) -> float:
        if mean_scale_deg <= 0.0 or max_scale_deg <= 0.0:
            raise ValueError("torsion scales must be positive")
        return max(self.mean_deg / mean_scale_deg, self.max_deg / max_scale_deg)


def torsion_distance(
    left: Sequence[float],
    right: Sequence[float],
    periodic: Sequence[bool] | None = None,
) -> TorsionDistance:
    """Return distances for a circular, or mixed circular/linear, genotype."""

    if len(left) != len(right) or not left:
        return TorsionDistance(float("inf"), float("inf"), float("inf"))
    mask = tuple(True for _ in left) if periodic is None else tuple(periodic)
    if len(mask) != len(left):
        return TorsionDistance(float("inf"), float("inf"), float("inf"))
    values = [
        circular_distance_deg(a, b) if is_periodic else abs(float(a) - float(b))
        for a, b, is_periodic in zip(left, right, mask)
    ]
    return TorsionDistance(
        sum(values) / len(values),
        math.sqrt(sum(value * value for value in values) / len(values)),
        max(values),
    )


def torsion_distance_score(
    left: Sequence[float],
    right: Sequence[float],
    mean_scale_deg: float,
    max_scale_deg: float,
    periodic: Sequence[bool] | None = None,
) -> float:
    return torsion_distance(left, right, periodic).score(mean_scale_deg, max_scale_deg)


def torsionally_similar(
    left: Sequence[float],
    right: Sequence[float],
    mean_threshold_deg: float,
    max_threshold_deg: float,
    periodic: Sequence[bool] | None = None,
) -> bool:
    components = torsion_distance(left, right, periodic)
    return (
        components.mean_deg < mean_threshold_deg
        and components.max_deg < max_threshold_deg
    )


def genotype_key(
    alleles: Sequence[float],
    precision_digits: int = 8,
    periodic: Sequence[bool] | None = None,
) -> tuple[float, ...]:
    """Canonical exact-genotype key consistent with serialized precision."""

    key: list[float] = []
    epsilon = 10.0 ** (-precision_digits)
    mask = tuple(True for _ in alleles) if periodic is None else tuple(periodic)
    if len(mask) != len(alleles):
        raise ValueError("inconsistent number of alleles and periodicity flags")
    for value, is_periodic in zip(alleles, mask):
        number = float(value) % 360.0 if is_periodic else min(360.0, max(0.0, float(value)))
        canonical = round(number, precision_digits)
        if is_periodic and abs(canonical - 360.0) < epsilon:
            canonical = 0.0
        key.append(canonical)
    return tuple(key)


def maximin_select_genotypes(
    candidates: Sequence[Sequence[float]],
    size: int,
    mean_scale_deg: float,
    max_scale_deg: float,
    first_index: int = 0,
    periodic: Sequence[bool] | None = None,
) -> list[list[float]]:
    """Select a deterministic farthest-point subset from a fixed pool."""

    if size < 1 or size > len(candidates):
        raise ValueError("maximin size is incompatible with the pool")
    if first_index < 0 or first_index >= len(candidates):
        raise ValueError("first_index is outside the pool")
    selected_indices = [first_index]
    remaining = [index for index in range(len(candidates)) if index != first_index]
    while len(selected_indices) < size:
        best_index = max(
            remaining,
            key=lambda index: (
                min(
                    torsion_distance_score(
                        candidates[index],
                        candidates[chosen],
                        mean_scale_deg,
                        max_scale_deg,
                        periodic,
                    )
                    for chosen in selected_indices
                ),
                tuple(-value for value in genotype_key(candidates[index], periodic=periodic)),
            ),
        )
        selected_indices.append(best_index)
        remaining.remove(best_index)
    mask = tuple(True for _ in candidates[0]) if periodic is None else tuple(periodic)
    return [
        [
            float(value) % 360.0 if is_periodic else min(360.0, max(0.0, float(value)))
            for value, is_periodic in zip(candidates[index], mask)
        ]
        for index in selected_indices
    ]


def dihedral_deg(p0: Vector, p1: Vector, p2: Vector, p3: Vector) -> float:
    """Return the signed p0-p1-p2-p3 dihedral in [-180, 180)."""

    b0 = _sub(p0, p1)
    b1 = _sub(p2, p1)
    b2 = _sub(p3, p2)
    b1u = unit(b1)
    v = _sub(b0, _scale(b1u, dot(b0, b1u)))
    w = _sub(b2, _scale(b1u, dot(b2, b1u)))
    x = dot(v, w)
    y = dot(cross(b1u, v), w)
    return math.degrees(math.atan2(y, x))


def quaternion(axis: Vector, angle_rad: float) -> tuple[float, float, float, float]:
    axis_u = unit(axis)
    half = 0.5 * angle_rad
    sine = math.sin(half)
    return (
        math.cos(half),
        axis_u[0] * sine,
        axis_u[1] * sine,
        axis_u[2] * sine,
    )


def rotate_vector(q: tuple[float, float, float, float], value: Vector) -> Vector:
    """Rotate a vector with q * (0, value) * conjugate(q)."""

    w, x, y, z = q
    uv = cross((x, y, z), value)
    uuv = cross((x, y, z), uv)
    return _add(value, _add(_scale(uv, 2.0 * w), _scale(uuv, 2.0)))


def build_bond_graph(molecule: Molecule, tolerance: float = 0.45) -> BondGraph:
    adjacency: list[set[int]] = [set() for _ in molecule.atoms]
    for i, atom_i in enumerate(molecule.atoms):
        radius_i = COVALENT_RADII.get(atom_i.element.upper(), 0.77)
        for j in range(i + 1, len(molecule.atoms)):
            atom_j = molecule.atoms[j]
            radius_j = COVALENT_RADII.get(atom_j.element.upper(), 0.77)
            if distance(atom_i.position, atom_j.position) <= radius_i + radius_j + tolerance:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return tuple(frozenset(neighbours) for neighbours in adjacency)


def graph_distance_leq(graph: BondGraph, start: int, end: int, max_hops: int) -> bool:
    if start == end:
        return True
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for neighbour in graph[node]:
            if neighbour == end:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, hops + 1))
    return False


def same_topology(left: BondGraph, right: BondGraph) -> bool:
    return left == right


@dataclass(frozen=True)
class PreparedGene:
    gene: Gene
    rotate_indices: frozenset[int]


def _component_without_edge(graph: BondGraph, start: int, edge: frozenset[int]) -> set[int]:
    component: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        for neighbour in graph[node]:
            if frozenset((node, neighbour)) == edge:
                continue
            if neighbour not in component:
                stack.append(neighbour)
    return component


def prepare_genes(genes: Sequence[Gene], graph: BondGraph) -> tuple[PreparedGene, ...]:
    """Validate genes and determine the rigid fragment moved by each torsion.

    A central bond that remains connected after removing its edge belongs to a
    ring. Such a bond cannot be moved with a rigid one-sided rotation and is
    rejected explicitly instead of silently distorting the ring.
    """

    prepared: list[PreparedGene] = []
    n_atoms = len(graph)
    for gene in genes:
        _i_idx, j_idx, k_idx, l_idx = gene.atoms
        if any(index < 0 or index >= n_atoms for index in gene.atoms):
            raise ValueError(f"{gene.name}: atom index out of range")
        if k_idx not in graph[j_idx]:
            raise ValueError(
                f"{gene.name}: central bond {j_idx + 1}-{k_idx + 1} does not exist"
            )
        edge = frozenset((j_idx, k_idx))
        k_side = _component_without_edge(graph, k_idx, edge)
        if j_idx in k_side:
            raise ValueError(
                f"{gene.name}: central bond {j_idx + 1}-{k_idx + 1} belongs to a ring; "
                "use a ring-puckering coordinate, not a rigid rotation"
            )
        if l_idx not in k_side:
            raise ValueError(f"{gene.name}: atom l={l_idx + 1} is not on the k side of the bond")
        rotate = frozenset(index for index in k_side if index not in (j_idx, k_idx))
        if not rotate:
            raise ValueError(f"{gene.name}: no movable atoms on the k side")
        prepared.append(PreparedGene(gene, rotate))
    return tuple(prepared)


def apply_torsions(
    reference: Molecule,
    prepared_genes: Sequence[PreparedGene],
    alleles_deg: Sequence[float],
) -> Molecule:
    if len(prepared_genes) != len(alleles_deg):
        raise ValueError("the number of alleles differs from the number of genes")

    positions = [atom.position for atom in reference.atoms]
    for prepared, target in zip(prepared_genes, alleles_deg):
        i_idx, j_idx, k_idx, l_idx = prepared.gene.atoms
        current = dihedral_deg(
            positions[i_idx], positions[j_idx], positions[k_idx], positions[l_idx]
        )
        delta = circular_delta_deg(float(target), current)
        if abs(delta) < 1.0e-10:
            continue
        pivot = positions[k_idx]
        axis = _sub(positions[k_idx], positions[j_idx])
        q = quaternion(axis, math.radians(delta))
        for index in prepared.rotate_indices:
            relative = _sub(positions[index], pivot)
            positions[index] = _add(pivot, rotate_vector(q, relative))

    atoms = [atom.moved(position) for atom, position in zip(reference.atoms, positions)]
    return reference.with_atoms(atoms)


def pair_distances(molecule: Molecule, indices: Iterable[int]) -> dict[tuple[int, int], float]:
    selected = sorted(set(indices))
    return {
        (i, j): distance(molecule.atoms[i].position, molecule.atoms[j].position)
        for offset, i in enumerate(selected)
        for j in selected[offset + 1 :]
    }


@dataclass(frozen=True)
class GeometryScreenResult:
    valid: bool
    reason: str = ""
    minimum_offending_distance: float | None = None
    offending_pair: tuple[int, int] | None = None


def steric_prescreen(
    molecule: Molecule,
    reference_graph: BondGraph,
    hh_scale: float = 0.55,
    heavy_heavy_scale: float = 0.55,
    hydrogen_heavy_scale: float = 0.50,
    exclude_hops: int = 3,
    hh_min_distance_angstrom: float | None = None,
) -> GeometryScreenResult:
    """Reject manifest steric clashes without changing the geometry.

    Bonded pairs and pairs separated by at most ``exclude_hops`` graph edges
    are excluded.  The remaining distances are compared with scaled sums of
    van der Waals radii; this is a conservative geometric filter, not an
    energy model or a new bond-perception step.
    """

    scales = (hh_scale, heavy_heavy_scale, hydrogen_heavy_scale)
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("van der Waals factors must be positive and finite")
    if exclude_hops < 1:
        raise ValueError("exclude_hops must be at least 1")
    if hh_min_distance_angstrom is not None and (
        not math.isfinite(hh_min_distance_angstrom) or hh_min_distance_angstrom <= 0.0
    ):
        raise ValueError("hh_min_distance_angstrom must be positive and finite")
    if len(reference_graph) != len(molecule.atoms):
        raise ValueError("reference graph is incompatible with the molecule")

    worst: tuple[float, float, int, int, str] | None = None
    for first, atom_a in enumerate(molecule.atoms):
        radius_a = VDW_RADII.get(atom_a.element.upper(), 1.70)
        is_h_a = atom_a.element.upper() == "H"
        for second in range(first + 1, len(molecule.atoms)):
            if graph_distance_leq(reference_graph, first, second, exclude_hops):
                continue
            atom_b = molecule.atoms[second]
            is_h_b = atom_b.element.upper() == "H"
            if is_h_a and is_h_b:
                scale, reason = hh_scale, "clash H-H"
            elif is_h_a or is_h_b:
                scale, reason = hydrogen_heavy_scale, "clash H-heavy"
            else:
                scale, reason = heavy_heavy_scale, "clash heavy-heavy"
            radius_b = VDW_RADII.get(atom_b.element.upper(), 1.70)
            cutoff = scale * (radius_a + radius_b)
            if is_h_a and is_h_b and hh_min_distance_angstrom is not None:
                cutoff = max(cutoff, hh_min_distance_angstrom)
            separation = distance(atom_a.position, atom_b.position)
            if separation < cutoff:
                severity = separation / cutoff
                candidate = (severity, separation, first, second, reason)
                if worst is None or candidate < worst:
                    worst = candidate
    if worst is None:
        return GeometryScreenResult(True)
    _severity, separation, first, second, reason = worst
    return GeometryScreenResult(False, reason, separation, (first, second))
