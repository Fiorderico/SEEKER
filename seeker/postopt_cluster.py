"""Fragment-aware structural clustering after geometry optimization."""

from __future__ import annotations

import csv
from functools import lru_cache
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import BondGraph, build_bond_graph
from .input import read_xyz
from .models import Molecule
from .symmetry import component_isomorphisms, connected_components


HARTREE_TO_KCAL_MOL = 627.509474


@dataclass(frozen=True)
class OptimizedStructure:
    source: Path
    optimized: Path
    energy_hartree: float
    molecule: Molecule


@dataclass(frozen=True)
class StructuralComparison:
    """Rigid-body-invariant comparison between two XYZ structures."""

    rmsd_angstrom: float
    same_topology: bool


def _coordinates(molecule: Molecule) -> np.ndarray:
    return np.asarray([atom.position for atom in molecule.atoms], dtype=float)


def _rank(coordinates: np.ndarray) -> int:
    if len(coordinates) < 2:
        return 0
    return int(np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0)))


def _alignment_and_comparison_indices(
    molecule: Molecule,
    graph: BondGraph,
    atom_mode: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    if atom_mode not in {"heavy", "all"}:
        raise ValueError("post-optimization RMSD atom mode must be heavy or all")
    components = connected_components(graph)
    symbols = [atom.element.upper() for atom in molecule.atoms]
    heavy = tuple(index for index, symbol in enumerate(symbols) if symbol != "H")
    comparison = heavy if atom_mode == "heavy" and heavy else tuple(range(len(symbols)))
    ranked = sorted(
        components,
        key=lambda item: (
            -sum(symbols[index] != "H" for index in item),
            -len(item),
            min(item),
        ),
    )
    reference = ranked[0]
    reference_heavy = tuple(index for index in reference if symbols[index] != "H")
    coordinates = _coordinates(molecule)
    candidates = (
        reference_heavy,
        reference,
        comparison,
        tuple(range(len(symbols))),
    )
    alignment = next(
        (
            indices
            for indices in candidates
            if len(indices) >= 3 and _rank(coordinates[list(indices)]) >= 2
        ),
        candidates[-1],
    )
    return tuple(alignment), tuple(comparison), components


def _aligned_coordinates(
    mobile: np.ndarray,
    reference: np.ndarray,
    alignment_indices: Sequence[int],
) -> np.ndarray:
    indices = np.asarray(tuple(alignment_indices), dtype=int)
    moving_fit = mobile[indices]
    reference_fit = reference[indices]
    moving_center = moving_fit.mean(axis=0)
    reference_center = reference_fit.mean(axis=0)
    covariance = (moving_fit - moving_center).T @ (reference_fit - reference_center)
    left, _singular, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return (mobile - moving_center) @ rotation + reference_center


def _aligned_coordinates_from_mapping(
    mobile: np.ndarray,
    reference: np.ndarray,
    reference_alignment_indices: Sequence[int],
    reference_to_mobile: dict[int, int],
) -> np.ndarray:
    reference_indices = np.asarray(tuple(reference_alignment_indices), dtype=int)
    mobile_indices = np.asarray(
        tuple(reference_to_mobile[index] for index in reference_alignment_indices),
        dtype=int,
    )
    moving_fit = mobile[mobile_indices]
    reference_fit = reference[reference_indices]
    moving_center = moving_fit.mean(axis=0)
    reference_center = reference_fit.mean(axis=0)
    covariance = (moving_fit - moving_center).T @ (reference_fit - reference_center)
    left, _singular, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return (mobile - moving_center) @ rotation + reference_center


def fragment_aligned_rmsd(
    left: Molecule,
    right: Molecule,
    alignment_indices: Sequence[int],
    comparison_indices: Sequence[int],
) -> float:
    if [atom.element for atom in left.atoms] != [atom.element for atom in right.atoms]:
        raise ValueError("optimized structures have incompatible atom ordering")
    left_xyz = _coordinates(left)
    right_xyz = _coordinates(right)
    aligned = _aligned_coordinates(left_xyz, right_xyz, alignment_indices)
    indices = np.asarray(tuple(comparison_indices), dtype=int)
    differences = aligned[indices] - right_xyz[indices]
    return float(math.sqrt(float(np.mean(np.sum(differences * differences, axis=1)))))


def _minimum_component_assignment_cost(
    reference: Molecule,
    candidate: Molecule,
    reference_graph: BondGraph,
    candidate_graph: BondGraph,
    reference_components: Sequence[Sequence[int]],
    candidate_components: Sequence[Sequence[int]],
    aligned_candidate_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    comparison_indices: Sequence[int],
) -> float:
    """Return the minimum squared displacement over component permutations."""

    if len(reference_components) != len(candidate_components):
        return float("inf")
    compared = set(int(index) for index in comparison_indices)
    costs: list[list[float]] = []
    for reference_component in reference_components:
        row: list[float] = []
        for candidate_component in candidate_components:
            mappings = component_isomorphisms(
                reference,
                reference_graph,
                reference_component,
                candidate,
                candidate_graph,
                candidate_component,
            )
            best = float("inf")
            for mapping in mappings:
                total = 0.0
                for reference_index, candidate_index in zip(
                    reference_component, mapping
                ):
                    if reference_index not in compared:
                        continue
                    difference = (
                        aligned_candidate_xyz[candidate_index]
                        - reference_xyz[reference_index]
                    )
                    total += float(difference @ difference)
                best = min(best, total)
            row.append(best)
        costs.append(row)

    component_count = len(reference_components)

    @lru_cache(maxsize=None)
    def assign(reference_position: int, used_mask: int) -> float:
        if reference_position == component_count:
            return 0.0
        best = float("inf")
        for candidate_position, cost in enumerate(costs[reference_position]):
            if used_mask & (1 << candidate_position) or not math.isfinite(cost):
                continue
            best = min(
                best,
                cost
                + assign(
                    reference_position + 1,
                    used_mask | (1 << candidate_position),
                ),
            )
        return best

    return assign(0, 0)


def _permutation_aware_comparison(
    reference: Molecule,
    candidate: Molecule,
    reference_graph: BondGraph,
    candidate_graph: BondGraph,
    atom_mode: str,
) -> StructuralComparison:
    """Minimize RMSD over identical fragments and graph-equivalent atoms."""

    if len(reference.atoms) != len(candidate.atoms):
        raise ValueError("structures do not contain the same number of atoms")
    _ordered_alignment, comparison_indices, reference_components = (
        _alignment_and_comparison_indices(reference, reference_graph, atom_mode)
    )
    candidate_components = connected_components(candidate_graph)
    symbols = [atom.element.upper() for atom in reference.atoms]
    anchor = min(
        reference_components,
        key=lambda component: (
            -sum(symbols[index] != "H" for index in component),
            -len(component),
            min(component),
        ),
    )
    reference_xyz = _coordinates(reference)
    candidate_xyz = _coordinates(candidate)
    anchor_heavy = tuple(index for index in anchor if symbols[index] != "H")
    alignment_indices = next(
        (
            indices
            for indices in (anchor_heavy, anchor)
            if len(indices) >= 3 and _rank(reference_xyz[list(indices)]) >= 2
        ),
        anchor,
    )
    best_squared = float("inf")
    found_topology = False
    for candidate_anchor in candidate_components:
        anchor_mappings = component_isomorphisms(
            reference,
            reference_graph,
            anchor,
            candidate,
            candidate_graph,
            candidate_anchor,
        )
        for mapping in anchor_mappings:
            found_topology = True
            reference_to_candidate = dict(zip(anchor, mapping))
            aligned = _aligned_coordinates_from_mapping(
                candidate_xyz,
                reference_xyz,
                alignment_indices,
                reference_to_candidate,
            )
            squared = _minimum_component_assignment_cost(
                reference,
                candidate,
                reference_graph,
                candidate_graph,
                reference_components,
                candidate_components,
                aligned,
                reference_xyz,
                comparison_indices,
            )
            best_squared = min(best_squared, squared)
    same_topology = found_topology and math.isfinite(best_squared)
    rmsd = (
        math.sqrt(best_squared / len(comparison_indices))
        if same_topology and comparison_indices
        else float("inf")
    )
    return StructuralComparison(float(rmsd), same_topology)


def _ordered_comparison(
    reference: Molecule,
    candidate: Molecule,
    reference_graph: BondGraph,
    candidate_graph: BondGraph,
    atom_mode: str,
) -> StructuralComparison:
    if [atom.element for atom in reference.atoms] != [
        atom.element for atom in candidate.atoms
    ]:
        raise ValueError("structures do not share the same atoms and ordering")
    alignment_indices, comparison_indices, _components_found = (
        _alignment_and_comparison_indices(reference, reference_graph, atom_mode)
    )
    reference_signature = tuple(
        tuple(sorted(neighbours)) for neighbours in reference_graph
    )
    candidate_signature = tuple(
        tuple(sorted(neighbours)) for neighbours in candidate_graph
    )
    return StructuralComparison(
        rmsd_angstrom=fragment_aligned_rmsd(
            candidate,
            reference,
            alignment_indices,
            comparison_indices,
        ),
        same_topology=reference_signature == candidate_signature,
    )


def compare_structures(
    reference: Molecule,
    candidate: Molecule,
    *,
    atom_mode: str = "all",
    topology_tolerance: float = 0.45,
    permutation_mode: str = "equivalent",
) -> StructuralComparison:
    """Compare structures after alignment of the main covalent fragment.

    RMSD alone is not sufficient when optimization makes or breaks a bond, so
    the inferred covalent topology is reported independently.  The default
    minimizes over exchanges of graph-isomorphic fragments and their internal
    graph automorphisms; ``ordered`` preserves strict XYZ-index correspondence.
    """

    if permutation_mode not in {"equivalent", "ordered"}:
        raise ValueError("permutation mode must be equivalent or ordered")
    reference_graph = build_bond_graph(reference, topology_tolerance)
    candidate_graph = build_bond_graph(candidate, topology_tolerance)
    return (
        _permutation_aware_comparison(
            reference,
            candidate,
            reference_graph,
            candidate_graph,
            atom_mode,
        )
        if permutation_mode == "equivalent"
        else _ordered_comparison(
            reference,
            candidate,
            reference_graph,
            candidate_graph,
            atom_mode,
        )
    )


def _complete_linkage(
    distance_matrix: np.ndarray,
    threshold_angstrom: float,
) -> list[list[int]]:
    clusters: list[list[int]] = [[index] for index in range(len(distance_matrix))]
    while True:
        possible: list[tuple[float, tuple[int, ...], tuple[int, ...], int, int]] = []
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                maximum = max(distance_matrix[i, j] for i in left for j in right)
                if maximum <= threshold_angstrom:
                    possible.append(
                        (
                            float(maximum),
                            tuple(left),
                            tuple(right),
                            left_index,
                            right_index,
                        )
                    )
        if not possible:
            break
        _distance, _left_key, _right_key, left_index, right_index = min(possible)
        clusters[left_index] = sorted([*clusters[left_index], *clusters[right_index]])
        del clusters[right_index]
    return clusters


def _load_optimized(optimization_csv: Path) -> list[OptimizedStructure]:
    rows: list[OptimizedStructure] = []
    with optimization_csv.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            optimized = str(raw.get("optimized_xyz", "")).strip()
            energy = str(raw.get("energy_hartree", "")).strip()
            if raw.get("status") != "ok" or not optimized or not energy:
                continue
            optimized_path = Path(optimized).resolve()
            if not optimized_path.is_file():
                raise FileNotFoundError(f"optimized XYZ not found: {optimized_path}")
            rows.append(
                OptimizedStructure(
                    source=Path(str(raw["source"])).resolve(),
                    optimized=optimized_path,
                    energy_hartree=float(energy),
                    molecule=read_xyz(optimized_path),
                )
            )
    if not rows:
        raise ValueError(f"no successful optimized structures in {optimization_csv}")
    rows.sort(key=lambda item: (item.energy_hartree, str(item.optimized)))
    return rows


def cluster_optimized_candidates(
    optimization: str | Path,
    output_dir: str | Path,
    *,
    rmsd_threshold_angstrom: float = 0.30,
    energy_window_kcal_mol: float | None = 10.0,
    atom_mode: str = "all",
    topology_tolerance: float = 0.45,
    permutation_mode: str = "equivalent",
    unique_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Cluster optimized minima and retain the lowest-energy representative.

    The default comparison includes hydrogens because geometry optimization can
    produce distinct OH, SH, or NH arrangements without materially moving the
    heavy atom skeleton.  By default, only graph-equivalent terminal atoms and
    graph-isomorphic disconnected fragments may exchange labels.
    """

    if not math.isfinite(rmsd_threshold_angstrom) or rmsd_threshold_angstrom <= 0.0:
        raise ValueError("post-optimization RMSD threshold must be positive and finite")
    if energy_window_kcal_mol is not None and (
        not math.isfinite(energy_window_kcal_mol) or energy_window_kcal_mol < 0.0
    ):
        raise ValueError("post-optimization energy window must be non-negative and finite")
    if permutation_mode not in {"equivalent", "ordered"}:
        raise ValueError("permutation mode must be equivalent or ordered")
    optimization_csv = Path(optimization).resolve()
    destination = Path(output_dir).resolve()
    structures = _load_optimized(optimization_csv)
    minimum_energy = min(item.energy_hartree for item in structures)
    eligible = (
        list(structures)
        if energy_window_kcal_mol is None
        else [
            item
            for item in structures
            if (item.energy_hartree - minimum_energy) * HARTREE_TO_KCAL_MOL
            <= energy_window_kcal_mol + 1.0e-12
        ]
    )
    reference = eligible[0].molecule
    graph = build_bond_graph(reference, topology_tolerance)
    alignment_indices, comparison_indices, components = _alignment_and_comparison_indices(
        reference, graph, atom_mode
    )
    reference_elements = sorted(atom.element.upper() for atom in reference.atoms)
    for item in eligible[1:]:
        if sorted(atom.element.upper() for atom in item.molecule.atoms) != reference_elements:
            raise ValueError("optimized structures do not share the same atoms")
    topology_graphs = [
        build_bond_graph(item.molecule, topology_tolerance) for item in eligible
    ]
    distance_matrix = np.zeros((len(eligible), len(eligible)), dtype=float)
    topology_matrix = np.eye(len(eligible), dtype=bool)
    for left_index, left in enumerate(eligible):
        for right_index in range(left_index + 1, len(eligible)):
            right = eligible[right_index]
            comparison = (
                _permutation_aware_comparison(
                    left.molecule,
                    right.molecule,
                    topology_graphs[left_index],
                    topology_graphs[right_index],
                    atom_mode,
                )
                if permutation_mode == "equivalent"
                else _ordered_comparison(
                    left.molecule,
                    right.molecule,
                    topology_graphs[left_index],
                    topology_graphs[right_index],
                    atom_mode,
                )
            )
            topology_matrix[left_index, right_index] = comparison.same_topology
            topology_matrix[right_index, left_index] = comparison.same_topology
            distance = (
                comparison.rmsd_angstrom
                if comparison.same_topology
                else float("inf")
            )
            distance_matrix[left_index, right_index] = distance
            distance_matrix[right_index, left_index] = distance
    clusters = _complete_linkage(distance_matrix, rmsd_threshold_angstrom)
    clusters.sort(
        key=lambda members: (
            min(eligible[index].energy_hartree for index in members),
            min(str(eligible[index].optimized) for index in members),
        )
    )

    unique_dir = (
        Path(unique_output_dir).resolve()
        if unique_output_dir is not None
        else destination / "unique_optimized_xyz"
    )
    if unique_dir.exists():
        shutil.rmtree(unique_dir)
    unique_dir.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    cluster_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(clusters, start=1):
        representative_index = min(
            members,
            key=lambda index: (eligible[index].energy_hartree, str(eligible[index].optimized)),
        )
        representative = eligible[representative_index]
        delta = (representative.energy_hartree - minimum_energy) * HARTREE_TO_KCAL_MOL
        destination_xyz = unique_dir / (
            f"cluster_{cluster_id:04d}_dE_{delta:08.3f}_{representative.optimized.name}"
        )
        shutil.copy2(representative.optimized, destination_xyz)
        maximum_rmsd = max(
            distance_matrix[left, right]
            for left in members
            for right in members
        )
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "representative_optimized_xyz": str(representative.optimized),
                "representative_unique_xyz": str(destination_xyz),
                "energy_hartree": f"{representative.energy_hartree:.12f}",
                "delta_energy_kcal_mol": f"{delta:.8f}",
                "maximum_pairwise_rmsd_angstrom": f"{maximum_rmsd:.8f}",
                "members": json.dumps(
                    [eligible[index].optimized.name for index in members]
                ),
            }
        )
        for index in members:
            item = eligible[index]
            assignment_rows.append(
                {
                    "cluster_id": cluster_id,
                    "source_candidate": str(item.source),
                    "optimized_xyz": str(item.optimized),
                    "energy_hartree": f"{item.energy_hartree:.12f}",
                    "delta_energy_kcal_mol": f"{(item.energy_hartree - minimum_energy) * HARTREE_TO_KCAL_MOL:.8f}",
                    "rmsd_to_representative_angstrom": f"{distance_matrix[index, representative_index]:.8f}",
                    "is_representative": index == representative_index,
                    "representative_unique_xyz": str(destination_xyz),
                }
            )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(destination / "clusters.csv", cluster_rows)
    write_csv(destination / "assignments.csv", assignment_rows)
    remaining_topologies = set(range(len(eligible)))
    topology_groups = 0
    while remaining_topologies:
        start = min(remaining_topologies)
        stack = [start]
        group: set[int] = set()
        while stack:
            index = stack.pop()
            if index in group:
                continue
            group.add(index)
            stack.extend(
                other
                for other in remaining_topologies
                if topology_matrix[index, other]
            )
        remaining_topologies -= group
        topology_groups += 1

    summary: dict[str, Any] = {
        "schema": "seeker.postoptimization.clustering.v1",
        "optimization_csv": str(optimization_csv),
        "successful_optimized_structures": len(structures),
        "eligible_structures": len(eligible),
        "energy_window_kcal_mol": energy_window_kcal_mol,
        "rmsd_threshold_angstrom": rmsd_threshold_angstrom,
        "atom_mode": atom_mode,
        "permutation_mode": permutation_mode,
        "permitted_permutations": (
            "graph_isomorphic_fragments_and_internal_graph_automorphisms"
            if permutation_mode == "equivalent"
            else "none_ordered_xyz_indices"
        ),
        "alignment": "largest_covalent_fragment",
        "alignment_atom_indices_1based": [index + 1 for index in alignment_indices],
        "comparison_atom_indices_1based": [index + 1 for index in comparison_indices],
        "covalent_fragments_1based": [
            [index + 1 for index in component] for component in components
        ],
        "clusters": len(clusters),
        "cluster_sizes": sorted((len(item) for item in clusters), reverse=True),
        "topology_groups": topology_groups,
        "unique_optimized_structures": len(cluster_rows),
        "reduction": len(eligible) - len(cluster_rows),
        "unique_optimized_xyz": str(unique_dir),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
