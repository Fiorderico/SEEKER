"""Post-analysis for completed SEEKER runs."""

from __future__ import annotations

import csv
from functools import lru_cache
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from .fragment_pose import (
    FragmentPoseBlock,
    prepare_native_fragment_poses,
    quaternion_distance,
    quaternion_from_rotation_vector,
    vector_angle,
)
from .geometry import TorsionDistance, build_bond_graph, torsion_distance
from .input import read_coordinate_specs, read_xyz
from .models import FragmentPoseGene
from .operators import periodic_angle_modes
from .symmetry import equivalent_fragment_groups

T = TypeVar("T")


def _read_population(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["id"] = int(raw["id"])
            row["valid"] = raw["valid"].lower() == "true"
            row["rank"] = int(raw["rank"])
            row["energy_hartree"] = float(raw["energy_hartree"])
            row["delta_energy_kcal_mol"] = float(raw["delta_energy_kcal_mol"])
            row["hbond_score"] = float(raw["hbond_score"])
            row["hbond_count"] = int(raw["hbond_count"])
            row["alleles"] = [float(value) for value in json.loads(raw["alleles_deg"])]
            row["fitness_values"] = json.loads(raw.get("fitness_values_minimized", "{}") or "{}")
            rows.append(row)
    return rows


def complete_linkage_torsional(
    candidates: Sequence[T],
    alleles: Callable[[T], Sequence[float]],
    mean_threshold_deg: float,
    max_threshold_deg: float,
    stable_key: Callable[[T], Any],
    periodic: Sequence[bool] | None = None,
) -> list[list[T]]:
    """Deterministic complete-linkage clustering under mean + max limits."""

    clusters: list[list[T]] = [[item] for item in sorted(candidates, key=stable_key)]
    while True:
        possible: list[tuple[float, tuple[Any, ...], tuple[Any, ...], int, int]] = []
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                distances = [
                    torsion_distance(alleles(a), alleles(b), periodic)
                    for a in left
                    for b in right
                ]
                if all(
                    value.mean_deg < mean_threshold_deg
                    and value.max_deg < max_threshold_deg
                    for value in distances
                ):
                    score = max(
                        max(
                            value.mean_deg / mean_threshold_deg,
                            value.max_deg / max_threshold_deg,
                        )
                        for value in distances
                    )
                    left_key = tuple(stable_key(item) for item in left)
                    right_key = tuple(stable_key(item) for item in right)
                    possible.append((score, left_key, right_key, left_index, right_index))
        if not possible:
            break
        _score, _left_key, _right_key, left_index, right_index = min(possible)
        merged = sorted([*clusters[left_index], *clusters[right_index]], key=stable_key)
        clusters[left_index] = merged
        del clusters[right_index]
    return sorted(clusters, key=lambda cluster: tuple(stable_key(item) for item in cluster))


def _cluster_candidates(
    candidates: list[dict[str, Any]],
    mean_threshold_deg: float,
    max_threshold_deg: float | None = None,
    periodic: Sequence[bool] | None = None,
) -> list[list[dict[str, Any]]]:
    return complete_linkage_torsional(
        candidates,
        lambda row: row["alleles"],
        mean_threshold_deg,
        mean_threshold_deg if max_threshold_deg is None else max_threshold_deg,
        lambda row: row["id"],
        periodic,
    )


def toroidal_embedding(alleles: Sequence[float]) -> list[float]:
    """Embed circular torsions without introducing a discontinuity at 0/360°."""

    embedded: list[float] = []
    for angle in alleles:
        radians = math.radians(float(angle))
        embedded.extend((math.cos(radians), math.sin(radians)))
    return embedded


def mixed_coordinate_embedding(
    alleles: Sequence[float], periodic: Sequence[bool]
) -> list[float]:
    """Embed circular and bounded normalized alleles in one Euclidean space."""

    if len(alleles) != len(periodic):
        raise ValueError("alleli e periodicità incompatibili nell'embedding mixed")
    embedded: list[float] = []
    for value, is_periodic in zip(alleles, periodic):
        normalized = float(value)
        if is_periodic:
            radians = math.radians(normalized % 360.0)
            embedded.extend((math.cos(radians), math.sin(radians)))
        else:
            # The native chromosome maps every bounded physical interval to
            # 0..360.  Scaling to [-1, 1] gives it the same maximum diameter
            # as one circular coordinate without introducing a seam.
            embedded.append(2.0 * min(360.0, max(0.0, normalized)) / 360.0 - 1.0)
    return embedded


def _physical_values_from_pose_alleles(
    alleles: Sequence[float], variables: Sequence[dict[str, Any]]
) -> list[float]:
    if len(alleles) != len(variables):
        raise ValueError("alleles and POSE variables are incompatible")
    values: list[float] = []
    for allele, variable in zip(alleles, variables):
        lower = float(variable["lower"])
        upper = float(variable["upper"])
        span = upper - lower
        if span <= 0.0:
            raise ValueError("invalid POSE bounds in manifest")
        periodic = bool(variable.get("periodic", False))
        units = str(variable.get("units", "")).strip().lower()
        offset = float(variable.get("allele_offset_degrees", 0.0))
        if periodic and units in {"radian", "radians", "rad"} and math.isclose(
            span, 2.0 * math.pi, rel_tol=0.0, abs_tol=1.0e-9
        ):
            values.append(lower + math.radians((float(allele) - offset) % 360.0))
        elif periodic and units in {"degree", "degrees", "deg"} and math.isclose(
            span, 360.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            values.append(lower + (float(allele) - offset) % 360.0)
        else:
            normalized = min(360.0, max(0.0, float(allele))) / 360.0
            values.append(lower + span * normalized)
    return values


def _rotation_matrix_from_quaternion(
    quaternion: Sequence[float],
) -> tuple[float, ...]:
    w, x, y, z = (float(item) for item in quaternion)
    return (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


def pose_manifold_embedding(
    alleles: Sequence[float],
    variables: Sequence[dict[str, Any]],
    pose_blocks: Sequence[FragmentPoseBlock],
    equivalent_block_groups: Sequence[Sequence[int]] | None = None,
) -> list[float]:
    """Embed bounded fragment poses without rotation-vector seams.

    Each POSE contributes one radial coordinate, a unit direction on S2 and a
    rotation matrix on SO(3).  Direction and rotation-matrix blocks are scaled
    to unit maximum chordal diameter, so no arbitrary FTRANS/FROT component can
    dominate density clustering.  Scalar coordinates outside POSE retain the
    historical circular/bounded embedding.
    """

    values = _physical_values_from_pose_alleles(alleles, variables)
    embedded: list[float] = []
    pose_indices = {index for block in pose_blocks for index in block.variable_indices}
    block_features: list[tuple[float, ...]] = []
    for block in pose_blocks:
        feature: list[float] = []
        translation = tuple(values[index] for index in block.variable_indices[:3])
        radius = math.sqrt(sum(item * item for item in translation))
        if radius <= 1.0e-14:
            raise ValueError(f"{block.name}: direzione POSE nulla nel clustering")
        radial_span = block.distance_bounds[1] - block.distance_bounds[0]
        feature.append((radius - block.distance_bounds[0]) / radial_span)
        feature.extend(item / radius / 2.0 for item in translation)
        rotation = tuple(values[index] for index in block.variable_indices[3:])
        matrix = _rotation_matrix_from_quaternion(
            quaternion_from_rotation_vector(rotation)
        )
        feature.extend(item / math.sqrt(8.0) for item in matrix)
        block_features.append(tuple(feature))
    groups = (
        tuple(tuple(int(index) for index in group) for group in equivalent_block_groups)
        if equivalent_block_groups is not None
        else tuple((index,) for index in range(len(pose_blocks)))
    )
    if sorted(index for group in groups for index in group) != list(
        range(len(pose_blocks))
    ):
        raise ValueError("gruppi di equivalenza POSE non validi")
    for group in groups:
        for feature in sorted(block_features[index] for index in group):
            embedded.extend(feature)
    for index, (allele, variable) in enumerate(zip(alleles, variables)):
        if index in pose_indices:
            continue
        normalized = float(allele)
        if bool(variable.get("periodic", True)):
            radians = math.radians(normalized % 360.0)
            embedded.extend((0.5 * math.cos(radians), 0.5 * math.sin(radians)))
        else:
            embedded.append(min(360.0, max(0.0, normalized)) / 360.0)
    return embedded


def pose_manifold_distance(
    left: Sequence[float],
    right: Sequence[float],
    variables: Sequence[dict[str, Any]],
    pose_blocks: Sequence[FragmentPoseBlock],
    equivalent_block_groups: Sequence[Sequence[int]] | None = None,
) -> TorsionDistance:
    """Return normalized S2/SO(3) distance, optionally quotienting labels."""

    left_values = _physical_values_from_pose_alleles(left, variables)
    right_values = _physical_values_from_pose_alleles(right, variables)
    normalized: list[float] = []
    pose_indices = {index for block in pose_blocks for index in block.variable_indices}
    groups = (
        tuple(tuple(int(index) for index in group) for group in equivalent_block_groups)
        if equivalent_block_groups is not None
        else tuple((index,) for index in range(len(pose_blocks)))
    )
    if sorted(index for group in groups for index in group) != list(
        range(len(pose_blocks))
    ):
        raise ValueError("gruppi di equivalenza POSE non validi")

    def pair_components(
        left_block: FragmentPoseBlock, right_block: FragmentPoseBlock
    ) -> tuple[float, float, float]:
        left_translation = tuple(
            left_values[index] for index in left_block.variable_indices[:3]
        )
        right_translation = tuple(
            right_values[index] for index in right_block.variable_indices[:3]
        )
        radial_span = left_block.distance_bounds[1] - left_block.distance_bounds[0]
        radial = abs(
            math.sqrt(sum(item * item for item in left_translation))
            - math.sqrt(sum(item * item for item in right_translation))
        ) / radial_span
        direction_scale = left_block.direction_max_radian or math.pi
        direction = vector_angle(left_translation, right_translation) / direction_scale
        left_rotation = quaternion_from_rotation_vector(
            tuple(left_values[index] for index in left_block.variable_indices[3:])
        )
        right_rotation = quaternion_from_rotation_vector(
            tuple(right_values[index] for index in right_block.variable_indices[3:])
        )
        orientation_scale = left_block.orientation_max_radian or math.pi
        orientation = quaternion_distance(left_rotation, right_rotation) / orientation_scale
        return tuple(
            min(1.0, max(0.0, item))
            for item in (radial, direction, orientation)
        )  # type: ignore[return-value]

    for group in groups:
        costs = [
            [
                sum(
                    item * item
                    for item in pair_components(
                        pose_blocks[left_index], pose_blocks[right_index]
                    )
                )
                for right_index in group
            ]
            for left_index in group
        ]

        @lru_cache(maxsize=None)
        def assign(position: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
            if position == len(group):
                return 0.0, ()
            options: list[tuple[float, tuple[int, ...]]] = []
            for relative_right in range(len(group)):
                if used_mask & (1 << relative_right):
                    continue
                remaining_cost, remaining_assignment = assign(
                    position + 1, used_mask | (1 << relative_right)
                )
                options.append(
                    (
                        costs[position][relative_right] + remaining_cost,
                        (relative_right, *remaining_assignment),
                    )
                )
            return min(options, key=lambda item: (item[0], item[1]))

        _cost, assignment = assign(0, 0)
        for relative_left, relative_right in enumerate(assignment):
            normalized.extend(
                pair_components(
                    pose_blocks[group[relative_left]],
                    pose_blocks[group[relative_right]],
                )
            )
    for index, variable in enumerate(variables):
        if index in pose_indices:
            continue
        component = torsion_distance(
            [left[index]],
            [right[index]],
            [bool(variable.get("periodic", True))],
        ).mean_deg
        normalized.append(min(1.0, component / 360.0))
    if not normalized:
        return TorsionDistance(float("inf"), float("inf"), float("inf"))
    scaled = [360.0 * min(1.0, max(0.0, item)) for item in normalized]
    return TorsionDistance(
        sum(scaled) / len(scaled),
        math.sqrt(sum(item * item for item in scaled) / len(scaled)),
        max(scaled),
    )


class _PoseManifoldNearestIndex:
    """Exact nearest-neighbour queries with a seam-free embedding bound.

    The Euclidean POSE embedding is not itself the geodesic selection metric,
    but it supplies a conservative lower bound for that metric.  Querying the
    embedding tree in progressively larger prefixes therefore lets us stop as
    soon as every unvisited point is provably farther away than the best exact
    distance found so far.
    """

    def __init__(
        self,
        candidates: Sequence[Sequence[float]],
        variables: Sequence[dict[str, Any]],
        pose_blocks: Sequence[FragmentPoseBlock],
    ) -> None:
        if not candidates:
            raise ValueError("pose nearest-neighbour index requires candidates")
        try:
            import numpy as np
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover - pose_hybrid requires scipy
            raise RuntimeError(
                "indexed POSE distances require scipy"
            ) from exc

        self._np = np
        self._variables = tuple(dict(item) for item in variables)
        self._pose_blocks = tuple(pose_blocks)
        self._candidates = tuple(tuple(float(value) for value in item) for item in candidates)
        self._points = np.asarray(
            [
                pose_manifold_embedding(item, self._variables, self._pose_blocks)
                for item in self._candidates
            ],
            dtype=float,
        )
        self._tree = cKDTree(self._points)

        pose_indices = {
            index
            for block in self._pose_blocks
            for index in block.variable_indices
        }
        scalar_indices = [
            index for index in range(len(self._variables)) if index not in pose_indices
        ]
        component_count = 3 * len(self._pose_blocks) + len(scalar_indices)
        if component_count < 1:
            raise ValueError("pose nearest-neighbour index has no metric components")

        # For a unit-vector or SO(3) chord c=sin(theta/2), the normalized
        # geodesic component is at least min(1, 2/scale) * c.  Circular scalar
        # alleles use a radius-0.5 embedding and contribute the weaker bound
        # delta/(2*pi) >= chord/pi.  Radial and bounded scalar components have
        # an exact linear embedding, hence factor 1.
        component_factors = [1.0]
        for block in self._pose_blocks:
            direction_scale = block.direction_max_radian or math.pi
            orientation_scale = block.orientation_max_radian or math.pi
            component_factors.extend(
                (
                    min(1.0, 2.0 / direction_scale),
                    min(1.0, 2.0 / orientation_scale),
                )
            )
        component_factors.extend(
            1.0 / math.pi
            if bool(self._variables[index].get("periodic", True))
            else 1.0
            for index in scalar_indices
        )
        self._rms_lower_bound_scale = (
            360.0 * min(component_factors) / math.sqrt(component_count)
        )

    @property
    def size(self) -> int:
        return len(self._candidates)

    def nearest(
        self, query: Sequence[float]
    ) -> tuple[TorsionDistance, int]:
        """Return the exact nearest POSE distance and exact evaluations used."""

        query_values = tuple(float(value) for value in query)
        query_point = self._np.asarray(
            pose_manifold_embedding(
                query_values, self._variables, self._pose_blocks
            ),
            dtype=float,
        )
        examined: set[int] = set()
        best = TorsionDistance(float("inf"), float("inf"), float("inf"))
        requested = 1
        while True:
            embedding_distances, indices = self._tree.query(
                query_point, k=requested
            )
            distance_values = self._np.atleast_1d(embedding_distances)
            index_values = self._np.atleast_1d(indices)
            for raw_index in index_values:
                index = int(raw_index)
                if index in examined:
                    continue
                examined.add(index)
                exact = pose_manifold_distance(
                    query_values,
                    self._candidates[index],
                    self._variables,
                    self._pose_blocks,
                )
                if exact.rms_deg < best.rms_deg:
                    best = exact

            boundary = float(max(distance_values))
            unvisited_lower_bound = boundary * self._rms_lower_bound_scale
            if best.rms_deg <= unvisited_lower_bound + 1.0e-12:
                break
            if requested >= self.size:
                break
            requested = min(self.size, 2 * requested)
        return best, len(examined)


def hdbscan_torsional_labels(
    candidates: Sequence[T],
    alleles: Callable[[T], Sequence[float]],
    min_cluster_size: int,
    min_samples: int,
    stable_key: Callable[[T], Any],
    periodic: Sequence[bool] | None = None,
    embedding: Callable[[Sequence[float]], Sequence[float]] | None = None,
) -> list[int]:
    """Return deterministic, one-based HDBSCAN labels in toroidal space.

    Noise keeps label ``-1``.  The labels are renumbered by the smallest stable
    key in each cluster because native HDBSCAN labels are implementation details.
    """

    if len(candidates) < min_cluster_size:
        return [-1] * len(candidates)
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "clustering hybrid richiede scikit-learn; installa "
            "seeker-conformer[discovery]"
        ) from exc

    indexed = sorted(
        enumerate(candidates), key=lambda item: (stable_key(item[1]), item[0])
    )
    first_values = alleles(indexed[0][1])
    mask = (
        tuple(True for _ in first_values)
        if periodic is None
        else tuple(bool(value) for value in periodic)
    )
    features = [
        list(embedding(alleles(candidate)))
        if embedding is not None
        else mixed_coordinate_embedding(alleles(candidate), mask)
        for _index, candidate in indexed
    ]
    raw_labels = [
        int(value)
        for value in HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        ).fit_predict(features)
    ]
    raw_clusters = sorted(
        {label for label in raw_labels if label >= 0},
        key=lambda label: min(
            stable_key(candidate)
            for (_index, candidate), assigned in zip(indexed, raw_labels)
            if assigned == label
        ),
    )
    normalized = {label: index for index, label in enumerate(raw_clusters, start=1)}
    result = [-1] * len(candidates)
    for (original_index, _candidate), label in zip(indexed, raw_labels):
        result[original_index] = normalized.get(label, -1)
    return result


def energy_graph_local_minima(
    candidates: Sequence[T],
    alleles: Callable[[T], Sequence[float]],
    energy: Callable[[T], float],
    neighbors: int,
    stable_key: Callable[[T], Any],
    periodic: Sequence[bool] | None = None,
    embedding: Callable[[Sequence[float]], Sequence[float]] | None = None,
) -> list[T]:
    """Find sampled energy minima on a circular k-nearest-neighbour graph."""

    ordered = sorted(candidates, key=stable_key)
    if len(ordered) < 2:
        return ordered
    neighbor_count = min(int(neighbors), len(ordered) - 1)
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - hybrid already needs sklearn
        raise RuntimeError(
            "il grafo energetico hybrid richiede scipy; installa "
            "seeker-conformer[discovery]"
        ) from exc

    first_values = alleles(ordered[0])
    mask = (
        tuple(True for _ in first_values)
        if periodic is None
        else tuple(bool(value) for value in periodic)
    )
    all_periodic = all(mask) and embedding is None
    if embedding is not None:
        points = np.asarray(
            [list(embedding(alleles(item))) for item in ordered], dtype=float
        )
        tree = cKDTree(points)
    elif all_periodic:
        points = np.asarray(
            [[float(value) % 360.0 for value in alleles(item)] for item in ordered],
            dtype=float,
        )
        tree = cKDTree(points, boxsize=360.0)
    else:
        points = np.asarray(
            [mixed_coordinate_embedding(alleles(item), mask) for item in ordered],
            dtype=float,
        )
        tree = cKDTree(points)
    query_count = min(len(ordered), neighbor_count + 2)
    distances, indices = tree.query(points, k=query_count, workers=-1)
    if query_count == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    minima: list[T] = []
    for candidate_index, candidate in enumerate(ordered):
        nearby = [
            (float(distance), int(index))
            for distance, index in zip(
                np.atleast_1d(distances[candidate_index]),
                np.atleast_1d(indices[candidate_index]),
            )
            if int(index) != candidate_index
        ]
        nearby.sort(key=lambda pair: (pair[0], stable_key(ordered[pair[1]])))

        # Resolve a distance tie at the k-th-neighbour boundary by stable key;
        # cKDTree is free to return tied points in any order.
        if len(nearby) > neighbor_count and math.isclose(
            nearby[neighbor_count - 1][0], nearby[neighbor_count][0], abs_tol=1e-12
        ):
            radius = nearby[neighbor_count - 1][0]
            tied_indices = tree.query_ball_point(
                points[candidate_index], radius + 1e-12
            )
            nearby = [
                (
                    math.sqrt(
                        sum(
                            min(abs(left - right), 360.0 - abs(left - right)) ** 2
                            for left, right in zip(
                                points[candidate_index], points[index]
                            )
                        )
                    )
                    if all_periodic
                    else float(
                        np.linalg.norm(points[candidate_index] - points[index])
                    ),
                    int(index),
                )
                for index in tied_indices
                if int(index) != candidate_index
            ]
            nearby.sort(key=lambda pair: (pair[0], stable_key(ordered[pair[1]])))
        nearest = [ordered[index] for _distance, index in nearby[:neighbor_count]]
        candidate_key = (float(energy(candidate)), stable_key(candidate))
        if all(
            candidate_key < (float(energy(other)), stable_key(other))
            for other in nearest
        ):
            minima.append(candidate)
    return minima


def adaptive_hybrid_selection(
    candidate_records: Sequence[dict[str, Any]],
    max_candidates: int,
    min_separation_deg: float,
    energy_window_kcal_mol: float,
    periodic: Sequence[bool] | None = None,
    distance: Callable[[Sequence[float], Sequence[float]], TorsionDistance]
    | None = None,
    objectives: Sequence[str] = ("energy",),
) -> list[dict[str, Any]]:
    """Select a Pareto-aware subset diverse in geometry and objective space.

    Every global ``objective_best:<name>`` record is mandatory, even when it is
    geometrically close to an already selected structure.  The energy minimum
    remains mandatory for backward-compatible callers.  Remaining points use
    a deterministic score combining structural novelty, normalized fitness
    novelty, energy quality and independent Pareto/density/minimum support.
    """

    if not candidate_records or max_candidates < 1:
        return []

    records = sorted(
        candidate_records,
        key=lambda record: (
            record["candidate"]["energy_hartree"],
            record["candidate"]["id"],
        ),
    )
    objective_names = tuple(objectives) or ("energy",)
    raw_objectives = {
        int(record["candidate"]["id"]): tuple(
            _objective_value(record["candidate"], name) for name in objective_names
        )
        for record in records
    }
    bounds: list[tuple[float, float]] = []
    for position in range(len(objective_names)):
        finite = [
            values[position]
            for values in raw_objectives.values()
            if math.isfinite(values[position])
        ]
        bounds.append(
            (min(finite), max(finite)) if finite else (0.0, 0.0)
        )
    normalized_objectives: dict[int, tuple[float, ...]] = {}
    for identifier, values in raw_objectives.items():
        normalized: list[float] = []
        for value, (lower, upper) in zip(values, bounds):
            if not math.isfinite(value):
                normalized.append(1.0)
            elif math.isclose(lower, upper, abs_tol=1.0e-15):
                normalized.append(0.0)
            else:
                normalized.append((value - lower) / (upper - lower))
        normalized_objectives[identifier] = tuple(normalized)

    def structural_novelty(
        record: dict[str, Any], chosen: Sequence[dict[str, Any]]
    ) -> float:
        if not chosen:
            return float("inf")
        candidate = record["candidate"]
        return min(
            (
                distance(candidate["alleles"], item["candidate"]["alleles"])
                if distance is not None
                else torsion_distance(
                    candidate["alleles"], item["candidate"]["alleles"], periodic
                )
            ).rms_deg
            for item in chosen
        )

    def fitness_novelty(
        record: dict[str, Any], chosen: Sequence[dict[str, Any]]
    ) -> float:
        if not chosen:
            return float("inf")
        vector = normalized_objectives[int(record["candidate"]["id"])]
        return min(
            math.sqrt(
                sum((left - right) ** 2 for left, right in zip(vector, other))
                / max(1, len(vector))
            )
            for other in (
                normalized_objectives[int(item["candidate"]["id"])]
                for item in chosen
            )
        )

    objective_order = {name: index for index, name in enumerate(objective_names)}
    mandatory: list[tuple[int, float, int, dict[str, Any], tuple[str, ...]]] = []
    for record in records:
        sources = set(record["sources"])
        roles = tuple(
            sorted(
                (
                    source.split(":", 1)[1]
                    for source in sources
                    if source.startswith("objective_best:")
                ),
                key=lambda name: (objective_order.get(name, len(objective_order)), name),
            )
        )
        if roles:
            mandatory.append(
                (
                    min(objective_order.get(name, len(objective_order)) for name in roles),
                    float(record["candidate"]["energy_hartree"]),
                    int(record["candidate"]["id"]),
                    record,
                    roles,
                )
            )
    energy_minimum = records[0]
    if all(item[3]["candidate"]["id"] != energy_minimum["candidate"]["id"] for item in mandatory):
        mandatory.append(
            (
                objective_order.get("energy", -1),
                float(energy_minimum["candidate"]["energy_hartree"]),
                int(energy_minimum["candidate"]["id"]),
                energy_minimum,
                ("energy",),
            )
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for _order, _energy, _identifier, record, roles in sorted(mandatory):
        identifier = int(record["candidate"]["id"])
        if identifier in selected_ids:
            continue
        geometry_novelty = structural_novelty(record, selected)
        objective_novelty = fitness_novelty(record, selected)
        selected.append(
            {
                **record,
                "selection_score": 1.0,
                "selection_reason": "objective_extreme",
                "mandatory_objectives": roles,
                "novelty_at_selection_rms_deg": geometry_novelty,
                "fitness_novelty_at_selection": objective_novelty,
                "objective_values": dict(
                    zip(objective_names, raw_objectives[identifier])
                ),
            }
        )
        selected_ids.add(identifier)

    selection_limit = max(max_candidates, len(selected))
    while len(selected) < selection_limit:
        possible: list[
            tuple[float, float, float, float, int, dict[str, Any]]
        ] = []
        for record in records:
            candidate = record["candidate"]
            if candidate["id"] in selected_ids:
                continue
            novelty = structural_novelty(record, selected)
            if novelty < min_separation_deg:
                continue
            objective_novelty = fitness_novelty(record, selected)
            if energy_window_kcal_mol > 0.0:
                energy_quality = max(
                    0.0,
                    1.0
                    - float(candidate["delta_energy_kcal_mol"])
                    / energy_window_kcal_mol,
                )
            else:
                energy_quality = (
                    1.0 if candidate["delta_energy_kcal_mol"] <= 0.0 else 0.0
                )
            sources = set(record["sources"])
            source_support = (
                int("hdbscan_density_minimum" in sources)
                + int("energy_graph_minimum" in sources)
            ) / 2.0
            evidence_support = (
                0.5 * int("global_pareto" in sources)
                + 0.5 * source_support
            )
            score = (
                0.45 * min(novelty / 180.0, 1.0)
                + 0.35 * min(objective_novelty, 1.0)
                + 0.10 * energy_quality
                + 0.10 * evidence_support
            )
            possible.append(
                (
                    score,
                    objective_novelty,
                    novelty,
                    -float(candidate["energy_hartree"]),
                    -int(candidate["id"]),
                    record,
                )
            )
        if not possible:
            break
        (
            score,
            objective_novelty,
            novelty,
            _negative_energy,
            _negative_id,
            chosen,
        ) = max(
            possible, key=lambda item: item[:5]
        )
        identifier = int(chosen["candidate"]["id"])
        selected.append(
            {
                **chosen,
                "selection_score": score,
                "selection_reason": (
                    "pareto_fitness_geometry_diversity"
                    if "global_pareto" in set(chosen["sources"])
                    else "fitness_geometry_diversity"
                ),
                "mandatory_objectives": (),
                "novelty_at_selection_rms_deg": novelty,
                "fitness_novelty_at_selection": objective_novelty,
                "objective_values": dict(
                    zip(objective_names, raw_objectives[identifier])
                ),
            }
        )
        selected_ids.add(identifier)
    return selected


def periodicity_cell_torsional(
    candidates: Sequence[T],
    alleles: Callable[[T], Sequence[float]],
    periodicities: Sequence[int],
    stable_key: Callable[[T], Any],
) -> list[list[T]]:
    """Assign candidates to the nearest mode of every declared periodic prior.

    Unlike symmetry reduction, this partitions the complete 0..360 degree
    domain into conformational basins and never treats two alleles as the same
    genotype.  The maximum number of cells is the product of periodicities.
    """

    if not periodicities or any(int(value) < 1 for value in periodicities):
        raise ValueError("periodicità di clustering non valide")
    modes = [periodic_angle_modes(int(value)) for value in periodicities]
    grouped: dict[tuple[int, ...], list[T]] = {}
    for candidate in sorted(candidates, key=stable_key):
        values = alleles(candidate)
        if len(values) != len(modes):
            raise ValueError("numero di alleli e periodicità di clustering incoerente")
        cell = tuple(
            min(
                range(len(gene_modes)),
                key=lambda index: (
                    abs((float(angle) - gene_modes[index] + 180.0) % 360.0 - 180.0),
                    index,
                ),
            )
            for angle, gene_modes in zip(values, modes)
        )
        grouped.setdefault(cell, []).append(candidate)
    return [grouped[cell] for cell in sorted(grouped)]


def subdivide_torsional_cells(
    cells: Sequence[Sequence[T]],
    alleles: Callable[[T], Sequence[float]],
    mean_threshold_deg: float,
    max_threshold_deg: float,
    stable_key: Callable[[T], Any],
    periodic: Sequence[bool] | None = None,
) -> list[list[T]]:
    """Split coarse periodicity cells into genuine torsional clusters.

    A periodicity cell says which prior mode is nearest; it does not guarantee
    that all geometries inside it are close.  In particular, a broad cell can
    otherwise collapse different X-H orientations into one representative.
    """

    clusters: list[list[T]] = []
    try:
        import numpy as np
        from scipy.cluster.hierarchy import fcluster, linkage
    except ImportError:  # pragma: no cover - hybrid installs the discovery extra
        np = None

    for raw_cell in cells:
        cell = sorted(raw_cell, key=stable_key)
        if len(cell) < 2:
            clusters.append(cell)
            continue
        if np is None:
            clusters.extend(
                complete_linkage_torsional(
                    cell,
                    alleles,
                    mean_threshold_deg,
                    max_threshold_deg,
                    stable_key,
                    periodic,
                )
            )
            continue

        condensed = []
        for left_index, left in enumerate(cell[:-1]):
            for right in cell[left_index + 1 :]:
                value = torsion_distance(alleles(left), alleles(right), periodic)
                condensed.append(
                    max(
                        value.mean_deg / mean_threshold_deg,
                        value.max_deg / max_threshold_deg,
                    )
                )
        hierarchy = linkage(np.asarray(condensed, dtype=float), method="complete")
        labels = fcluster(
            hierarchy,
            t=float(np.nextafter(1.0, 0.0)),
            criterion="distance",
        )
        grouped: dict[int, list[T]] = {}
        for item, label in zip(cell, labels):
            grouped.setdefault(int(label), []).append(item)
        clusters.extend(grouped.values())
    return sorted(
        clusters,
        key=lambda cluster: tuple(stable_key(item) for item in cluster),
    )


def _manifest_periodicities(
    manifest: dict[str, Any], expected_dimensions: int
) -> tuple[int, ...]:
    raw = manifest.get("active_variables") or manifest.get("genes") or []
    if not isinstance(raw, list):
        raise ValueError("metadati delle periodicità non validi nel manifest")
    periodicities = tuple(
        int(item.get("periodicity", 1))
        for item in raw
        if isinstance(item, dict)
    )
    if len(periodicities) != expected_dimensions:
        raise ValueError(
            "clustering periodicity_cells richiede una periodicità per ogni allele"
        )
    if any(value < 1 for value in periodicities):
        raise ValueError("periodicità non positive nel manifest")
    return periodicities


def _manifest_periodic_flags(
    manifest: dict[str, Any], expected_dimensions: int
) -> tuple[bool, ...]:
    raw = manifest.get("active_variables") or manifest.get("genes") or []
    if not isinstance(raw, list):
        raise ValueError("metadati delle variabili non validi nel manifest")
    flags = tuple(bool(item.get("periodic", True)) for item in raw)
    if len(flags) != expected_dimensions:
        return tuple(True for _ in range(expected_dimensions))
    return flags


def _manifest_bounded_bins(
    manifest: dict[str, Any], expected_dimensions: int
) -> tuple[int, ...]:
    raw = manifest.get("active_variables") or manifest.get("genes") or []
    if not isinstance(raw, list) or len(raw) != expected_dimensions:
        return tuple(1 for _ in range(expected_dimensions))
    return tuple(
        max(1, int(item.get("scan_points", item.get("points", 5))))
        if isinstance(item, dict) and not bool(item.get("periodic", True))
        else 1
        for item in raw
    )


def _manifest_pose_blocks(
    manifest: dict[str, Any], root: Path
) -> tuple[FragmentPoseBlock, ...]:
    raw_blocks = manifest.get("fragment_pose_blocks", [])
    if isinstance(raw_blocks, list) and raw_blocks:
        blocks: list[FragmentPoseBlock] = []
        for item in raw_blocks:
            if not isinstance(item, dict):
                raise ValueError("metadati fragment_pose non validi nel manifest")
            indices = tuple(int(value) for value in item["variable_indices"])
            distance_bounds = tuple(float(value) for value in item["distance_bounds"])
            translation = tuple(float(value) for value in item["reference_translation"])
            rotation = tuple(float(value) for value in item["reference_rotation"])
            blocks.append(
                FragmentPoseBlock(
                    name=str(item["name"]),
                    variable_indices=indices,  # type: ignore[arg-type]
                    distance_bounds=distance_bounds,  # type: ignore[arg-type]
                    direction_max_radian=float(item["direction_max_radian"]),
                    orientation_max_radian=float(item["orientation_max_radian"]),
                    reference_translation=translation,  # type: ignore[arg-type]
                    reference_rotation=rotation,  # type: ignore[arg-type]
                    reference_atoms=tuple(int(value) for value in item["reference_atoms"]),
                    moving_atoms=tuple(int(value) for value in item["moving_atoms"]),
                )
            )
        return tuple(blocks)

    inputs = manifest.get("input", {})
    if not isinstance(inputs, dict):
        return ()
    xyz_path = Path(str(inputs.get("xyz", "")))
    genes_path = Path(str(inputs.get("genes", "")))
    if not xyz_path.is_file() or not genes_path.is_file():
        return ()
    molecule = read_xyz(xyz_path)
    poses = tuple(
        item
        for item in read_coordinate_specs(genes_path)
        if isinstance(item, FragmentPoseGene)
    )
    if not poses:
        return ()
    graph = build_bond_graph(
        molecule, float(manifest.get("config", {}).get("topology_tolerance", 0.45))
    )
    return prepare_native_fragment_poses(molecule, poses, graph).pose_blocks


def _pose_equivalence_groups(
    pose_blocks: Sequence[FragmentPoseBlock],
    molecule_path: Path | None,
    topology_tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    """Find POSE blocks whose moving fragments may exchange physical labels."""

    if not pose_blocks:
        return ()
    if molecule_path is None or not molecule_path.is_file():
        return tuple((index,) for index in range(len(pose_blocks)))
    molecule = read_xyz(molecule_path)
    graph = build_bond_graph(molecule, topology_tolerance)
    graph_groups = equivalent_fragment_groups(
        molecule,
        graph,
        [block.moving_atoms for block in pose_blocks],
    )
    groups: list[tuple[int, ...]] = []
    for graph_group in graph_groups:
        compatible: dict[tuple[Any, ...], list[int]] = {}
        for index in graph_group:
            block = pose_blocks[index]
            key = (
                tuple(block.reference_atoms),
                tuple(round(value, 12) for value in block.distance_bounds),
                round(block.direction_max_radian, 12),
                round(block.orientation_max_radian, 12),
            )
            compatible.setdefault(key, []).append(index)
        groups.extend(tuple(indices) for indices in compatible.values())
    return tuple(sorted(groups, key=lambda group: group[0]))


def mixed_cell_torsional(
    candidates: Sequence[T],
    alleles: Callable[[T], Sequence[float]],
    periodicities: Sequence[int],
    periodic: Sequence[bool],
    bounded_bins: Sequence[int],
    stable_key: Callable[[T], Any],
) -> list[list[T]]:
    """Partition a mixed chromosome into modal and bounded Cartesian cells."""

    dimensions = len(periodic)
    if not (
        dimensions == len(periodicities) == len(bounded_bins)
        and dimensions > 0
    ):
        raise ValueError("metadati delle celle mixed incompatibili")
    modes = [
        periodic_angle_modes(int(order)) if is_periodic else ()
        for order, is_periodic in zip(periodicities, periodic)
    ]
    grouped: dict[tuple[int, ...], list[T]] = {}
    for candidate in sorted(candidates, key=stable_key):
        values = alleles(candidate)
        if len(values) != dimensions:
            raise ValueError("numero di alleli incoerente nelle celle mixed")
        cell: list[int] = []
        for value, is_periodic, axis_modes, bins in zip(
            values, periodic, modes, bounded_bins
        ):
            if is_periodic:
                cell.append(
                    min(
                        range(len(axis_modes)),
                        key=lambda index: (
                            abs(
                                (float(value) - axis_modes[index] + 180.0)
                                % 360.0
                                - 180.0
                            ),
                            index,
                        ),
                    )
                )
            else:
                normalized = min(360.0, max(0.0, float(value)))
                cell.append(min(bins - 1, int(normalized * bins / 360.0)))
        grouped.setdefault(tuple(cell), []).append(candidate)
    return [grouped[cell] for cell in sorted(grouped)]


def mixed_cell_centre(
    alleles: Sequence[float],
    periodicities: Sequence[int],
    periodic: Sequence[bool],
    bounded_bins: Sequence[int],
) -> list[float]:
    centre: list[float] = []
    for value, order, is_periodic, bins in zip(
        alleles, periodicities, periodic, bounded_bins
    ):
        if is_periodic:
            centre.append(
                min(
                    periodic_angle_modes(order),
                    key=lambda mode: (
                        abs((float(value) - mode + 180.0) % 360.0 - 180.0),
                        mode,
                    ),
                )
            )
        else:
            normalized = min(360.0, max(0.0, float(value)))
            index = min(bins - 1, int(normalized * bins / 360.0))
            centre.append(360.0 * (index + 0.5) / bins)
    return centre


def mixed_core_representative(
    members: Sequence[dict[str, Any]],
    periodicities: Sequence[int],
    periodic: Sequence[bool],
    bounded_bins: Sequence[int],
) -> dict[str, Any]:
    if not members:
        raise ValueError("cella mixed priva di membri")
    centre = mixed_cell_centre(
        members[0]["alleles"], periodicities, periodic, bounded_bins
    )
    radii = [
        180.0 / order if is_periodic else 180.0 / bins
        for order, is_periodic, bins in zip(
            periodicities, periodic, bounded_bins
        )
    ]

    def deltas(row: dict[str, Any]) -> list[float]:
        return [
            abs((value - target + 180.0) % 360.0 - 180.0)
            if is_periodic
            else abs(value - target)
            for value, target, is_periodic in zip(
                row["alleles"], centre, periodic
            )
        ]

    core = [
        row
        for row in members
        if all(delta <= radius + 1.0e-9 for delta, radius in zip(deltas(row), radii))
    ]
    if core:
        return min(core, key=lambda row: (row["energy_hartree"], row["id"]))
    return min(
        members,
        key=lambda row: (
            max(deltas(row)),
            torsion_distance(row["alleles"], centre, periodic).rms_deg,
            row["energy_hartree"],
            row["id"],
        ),
    )


def periodicity_cell_centre(
    alleles: Sequence[float], periodicities: Sequence[int]
) -> list[float]:
    return [
        min(
            periodic_angle_modes(periodicity),
            key=lambda mode: (
                abs((float(angle) - mode + 180.0) % 360.0 - 180.0),
                mode,
            ),
        )
        for angle, periodicity in zip(alleles, periodicities)
    ]


def periodicity_modal_representative(
    members: Sequence[dict[str, Any]], periodicities: Sequence[int]
) -> dict[str, Any]:
    """Choose the member nearest to its periodic cell's modal centre."""

    if not members:
        raise ValueError("cella periodica priva di membri")
    centre = periodicity_cell_centre(members[0]["alleles"], periodicities)
    return min(
        members,
        key=lambda row: (
            torsion_distance(row["alleles"], centre).max_deg,
            torsion_distance(row["alleles"], centre).rms_deg,
            row["energy_hartree"],
            row["id"],
        ),
    )


def periodicity_core_representative(
    members: Sequence[dict[str, Any]], periodicities: Sequence[int]
) -> dict[str, Any]:
    """Choose the lowest-energy member inside the common modal-cell core."""

    if not members:
        raise ValueError("cella periodica priva di membri")
    centre = periodicity_cell_centre(members[0]["alleles"], periodicities)
    core_radius = min(180.0 / int(value) for value in periodicities)
    core = [
        row
        for row in members
        if torsion_distance(row["alleles"], centre).max_deg <= core_radius + 1.0e-9
    ]
    if not core:
        return periodicity_modal_representative(members, periodicities)
    return min(core, key=lambda row: (row["energy_hartree"], row["id"]))


def _objective_value(row: dict[str, Any], name: str) -> float:
    if name == "energy":
        return float(row["energy_hartree"])
    if name == "hbond":
        return float(row["hbond_score"])
    return float(row["fitness_values"].get(name, float("inf")))


def _local_pareto(
    members: Sequence[dict[str, Any]], objectives: Sequence[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in members:
        values = [_objective_value(candidate, name) for name in objectives]
        dominated = False
        for other in members:
            if other["id"] == candidate["id"]:
                continue
            other_values = [_objective_value(other, name) for name in objectives]
            if all(a <= b for a, b in zip(other_values, values)) and any(
                a < b for a, b in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda row: row["id"])


def _write_scatter_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row["valid"] and math.isfinite(row["delta_energy_kcal_mol"])]
    width, height = 900, 560
    left, right, top, bottom = 85, 30, 40, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    if not valid:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='200'><text x='20' y='40'>No valid candidates</text></svg>\n", encoding="utf-8")
        return
    x_values = [float(row["delta_energy_kcal_mol"]) for row in valid]
    y_values = [float(row["hbond_score"]) for row in valid]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' y2='{top + plot_height}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' stroke='#333'/>",
        f"<text x='{width / 2}' y='{height - 20}' text-anchor='middle' font-family='sans-serif'>ΔE (kcal/mol)</text>",
        f"<text x='22' y='{height / 2}' text-anchor='middle' transform='rotate(-90 22 {height / 2})' font-family='sans-serif'>Hydrogen-bond score (lower is better)</text>",
        f"<text x='{left}' y='{top - 14}' font-family='sans-serif' font-size='18'>SEEKER final population</text>",
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = x_position(x_value)
        lines.append(f"<text x='{x:.1f}' y='{top + plot_height + 25}' text-anchor='middle' font-size='11' font-family='sans-serif'>{x_value:.2f}</text>")
        y_value = y_min + fraction * (y_max - y_min)
        y = y_position(y_value)
        lines.append(f"<text x='{left - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='11' font-family='sans-serif'>{y_value:.2f}</text>")
    for row in valid:
        x = x_position(float(row["delta_energy_kcal_mol"]))
        y = y_position(float(row["hbond_score"]))
        color = "#d62728" if row["rank"] == 0 else "#4c78a8"
        radius = 5 if row["rank"] == 0 else 3
        title = html.escape(
            f"id={row['id']} rank={row['rank']} ΔE={row['delta_energy_kcal_mol']:.3f} HB={row['hbond_score']:.4f}"
        )
        lines.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius}' fill='{color}' opacity='0.85'><title>{title}</title></circle>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cluster_feature_space_svg(
    path: Path,
    rows: Sequence[dict[str, Any]],
    representatives: Sequence[tuple[int, dict[str, Any]]],
) -> None:
    """Plot the energy/H-bond feature space and label cluster representatives."""

    valid = [
        row
        for row in rows
        if row["valid"] and math.isfinite(float(row["delta_energy_kcal_mol"]))
    ]
    width, height = 900, 560
    left, right, top, bottom = 85, 30, 40, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    if not valid:
        path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='200'>"
            "<text x='20' y='40'>No valid candidates</text></svg>\n",
            encoding="utf-8",
        )
        return
    x_values = [float(row["delta_energy_kcal_mol"]) for row in valid]
    y_values = [float(row["hbond_score"]) for row in valid]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' y2='{top + plot_height}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' stroke='#333'/>",
        f"<text x='{width / 2}' y='{height - 20}' text-anchor='middle' font-family='sans-serif'>ΔE (kcal/mol)</text>",
        f"<text x='22' y='{height / 2}' text-anchor='middle' transform='rotate(-90 22 {height / 2})' font-family='sans-serif'>Hydrogen-bond score</text>",
        f"<text x='{left}' y='{top - 14}' font-family='sans-serif' font-size='18'>SEEKER feature space — cluster representatives</text>",
    ]
    for row in valid:
        x = x_position(float(row["delta_energy_kcal_mol"]))
        y = y_position(float(row["hbond_score"]))
        color = "#d62728" if row["rank"] == 0 else "#a0a0a0"
        lines.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color}' opacity='0.65'/>"
        )
    for cluster_id, representative in representatives:
        x = x_position(float(representative["delta_energy_kcal_mol"]))
        y = y_position(float(representative["hbond_score"]))
        lines.extend(
            (
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='8' fill='none' stroke='#111' stroke-width='2'/>",
                f"<path d='M {x - 5:.2f} {y:.2f} H {x + 5:.2f} M {x:.2f} {y - 5:.2f} V {y + 5:.2f}' stroke='#111' stroke-width='1.5'/>",
                f"<text x='{x + 10:.2f}' y='{y - 9:.2f}' font-family='sans-serif' font-size='12'>C{cluster_id}</text>",
            )
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_CLUSTER_PALETTE = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#aec7e8",
)


def _cluster_colour(cluster_id: int) -> str:
    if cluster_id < 0:
        return "#b8b8b8"
    return _CLUSTER_PALETTE[(cluster_id - 1) % len(_CLUSTER_PALETTE)]


def _feature_centroid(
    members: Sequence[dict[str, Any]],
) -> list[float]:
    features = [toroidal_embedding(row["alleles"]) for row in members]
    return [
        sum(feature[index] for feature in features) / len(features)
        for index in range(len(features[0]))
    ]


def _centroid_angles(feature_centroid: Sequence[float]) -> list[float]:
    angles: list[float] = []
    for index in range(0, len(feature_centroid), 2):
        cosine = float(feature_centroid[index])
        sine = float(feature_centroid[index + 1])
        angle = math.degrees(math.atan2(sine, cosine)) % 360.0
        angles.append(0.0 if abs(angle - 360.0) < 1.0e-10 else angle)
    return angles


def _nearest_to_feature_centroid(
    members: Sequence[dict[str, Any]], feature_centroid: Sequence[float]
) -> tuple[dict[str, Any], float]:
    def distance(row: dict[str, Any]) -> float:
        feature = toroidal_embedding(row["alleles"])
        return math.sqrt(
            sum(
                (float(value) - float(centre)) ** 2
                for value, centre in zip(feature, feature_centroid)
            )
        )

    nearest = min(members, key=lambda row: (distance(row), row["id"]))
    return nearest, distance(nearest)


def _write_torsional_projection_svg(
    path: Path,
    points: Sequence[dict[str, Any]],
    centroids: Sequence[dict[str, Any]],
    *,
    title: str,
    cluster_prefix: str,
    centroid_description: str,
    explained_variance: Sequence[float],
    bounds: tuple[float, float, float, float],
) -> None:
    """Plot a shared PCA projection of the toroidal torsional feature space."""

    width, height = 1180, 720
    left, right, top, bottom = 85, 365, 72, 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max, y_min, y_max = bounds

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    pc1_percent = 100.0 * float(explained_variance[0])
    pc2_percent = 100.0 * float(explained_variance[1])
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='30' font-family='sans-serif' font-size='20' font-weight='600'>{html.escape(title)}</text>",
        f"<text x='{left}' y='52' font-family='sans-serif' font-size='12' fill='#555'>{html.escape(centroid_description)}</text>",
        f"<rect x='{left}' y='{top}' width='{plot_width}' height='{plot_height}' fill='#fafafa' stroke='#333'/>",
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = x_position(x_value)
        y_value = y_min + fraction * (y_max - y_min)
        y = y_position(y_value)
        lines.extend(
            (
                f"<line x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{top + plot_height}' stroke='#e4e4e4'/>",
                f"<text x='{x:.2f}' y='{top + plot_height + 24}' text-anchor='middle' font-family='sans-serif' font-size='11'>{x_value:.2f}</text>",
                f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_width}' y2='{y:.2f}' stroke='#e4e4e4'/>",
                f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-family='sans-serif' font-size='11'>{y_value:.2f}</text>",
            )
        )
    lines.extend(
        (
            f"<text x='{left + plot_width / 2}' y='{height - 22}' text-anchor='middle' font-family='sans-serif'>PC1 ({pc1_percent:.1f}% varianza)</text>",
            f"<text x='23' y='{top + plot_height / 2}' text-anchor='middle' transform='rotate(-90 23 {top + plot_height / 2})' font-family='sans-serif'>PC2 ({pc2_percent:.1f}% varianza)</text>",
        )
    )
    for point in points:
        x = x_position(float(point["pc1"]))
        y = y_position(float(point["pc2"]))
        cluster_id = int(point["cluster_id"])
        colour = _cluster_colour(cluster_id)
        tooltip = html.escape(
            f"id={point['individual_id']} cluster={cluster_id} "
            f"alleles={point['alleles_deg']} ΔE={point['delta_energy_kcal_mol']:.3f}"
        )
        if cluster_id < 0:
            lines.append(
                f"<path d='M {x - 2.5:.2f} {y - 2.5:.2f} L {x + 2.5:.2f} {y + 2.5:.2f} M {x - 2.5:.2f} {y + 2.5:.2f} L {x + 2.5:.2f} {y - 2.5:.2f}' stroke='{colour}' stroke-width='1.2'><title>{tooltip}</title></path>"
            )
        else:
            lines.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.2' fill='{colour}' opacity='0.66'><title>{tooltip}</title></circle>"
            )
    for centroid in centroids:
        cluster_id = int(centroid["cluster_id"])
        x = x_position(float(centroid["pc1"]))
        y = y_position(float(centroid["pc2"]))
        colour = _cluster_colour(cluster_id)
        label = f"{cluster_prefix}{cluster_id}"
        tooltip = html.escape(
            f"{label} centroid angles={centroid['centroid_alleles_deg']} "
            f"nearest id={centroid['nearest_individual_id']}"
        )
        lines.extend(
            (
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='9' fill='{colour}' stroke='#111' stroke-width='2.2'><title>{tooltip}</title></circle>",
                f"<path d='M {x - 5:.2f} {y:.2f} H {x + 5:.2f} M {x:.2f} {y - 5:.2f} V {y + 5:.2f}' stroke='white' stroke-width='1.7'/>",
            )
        )

    legend_x = left + plot_width + 30
    lines.append(
        f"<text x='{legend_x}' y='{top}' font-family='sans-serif' font-size='15' font-weight='600'>Centri dei cluster</text>"
    )
    if len(centroids) <= 18:
        for index, centroid in enumerate(centroids):
            cluster_id = int(centroid["cluster_id"])
            y = top + 30 + index * 34
            colour = _cluster_colour(cluster_id)
            angles = ", ".join(
                f"{float(value):.1f}°" for value in centroid["centroid_alleles_deg"]
            )
            lines.extend(
                (
                    f"<circle cx='{legend_x + 8}' cy='{y - 4}' r='7' fill='{colour}' stroke='#222'/>",
                    f"<text x='{legend_x + 22}' y='{y}' font-family='sans-serif' font-size='12' font-weight='600'>{cluster_prefix}{cluster_id} · n={centroid['size']}</text>",
                    f"<text x='{legend_x + 22}' y='{y + 15}' font-family='sans-serif' font-size='10' fill='#555'>θ=({angles}) · vicino ID {centroid['nearest_individual_id']}</text>",
                )
            )
    else:
        lines.extend(
            (
                f"<text x='{legend_x}' y='{top + 30}' font-family='sans-serif' font-size='12'>{len(centroids)} centroidi visualizzati.</text>",
                f"<text x='{legend_x}' y='{top + 50}' font-family='sans-serif' font-size='11' fill='#555'>Colori ciclici; passa il mouse sui centri</text>",
                f"<text x='{legend_x}' y='{top + 66}' font-family='sans-serif' font-size='11' fill='#555'>oppure consulta il CSV dei centroidi.</text>",
            )
        )
    if any(int(point["cluster_id"]) < 0 for point in points):
        noise_y = (
            top + 30 + len(centroids) * 34
            if len(centroids) <= 18
            else top + 96
        )
        lines.append(
            f"<text x='{legend_x}' y='{noise_y}' font-family='sans-serif' font-size='11' fill='#777'>× = rumore HDBSCAN</text>"
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_centroid_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "cluster_id",
        "size",
        "centroid_kind",
        "centroid_alleles_deg",
        "centroid_feature_vector",
        "pc1",
        "pc2",
        "nearest_individual_id",
        "nearest_feature_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "centroid_alleles_deg": json.dumps(
                        row["centroid_alleles_deg"]
                    ),
                    "centroid_feature_vector": json.dumps(
                        row["centroid_feature_vector"]
                    ),
                }
            )


def _write_hybrid_torsional_feature_spaces(
    analysis_dir: Path,
    eligible: Sequence[dict[str, Any]],
    periodic_clusters: Sequence[Sequence[dict[str, Any]]],
    periodicities: Sequence[int],
    density_labels: Sequence[int],
) -> dict[str, Any]:
    """Write comparable grid/HDBSCAN PCA views and exact centroid tables."""

    if len(eligible) < 2:
        return {}
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - guarded with HDBSCAN import
        raise RuntimeError(
            "le immagini del feature space hybrid richiedono scikit-learn"
        ) from exc

    features = [toroidal_embedding(row["alleles"]) for row in eligible]
    projection = PCA(n_components=2, svd_solver="full")
    point_scores = projection.fit_transform(features)
    explained = [float(value) for value in projection.explained_variance_ratio_]
    grid_cluster_by_id = {
        row["id"]: cluster_id
        for cluster_id, members in enumerate(periodic_clusters, start=1)
        for row in members
    }

    grid_centroids: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(periodic_clusters, start=1):
        angles = periodicity_cell_centre(members[0]["alleles"], periodicities)
        feature_centroid = toroidal_embedding(angles)
        score = projection.transform([feature_centroid])[0]
        nearest, nearest_distance = _nearest_to_feature_centroid(
            members, feature_centroid
        )
        grid_centroids.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "centroid_kind": "scan_reference_mode",
                "centroid_alleles_deg": angles,
                "centroid_feature_vector": feature_centroid,
                "pc1": float(score[0]),
                "pc2": float(score[1]),
                "nearest_individual_id": nearest["id"],
                "nearest_feature_distance": nearest_distance,
            }
        )

    hdbscan_centroids: list[dict[str, Any]] = []
    for cluster_id in sorted({label for label in density_labels if label >= 0}):
        members = [
            row
            for row, label in zip(eligible, density_labels)
            if label == cluster_id
        ]
        feature_centroid = _feature_centroid(members)
        angles = _centroid_angles(feature_centroid)
        score = projection.transform([feature_centroid])[0]
        nearest, nearest_distance = _nearest_to_feature_centroid(
            members, feature_centroid
        )
        hdbscan_centroids.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "centroid_kind": "mean_toroidal_feature",
                "centroid_alleles_deg": angles,
                "centroid_feature_vector": feature_centroid,
                "pc1": float(score[0]),
                "pc2": float(score[1]),
                "nearest_individual_id": nearest["id"],
                "nearest_feature_distance": nearest_distance,
            }
        )

    grid_points = [
        {
            "individual_id": row["id"],
            "cluster_id": grid_cluster_by_id[row["id"]],
            "alleles_deg": row["alleles"],
            "delta_energy_kcal_mol": row["delta_energy_kcal_mol"],
            "pc1": float(score[0]),
            "pc2": float(score[1]),
        }
        for row, score in zip(eligible, point_scores)
    ]
    hdbscan_points = [
        {
            "individual_id": row["id"],
            "cluster_id": label,
            "alleles_deg": row["alleles"],
            "delta_energy_kcal_mol": row["delta_energy_kcal_mol"],
            "pc1": float(score[0]),
            "pc2": float(score[1]),
        }
        for row, label, score in zip(eligible, density_labels, point_scores)
    ]
    all_x = [float(score[0]) for score in point_scores] + [
        float(row["pc1"]) for row in [*grid_centroids, *hdbscan_centroids]
    ]
    all_y = [float(score[1]) for score in point_scores] + [
        float(row["pc2"]) for row in [*grid_centroids, *hdbscan_centroids]
    ]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = max(x_max - x_min, 1.0e-6)
    y_span = max(y_max - y_min, 1.0e-6)
    bounds = (
        x_min - 0.08 * x_span,
        x_max + 0.08 * x_span,
        y_min - 0.08 * y_span,
        y_max + 0.08 * y_span,
    )
    grid_svg = analysis_dir / "torsional_feature_space_grid.svg"
    hdbscan_svg = analysis_dir / "torsional_feature_space_hdbscan.svg"
    grid_csv = analysis_dir / "grid_centroids.csv"
    hdbscan_csv = analysis_dir / "hdbscan_centroids.csv"
    _write_torsional_projection_svg(
        grid_svg,
        grid_points,
        grid_centroids,
        title="Spazio torsionale PCA — celle della griglia periodica",
        cluster_prefix="G",
        centroid_description=(
            "I cerchi grandi sono i modi esatti dello SCAN definiti dalle periodicità."
        ),
        explained_variance=explained,
        bounds=bounds,
    )
    _write_torsional_projection_svg(
        hdbscan_svg,
        hdbscan_points,
        hdbscan_centroids,
        title="Spazio torsionale PCA — cluster HDBSCAN",
        cluster_prefix="H",
        centroid_description=(
            "I cerchi grandi sono le medie nello spazio toroidale; × indica rumore."
        ),
        explained_variance=explained,
        bounds=bounds,
    )
    _write_centroid_csv(grid_csv, grid_centroids)
    _write_centroid_csv(hdbscan_csv, hdbscan_centroids)
    return {
        "projection": "PCA of [cos(theta_i), sin(theta_i)]",
        "explained_variance_ratio": explained,
        "shared_axis_bounds": list(bounds),
        "grid_feature_space_svg": str(grid_svg.name),
        "hdbscan_feature_space_svg": str(hdbscan_svg.name),
        "grid_centroids_csv": str(grid_csv.name),
        "hdbscan_centroids_csv": str(hdbscan_csv.name),
    }


def analyze_run(
    run_dir: str | Path,
    max_delta_energy_kcal_mol: float | None = None,
    torsion_threshold_deg: float = 15.0,
    torsion_max_threshold_deg: float | None = None,
    clustering_source: str | None = None,
    clustering_method: str | None = None,
    hybrid_max_candidates: int | None = None,
    hybrid_min_cluster_size: int | None = None,
    hybrid_min_samples: int | None = None,
    hybrid_energy_neighbors: int | None = None,
    hybrid_min_separation_deg: float | None = None,
    pose_permutation_mode: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    config = manifest.get("config", {})
    source = clustering_source or config.get("clustering_source", "archive")
    method = clustering_method or config.get("clustering_method", "complete_linkage")
    resolved_pose_permutation_mode = str(
        config.get("pose_permutation_mode", "equivalent")
        if pose_permutation_mode is None
        else pose_permutation_mode
    )
    if resolved_pose_permutation_mode not in {"equivalent", "ordered"}:
        raise ValueError("pose_permutation_mode deve essere equivalent oppure ordered")
    energy_window = (
        float(config.get("cluster_energy_window_kcal_mol", 10.0))
        if max_delta_energy_kcal_mol is None
        else float(max_delta_energy_kcal_mol)
    )
    hybrid_options = {
        "max_candidates": int(
            config.get("hybrid_max_candidates", 16)
            if hybrid_max_candidates is None
            else hybrid_max_candidates
        ),
        "min_cluster_size": int(
            config.get("hybrid_min_cluster_size", 5)
            if hybrid_min_cluster_size is None
            else hybrid_min_cluster_size
        ),
        "min_samples": int(
            config.get("hybrid_min_samples", 2)
            if hybrid_min_samples is None
            else hybrid_min_samples
        ),
        "energy_neighbors": int(
            config.get("hybrid_energy_neighbors", 8)
            if hybrid_energy_neighbors is None
            else hybrid_energy_neighbors
        ),
        "min_separation_deg": float(
            config.get("hybrid_min_separation_deg", 25.0)
            if hybrid_min_separation_deg is None
            else hybrid_min_separation_deg
        ),
    }
    if hybrid_options["max_candidates"] < 1:
        raise ValueError("hybrid_max_candidates deve essere almeno 1")
    if hybrid_options["min_cluster_size"] < 2:
        raise ValueError("hybrid_min_cluster_size deve essere almeno 2")
    if hybrid_options["min_samples"] < 1:
        raise ValueError("hybrid_min_samples deve essere almeno 1")
    if hybrid_options["energy_neighbors"] < 1:
        raise ValueError("hybrid_energy_neighbors deve essere almeno 1")
    if not math.isfinite(hybrid_options["min_separation_deg"]) or hybrid_options[
        "min_separation_deg"
    ] <= 0.0:
        raise ValueError("hybrid_min_separation_deg deve essere positivo e finito")
    filenames = {
        "archive": "evaluated_archive.csv",
        "final_population": "final_population.csv",
        "pareto_front": "pareto_front.csv",
    }
    if source not in filenames:
        raise ValueError("clustering_source non valido")
    population_path = root / filenames[source]
    if source == "archive" and not population_path.exists():
        population_path = root / "final_population.csv"
        source = "final_population"
    if not population_path.exists():
        raise FileNotFoundError(f"risultati non trovati: {population_path}")
    rows = _read_population(population_path)
    eligible = [
        row
        for row in rows
        if row["valid"] and row["delta_energy_kcal_mol"] <= energy_window
    ]
    objectives = tuple(manifest.get("objectives", ["energy", "hbond"]))
    global_pareto = _local_pareto(eligible, objectives) if eligible else []
    global_pareto_ids = {row["id"] for row in global_pareto}
    objective_best: dict[str, dict[str, Any]] = {
        name: min(
            eligible,
            key=lambda row: (_objective_value(row, name), row["id"]),
        )
        for name in objectives
    } if eligible else {}
    objective_best_roles_by_id: dict[int, set[str]] = {}
    for name, row in objective_best.items():
        objective_best_roles_by_id.setdefault(row["id"], set()).add(name)
    max_threshold = (
        torsion_threshold_deg
        if torsion_max_threshold_deg is None
        else torsion_max_threshold_deg
    )
    periodicities: tuple[int, ...] = ()
    dimensions = len(eligible[0]["alleles"]) if eligible else len(
        manifest.get("active_variables") or manifest.get("genes") or []
    )
    periodic_flags = _manifest_periodic_flags(manifest, dimensions)
    bounded_bins = _manifest_bounded_bins(manifest, dimensions)
    raw_active_variables = manifest.get("active_variables", [])
    if method == "pose_hybrid" and not raw_active_variables:
        raw_active_variables = manifest.get("genes", [])
    active_variables = (
        [
            {
                **dict(item),
                **(
                    {
                        "lower": float(item["physical_bounds"][0]),
                        "upper": float(item["physical_bounds"][1]),
                    }
                    if "lower" not in item
                    and isinstance(item.get("physical_bounds"), list)
                    and len(item["physical_bounds"]) == 2
                    else {}
                ),
            }
            for item in raw_active_variables
            if isinstance(item, dict)
        ]
        if isinstance(raw_active_variables, list)
        else []
    )
    pose_blocks = _manifest_pose_blocks(manifest, root)
    if method == "pose_hybrid" and (
        not pose_blocks or len(active_variables) != dimensions
    ):
        raise ValueError(
            "pose_hybrid richiede i metadati completi fragment_pose e active_variables"
        )
    pose_equivalence_groups = _pose_equivalence_groups(
        pose_blocks,
        root / eligible[0]["structure_file"] if eligible else None,
        float(config.get("topology_tolerance", 0.45)),
    )
    if resolved_pose_permutation_mode == "ordered":
        pose_equivalence_groups = tuple(
            (index,) for index in range(len(pose_blocks))
        )
    pose_permutation_aware = any(
        len(group) > 1 for group in pose_equivalence_groups
    )

    def pose_embedding(values: Sequence[float]) -> Sequence[float]:
        return pose_manifold_embedding(
            values,
            active_variables,
            pose_blocks,
            pose_equivalence_groups,
        )

    def genotype_distance(
        left: Sequence[float], right: Sequence[float]
    ) -> TorsionDistance:
        if method == "pose_hybrid":
            return pose_manifold_distance(
                left,
                right,
                active_variables,
                pose_blocks,
                pose_equivalence_groups,
            )
        return torsion_distance(left, right, periodic_flags)
    is_hybrid = method in {"hybrid", "mixed_hybrid", "pose_hybrid"}
    coarse_cell_count: int | None = None
    coarse_cells_for_visualization: list[list[dict[str, Any]]] = []
    density_labels: list[int] = []
    if method == "complete_linkage":
        clusters = _cluster_candidates(
            eligible,
            torsion_threshold_deg,
            max_threshold,
            periodic_flags,
        )
    elif method in {"periodicity_cells", "hybrid"}:
        if not all(periodic_flags):
            raise ValueError(
                "periodicity_cells/hybrid non è definito per coordinate bounded; "
                "usare --method complete_linkage"
            )
        periodicities = _manifest_periodicities(manifest, dimensions)
        coarse_cells = periodicity_cell_torsional(
            eligible,
            lambda row: row["alleles"],
            periodicities,
            lambda row: row["id"],
        )
        coarse_cell_count = len(coarse_cells)
        coarse_cells_for_visualization = coarse_cells
        clusters = (
            subdivide_torsional_cells(
                coarse_cells,
                lambda row: row["alleles"],
                torsion_threshold_deg,
                max_threshold,
                lambda row: row["id"],
                periodic_flags,
            )
            if method == "hybrid"
            else coarse_cells
        )
    elif method == "mixed_hybrid":
        periodicities = _manifest_periodicities(manifest, dimensions)
        coarse_cells = mixed_cell_torsional(
            eligible,
            lambda row: row["alleles"],
            periodicities,
            periodic_flags,
            bounded_bins,
            lambda row: row["id"],
        )
        coarse_cell_count = len(coarse_cells)
        coarse_cells_for_visualization = coarse_cells
        clusters = subdivide_torsional_cells(
            coarse_cells,
            lambda row: row["alleles"],
            torsion_threshold_deg,
            max_threshold,
            lambda row: row["id"],
            periodic_flags,
        )
    elif method == "pose_hybrid":
        density_labels = hdbscan_torsional_labels(
            eligible,
            lambda row: row["alleles"],
            hybrid_options["min_cluster_size"],
            hybrid_options["min_samples"],
            lambda row: row["id"],
            periodic_flags,
            embedding=pose_embedding,
        )
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row, label in zip(eligible, density_labels):
            key = ("density", label) if label >= 0 else ("noise", row["id"])
            grouped.setdefault(key, []).append(row)
        clusters = sorted(
            grouped.values(),
            key=lambda members: min(row["id"] for row in members),
        )
    else:
        raise ValueError("clustering_method non valido")

    def selected_cell_centre(values: Sequence[float]) -> list[float]:
        if method == "mixed_hybrid":
            return mixed_cell_centre(
                values, periodicities, periodic_flags, bounded_bins
            )
        return periodicity_cell_centre(values, periodicities) if periodicities else []

    density_label_by_id: dict[int, int] = {}
    density_minimum_ids: set[int] = set()
    graph_minimum_ids: set[int] = set()
    if is_hybrid:
        if not density_labels:
            density_labels = hdbscan_torsional_labels(
                eligible,
                lambda row: row["alleles"],
                hybrid_options["min_cluster_size"],
                hybrid_options["min_samples"],
                lambda row: row["id"],
                periodic_flags,
                embedding=pose_embedding if method == "pose_hybrid" else None,
            )
        density_label_by_id = {
            row["id"]: label for row, label in zip(eligible, density_labels)
        }
        for label in sorted({value for value in density_labels if value >= 0}):
            members = [
                row
                for row, assigned in zip(eligible, density_labels)
                if assigned == label
            ]
            density_minimum_ids.add(
                min(members, key=lambda row: (row["energy_hartree"], row["id"]))[
                    "id"
                ]
            )
        graph_minimum_ids = {
            row["id"]
            for row in energy_graph_local_minima(
                eligible,
                lambda row: row["alleles"],
                lambda row: row["energy_hartree"],
                hybrid_options["energy_neighbors"],
                lambda row: row["id"],
                periodic_flags,
                embedding=pose_embedding if method == "pose_hybrid" else None,
            )
        }

    primary_cluster_by_id = {
        row["id"]: cluster_id
        for cluster_id, members in enumerate(clusters, start=1)
        for row in members
    }
    primary_representative_by_cluster: dict[int, dict[str, Any]] = {}
    hybrid_pool_by_id: dict[int, dict[str, Any]] = {}
    adaptive_selected_records: list[dict[str, Any]] = []
    compressed_hybrid = False
    adaptive_hybrid = False
    if is_hybrid:
        for cluster_id, members in enumerate(clusters, start=1):
            representative = (
                min(members, key=lambda row: (row["energy_hartree"], row["id"]))
                if method == "pose_hybrid"
                else
                mixed_core_representative(
                    members, periodicities, periodic_flags, bounded_bins
                )
                if method == "mixed_hybrid"
                else periodicity_core_representative(members, periodicities)
            )
            primary_representative_by_cluster[cluster_id] = representative
            hybrid_pool_by_id.setdefault(
                representative["id"],
                {"candidate": representative, "sources": set()},
            )["sources"].add(
                "pose_density_energy"
                if method == "pose_hybrid"
                else "periodicity_core_energy"
            )
        eligible_by_id = {row["id"]: row for row in eligible}
        for individual_id in density_minimum_ids:
            hybrid_pool_by_id.setdefault(
                individual_id,
                {"candidate": eligible_by_id[individual_id], "sources": set()},
            )["sources"].add("hdbscan_density_minimum")
        for individual_id in graph_minimum_ids:
            hybrid_pool_by_id.setdefault(
                individual_id,
                {"candidate": eligible_by_id[individual_id], "sources": set()},
            )["sources"].add("energy_graph_minimum")
        for row in global_pareto:
            hybrid_pool_by_id.setdefault(
                row["id"],
                {"candidate": row, "sources": set()},
            )["sources"].add("global_pareto")
        for individual_id, names in objective_best_roles_by_id.items():
            record = hybrid_pool_by_id.setdefault(
                individual_id,
                {"candidate": eligible_by_id[individual_id], "sources": set()},
            )
            record["sources"].update(
                f"objective_best:{name}" for name in names
            )
        compressed_hybrid = (
            len(hybrid_pool_by_id) > hybrid_options["max_candidates"]
        )
        adaptive_hybrid = compressed_hybrid or (
            method == "pose_hybrid" and pose_permutation_aware
        )
        if adaptive_hybrid:
            adaptive_selected_records = adaptive_hybrid_selection(
                list(hybrid_pool_by_id.values()),
                hybrid_options["max_candidates"],
                hybrid_options["min_separation_deg"],
                energy_window,
                periodic_flags,
                distance=genotype_distance,
                objectives=objectives,
            )
    analysis_dir = root / "analysis"
    representatives_dir = root / "cluster_representatives"
    selected_dir = root / "selected_candidates"
    legacy_representatives_dir = analysis_dir / "representatives"
    for directory in (representatives_dir, selected_dir, legacy_representatives_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    cluster_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    local_pareto_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    feature_representatives: list[tuple[int, dict[str, Any]]] = []
    selected_candidate_by_id: dict[int, dict[str, Any]] = {}
    for cluster_id, members in enumerate(clusters, start=1):
        local_pareto = _local_pareto(members, objectives)
        local_ids = {row["id"] for row in local_pareto}
        roles: dict[int, set[str]] = {row["id"]: {"local_pareto"} for row in local_pareto}
        energy_best = min(members, key=lambda row: (row["energy_hartree"], row["id"]))
        feature_representatives.append((cluster_id, energy_best))
        selected_best = (
            primary_representative_by_cluster[cluster_id]
            if is_hybrid
            else periodicity_core_representative(members, periodicities)
            if method == "periodicity_cells"
            else energy_best
        )
        if not adaptive_hybrid:
            selected_candidate_by_id[selected_best["id"]] = selected_best
            selected_destination = (
                selected_dir
                / f"cluster_{cluster_id:04d}_candidate_{selected_best['id']:06d}.xyz"
            )
            selected_source = root / selected_best["structure_file"]
            if selected_source.exists():
                shutil.copy2(selected_source, selected_destination)
            selected_rows.append(
                {
                    "cluster_id": cluster_id,
                    "individual_id": selected_best["id"],
                    "selection_role": (
                        "pose_density_energy"
                        if method == "pose_hybrid"
                        else "periodicity_core_energy"
                        if method in {"periodicity_cells", "hybrid", "mixed_hybrid"}
                        else "minimum_energy"
                    ),
                    "discovery_methods": json.dumps([]),
                    "density_cluster_id": density_label_by_id.get(
                        selected_best["id"], ""
                    ),
                    "novelty_rms_deg": 0.0,
                    "cell_modes_deg": json.dumps(
                        selected_cell_centre(selected_best["alleles"])
                    ),
                    "energy_hartree": selected_best["energy_hartree"],
                    "delta_energy_kcal_mol": selected_best[
                        "delta_energy_kcal_mol"
                    ],
                    "hbond_score": selected_best["hbond_score"],
                    "alleles_deg": json.dumps(selected_best["alleles"]),
                    "selected_file": str(selected_destination.relative_to(root)),
                }
            )
        roles.setdefault(energy_best["id"], set()).add("minimum_energy")
        for name in objectives:
            best = min(members, key=lambda row: (_objective_value(row, name), row["id"]))
            roles.setdefault(best["id"], set()).add(f"best_{name}")
        by_id = {row["id"]: row for row in members}
        for member in sorted(members, key=lambda row: row["id"]):
            member_roles = sorted(roles.get(member["id"], set()))
            cluster_rows.append(
                {
                    "cluster": cluster_id,
                    "size": len(members),
                    "representative_id": energy_best["id"],
                    "members": json.dumps([row["id"] for row in members]),
                    "cluster_id": cluster_id,
                    "individual_id": member["id"],
                    "cluster_size": len(members),
                    "is_local_pareto": member["id"] in local_ids,
                    "representative_roles": json.dumps(member_roles),
                    "density_cluster_id": density_label_by_id.get(member["id"], ""),
                    "energy_hartree": member["energy_hartree"],
                    "delta_energy_kcal_mol": member["delta_energy_kcal_mol"],
                    "hbond_score": member["hbond_score"],
                    "hbond_count": member["hbond_count"],
                    "rank": member["rank"],
                    "structure_file": member["structure_file"],
                }
            )
        for individual_id, role_set in sorted(roles.items()):
            representative = by_id[individual_id]
            source_path = root / representative["structure_file"]
            destination = representatives_dir / f"cluster_{cluster_id:04d}_candidate_{individual_id:06d}.xyz"
            if source_path.exists():
                shutil.copy2(source_path, destination)
                shutil.copy2(source_path, legacy_representatives_dir / destination.name)
            representative_rows.append(
                {
                    "cluster_id": cluster_id,
                    "individual_id": individual_id,
                    "roles": json.dumps(sorted(role_set)),
                    "representative_file": str(destination.relative_to(root)),
                }
            )
        local_pareto_rows.extend(
            {
                "cluster_id": cluster_id,
                "individual_id": row["id"],
                "energy_hartree": row["energy_hartree"],
                "hbond_score": row["hbond_score"],
            }
            for row in local_pareto
        )

    discovery_rows: list[dict[str, Any]] = []
    primary_distance_strategy = "exhaustive"
    primary_distance_exact_evaluations = 0
    primary_distance_possible_pairs = 0
    if is_hybrid and adaptive_hybrid:
        adaptive_by_id = {
            record["candidate"]["id"]: record
            for record in adaptive_selected_records
        }
        for selection_index, record in enumerate(adaptive_selected_records, start=1):
            candidate = record["candidate"]
            sources = sorted(record["sources"])
            discovery_methods = [
                source
                for source in sources
                if source not in {"periodicity_core_energy", "pose_density_energy"}
            ]
            selected_candidate_by_id[candidate["id"]] = candidate
            destination = (
                selected_dir
                / f"selected_{selection_index:04d}_candidate_{candidate['id']:06d}.xyz"
            )
            source_path = root / candidate["structure_file"]
            if source_path.exists():
                shutil.copy2(source_path, destination)
            novelty = record["novelty_at_selection_rms_deg"]
            mandatory_objectives = tuple(record["mandatory_objectives"])
            selected_rows.append(
                {
                    "cluster_id": primary_cluster_by_id[candidate["id"]],
                    "individual_id": candidate["id"],
                    "selection_role": (
                        "objective_extreme["
                        + "+".join(mandatory_objectives)
                        + "]"
                        if mandatory_objectives
                        else "adaptive_pareto_fitness_geometry["
                        + "+".join(sources)
                        + "]"
                    ),
                    "discovery_methods": json.dumps(discovery_methods),
                    "density_cluster_id": density_label_by_id.get(
                        candidate["id"], -1
                    ),
                    "novelty_rms_deg": 0.0 if math.isinf(novelty) else novelty,
                    "fitness_novelty": (
                        0.0
                        if math.isinf(record["fitness_novelty_at_selection"])
                        else record["fitness_novelty_at_selection"]
                    ),
                    "objective_values": json.dumps(
                        record["objective_values"], sort_keys=True
                    ),
                    "is_global_pareto": candidate["id"] in global_pareto_ids,
                    "cell_modes_deg": json.dumps(
                        selected_cell_centre(candidate["alleles"])
                    ),
                    "energy_hartree": candidate["energy_hartree"],
                    "delta_energy_kcal_mol": candidate["delta_energy_kcal_mol"],
                    "hbond_score": candidate["hbond_score"],
                    "alleles_deg": json.dumps(candidate["alleles"]),
                    "selected_file": str(destination.relative_to(root)),
                }
            )

        primary_representatives = list(
            primary_representative_by_cluster.values()
        )
        primary_pose_index = (
            _PoseManifoldNearestIndex(
                [representative["alleles"] for representative in primary_representatives],
                active_variables,
                pose_blocks,
            )
            if (
                method == "pose_hybrid"
                and primary_representatives
                and not pose_permutation_aware
            )
            else None
        )
        if primary_pose_index is not None:
            primary_distance_strategy = "exact_indexed_pose_embedding"
        final_selected = list(selected_candidate_by_id.values())
        discovery_pool = sorted(
            hybrid_pool_by_id.values(),
            key=lambda record: (
                record["candidate"]["energy_hartree"],
                record["candidate"]["id"],
            ),
        )
        primary_distance_possible_pairs = (
            len(discovery_pool) * len(primary_representatives)
        )
        for pool_record in discovery_pool:
            candidate = pool_record["candidate"]
            methods = sorted(
                source
                for source in pool_record["sources"]
                if source not in {"periodicity_core_energy", "pose_density_energy"}
            )
            if primary_pose_index is not None:
                nearest_primary, exact_evaluations = primary_pose_index.nearest(
                    candidate["alleles"]
                )
                novelty_to_primary = nearest_primary.rms_deg
                primary_distance_exact_evaluations += exact_evaluations
            else:
                novelty_to_primary = min(
                    genotype_distance(
                        candidate["alleles"], representative["alleles"]
                    ).rms_deg
                    for representative in primary_representatives
                )
                primary_distance_exact_evaluations += len(primary_representatives)
            selected = candidate["id"] in selected_candidate_by_id
            if selected:
                novelty_to_selected = adaptive_by_id[candidate["id"]][
                    "novelty_at_selection_rms_deg"
                ]
                if math.isinf(novelty_to_selected):
                    novelty_to_selected = 0.0
                rejection_reason = ""
            else:
                novelty_to_selected = min(
                    genotype_distance(
                        candidate["alleles"], representative["alleles"]
                    ).rms_deg
                    for representative in final_selected
                )
                rejection_reason = (
                    "below_minimum_separation"
                    if novelty_to_selected < hybrid_options["min_separation_deg"]
                    else "adaptive_candidate_cap_reached"
                )
            discovery_rows.append(
                {
                    "individual_id": candidate["id"],
                    "primary_cluster_id": primary_cluster_by_id[candidate["id"]],
                    "density_cluster_id": density_label_by_id.get(
                        candidate["id"], -1
                    ),
                    "discovery_methods": json.dumps(methods),
                    "energy_hartree": candidate["energy_hartree"],
                    "delta_energy_kcal_mol": candidate[
                        "delta_energy_kcal_mol"
                    ],
                    "alleles_deg": json.dumps(candidate["alleles"]),
                    "novelty_to_primary_rms_deg": novelty_to_primary,
                    "novelty_to_selected_rms_deg": novelty_to_selected,
                    "selected": selected,
                    "rejection_reason": rejection_reason,
                }
            )

    if is_hybrid and not adaptive_hybrid:
        primary_representatives = list(selected_candidate_by_id.values())
        discovery_pool = list(hybrid_pool_by_id.values())
        discovery_pool.sort(
            key=lambda record: (
                record["candidate"]["energy_hartree"],
                record["candidate"]["id"],
            )
        )
        remaining_slots = hybrid_options["max_candidates"] - len(selected_rows)
        accepted_discoveries: list[dict[str, Any]] = []
        for pool_record in discovery_pool:
            candidate = pool_record["candidate"]
            methods = sorted(
                source
                for source in pool_record["sources"]
                if source not in {"periodicity_core_energy", "pose_density_energy"}
            )
            novelty_to_primary = min(
                (
                    genotype_distance(
                        candidate["alleles"], representative["alleles"]
                    ).rms_deg
                    for representative in primary_representatives
                ),
                default=float("inf"),
            )
            novelty_to_selected = min(
                (
                    genotype_distance(
                        candidate["alleles"], representative["alleles"]
                    ).rms_deg
                    for representative in [
                        *primary_representatives,
                        *accepted_discoveries,
                    ]
                ),
                default=float("inf"),
            )
            selected = False
            if candidate["id"] in selected_candidate_by_id:
                rejection_reason = "primary_representative"
            elif remaining_slots <= 0:
                rejection_reason = "candidate_cap_reached"
            else:
                selected = True
                rejection_reason = ""
                remaining_slots -= 1
                accepted_discoveries.append(candidate)
                selected_candidate_by_id[candidate["id"]] = candidate
                discovery_index = len(accepted_discoveries)
                destination = (
                    selected_dir
                    / f"discovery_{discovery_index:04d}_candidate_{candidate['id']:06d}.xyz"
                )
                source_path = root / candidate["structure_file"]
                if source_path.exists():
                    shutil.copy2(source_path, destination)
                selected_rows.append(
                    {
                        "cluster_id": primary_cluster_by_id[candidate["id"]],
                        "individual_id": candidate["id"],
                        "selection_role": "+".join(methods),
                        "discovery_methods": json.dumps(methods),
                        "density_cluster_id": density_label_by_id.get(
                            candidate["id"], -1
                        ),
                        "novelty_rms_deg": novelty_to_primary,
                        "cell_modes_deg": json.dumps(
                            selected_cell_centre(candidate["alleles"])
                        ),
                        "energy_hartree": candidate["energy_hartree"],
                        "delta_energy_kcal_mol": candidate[
                            "delta_energy_kcal_mol"
                        ],
                        "hbond_score": candidate["hbond_score"],
                        "alleles_deg": json.dumps(candidate["alleles"]),
                        "selected_file": str(destination.relative_to(root)),
                    }
                )
            discovery_rows.append(
                {
                    "individual_id": candidate["id"],
                    "primary_cluster_id": primary_cluster_by_id[candidate["id"]],
                    "density_cluster_id": density_label_by_id.get(candidate["id"], -1),
                    "discovery_methods": json.dumps(methods),
                    "energy_hartree": candidate["energy_hartree"],
                    "delta_energy_kcal_mol": candidate["delta_energy_kcal_mol"],
                    "alleles_deg": json.dumps(candidate["alleles"]),
                    "novelty_to_primary_rms_deg": novelty_to_primary,
                    "novelty_to_selected_rms_deg": novelty_to_selected,
                    "selected": selected,
                    "rejection_reason": rejection_reason,
                }
            )

    for selection_index, row in enumerate(selected_rows, start=1):
        row["selection_index"] = selection_index

    hybrid_selection_rows: list[dict[str, Any]] = []
    if is_hybrid:
        selected_order = {
            row["individual_id"]: row["selection_index"] for row in selected_rows
        }
        final_selected = list(selected_candidate_by_id.values())
        adaptive_by_id = {
            record["candidate"]["id"]: record
            for record in adaptive_selected_records
        }
        discovery_reason_by_id = {
            row["individual_id"]: row["rejection_reason"]
            for row in discovery_rows
        }
        for record in sorted(
            hybrid_pool_by_id.values(),
            key=lambda item: (
                item["candidate"]["energy_hartree"],
                item["candidate"]["id"],
            ),
        ):
            candidate = record["candidate"]
            selected = candidate["id"] in selected_order
            if selected:
                minimum_distance = 0.0
                rejection_reason = ""
            else:
                minimum_distance = min(
                    (
                        genotype_distance(
                            candidate["alleles"], representative["alleles"]
                        ).rms_deg
                        for representative in final_selected
                    ),
                    default=float("inf"),
                )
                rejection_reason = discovery_reason_by_id.get(candidate["id"], "")
                if not rejection_reason:
                    rejection_reason = (
                        "below_minimum_separation"
                        if minimum_distance < hybrid_options["min_separation_deg"]
                        else "adaptive_candidate_cap_reached"
                        if adaptive_hybrid
                        else "not_selected"
                    )
            adaptive_record = adaptive_by_id.get(candidate["id"])
            novelty_at_selection = (
                adaptive_record["novelty_at_selection_rms_deg"]
                if adaptive_record is not None
                else ""
            )
            if isinstance(novelty_at_selection, float) and math.isinf(
                novelty_at_selection
            ):
                novelty_at_selection = 0.0
            fitness_novelty_at_selection: float | str = (
                adaptive_record["fitness_novelty_at_selection"]
                if adaptive_record is not None
                else ""
            )
            if isinstance(fitness_novelty_at_selection, float) and math.isinf(
                fitness_novelty_at_selection
            ):
                fitness_novelty_at_selection = 0.0
            hybrid_selection_rows.append(
                {
                    "individual_id": candidate["id"],
                    "primary_cluster_id": primary_cluster_by_id[candidate["id"]],
                    "density_cluster_id": density_label_by_id.get(
                        candidate["id"], -1
                    ),
                    "sources": json.dumps(sorted(record["sources"])),
                    "energy_hartree": candidate["energy_hartree"],
                    "delta_energy_kcal_mol": candidate[
                        "delta_energy_kcal_mol"
                    ],
                    "alleles_deg": json.dumps(candidate["alleles"]),
                    "selected": selected,
                    "selection_order": selected_order.get(candidate["id"], ""),
                    "selection_score": (
                        adaptive_record["selection_score"]
                        if adaptive_record is not None
                        else ""
                    ),
                    "selection_reason": (
                        adaptive_record["selection_reason"]
                        if adaptive_record is not None
                        else ""
                    ),
                    "mandatory_objectives": json.dumps(
                        list(adaptive_record["mandatory_objectives"])
                        if adaptive_record is not None
                        else []
                    ),
                    "objective_values": json.dumps(
                        {
                            name: _objective_value(candidate, name)
                            for name in objectives
                        },
                        sort_keys=True,
                    ),
                    "is_global_pareto": candidate["id"] in global_pareto_ids,
                    "novelty_at_selection_rms_deg": novelty_at_selection,
                    "fitness_novelty_at_selection": fitness_novelty_at_selection,
                    "minimum_distance_to_final_selection_rms_deg": minimum_distance,
                    "rejection_reason": rejection_reason,
                }
            )

    clusters_path = analysis_dir / "clusters.csv"
    fields = list(cluster_rows[0]) if cluster_rows else [
        "cluster", "size", "representative_id", "members", "cluster_id",
        "individual_id", "cluster_size", "is_local_pareto", "representative_roles",
        "density_cluster_id",
        "energy_hartree", "delta_energy_kcal_mol", "hbond_score", "hbond_count",
        "rank", "structure_file",
    ]
    with clusters_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cluster_rows)

    output_tables = [
        ("cluster_representatives.csv", representative_rows, ["cluster_id", "individual_id", "roles", "representative_file"]),
        ("cluster_local_pareto.csv", local_pareto_rows, ["cluster_id", "individual_id", "energy_hartree", "hbond_score"]),
        (
            "selected_candidates.csv",
            selected_rows,
            [
                "cluster_id", "individual_id", "selection_role",
                "discovery_methods", "density_cluster_id", "novelty_rms_deg",
                "cell_modes_deg",
                "energy_hartree",
                "delta_energy_kcal_mol", "hbond_score", "alleles_deg",
                "selected_file", "selection_index",
            ],
        ),
    ]
    if is_hybrid:
        output_tables.extend(
            [
                (
                "hybrid_discovery.csv",
                discovery_rows,
                [
                    "individual_id",
                    "primary_cluster_id",
                    "density_cluster_id",
                    "discovery_methods",
                    "energy_hartree",
                    "delta_energy_kcal_mol",
                    "alleles_deg",
                    "novelty_to_primary_rms_deg",
                    "novelty_to_selected_rms_deg",
                    "selected",
                    "rejection_reason",
                ],
                ),
                (
                    "hybrid_selection.csv",
                    hybrid_selection_rows,
                    [
                        "individual_id",
                        "primary_cluster_id",
                        "density_cluster_id",
                        "sources",
                        "energy_hartree",
                        "delta_energy_kcal_mol",
                        "alleles_deg",
                        "selected",
                        "selection_order",
                        "selection_score",
                        "selection_reason",
                        "mandatory_objectives",
                        "objective_values",
                        "is_global_pareto",
                        "novelty_at_selection_rms_deg",
                        "fitness_novelty_at_selection",
                        "minimum_distance_to_final_selection_rms_deg",
                        "rejection_reason",
                    ],
                ),
            ]
        )
    for filename, data, empty_fields in output_tables:
        with (analysis_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            fields_for_file = list(data[0]) if data else empty_fields
            writer = csv.DictWriter(handle, fieldnames=fields_for_file)
            writer.writeheader()
            writer.writerows(data)

    _write_scatter_svg(analysis_dir / "pareto_scatter.svg", rows)
    _write_cluster_feature_space_svg(
        analysis_dir / "cluster_feature_space.svg", rows, feature_representatives
    )
    torsional_feature_spaces: dict[str, Any] = {}
    if method == "hybrid":
        torsional_feature_spaces = _write_hybrid_torsional_feature_spaces(
            analysis_dir,
            eligible,
            coarse_cells_for_visualization,
            periodicities,
            density_labels,
        )
    summary = {
        "valid_final_candidates": sum(1 for row in rows if row["valid"]),
        "pareto_candidates": sum(1 for row in rows if row["valid"] and row["rank"] == 0),
        "energy_window_kcal_mol": energy_window,
        "candidates_in_energy_window": len(eligible),
        "torsion_cluster_threshold_deg": torsion_threshold_deg,
        "torsion_cluster_max_threshold_deg": max_threshold,
        "clustering_source": source,
        "clustering_method": method,
        "clusters": len(clusters),
        "coarse_periodicity_cells": coarse_cell_count,
        "selected_candidates": len(selected_rows),
        "periodicities": list(periodicities),
        "maximum_periodicity_cells": (
            math.prod(
                order if is_periodic else bins
                for order, is_periodic, bins in zip(
                    periodicities, periodic_flags, bounded_bins
                )
            )
            if periodicities
            else None
        ),
        "objectives": manifest.get("objectives", ["energy", "hbond"]),
        "torsional_feature_spaces": torsional_feature_spaces or None,
        "pose_feature_space": (
            {
                "embedding": "radial + S2 unit direction + sign-invariant SO3 rotation matrix",
                "distance": (
                    "permutation-minimized normalized radial/S2/SO3 geodesic RMS"
                    if pose_permutation_aware
                    else "normalized radial/S2/SO3 geodesic RMS"
                ),
                "fragment_pose_blocks": [block.name for block in pose_blocks],
                "equivalent_fragment_groups": [
                    [pose_blocks[index].name for index in group]
                    for group in pose_equivalence_groups
                ],
                "permutation_aware": pose_permutation_aware,
                "permutation_mode": resolved_pose_permutation_mode,
            }
            if method == "pose_hybrid"
            else None
        ),
        "hybrid": (
            {
                **hybrid_options,
                "primary_periodicity_representatives": (
                    None if method == "pose_hybrid" else len(clusters)
                ),
                "coarse_periodicity_cells": coarse_cell_count,
                "torsional_subclusters": (
                    None if method == "pose_hybrid" else len(clusters)
                ),
                "primary_manifold_representatives": (
                    len(clusters) if method == "pose_hybrid" else None
                ),
                "density_clusters": len(
                    {label for label in density_labels if label >= 0}
                ),
                "density_noise_points": sum(
                    1 for label in density_labels if label < 0
                ),
                "density_minima": len(density_minimum_ids),
                "energy_graph_minima": len(graph_minimum_ids),
                "global_pareto_candidates": len(global_pareto_ids),
                "objective_extrema": {
                    name: row["id"] for name, row in objective_best.items()
                },
                "objective_extrema_selected": {
                    name: row["id"] in selected_candidate_by_id
                    for name, row in objective_best.items()
                },
                "discovery_candidates_examined": len(discovery_rows),
                "primary_distance_strategy": primary_distance_strategy,
                "primary_distance_exact_evaluations": (
                    primary_distance_exact_evaluations
                ),
                "primary_distance_possible_pairs": (
                    primary_distance_possible_pairs
                ),
                "compressed_periodicity_cells": compressed_hybrid,
                "compressed_primary_clusters": (
                    len(clusters) > hybrid_options["max_candidates"]
                ),
                "compressed_candidate_pool": compressed_hybrid,
                "adaptive_selection_applied": adaptive_hybrid,
                "selection_strategy": (
                    "mandatory_objective_extrema_plus_pareto_fitness_geometry_diversity"
                    if adaptive_hybrid
                    else "complete_hybrid_pool_including_pareto"
                ),
                "hybrid_pool_candidates": len(hybrid_pool_by_id),
                "primary_representatives_selected": sum(
                    1
                    for representative in primary_representative_by_cluster.values()
                    if representative["id"] in selected_candidate_by_id
                ),
                "data_supported_selected": sum(
                    1
                    for individual_id in selected_candidate_by_id
                    if individual_id in density_minimum_ids
                    or individual_id in graph_minimum_ids
                ),
                "discoveries_selected": sum(
                    1
                    for individual_id in selected_candidate_by_id
                    if individual_id not in {
                        representative["id"]
                        for representative in primary_representative_by_cluster.values()
                    }
                ),
            }
            if is_hybrid
            else None
        ),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
