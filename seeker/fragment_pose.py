"""Manifold-aware genetic operations for rigid inter-fragment poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

import numpy as np

from .geometry import BondGraph
from .models import FragmentPoseGene, Molecule, NativePoseCoordinate


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Sequence[float]) -> Vector3:
    length = _norm(value)
    if length <= 1.0e-14:
        raise ValueError("zero-length fragment direction")
    return tuple(float(item) / length for item in value)  # type: ignore[return-value]


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _scale(value: Sequence[float], factor: float) -> Vector3:
    return tuple(float(item) * factor for item in value)  # type: ignore[return-value]


def _add(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(float(a) + float(b) for a, b in zip(left, right))  # type: ignore[return-value]


def quaternion_normalize(value: Sequence[float]) -> Quaternion:
    length = _norm(value)
    if length <= 1.0e-14:
        return (1.0, 0.0, 0.0, 0.0)
    result = tuple(float(item) / length for item in value)
    if result[0] < 0.0:
        result = tuple(-item for item in result)
    return result  # type: ignore[return-value]


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return quaternion_normalize(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quaternion_conjugate(value: Quaternion) -> Quaternion:
    return (value[0], -value[1], -value[2], -value[3])


def quaternion_from_rotation_vector(value: Sequence[float]) -> Quaternion:
    angle = _norm(value)
    if angle <= 1.0e-14:
        return (1.0, 0.0, 0.0, 0.0)
    half = 0.5 * angle
    factor = math.sin(half) / angle
    return quaternion_normalize(
        (math.cos(half), factor * value[0], factor * value[1], factor * value[2])
    )


def rotation_vector_from_quaternion(value: Quaternion) -> Vector3:
    w, x, y, z = quaternion_normalize(value)
    sine = math.sqrt(x * x + y * y + z * z)
    if sine <= 1.0e-14:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(sine, max(0.0, w))
    return (angle * x / sine, angle * y / sine, angle * z / sine)


def quaternion_distance(left: Quaternion, right: Quaternion) -> float:
    cosine = min(1.0, max(0.0, abs(_dot(left, right))))
    return 2.0 * math.acos(cosine)


def quaternion_slerp(left: Quaternion, right: Quaternion, fraction: float) -> Quaternion:
    first = quaternion_normalize(left)
    second = quaternion_normalize(right)
    cosine = _dot(first, second)
    if cosine < 0.0:
        second = tuple(-item for item in second)  # type: ignore[assignment]
        cosine = -cosine
    cosine = min(1.0, max(-1.0, cosine))
    if cosine > 0.999999:
        return quaternion_normalize(
            tuple((1.0 - fraction) * a + fraction * b for a, b in zip(first, second))
        )
    angle = math.acos(cosine)
    denominator = math.sin(angle)
    left_weight = math.sin((1.0 - fraction) * angle) / denominator
    right_weight = math.sin(fraction * angle) / denominator
    return quaternion_normalize(
        tuple(left_weight * a + right_weight * b for a, b in zip(first, second))
    )


def vector_angle(left: Sequence[float], right: Sequence[float]) -> float:
    cosine = min(1.0, max(-1.0, _dot(_unit(left), _unit(right))))
    return math.acos(cosine)


def direction_slerp(left: Sequence[float], right: Sequence[float], fraction: float) -> Vector3:
    first = _unit(left)
    second = _unit(right)
    cosine = min(1.0, max(-1.0, _dot(first, second)))
    angle = math.acos(cosine)
    if angle <= 1.0e-12:
        return first
    if math.pi - angle <= 1.0e-8:
        helper = (1.0, 0.0, 0.0) if abs(first[0]) < 0.8 else (0.0, 1.0, 0.0)
        perpendicular = _unit(_cross(first, helper))
        return _unit(
            _add(_scale(first, math.cos(math.pi * fraction)), _scale(perpendicular, math.sin(math.pi * fraction)))
        )
    denominator = math.sin(angle)
    return _unit(
        _add(
            _scale(first, math.sin((1.0 - fraction) * angle) / denominator),
            _scale(second, math.sin(fraction * angle) / denominator),
        )
    )


def sample_unit_vector(rng: random.Random) -> Vector3:
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    radial = math.sqrt(max(0.0, 1.0 - z * z))
    return (radial * math.cos(phi), radial * math.sin(phi), z)


def sample_direction_in_cone(
    reference: Sequence[float], maximum_angle: float, rng: random.Random
) -> Vector3:
    axis = _unit(reference)
    limit = min(math.pi, max(0.0, float(maximum_angle)))
    if limit <= 1.0e-14:
        return axis
    cosine = 1.0 - rng.random() * (1.0 - math.cos(limit))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    phi = 2.0 * math.pi * rng.random()
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.8 else (0.0, 1.0, 0.0)
    first = _unit(_cross(axis, helper))
    second = _cross(axis, first)
    tangent = _add(_scale(first, math.cos(phi)), _scale(second, math.sin(phi)))
    return _unit(_add(_scale(axis, cosine), _scale(tangent, sine)))


def _sample_haar_angle(maximum_angle: float, rng: random.Random) -> float:
    limit = min(math.pi, max(0.0, float(maximum_angle)))
    if limit <= 1.0e-14:
        return 0.0
    target = rng.random() * (limit - math.sin(limit))
    lower, upper = 0.0, limit
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if middle - math.sin(middle) < target:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def sample_orientation(
    reference_rotation: Sequence[float], maximum_angle: float, rng: random.Random
) -> Vector3:
    reference = quaternion_from_rotation_vector(reference_rotation)
    angle = _sample_haar_angle(maximum_angle, rng)
    axis = sample_unit_vector(rng)
    delta = quaternion_from_rotation_vector(_scale(axis, angle))
    return rotation_vector_from_quaternion(quaternion_multiply(delta, reference))


@dataclass(frozen=True)
class FragmentPoseBlock:
    name: str
    variable_indices: tuple[int, int, int, int, int, int]
    distance_bounds: tuple[float, float]
    direction_max_radian: float
    orientation_max_radian: float
    reference_translation: Vector3
    reference_rotation: Vector3
    reference_atoms: tuple[int, ...]
    moving_atoms: tuple[int, ...]

    @property
    def reference_direction(self) -> Vector3:
        return _unit(self.reference_translation)

    def validate_values(self, values: Sequence[float], tolerance: float = 1.0e-8) -> None:
        translation = tuple(float(values[index]) for index in self.variable_indices[:3])
        rotation = tuple(float(values[index]) for index in self.variable_indices[3:])
        radius = _norm(translation)
        if not self.distance_bounds[0] - tolerance <= radius <= self.distance_bounds[1] + tolerance:
            raise ValueError(f"{self.name}: center distance {radius:.8f} outside bounds")
        direction = vector_angle(translation, self.reference_translation)
        if direction > self.direction_max_radian + tolerance:
            raise ValueError(f"{self.name}: center direction outside angular bound")
        orientation = quaternion_distance(
            quaternion_from_rotation_vector(rotation),
            quaternion_from_rotation_vector(self.reference_rotation),
        )
        if orientation > self.orientation_max_radian + tolerance:
            raise ValueError(f"{self.name}: fragment orientation outside angular bound")

    def sample_values(self, values: Sequence[float], rng: random.Random) -> list[float]:
        result = [float(item) for item in values]
        lower, upper = self.distance_bounds
        radius = lower + (upper - lower) * rng.random()
        direction = sample_direction_in_cone(
            self.reference_translation, self.direction_max_radian, rng
        )
        rotation = sample_orientation(
            self.reference_rotation, self.orientation_max_radian, rng
        )
        for index, value in zip(self.variable_indices[:3], _scale(direction, radius)):
            result[index] = value
        for index, value in zip(self.variable_indices[3:], rotation):
            result[index] = value
        return result

    def mutate_values(self, values: Sequence[float], rng: random.Random) -> list[float]:
        result = [float(item) for item in values]
        mode = rng.randrange(3)
        translation = tuple(result[index] for index in self.variable_indices[:3])
        if mode == 0:
            lower, upper = self.distance_bounds
            radius = lower + (upper - lower) * rng.random()
            direction = _unit(translation)
            translation = _scale(direction, radius)
        elif mode == 1:
            radius = _norm(translation)
            direction = sample_direction_in_cone(
                self.reference_translation, self.direction_max_radian, rng
            )
            translation = _scale(direction, radius)
        else:
            rotation = sample_orientation(
                self.reference_rotation, self.orientation_max_radian, rng
            )
            for index, value in zip(self.variable_indices[3:], rotation):
                result[index] = value
        for index, value in zip(self.variable_indices[:3], translation):
            result[index] = value
        self.validate_values(result)
        return result

    def crossover_values(
        self, first: Sequence[float], second: Sequence[float], rng: random.Random
    ) -> list[float]:
        result = [float(item) for item in first]
        first_translation = tuple(first[index] for index in self.variable_indices[:3])
        second_translation = tuple(second[index] for index in self.variable_indices[:3])
        fraction = rng.random()
        radius = (1.0 - fraction) * _norm(first_translation) + fraction * _norm(second_translation)
        direction = direction_slerp(first_translation, second_translation, fraction)
        first_rotation = quaternion_from_rotation_vector(
            tuple(first[index] for index in self.variable_indices[3:])
        )
        second_rotation = quaternion_from_rotation_vector(
            tuple(second[index] for index in self.variable_indices[3:])
        )
        rotation = rotation_vector_from_quaternion(
            quaternion_slerp(first_rotation, second_rotation, fraction)
        )
        for index, value in zip(self.variable_indices[:3], _scale(direction, radius)):
            result[index] = value
        for index, value in zip(self.variable_indices[3:], rotation):
            result[index] = value
        try:
            self.validate_values(result)
        except ValueError:
            source = first if rng.random() < 0.5 else second
            for index in self.variable_indices:
                result[index] = float(source[index])
            self.validate_values(result)
        return result

    def normalized_distance_components(
        self, first: Sequence[float], second: Sequence[float]
    ) -> tuple[float, float, float]:
        first_translation = tuple(first[index] for index in self.variable_indices[:3])
        second_translation = tuple(second[index] for index in self.variable_indices[:3])
        radial_span = self.distance_bounds[1] - self.distance_bounds[0]
        radial = abs(_norm(first_translation) - _norm(second_translation)) / radial_span
        direction_scale = self.direction_max_radian or math.pi
        direction = vector_angle(first_translation, second_translation) / direction_scale
        first_rotation = quaternion_from_rotation_vector(
            tuple(first[index] for index in self.variable_indices[3:])
        )
        second_rotation = quaternion_from_rotation_vector(
            tuple(second[index] for index in self.variable_indices[3:])
        )
        orientation_scale = self.orientation_max_radian or math.pi
        orientation = quaternion_distance(first_rotation, second_rotation) / orientation_scale
        return min(1.0, radial), min(1.0, direction), min(1.0, orientation)


def fragment_pose_blocks_from_variables(
    raw_variables: Sequence[Mapping[str, object]], labels: Sequence[str]
) -> tuple[FragmentPoseBlock, ...]:
    groups: dict[str, dict[str, tuple[int, Mapping[str, object]]]] = {}
    for index, item in enumerate(raw_variables):
        metadata = item.get("genetic_block")
        if not isinstance(metadata, Mapping) or metadata.get("type") != "fragment_pose":
            continue
        name = str(metadata.get("name", "")).strip()
        component = str(metadata.get("component", "")).strip().lower()
        if not name or component not in {"tx", "ty", "tz", "rx", "ry", "rz"}:
            raise ValueError("invalid fragment_pose genetic_block metadata")
        if component in groups.setdefault(name, {}):
            raise ValueError(f"duplicate {name} fragment-pose component: {component}")
        groups[name][component] = (index, metadata)
    blocks: list[FragmentPoseBlock] = []
    components = ("tx", "ty", "tz", "rx", "ry", "rz")
    for name, group in groups.items():
        if set(group) != set(components):
            raise ValueError(f"{name}: fragment pose requires complete FTRANS/FROT triplets")
        metadata = group["tx"][1]
        indices = tuple(group[item][0] for item in components)
        bounds = tuple(float(item) for item in metadata["distance_bounds_angstrom"])
        reference_translation = tuple(
            float(item) for item in metadata["reference_translation_angstrom"]
        )
        reference_rotation = tuple(
            float(item) for item in metadata["reference_rotation_radian"]
        )
        blocks.append(
            FragmentPoseBlock(
                name=name,
                variable_indices=indices,  # type: ignore[arg-type]
                distance_bounds=(bounds[0], bounds[1]),
                direction_max_radian=math.radians(float(metadata["direction_max_degrees"])),
                orientation_max_radian=math.radians(
                    float(metadata["orientation_max_degrees"])
                ),
                reference_translation=reference_translation,  # type: ignore[arg-type]
                reference_rotation=reference_rotation,  # type: ignore[arg-type]
                reference_atoms=tuple(int(item) - 1 for item in metadata["reference_atoms"]),
                moving_atoms=tuple(int(item) - 1 for item in metadata["moving_atoms"]),
            )
        )
    return tuple(blocks)


@dataclass(frozen=True)
class PreparedNativePoseBlock:
    """Precomputed Cartesian data for one native rigid-fragment pose."""

    genetic_block: FragmentPoseBlock
    reference_center: np.ndarray
    centered_moving_coordinates: np.ndarray
    orientation_frame: np.ndarray


@dataclass(frozen=True)
class NativeFragmentPosePlan:
    """B-free, vectorized realization of complete native fragment poses."""

    reference: Molecule
    coordinates: tuple[NativePoseCoordinate, ...]
    pose_blocks: tuple[FragmentPoseBlock, ...]
    prepared_blocks: tuple[PreparedNativePoseBlock, ...]

    @property
    def reference_alleles(self) -> tuple[float, ...]:
        return tuple(coordinate.reference_allele for coordinate in self.coordinates)

    def values_from_alleles_batch(
        self, allele_batch: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        alleles = np.asarray(allele_batch, dtype=float)
        if alleles.ndim != 2 or alleles.shape[1] != len(self.coordinates):
            raise ValueError("native POSE alleles must have shape (ncandidate, ncoordinate)")
        if not np.all(np.isfinite(alleles)):
            raise ValueError("native POSE alleles must be finite")
        bounded = np.clip(alleles, 0.0, 360.0)
        lower = np.asarray([item.lower for item in self.coordinates], dtype=float)
        span = np.asarray([item.span for item in self.coordinates], dtype=float)
        values = lower[None, :] + bounded * span[None, :] / 360.0
        for block in self.pose_blocks:
            translation = values[:, list(block.variable_indices[:3])]
            radii = np.linalg.norm(translation, axis=1)
            if np.any(radii < block.distance_bounds[0] - 1.0e-8) or np.any(
                radii > block.distance_bounds[1] + 1.0e-8
            ):
                raise ValueError(f"{block.name}: native POSE center distance outside bounds")
            reference = np.asarray(block.reference_translation, dtype=float)
            denominator = radii * np.linalg.norm(reference)
            cosine = np.sum(translation * reference[None, :], axis=1) / denominator
            direction = np.arccos(np.clip(cosine, -1.0, 1.0))
            if np.any(direction > block.direction_max_radian + 1.0e-8):
                raise ValueError(f"{block.name}: native POSE direction outside bound")
            rotation = values[:, list(block.variable_indices[3:])]
            if np.any(
                np.linalg.norm(rotation, axis=1)
                > block.orientation_max_radian + 1.0e-8
            ):
                raise ValueError(f"{block.name}: native POSE orientation outside bound")
        return values

    def coordinates_from_alleles_batch(
        self, allele_batch: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        values = self.values_from_alleles_batch(allele_batch)
        reference = np.asarray(
            [atom.position for atom in self.reference.atoms], dtype=float
        )
        result = np.broadcast_to(
            reference,
            (values.shape[0], *reference.shape),
        ).copy()
        for prepared in self.prepared_blocks:
            block = prepared.genetic_block
            moving = np.asarray(block.moving_atoms, dtype=int)
            translation = values[:, list(block.variable_indices[:3])]
            rotations_local = _row_rotation_matrices_from_vectors(
                values[:, list(block.variable_indices[3:])]
            )
            frame = prepared.orientation_frame
            rotations = np.einsum(
                "ij,njk,kl->nil", frame, rotations_local, frame.T
            )
            centers = prepared.reference_center[None, :] + translation
            result[:, moving, :] = (
                np.einsum(
                    "aj,njk->nak",
                    prepared.centered_moving_coordinates,
                    rotations,
                )
                + centers[:, None, :]
            )
        return result

    def apply_batch(
        self, allele_batch: Sequence[Sequence[float]] | np.ndarray
    ) -> tuple[Molecule, ...]:
        batches = self.coordinates_from_alleles_batch(allele_batch)
        return tuple(
            self.reference.with_atoms(
                [
                    atom.moved(tuple(float(value) for value in xyz))
                    for atom, xyz in zip(self.reference.atoms, coordinates)
                ]
            )
            for coordinates in batches
        )

    def apply(self, alleles: Sequence[float]) -> Molecule:
        return self.apply_batch((alleles,))[0]


def prepare_native_fragment_poses(
    molecule: Molecule,
    poses: Sequence[FragmentPoseGene],
    graph: BondGraph,
) -> NativeFragmentPosePlan:
    """Compile pose-only input into native SE(3) blocks."""

    requested = tuple(poses)
    if not requested:
        raise ValueError("native POSE realization requires at least one pose")
    if len(graph) != len(molecule.atoms):
        raise ValueError("native POSE graph is incompatible with the molecule")
    reference_coordinates = np.asarray(
        [atom.position for atom in molecule.atoms], dtype=float
    )
    common_reference = requested[0].reference_atoms
    occupied: set[int] = set()
    coordinates: list[NativePoseCoordinate] = []
    genetic_blocks: list[FragmentPoseBlock] = []
    prepared_blocks: list[PreparedNativePoseBlock] = []

    for pose in requested:
        if pose.reference_atoms != common_reference:
            raise ValueError("native POSE blocks require one common reference fragment")
        _validate_complete_component(pose.name, pose.reference_atoms, graph, "reference")
        _validate_complete_component(pose.name, pose.moving_atoms, graph, "moving")
        if occupied.intersection(pose.moving_atoms):
            raise ValueError(f"{pose.name}: native POSE moving fragments overlap")
        occupied.update(pose.moving_atoms)
        _fragment_frame(reference_coordinates, pose.reference_atoms)
        moving_frame = _fragment_frame(reference_coordinates, pose.moving_atoms)
        reference_center = np.mean(
            reference_coordinates[list(pose.reference_atoms), :], axis=0
        )
        moving_center = np.mean(
            reference_coordinates[list(pose.moving_atoms), :], axis=0
        )
        reference_translation = moving_center - reference_center
        start = len(coordinates)
        components = ("tx", "ty", "tz", "rx", "ry", "rz")
        orientation_limit = max(math.radians(pose.orientation_max_degrees), 1.0e-9)
        for position, component in enumerate(components):
            translation = position < 3
            limit = float(pose.distance_bounds[1]) if translation else orientation_limit
            coordinates.append(
                NativePoseCoordinate(
                    name=f"{pose.name}_{component.upper()}",
                    atoms=pose.moving_atoms,
                    pose_name=pose.name,
                    component=component,
                    lower=-limit,
                    upper=limit,
                    reference_value=(
                        float(reference_translation[position]) if translation else 0.0
                    ),
                    reference_atoms=pose.reference_atoms,
                    moving_atoms=pose.moving_atoms,
                    units="angstrom" if translation else "radian",
                    scan_points=pose.scan_points,
                )
            )
        block = FragmentPoseBlock(
            name=pose.name,
            variable_indices=tuple(range(start, start + 6)),  # type: ignore[arg-type]
            distance_bounds=pose.distance_bounds,
            direction_max_radian=math.radians(pose.direction_max_degrees),
            orientation_max_radian=math.radians(pose.orientation_max_degrees),
            reference_translation=tuple(float(value) for value in reference_translation),
            reference_rotation=(0.0, 0.0, 0.0),
            reference_atoms=pose.reference_atoms,
            moving_atoms=pose.moving_atoms,
        )
        genetic_blocks.append(block)
        prepared_blocks.append(
            PreparedNativePoseBlock(
                block,
                reference_center,
                reference_coordinates[list(pose.moving_atoms), :] - moving_center,
                moving_frame,
            )
        )

    return NativeFragmentPosePlan(
        molecule,
        tuple(coordinates),
        tuple(genetic_blocks),
        tuple(prepared_blocks),
    )


def _validate_complete_component(
    name: str,
    atoms: Sequence[int],
    graph: BondGraph,
    role: str,
) -> None:
    selected = tuple(int(index) for index in atoms)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{name}: native POSE {role} fragment is empty or duplicated")
    if any(index < 0 or index >= len(graph) for index in selected):
        raise ValueError(f"{name}: native POSE {role} atom index is outside the molecule")
    component = {selected[0]}
    stack = [selected[0]]
    while stack:
        atom = stack.pop()
        for neighbour in graph[atom]:
            if neighbour not in component:
                component.add(neighbour)
                stack.append(neighbour)
    if component != set(selected):
        raise ValueError(
            f"{name}: native POSE {role} atoms must be one complete disconnected component"
        )


def _fragment_frame(coordinates: np.ndarray, atoms: Sequence[int]) -> np.ndarray:
    indices = tuple(int(index) for index in atoms)
    center = np.mean(coordinates[list(indices), :], axis=0)
    centered = coordinates[list(indices), :] - center
    if int(np.sum(np.linalg.svd(centered, compute_uv=False) > 1.0e-8)) < 2:
        raise ValueError("native POSE fragment orientation is underdefined")
    ranked = sorted(
        indices,
        key=lambda atom: (-float(np.linalg.norm(coordinates[atom] - center)), atom),
    )
    first_atom = ranked[0]
    first_axis = coordinates[first_atom] - center
    first_axis /= np.linalg.norm(first_axis)
    candidates = []
    for atom in indices:
        if atom == first_atom:
            continue
        vector = coordinates[atom] - center
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-8:
            continue
        candidates.append((abs(float(np.dot(first_axis, vector / length))), -length, atom))
    if not candidates:
        raise ValueError("native POSE fragment has no second orientation anchor")
    second_atom = min(candidates)[2]
    second_raw = np.cross(first_axis, coordinates[second_atom] - center)
    second_axis = second_raw / np.linalg.norm(second_raw)
    third_axis = np.cross(first_axis, second_axis)
    third_axis /= np.linalg.norm(third_axis)
    return np.column_stack((first_axis, second_axis, third_axis))


def _row_rotation_matrices_from_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("native POSE rotation batch must have shape (n, 3)")
    count = values.shape[0]
    theta = np.linalg.norm(values, axis=1)
    skew = np.zeros((count, 3, 3), dtype=float)
    skew[:, 0, 1] = -values[:, 2]
    skew[:, 0, 2] = values[:, 1]
    skew[:, 1, 0] = values[:, 2]
    skew[:, 1, 2] = -values[:, 0]
    skew[:, 2, 0] = -values[:, 1]
    skew[:, 2, 1] = values[:, 0]
    skew2 = np.einsum("nij,njk->nik", skew, skew)
    small = theta < 1.0e-8
    sine_scale = np.empty(count, dtype=float)
    cosine_scale = np.empty(count, dtype=float)
    sine_scale[small] = 1.0 - theta[small] ** 2 / 6.0
    cosine_scale[small] = 0.5 - theta[small] ** 2 / 24.0
    sine_scale[~small] = np.sin(theta[~small]) / theta[~small]
    cosine_scale[~small] = (1.0 - np.cos(theta[~small])) / theta[~small] ** 2
    return (
        np.eye(3, dtype=float)[None, :, :]
        - sine_scale[:, None, None] * skew
        + cosine_scale[:, None, None] * skew2
    )
