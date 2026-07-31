"""Domain models shared by the SEEKER engine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def moved(self, position: tuple[float, float, float]) -> "Atom":
        return Atom(self.element, *position)


@dataclass(frozen=True)
class Molecule:
    atoms: tuple[Atom, ...]
    comment: str = ""
    charge: int = 0
    multiplicity: int = 1

    def with_atoms(self, atoms: list[Atom] | tuple[Atom, ...]) -> "Molecule":
        return Molecule(tuple(atoms), self.comment, self.charge, self.multiplicity)


@dataclass(frozen=True)
class Gene:
    """A rotatable dihedral; atom indices are zero-based internally."""

    name: str
    atoms: tuple[int, int, int, int]
    periodicity: int = 1
    periodic: bool = True
    uniform_prior: bool = False


@dataclass(frozen=True)
class FragmentPoseGene:
    """One rigid relative pose between two disconnected molecular fragments.

    ``reference_atoms`` and ``moving_atoms`` are zero-based complete connected
    components.  The genetic block has six physical degrees of freedom and is
    realized natively as one complete translation/rotation block. Distance bounds
    are absolute geometric-centre separations in angstrom; the angular bounds
    are maximum geodesic displacements from the reference pose in degrees.
    """

    name: str
    reference_atoms: tuple[int, ...]
    moving_atoms: tuple[int, ...]
    distance_bounds: tuple[float, float]
    direction_max_degrees: float = 180.0
    orientation_max_degrees: float = 180.0
    scan_points: int = 3


@dataclass(frozen=True)
class HydrogenPiParameters:
    """Heuristic geometry parameters for one X-H donor family."""

    z0_angstrom: float
    sigma_z_angstrom: float
    rho_c_angstrom: float
    sigma_beta_degrees: float
    weight: float = 1.0

    def validate(self, label: str) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"HPI {label} parameter {name} must be positive and finite")


@dataclass(frozen=True)
class HydrogenDoubleBondParameters:
    """Heuristic geometry parameters for one X-H...double-bond contact."""

    r0_angstrom: float
    sigma_r_angstrom: float
    axial_c_angstrom: float
    sigma_beta_degrees: float
    weight: float = 1.0

    def validate(self, label: str) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"HBD {label} parameter {name} must be positive and finite"
                )


@dataclass(frozen=True)
class HydrogenPiRingSpec:
    """A named, zero-based ring atom selection read from genes.txt."""

    name: str
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class HydrogenPiConfig:
    """Non-genetic configuration of the optional X-H...pi objective."""

    mode: str = "auto"
    included_rings: tuple[HydrogenPiRingSpec, ...] = ()
    excluded_rings: tuple[HydrogenPiRingSpec, ...] = ()
    oh: HydrogenPiParameters = field(
        default_factory=lambda: HydrogenPiParameters(2.35, 0.30, 1.80, 20.0)
    )
    nh: HydrogenPiParameters = field(
        default_factory=lambda: HydrogenPiParameters(2.45, 0.35, 1.80, 25.0)
    )
    sh: HydrogenPiParameters = field(
        default_factory=lambda: HydrogenPiParameters(2.55, 0.40, 1.90, 30.0)
    )
    configured: bool = False

    def validate(self) -> None:
        if self.mode not in {"auto", "explicit"}:
            raise ValueError("HPI mode must be auto or explicit")
        if self.mode == "explicit" and not self.included_rings:
            raise ValueError("HPI explicit mode requires at least one HPI_RING directive")
        if self.mode == "explicit" and self.excluded_rings:
            raise ValueError("HPI_EXCLUDE directives are not allowed in explicit mode")
        self.oh.validate("OH")
        self.nh.validate("NH")
        self.sh.validate("SH")
        for spec in (*self.included_rings, *self.excluded_rings):
            if not 5 <= len(spec.atoms) <= 7 or len(set(spec.atoms)) != len(spec.atoms):
                raise ValueError(f"HPI ring {spec.name} requires 5 to 7 distinct atoms")
        included_names = [item.name.upper() for item in self.included_rings]
        excluded_names = [item.name.upper() for item in self.excluded_rings]
        if len(set(included_names)) != len(included_names):
            raise ValueError("duplicate HPI_RING name")
        if len(set(excluded_names)) != len(excluded_names):
            raise ValueError("duplicate HPI_EXCLUDE name")
        included = [frozenset(item.atoms) for item in self.included_rings]
        excluded = [frozenset(item.atoms) for item in self.excluded_rings]
        if len(set(included)) != len(included):
            raise ValueError("duplicate HPI_RING atom selection")
        if len(set(excluded)) != len(excluded):
            raise ValueError("duplicate HPI_EXCLUDE atom selection")
        if set(included) & set(excluded):
            raise ValueError("the same HPI ring cannot be both included and excluded")


@dataclass(frozen=True)
class NativePoseCoordinate:
    """One bounded scalar in a native rigid-fragment SE(3) pose block."""

    name: str
    atoms: tuple[int, ...]
    pose_name: str
    component: str
    lower: float
    upper: float
    reference_value: float
    reference_atoms: tuple[int, ...]
    moving_atoms: tuple[int, ...]
    units: str
    scan_points: int = 3
    periodic: bool = False
    periodicity: int = 1
    uniform_prior: bool = True

    @property
    def span(self) -> float:
        return self.upper - self.lower

    def normalize(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"coordinate {self.name}: non-finite value")
        return min(self.upper, max(self.lower, number))

    def scaled_distance(self, left: float, right: float) -> float:
        return abs(float(left) - float(right)) / self.span

    def value_from_allele(self, allele_deg: float) -> float:
        allele = min(360.0, max(0.0, float(allele_deg)))
        return self.lower + self.span * allele / 360.0

    def allele_from_value(self, value: float) -> float:
        normalized = self.normalize(value)
        if math.isclose(normalized, self.upper, abs_tol=1.0e-12):
            return math.nextafter(360.0, 0.0)
        return 360.0 * (normalized - self.lower) / self.span

    @property
    def reference_allele(self) -> float:
        return self.allele_from_value(self.reference_value)


@dataclass
class Individual:
    id: int
    alleles: list[float]
    energy: float = float("inf")
    hbond: float = float("inf")
    hbond_count: int = 0
    rank: int = 10**9
    crowding: float = 0.0
    valid: bool = False
    parents: tuple[int, ...] = ()
    operator: str = "initial"
    generation: int = 0
    error: str = ""
    objective_values: dict[str, float] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    island: int = 0
    origin_island: int = 0

    @property
    def objectives(self) -> tuple[float, float]:
        return (self.energy, self.hbond)

    def objective_value(self, name: str) -> float:
        if name == "energy":
            return self.energy
        if name == "hbond":
            return self.hbond
        return float(self.objective_values.get(name, float("inf")))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parents"] = list(self.parents)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Individual":
        payload = dict(data)
        payload["parents"] = tuple(int(value) for value in payload.get("parents", []))
        return cls(**payload)


@dataclass(frozen=True)
class RunConfig:
    population_size: int = 32
    offspring_size: int = 32
    generations: int = 30
    seed: int = 1
    workers: int = 1
    islands: int = 1
    migration_interval: int = 0
    migration_size: int = 1
    migration_selection: str = "pareto"
    base_mutation_weight: float = 0.45
    base_crossover_weight: float = 0.80
    mutation_weight_amplitude: float = 0.10
    crossover_weight_amplitude: float = 0.10
    operator_oscillations: int = 2
    operator_schedule: tuple[tuple[float, float], ...] = ()
    mutation_operator: str = "resample_one"
    crossover_operator: str = "mixed_sbx"
    mutation_sigma_deg: float = 20.0
    periodicity_grid_step_deg: float = 20.0
    sbx_eta: float = 15.0
    duplicate_threshold_deg: float = 3.0
    duplicate_mean_threshold_deg: float | None = None
    duplicate_max_threshold_deg: float | None = None
    initialization_strategy: str = "maximin"
    initial_pool_factor: int = 20
    initial_pool_sampling: str = "latin_hypercube"
    initial_scan_layout: str = "tensor"
    initial_scan_grid: str = "uniform"
    initial_scan_points_mode: str = "fixed"
    initial_scan_points: int = 3
    topology_tolerance: float = 0.45
    hbond_cutoff_angstrom: float = 3.2
    hbond_contact_threshold: float = -0.30
    hh_clash_distance_angstrom: float = 1.40
    geometric_prescreen: bool = True
    steric_hh_scale: float = 0.55
    steric_heavy_heavy_scale: float = 0.55
    steric_hydrogen_heavy_scale: float = 0.50
    steric_exclude_hops: int = 3
    checkpoint_every: int = 1
    max_duplicate_attempts: int = 30
    objectives: tuple[str, ...] = ("energy", "hbond")
    hbond_pi_config: HydrogenPiConfig = field(default_factory=HydrogenPiConfig)
    rotor_symmetry_sigma: float = 0.15
    rotor_anisotropy_sigma: float = 0.15
    early_stopping: bool = False
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1.0e-6
    early_stop_diversity_deg: float = 8.0
    early_stop_min_generations: int = 0
    archive_stagnation_patience: int = 0
    clustering_source: str = "archive"
    clustering_method: str = "complete_linkage"
    cluster_mean_threshold_deg: float = 15.0
    cluster_max_threshold_deg: float = 15.0
    cluster_energy_window_kcal_mol: float = 10.0
    hybrid_max_candidates: int = 16
    hybrid_min_cluster_size: int = 5
    hybrid_min_samples: int = 2
    hybrid_energy_neighbors: int = 8
    hybrid_min_separation_deg: float = 25.0

    @property
    def resolved_duplicate_mean_threshold_deg(self) -> float:
        return (
            self.duplicate_threshold_deg
            if self.duplicate_mean_threshold_deg is None
            else self.duplicate_mean_threshold_deg
        )

    @property
    def resolved_duplicate_max_threshold_deg(self) -> float:
        return (
            self.duplicate_threshold_deg
            if self.duplicate_max_threshold_deg is None
            else self.duplicate_max_threshold_deg
        )

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.offspring_size < 1:
            raise ValueError("offspring_size must be at least 1")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.islands < 1:
            raise ValueError("islands must be at least 1")
        if self.migration_interval < 0:
            raise ValueError("migration_interval cannot be negative")
        if self.migration_size < 1:
            raise ValueError("migration_size must be at least 1")
        if self.migration_size >= self.population_size:
            raise ValueError("migration_size must be smaller than population_size")
        if self.migration_selection not in {"pareto", "random"}:
            raise ValueError("migration_selection must be pareto or random")
        if self.islands > 1 and self.migration_interval == 0:
            raise ValueError(
                "multiple islands require migration_interval greater than zero"
            )
        for name in (
            "base_mutation_weight",
            "base_crossover_weight",
            "mutation_weight_amplitude",
            "crossover_weight_amplitude",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.base_mutation_weight + self.base_crossover_weight <= 0.0:
            raise ValueError("at least one base operator weight must be positive")
        critical_sines = {-1.0, 0.0, 1.0}
        if self.mutation_weight_amplitude > 0.0:
            critical_sines.add(-self.base_mutation_weight / self.mutation_weight_amplitude)
        if self.crossover_weight_amplitude > 0.0:
            critical_sines.add(self.base_crossover_weight / self.crossover_weight_amplitude)
        for sine in (value for value in critical_sines if -1.0 <= value <= 1.0):
            mutation = max(0.0, self.base_mutation_weight + self.mutation_weight_amplitude * sine)
            crossover = max(0.0, self.base_crossover_weight - self.crossover_weight_amplitude * sine)
            if mutation + crossover <= 0.0:
                raise ValueError("the schedule can produce zero total operator weight")
        if (
            isinstance(self.operator_oscillations, bool)
            or int(self.operator_oscillations) != self.operator_oscillations
            or self.operator_oscillations < 0
        ):
            raise ValueError("operator_oscillations must be a non-negative integer")
        if self.mutation_operator not in {"resample_one", "local_gaussian"}:
            raise ValueError("mutation_operator must be resample_one or local_gaussian")
        if self.crossover_operator not in {"mixed_sbx", "uniform_gene"}:
            raise ValueError("crossover_operator must be mixed_sbx or uniform_gene")
        if not math.isfinite(self.mutation_sigma_deg) or self.mutation_sigma_deg <= 0.0:
            raise ValueError("mutation_sigma_deg must be positive and finite")
        if self.operator_schedule:
            if len(self.operator_schedule) != self.generations:
                raise ValueError("operator_schedule must contain exactly one row per generation")
            for mutation, crossover in self.operator_schedule:
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (mutation, crossover)):
                    raise ValueError("operator_schedule probabilities must be between zero and one")
                if not math.isclose(mutation + crossover, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError("operator_schedule probabilities must sum to one")
        if (
            not math.isfinite(self.periodicity_grid_step_deg)
            or not 0.0 < self.periodicity_grid_step_deg < 360.0
        ):
            raise ValueError("periodicity_grid_step_deg must be between 0 and 360")
        if not math.isfinite(self.sbx_eta) or self.sbx_eta <= 0.0:
            raise ValueError("sbx_eta must be positive and finite")
        for name in (
            "duplicate_threshold_deg",
            "resolved_duplicate_mean_threshold_deg",
            "resolved_duplicate_max_threshold_deg",
            "cluster_mean_threshold_deg",
            "cluster_max_threshold_deg",
            "hybrid_min_separation_deg",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.initialization_strategy not in {"random", "maximin", "lhs", "scan"}:
            raise ValueError(
                "initialization_strategy must be random, maximin, lhs, or scan"
            )
        if self.initial_pool_sampling not in {"prior", "latin_hypercube"}:
            raise ValueError("initial_pool_sampling must be prior or latin_hypercube")
        if self.initial_pool_factor < 1:
            raise ValueError("initial_pool_factor must be at least 1")
        if self.initial_scan_layout not in {"tensor", "one-at-a-time"}:
            raise ValueError("initial_scan_layout must be tensor or one-at-a-time")
        if self.initial_scan_grid not in {"uniform", "periodicity-modes"}:
            raise ValueError("initial_scan_grid must be uniform or periodicity-modes")
        if self.initial_scan_points_mode not in {"fixed", "periodicity"}:
            raise ValueError("initial_scan_points_mode must be fixed or periodicity")
        if self.initial_scan_points < 1:
            raise ValueError("initial_scan_points must be at least 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if self.max_duplicate_attempts < 1:
            raise ValueError("max_duplicate_attempts must be at least 1")
        for name in (
            "steric_hh_scale",
            "steric_heavy_heavy_scale",
            "steric_hydrogen_heavy_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.steric_exclude_hops < 1:
            raise ValueError("steric_exclude_hops must be at least 1")
        if not self.objectives:
            raise ValueError("at least one objective must be active")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("the objective list contains duplicates")
        if self.objectives[:2] != ("energy", "hbond"):
            raise ValueError("energy and hbond must remain the first two objectives")
        self.hbond_pi_config.validate()
        if not math.isfinite(self.rotor_symmetry_sigma) or self.rotor_symmetry_sigma <= 0.0:
            raise ValueError("rotor_symmetry_sigma must be positive and finite")
        if not math.isfinite(self.rotor_anisotropy_sigma) or self.rotor_anisotropy_sigma <= 0.0:
            raise ValueError("rotor_anisotropy_sigma must be positive and finite")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be at least 1")
        if not math.isfinite(self.early_stop_min_delta) or self.early_stop_min_delta < 0.0:
            raise ValueError("early_stop_min_delta must be finite and non-negative")
        if not math.isfinite(self.early_stop_diversity_deg) or self.early_stop_diversity_deg < 0.0:
            raise ValueError("early_stop_diversity_deg must be finite and non-negative")
        if self.early_stop_min_generations < 0:
            raise ValueError("early_stop_min_generations cannot be negative")
        if self.archive_stagnation_patience < 0:
            raise ValueError("archive_stagnation_patience cannot be negative")
        if self.clustering_source not in {"archive", "final_population", "pareto_front"}:
            raise ValueError("invalid clustering_source")
        if self.clustering_method not in {
            "complete_linkage",
            "periodicity_cells",
            "hybrid",
            "mixed_hybrid",
            "pose_hybrid",
        }:
            raise ValueError("invalid clustering_method")
        if (
            not math.isfinite(self.cluster_energy_window_kcal_mol)
            or self.cluster_energy_window_kcal_mol < 0.0
        ):
            raise ValueError("cluster_energy_window_kcal_mol must be non-negative")
        if self.hybrid_max_candidates < 1:
            raise ValueError("hybrid_max_candidates must be at least 1")
        if self.hybrid_min_cluster_size < 2:
            raise ValueError("hybrid_min_cluster_size must be at least 2")
        if self.hybrid_min_samples < 1:
            raise ValueError("hybrid_min_samples must be at least 1")
        if self.hybrid_energy_neighbors < 1:
            raise ValueError("hybrid_energy_neighbors must be at least 1")


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    evaluations: int
    valid: int
    pareto_size: int
    min_energy_hartree: float
    min_hbond_score: float
    diversity_deg: float
    mutation_probability: float = 0.0
    crossover_probability: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    operator_counts: dict[str, int] = field(default_factory=dict)
    objective_minima: dict[str, float] = field(default_factory=dict)
    stagnant_generations: int = 0
    stop_reason: str = ""
    new_unique_individuals: int = 0
    archive_size: int = 0
    new_clusters: int = 0
    duplicate_rejection_rate: float = 0.0
    geometric_rejection_rate: float = 0.0
    energy_backend_calls: int = 0
    energy_backend_calls_saved: int = 0
    archive_stagnant_generations: int = 0
    geometry_screen_checks: int = 0
    migrations: int = 0
    migration_survivors: int = 0
    island_diversities_deg: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
