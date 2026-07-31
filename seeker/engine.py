"""Elitist multi-objective genetic conformer search engine."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from .backends import CachedEnergyBackend, EnergyBackend
from .descriptors import rotational_constants_mhz, rotor_shape_scores
from .fitness import (
    DisconnectedComponentsPenaltyModel,
    HydrogenBondModel,
    HydrogenDoubleBondModel,
    HydrogenPiModel,
)
from .geometry import (
    PreparedGene,
    apply_torsions,
    build_bond_graph,
    dihedral_deg,
    genotype_key,
    genotype_rms_deg,
    maximin_select_genotypes,
    prepare_genes,
    steric_prescreen,
    torsion_distance_score,
    torsionally_similar,
)
from .input import molecule_fingerprint
from .models import (
    GenerationStats,
    Gene,
    Individual,
    Molecule,
    NativePoseCoordinate,
    RunConfig,
)

from .nsga2 import assign_rank_and_crowding, environmental_selection, tournament
from .objectives import OBJECTIVES, validate_objectives
from .operators import (
    choose_operator,
    mixed_sbx_genotype,
    mutate_one_local_gaussian,
    mutate_one_periodic,
    oscillating_operator_probabilities,
    scan_initial_genotypes,
    sample_periodic_genotype,
    sample_periodic_genotypes_latin_hypercube,
    uniform_gene_crossover,
)
from .output import (
    save_checkpoint,
    write_final_results,
    write_history,
    write_migration_events,
)


GeneticCoordinate = Gene | NativePoseCoordinate


def run_fingerprint(
    molecule: Molecule,
    genes: Sequence[GeneticCoordinate],
    config: RunConfig,
    backend_signature: dict[str, Any],
    descriptor_signature: dict[str, Any] | None = None,
) -> str:
    scientific_config = asdict(config)
    # These settings affect execution only, not the scientific trajectory.
    # ``generations`` is intentionally retained because it defines the phase of
    # the oscillating mutation/crossover schedule.
    for key in ("workers", "checkpoint_every"):
        scientific_config.pop(key, None)
    payload = {
        "molecule": molecule_fingerprint(molecule, genes),
        "config": scientific_config,
        "backend": backend_signature,
        "descriptors": descriptor_signature or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GeneticConformerSearch:
    def __init__(
        self,
        reference: Molecule,
        genes: Sequence[GeneticCoordinate],
        backend: EnergyBackend,
        config: RunConfig,
        output_dir: str | Path,
        structure_provider: Callable[[Individual], Molecule] | None = None,
        allele_structure_provider: Callable[[Sequence[float]], Molecule] | None = None,
        allele_structure_batch_provider: (
            Callable[[Sequence[Sequence[float]]], Sequence[Molecule]] | None
        ) = None,
        reference_alleles: Sequence[float] | None = None,
        auto_analyze: bool = True,
        progress_callback: Callable[
            [GenerationStats, Sequence[Sequence[Individual]], Sequence[dict[str, Any]]],
            None,
        ]
        | None = None,
    ) -> None:
        config.validate()
        validate_objectives(config.objectives)
        self.reference = reference
        self.genes = tuple(genes)
        self.periodic_mask = tuple(gene.periodic for gene in self.genes)
        self.backend = backend
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.objective_names = tuple(config.objectives)
        self.rotor_objectives = tuple(
            name
            for name in self.objective_names
            if name.startswith("rotor_") or name.startswith("rotational_")
        )
        self._structure_provider = structure_provider
        self._allele_structure_provider = allele_structure_provider
        self._allele_structure_batch_provider = allele_structure_batch_provider
        self.auto_analyze = bool(auto_analyze)
        self.progress_callback = progress_callback
        self.reference_alleles = (
            tuple(float(value) for value in reference_alleles)
            if reference_alleles is not None
            else None
        )
        if self.reference_alleles is not None and len(self.reference_alleles) != len(self.genes):
            raise ValueError("the number of reference alleles is incompatible with the genes")

        self.reference_graph = build_bond_graph(reference, config.topology_tolerance)
        self.prepared_genes: tuple[PreparedGene, ...] = (
            prepare_genes(self.genes, self.reference_graph)
            if structure_provider is None
            and allele_structure_provider is None
            and allele_structure_batch_provider is None
            else ()
        )
        self.hbond_model = HydrogenBondModel.from_reference(
            reference,
            self.reference_graph,
            config.hbond_cutoff_angstrom,
            config.hbond_contact_threshold,
            config.hh_clash_distance_angstrom,
        )
        self.hbond_pi_model = (
            HydrogenPiModel.from_reference(
                reference,
                self.reference_graph,
                config.hbond_pi_config,
            )
            if (
                "hbond_pi" in self.objective_names
                or "disconnected_components_penalty" in self.objective_names
                or config.hbond_pi_config.configured
            )
            else None
        )
        self.hbond_double_model = (
            HydrogenDoubleBondModel.from_reference(reference, self.reference_graph)
            if "hbond_=" in self.objective_names
            else None
        )
        self.disconnected_components_model = (
            DisconnectedComponentsPenaltyModel.from_reference(
                reference,
                self.reference_graph,
            )
            if "disconnected_components_penalty" in self.objective_names
            else None
        )
        self.island_rngs = [
            random.Random(config.seed + 104729 * island)
            for island in range(config.islands)
        ]
        self.rng = self.island_rngs[0]
        self.current_island = 0
        self.migration_events: list[dict[str, Any]] = []
        self.next_individual_id = 0
        self.total_evaluations = 0
        self.archive: dict[tuple[float, ...], Individual] = {}
        self.duplicate_rejections = 0
        self.duplicate_attempts = 0
        self.geometric_rejections = 0
        self.energy_backend_calls = 0
        self.energy_backend_calls_saved = 0
        self.geometry_screen_checks = 0
        self._archive_added_since_stats = 0
        self._new_clusters_since_stats = 0
        self.fingerprint = run_fingerprint(
            reference,
            genes,
            config,
            backend.signature,
        )

    def _report_progress(
        self,
        stats: GenerationStats,
        islands: Sequence[Sequence[Individual]],
        migration_events: Sequence[dict[str, Any]] = (),
    ) -> bool:
        """Send a generation to the TUI, returning whether it consumed output."""

        if self.progress_callback is None:
            return False
        self.progress_callback(stats, islands, migration_events)
        return True

    def _new_individual(
        self,
        alleles: Sequence[float],
        parents: tuple[int, ...] = (),
        operator: str = "initial",
        generation: int = 0,
        island: int | None = None,
        origin_island: int | None = None,
    ) -> Individual:
        assigned_island = self.current_island if island is None else int(island)
        individual = Individual(
            id=self.next_individual_id,
            alleles=[
                float(value) % 360.0
                if gene.periodic
                else min(360.0, max(0.0, float(value)))
                for value, gene in zip(alleles, self.genes)
            ],
            parents=parents,
            operator=operator,
            generation=generation,
            island=assigned_island,
            origin_island=(
                assigned_island if origin_island is None else int(origin_island)
            ),
        )
        self.next_individual_id += 1
        return individual

    def _genotype_key(self, alleles: Sequence[float]) -> tuple[float, ...]:
        return genotype_key(alleles, periodic=self.periodic_mask)

    def _genotype_rms(
        self, first: Sequence[float], second: Sequence[float]
    ) -> float:
        return genotype_rms_deg(first, second, self.periodic_mask)

    def _genotype_score(
        self, first: Sequence[float], second: Sequence[float]
    ) -> float:
        return torsion_distance_score(
            first,
            second,
            self.config.resolved_duplicate_mean_threshold_deg,
            self.config.resolved_duplicate_max_threshold_deg,
            self.periodic_mask,
        )

    def _genotypes_similar(
        self,
        first: Sequence[float],
        second: Sequence[float],
        mean_threshold: float,
        max_threshold: float,
    ) -> bool:
        return torsionally_similar(
            first,
            second,
            mean_threshold,
            max_threshold,
            self.periodic_mask,
        )

    def _mutate(self, alleles: Sequence[float]) -> list[float]:
        if self.config.mutation_operator == "local_gaussian":
            mutated, _index = mutate_one_local_gaussian(
                alleles, self.genes, self.rng, self.config.mutation_sigma_deg
            )
        else:
            mutated, _index = mutate_one_periodic(
                alleles,
                self.genes,
                self.rng,
                self.config.periodicity_grid_step_deg,
            )
        return mutated

    def _population_diversity(self, population: Sequence[Individual]) -> float:
        if len(population) < 2:
            return 0.0
        values = [
            self._genotype_rms(left.alleles, right.alleles)
            for offset, left in enumerate(population)
            for right in population[offset + 1 :]
        ]
        finite = [value for value in values if math.isfinite(value)]
        return sum(finite) / len(finite) if finite else 0.0

    def _random_alleles(self) -> list[float]:
        return sample_periodic_genotype(
            self.genes,
            self.rng,
            self.config.periodicity_grid_step_deg,
        )

    def _initial_population(self) -> list[Individual]:
        if self.config.initialization_strategy == "random":
            return self._initial_population_random()

        if self.config.initialization_strategy == "lhs":
            samples = sample_periodic_genotypes_latin_hypercube(
                self.genes,
                self.config.population_size,
                self.rng,
                self.config.periodicity_grid_step_deg,
            )
            return [
                self._new_individual(alleles, operator="initial_lhs")
                for alleles in samples
            ]

        if self.config.initialization_strategy == "scan":
            reference_alleles = self._reference_genotype()
            samples = scan_initial_genotypes(
                self.genes,
                reference_alleles,
                self.config.initial_scan_layout,
                self.config.initial_scan_grid,
                self.config.initial_scan_points_mode,
                self.config.initial_scan_points,
            )
            if len(samples) != self.config.population_size:
                raise ValueError(
                    "lo SCAN quaternionico genera "
                    f"{len(samples)} punti, ma population_size={self.config.population_size}"
                )
            return [
                self._new_individual(alleles, operator="initial_scan")
                for alleles in samples
            ]

        target = self.config.population_size
        pool_size = max(target, target * self.config.initial_pool_factor)
        if self.config.initial_pool_sampling == "latin_hypercube":
            samples = sample_periodic_genotypes_latin_hypercube(
                self.genes,
                pool_size,
                self.rng,
                self.config.periodicity_grid_step_deg,
            )
        else:
            samples = [self._random_alleles() for _ in range(pool_size)]

        reference_alleles = self._reference_genotype()
        candidates: list[list[float]] = []
        seen: set[tuple[float, ...]] = set()
        for alleles in [reference_alleles, *samples]:
            key = self._genotype_key(alleles)
            if key in seen:
                self.duplicate_attempts += 1
                self.duplicate_rejections += 1
                continue
            seen.add(key)
            try:
                molecule = self._molecule_from_alleles(alleles)
            except (ValueError, RuntimeError):
                self.geometric_rejections += 1
                self.energy_backend_calls_saved += 1
                continue
            screen = self._screen_geometry(molecule)
            if not screen.valid:
                self.geometric_rejections += 1
                self.energy_backend_calls_saved += 1
                continue
            candidates.append([float(value) % 360.0 for value in alleles])
        if len(candidates) < target:
            raise RuntimeError(
                f"pool iniziale insufficiente: {len(candidates)} configurazioni valide e uniche per {target} richieste"
            )

        selected = maximin_select_genotypes(
            candidates,
            target,
            self.config.resolved_duplicate_mean_threshold_deg,
            self.config.resolved_duplicate_max_threshold_deg,
            periodic=self.periodic_mask,
        )
        return [self._new_individual(alleles, operator="initial_maximin") for alleles in selected]

    def _initial_population_random(self) -> list[Individual]:
        """Historical sequential initialization retained for ablation studies."""

        population: list[Individual] = []
        attempts = 0
        while len(population) < self.config.population_size:
            attempts += 1
            alleles = self._random_alleles()
            self.duplicate_attempts += 1
            if any(
                genotype_rms_deg(alleles, existing.alleles, self.periodic_mask)
                < self.config.duplicate_threshold_deg
                for existing in population
            ):
                self.duplicate_rejections += 1
                if attempts > self.config.population_size * 100:
                    raise RuntimeError("could not generate a duplicate-free initial population")
                continue
            population.append(self._new_individual(alleles))
        return population

    def _screen_geometry(self, molecule: Molecule):
        self.geometry_screen_checks += 1
        if not self.config.geometric_prescreen:
            from .geometry import GeometryScreenResult

            return GeometryScreenResult(True)
        return steric_prescreen(
            molecule,
            self.reference_graph,
            self.config.steric_hh_scale,
            self.config.steric_heavy_heavy_scale,
            self.config.steric_hydrogen_heavy_scale,
            self.config.steric_exclude_hops,
            self.config.hh_clash_distance_angstrom,
        )

    def _reference_genotype(self) -> list[float]:
        if self.reference_alleles is not None:
            return list(self.reference_alleles)
        return [
            dihedral_deg(*(self.reference.atoms[index].position for index in gene.atoms))
            % 360.0
            for gene in self.genes
        ]

    def _molecule_from_alleles(self, alleles: Sequence[float]) -> Molecule:
        if self._allele_structure_provider is not None:
            return self._allele_structure_provider(alleles)
        return apply_torsions(self.reference, self.prepared_genes, alleles)

    def structure_for(self, individual: Individual) -> Molecule:
        if self._structure_provider is not None:
            return self._structure_provider(individual)
        return self._molecule_from_alleles(individual.alleles)

    def _evaluate_one(
        self,
        individual: Individual,
        prepared_molecule: Molecule | None = None,
    ) -> Individual:
        individual.objective_values = {
            name: float("inf") for name in self.objective_names if name not in ("energy", "hbond")
        }
        individual.properties = {}
        try:
            molecule = (
                prepared_molecule
                if prepared_molecule is not None
                else self.structure_for(individual)
            )
            screen = self._screen_geometry(molecule)
            if not screen.valid:
                pair = screen.offending_pair
                pair_text = f" atomi {pair[0] + 1}-{pair[1] + 1}" if pair else ""
                distance_text = (
                    f" distanza={screen.minimum_offending_distance:.6f} Å"
                    if screen.minimum_offending_distance is not None
                    else ""
                )
                raise ValueError(f"[geometry] {screen.reason}{pair_text}{distance_text}")
            energy = self.backend.evaluate(molecule)
            hbond_score, hbond_count, hbond_details = self.hbond_model.evaluate(molecule)
            if not (math.isfinite(energy.energy_hartree) and math.isfinite(hbond_score)):
                raise ValueError("non-finite fitness")
            individual.energy = energy.energy_hartree
            individual.hbond = hbond_score
            individual.hbond_count = hbond_count
            pi_result: tuple[float, int, int, dict[str, object]] | None = None
            if (
                "hbond_pi" in self.objective_names
                or "disconnected_components_penalty" in self.objective_names
            ):
                if self.hbond_pi_model is None:
                    raise RuntimeError("H-bond/pi interaction model is not configured")
                pi_result = self.hbond_pi_model.evaluate(molecule)

            if "hbond_pi" in self.objective_names:
                if pi_result is None:
                    raise RuntimeError("hbond_pi objective is not configured")
                pi_score, possible_count, favorable_count, pi_details = pi_result
                individual.objective_values["hbond_pi"] = OBJECTIVES[
                    "hbond_pi"
                ].fitness_value(pi_score)
                individual.properties["hbond_pi_score"] = pi_score
                individual.properties["hbond_pi_possible_count"] = possible_count
                individual.properties["hbond_pi_favorable_count"] = favorable_count
                individual.properties["hbond_pi_details"] = pi_details

            if "hbond_=" in self.objective_names:
                if self.hbond_double_model is None:
                    raise RuntimeError("hbond_= objective is not configured")
                double_score, possible_count, favorable_count, double_details = (
                    self.hbond_double_model.evaluate(molecule)
                )
                individual.objective_values["hbond_="] = OBJECTIVES[
                    "hbond_="
                ].fitness_value(double_score)
                individual.properties["hbond_=_score"] = double_score
                individual.properties["hbond_=_possible_count"] = possible_count
                individual.properties["hbond_=_favorable_count"] = favorable_count
                individual.properties["hbond_=_details"] = double_details

            if "disconnected_components_penalty" in self.objective_names:
                if self.disconnected_components_model is None:
                    raise RuntimeError("disconnected_components_penalty objective is not configured")
                pi_details = pi_result[3] if pi_result is not None else {}
                network_penalty, component_count, detached_count, network_details = (
                    self.disconnected_components_model.evaluate_interactions(
                        hbond_details.get("active_contacts", []),
                        pi_details,
                    )
                )
                individual.objective_values["disconnected_components_penalty"] = OBJECTIVES[
                    "disconnected_components_penalty"
                ].fitness_value(network_penalty)
                individual.properties["disconnected_components_penalty"] = network_penalty
                individual.properties[
                    "disconnected_components_component_count"
                ] = component_count
                individual.properties[
                    "disconnected_components_detached_fragment_count"
                ] = detached_count
                individual.properties["disconnected_components_details"] = network_details

            if self.rotor_objectives:
                constants = rotational_constants_mhz(molecule)
                scores = rotor_shape_scores(
                    constants,
                    self.config.rotor_symmetry_sigma,
                    self.config.rotor_anisotropy_sigma,
                )
                individual.properties.update(constants)
                individual.properties.update(scores)
                rotational_values = {
                    "rotational_a": float(constants["A_mhz"]),
                    "rotational_b": float(constants["B_mhz"]),
                    "rotational_c": float(constants["C_mhz"]),
                }
                for name in self.rotor_objectives:
                    value = rotational_values[name] if name in rotational_values else scores[name]
                    individual.objective_values[name] = OBJECTIVES[name].fitness_value(value)

            finite_objectives = [
                individual.objective_value(name)
                for name in self.objective_names
            ]
            if not all(math.isfinite(value) for value in finite_objectives):
                raise ValueError("non-finite optional objective")
            individual.valid = True
            individual.error = ""
        except Exception as exc:
            individual.energy = float("inf")
            individual.hbond = float("inf")
            individual.hbond_count = 0
            individual.objective_values = {
                name: float("inf") for name in self.objective_names if name not in {"energy", "hbond"}
            }
            individual.properties = {}
            individual.valid = False
            individual.error = str(exc)[:1000]
        return individual

    def _evaluate(self, population: Sequence[Individual]) -> list[Individual]:
        self.total_evaluations += len(population)
        prepared = None
        if self._allele_structure_batch_provider is not None:
            prepared = tuple(
                self._allele_structure_batch_provider(
                    tuple(individual.alleles for individual in population)
                )
            )
            if len(prepared) != len(population):
                raise ValueError("batch structure provider returned the wrong population size")
        if self.config.workers == 1:
            evaluated = [
                self._evaluate_one(
                    individual,
                    None if prepared is None else prepared[index],
                )
                for index, individual in enumerate(population)
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                if prepared is None:
                    evaluated = list(executor.map(self._evaluate_one, population))
                else:
                    evaluated = list(
                        executor.map(self._evaluate_one, population, prepared)
                    )
        for individual in evaluated:
            if individual.error.startswith("[geometry]"):
                self.geometric_rejections += 1
                self.energy_backend_calls_saved += 1
            else:
                self.energy_backend_calls += 1
            if individual.valid:
                key = self._genotype_key(individual.alleles)
                if key not in self.archive:
                    if not any(
                        self._genotypes_similar(
                            individual.alleles,
                            archived.alleles,
                            self.config.cluster_mean_threshold_deg,
                            self.config.cluster_max_threshold_deg,
                        )
                        for archived in self.archive.values()
                    ):
                        self._new_clusters_since_stats += 1
                    self.archive[key] = individual
                    self._archive_added_since_stats += 1
        return evaluated

    def _crossover(self, first: Individual, second: Individual) -> list[float]:
        if self.config.crossover_operator == "uniform_gene":
            return uniform_gene_crossover(first.alleles, second.alleles, self.genes, self.rng)
        return mixed_sbx_genotype(
            first.alleles,
            second.alleles,
            self.genes,
            self.rng,
            self.config.sbx_eta,
        )

    def _operator_probabilities(self, generation: int) -> tuple[float, float]:
        if self.config.operator_schedule:
            return self.config.operator_schedule[generation]
        return oscillating_operator_probabilities(
            generation=generation,
            generations=self.config.generations,
            base_mutation_weight=self.config.base_mutation_weight,
            base_crossover_weight=self.config.base_crossover_weight,
            mutation_amplitude=self.config.mutation_weight_amplitude,
            crossover_amplitude=self.config.crossover_weight_amplitude,
            oscillations=self.config.operator_oscillations,
        )

    def _checkpoint_operator_configuration(self) -> dict[str, Any]:
        """Embed resolved treatment settings so resume/audit needs no schedule file."""
        return {
            "operator_schedule": [list(pair) for pair in self.config.operator_schedule],
            "mutation_operator": self.config.mutation_operator,
            "crossover_operator": self.config.crossover_operator,
            "mutation_sigma_deg": self.config.mutation_sigma_deg,
        }

    def _is_duplicate(
        self,
        alleles: Sequence[float],
        population: Sequence[Individual],
        include_archive: bool = True,
    ) -> bool:
        self.duplicate_attempts += 1
        key = self._genotype_key(alleles)
        exact = any(
            key == self._genotype_key(individual.alleles)
            for individual in population
        )
        if include_archive:
            exact = exact or key in self.archive
        similar = exact or any(
            self._genotypes_similar(
                alleles,
                individual.alleles,
                self.config.resolved_duplicate_mean_threshold_deg,
                self.config.resolved_duplicate_max_threshold_deg,
            )
            for individual in population
        )
        if similar:
            self.duplicate_rejections += 1
        return similar

    def _offspring(
        self,
        parents: Sequence[Individual],
        operator_probabilities: tuple[float, float],
        generation: int = 0,
        include_archive_duplicates: bool = True,
    ) -> list[Individual]:
        children: list[Individual] = []
        comparison_pool = list(parents)
        mutation_probability, crossover_probability = operator_probabilities
        while len(children) < self.config.offspring_size:
            first = tournament(parents, self.rng)
            second = tournament(parents, self.rng)
            if len(parents) > 1:
                for _ in range(5):
                    if second.id != first.id:
                        break
                    second = tournament(parents, self.rng)
                if second.id == first.id:
                    second = next(parent for parent in parents if parent.id != first.id)
            operator = choose_operator(
                mutation_probability,
                crossover_probability,
                self.rng,
            )
            if operator == "crossover":
                alleles = self._crossover(first, second)
            else:
                alleles = self._mutate(first.alleles)

            duplicate_attempts = 0
            while (
                self._is_duplicate(
                    alleles,
                    comparison_pool + children,
                    include_archive=include_archive_duplicates,
                )
                and duplicate_attempts < self.config.max_duplicate_attempts
            ):
                duplicate_attempts += 1
                alleles = self._mutate(alleles)
                operator = operator + "+dedup" if "+dedup" not in operator else operator
            if self._is_duplicate(
                alleles,
                comparison_pool + children,
                include_archive=include_archive_duplicates,
            ):
                for _ in range(100):
                    alleles = self._random_alleles()
                    if not self._is_duplicate(
                        alleles,
                        comparison_pool + children,
                        include_archive=include_archive_duplicates,
                    ):
                        operator = "random_injection"
                        break
                else:
                    raise RuntimeError("could not generate a non-duplicate child")

            children.append(
                self._new_individual(alleles, (first.id, second.id), operator, generation)
            )
        return children

    def _stats(
        self,
        generation: int,
        population: Sequence[Individual],
        operator_counts: dict[str, int],
        operator_probabilities: tuple[float, float],
        stagnant_generations: int = 0,
        stop_reason: str = "",
        archive_stagnant_generations: int = 0,
        migrations: int = 0,
        migration_survivors: int = 0,
        island_diversities_deg: Sequence[float] = (),
    ) -> GenerationStats:
        assign_rank_and_crowding(population, self.objective_names)
        valid = [individual for individual in population if individual.valid]
        cache = self.backend.cache if isinstance(self.backend, CachedEnergyBackend) else None
        objective_minima = self._objective_minima(valid)
        new_clusters = self._new_clusters_since_stats
        self._new_clusters_since_stats = 0
        new_unique = self._archive_added_since_stats
        self._archive_added_since_stats = 0
        return GenerationStats(
            generation=generation,
            evaluations=self.total_evaluations,
            valid=len(valid),
            pareto_size=sum(1 for individual in valid if individual.rank == 0),
            min_energy_hartree=min((individual.energy for individual in valid), default=float("inf")),
            min_hbond_score=min((individual.hbond for individual in valid), default=float("inf")),
            diversity_deg=self._population_diversity(valid),
            mutation_probability=operator_probabilities[0],
            crossover_probability=operator_probabilities[1],
            cache_hits=cache.hits if cache else 0,
            cache_misses=cache.misses if cache else self.total_evaluations,
            operator_counts=operator_counts,
            objective_minima=objective_minima,
            stagnant_generations=stagnant_generations,
            stop_reason=stop_reason,
            new_unique_individuals=new_unique,
            archive_size=len(self.archive),
            new_clusters=new_clusters,
            duplicate_rejection_rate=(
                self.duplicate_rejections / self.duplicate_attempts
                if self.duplicate_attempts
                else 0.0
            ),
            geometric_rejection_rate=(
                self.geometric_rejections / self.geometry_screen_checks
                if self.geometry_screen_checks
                else 0.0
            ),
            energy_backend_calls=cache.misses if cache else self.energy_backend_calls,
            energy_backend_calls_saved=self.energy_backend_calls_saved,
            archive_stagnant_generations=archive_stagnant_generations,
            geometry_screen_checks=self.geometry_screen_checks,
            migrations=migrations,
            migration_survivors=migration_survivors,
            island_diversities_deg=tuple(float(value) for value in island_diversities_deg),
        )

    def _objective_minima(self, population: Sequence[Individual]) -> dict[str, float]:
        return {
            name: min(
                (
                    individual.objective_value(name)
                    for individual in population
                    if individual.valid and math.isfinite(individual.objective_value(name))
                ),
                default=float("inf"),
            )
            for name in self.objective_names
        }

    def _update_best_objectives(
        self,
        current: dict[str, float],
        best: dict[str, float],
    ) -> bool:
        improved = False
        for name, value in current.items():
            previous = best.get(name, float("inf"))
            if value < previous - self.config.early_stop_min_delta:
                improved = True
            if value < previous:
                best[name] = value
        return improved

    def _migration_candidates(
        self, population: Sequence[Individual], island_index: int
    ) -> list[Individual]:
        valid = [individual for individual in population if individual.valid]
        if self.config.migration_selection == "random":
            ordered = list(valid)
            self.island_rngs[island_index].shuffle(ordered)
            return ordered
        assign_rank_and_crowding(valid, self.objective_names)
        return sorted(
            valid,
            key=lambda individual: (
                individual.rank,
                -individual.crowding,
                individual.id,
            ),
        )

    def _migrate_islands(
        self,
        islands: Sequence[Sequence[Individual]],
        generation: int,
    ) -> tuple[list[list[Individual]], int, int]:
        """Perform simultaneous ring migration followed by local NSGA-II selection."""

        if (
            self.config.islands < 2
            or self.config.migration_interval <= 0
            or generation % self.config.migration_interval != 0
        ):
            return [list(population) for population in islands], 0, 0

        incoming: list[list[tuple[Individual, dict[str, Any]]]] = [
            [] for _ in islands
        ]
        for source_index, source_population in enumerate(islands):
            destination_index = (source_index + 1) % len(islands)
            destination_keys = {
                self._genotype_key(individual.alleles)
                for individual in islands[destination_index]
            }
            chosen = 0
            for source in self._migration_candidates(source_population, source_index):
                key = self._genotype_key(source.alleles)
                if key in destination_keys:
                    continue
                migrant = Individual.from_dict(source.to_dict())
                migrant.id = self.next_individual_id
                self.next_individual_id += 1
                migrant.parents = (source.id,)
                migrant.operator = f"migration:ring:{source_index}->{destination_index}"
                migrant.generation = generation
                migrant.island = destination_index
                event = {
                    "generation": generation,
                    "source_island": source_index,
                    "destination_island": destination_index,
                    "source_individual_id": source.id,
                    "migrant_individual_id": migrant.id,
                    "survived": False,
                    "alleles_deg": [float(value) for value in migrant.alleles],
                }
                incoming[destination_index].append((migrant, event))
                destination_keys.add(key)
                chosen += 1
                if chosen >= self.config.migration_size:
                    break

        migrated: list[list[Individual]] = []
        survivors = 0
        for destination_index, residents in enumerate(islands):
            arrivals = incoming[destination_index]
            combined = [*residents, *(migrant for migrant, _event in arrivals)]
            selected = environmental_selection(
                combined,
                self.config.population_size,
                self.objective_names,
            )
            survivor_ids = {individual.id for individual in selected}
            for migrant, event in arrivals:
                event["survived"] = migrant.id in survivor_ids
                survivors += int(event["survived"])
                self.migration_events.append(event)
            migrated.append(selected)
        return migrated, sum(len(items) for items in incoming), survivors

    def run(self, checkpoint: dict[str, Any] | None = None) -> list[Individual]:
        if self.config.islands > 1:
            return self._run_islands(checkpoint)
        return self._run_single(checkpoint)

    def _run_single(self, checkpoint: dict[str, Any] | None = None) -> list[Individual]:
        if checkpoint is None:
            generation = 0
            population = self._evaluate(self._initial_population())
            if not any(individual.valid for individual in population):
                errors = sorted({individual.error for individual in population if individual.error})
                raise RuntimeError("no valid initial individual: " + " | ".join(errors[:3]))
            population = environmental_selection(
                population,
                self.config.population_size,
                self.objective_names,
            )
            best_objectives = self._objective_minima(population)
            stagnant_generations = 0
            archive_stagnant_generations = 0
            history = [
                self._stats(
                    0,
                    population,
                    {"initial": len(population)},
                    self._operator_probabilities(0),
                )
            ]
        else:
            if checkpoint["run_fingerprint"] != self.fingerprint:
                raise ValueError("checkpoint is incompatible with the current input, configuration, or backend")
            generation = int(checkpoint["generation"])
            population = list(checkpoint["population"])
            archived_items = checkpoint.get("archive") or population
            self.archive = {
                self._genotype_key(individual.alleles): individual
                for individual in archived_items
                if individual.valid
            }
            history = list(checkpoint["history"])
            self.rng.setstate(checkpoint["rng_state"])
            self.next_individual_id = int(checkpoint["next_individual_id"])
            self.total_evaluations = history[-1].evaluations if history else 0
            if isinstance(self.backend, CachedEnergyBackend) and history:
                self.backend.cache.hits = history[-1].cache_hits
                self.backend.cache.misses = history[-1].cache_misses
            assign_rank_and_crowding(population, self.objective_names)
            best_objectives: dict[str, float] = {}
            for stats in history:
                for name, value in stats.objective_minima.items():
                    if value < best_objectives.get(name, float("inf")):
                        best_objectives[name] = value
            stagnant_generations = history[-1].stagnant_generations if history else 0
            archive_stagnant_generations = (
                history[-1].archive_stagnant_generations if history else 0
            )
            self.energy_backend_calls = history[-1].energy_backend_calls if history else 0
            self.energy_backend_calls_saved = (
                history[-1].energy_backend_calls_saved if history else 0
            )
            self.geometry_screen_checks = history[-1].geometry_screen_checks if history else 0

        save_checkpoint(
            self.output_dir,
            generation,
            population,
            history,
            self.rng,
            self.next_individual_id,
            self.fingerprint,
            list(self.archive.values()),
            operator_configuration=self._checkpoint_operator_configuration(),
        )
        write_history(self.output_dir, history)
        if history:
            self._report_progress(history[-1], [population])

        end_generation = (
            generation
            if history and history[-1].stop_reason
            else self.config.generations
        )
        for current_generation in range(generation + 1, end_generation + 1):
            probabilities = self._operator_probabilities(current_generation - 1)
            offspring = self._evaluate(
                self._offspring(population, probabilities, current_generation)
            )
            operator_counts = dict(Counter(child.operator for child in offspring))
            combined = [*population, *offspring]
            population = environmental_selection(
                combined,
                self.config.population_size,
                self.objective_names,
            )
            current_minima = self._objective_minima(population)
            improved = self._update_best_objectives(current_minima, best_objectives)
            diversity = self._population_diversity(
                [individual for individual in population if individual.valid]
            )
            if improved:
                stagnant_generations = 0
            elif diversity <= self.config.early_stop_diversity_deg:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
            should_stop = (
                self.config.early_stopping
                and current_generation >= self.config.early_stop_min_generations
                and stagnant_generations >= self.config.early_stop_patience
            )
            if self._archive_added_since_stats:
                archive_stagnant_generations = 0
            else:
                archive_stagnant_generations += 1
            archive_should_stop = (
                self.config.archive_stagnation_patience > 0
                and current_generation >= self.config.early_stop_min_generations
                and archive_stagnant_generations >= self.config.archive_stagnation_patience
            )
            stop_reason = ""
            if should_stop:
                stop_reason = (
                    f"stagnation for {stagnant_generations} generations with diversity "
                    f"{diversity:.6f}° <= {self.config.early_stop_diversity_deg:.6f}°"
                )
            elif archive_should_stop:
                stop_reason = (
                    f"archive without new unique individuals for "
                    f"{archive_stagnant_generations} generations"
                )
            history.append(
                self._stats(
                    current_generation,
                    population,
                    operator_counts,
                    probabilities,
                    stagnant_generations,
                    stop_reason,
                    archive_stagnant_generations,
                )
            )
            write_history(self.output_dir, history)
            if current_generation % self.config.checkpoint_every == 0:
                save_checkpoint(
                    self.output_dir,
                    current_generation,
                    population,
                    history,
                    self.rng,
                    self.next_individual_id,
                    self.fingerprint,
                    list(self.archive.values()),
                    operator_configuration=self._checkpoint_operator_configuration(),
                )
            latest = history[-1]
            if not self._report_progress(latest, [population]):
                print(
                    f"generation={current_generation} valid={latest.valid}/{len(population)} "
                    f"pareto={latest.pareto_size} Emin={latest.min_energy_hartree:.10f} "
                    f"HBmin={latest.min_hbond_score:.6f} diversity={latest.diversity_deg:.2f}° "
                    f"Pmut={latest.mutation_probability:.4f} "
                    f"Pcross={latest.crossover_probability:.4f}",
                    flush=True,
                )
            generation = current_generation
            if should_stop or archive_should_stop:
                print(f"early-stop: {stop_reason}", flush=True)
                break

        save_checkpoint(
            self.output_dir,
            generation,
            population,
            history,
            self.rng,
            self.next_individual_id,
            self.fingerprint,
            list(self.archive.values()),
            operator_configuration=self._checkpoint_operator_configuration(),
        )

        archive = list(self.archive.values())
        structures = {
            individual.id: self.structure_for(individual)
            for individual in archive
            if individual.valid
        }
        write_final_results(self.output_dir, population, structures, history, archive)
        if self.auto_analyze:
            from .analysis import analyze_run

            analyze_run(
                self.output_dir,
                self.config.cluster_energy_window_kcal_mol,
                self.config.cluster_mean_threshold_deg,
                self.config.cluster_max_threshold_deg,
                self.config.clustering_source,
                self.config.clustering_method,
            )
        return population

    def _run_islands(
        self, checkpoint: dict[str, Any] | None = None
    ) -> list[Individual]:
        if checkpoint is None:
            generation = 0
            islands: list[list[Individual]] = []
            for island_index in range(self.config.islands):
                self.current_island = island_index
                self.rng = self.island_rngs[island_index]
                population = self._evaluate(self._initial_population())
                if not any(individual.valid for individual in population):
                    errors = sorted(
                        {individual.error for individual in population if individual.error}
                    )
                    raise RuntimeError(
                        f"no valid initial individual on island {island_index}: "
                        + " | ".join(errors[:3])
                    )
                islands.append(
                    environmental_selection(
                        population,
                        self.config.population_size,
                        self.objective_names,
                    )
                )
            flattened = [individual for island in islands for individual in island]
            best_objectives = self._objective_minima(flattened)
            stagnant_generations = 0
            archive_stagnant_generations = 0
            stats_population = [
                Individual.from_dict(individual.to_dict()) for individual in flattened
            ]
            history = [
                self._stats(
                    0,
                    stats_population,
                    {"initial": len(flattened)},
                    self._operator_probabilities(0),
                    island_diversities_deg=[
                        self._population_diversity(
                            [individual for individual in island if individual.valid]
                        )
                        for island in islands
                    ],
                )
            ]
        else:
            if checkpoint["run_fingerprint"] != self.fingerprint:
                raise ValueError(
                    "checkpoint is incompatible with the current input, configuration, or backend"
                )
            raw_islands = checkpoint.get("islands")
            raw_rng_states = checkpoint.get("island_rng_states")
            if raw_islands is None or raw_rng_states is None:
                raise ValueError("checkpoint does not contain island-model state")
            if len(raw_islands) != self.config.islands or len(raw_rng_states) != self.config.islands:
                raise ValueError("checkpoint contains an inconsistent number of islands")
            generation = int(checkpoint["generation"])
            islands = [list(population) for population in raw_islands]
            for island_rng, state in zip(self.island_rngs, raw_rng_states):
                island_rng.setstate(state)
            self.rng = self.island_rngs[0]
            archived_items = checkpoint.get("archive") or checkpoint["population"]
            self.archive = {
                self._genotype_key(individual.alleles): individual
                for individual in archived_items
                if individual.valid
            }
            history = list(checkpoint["history"])
            self.migration_events = list(checkpoint.get("migration_events", []))
            self.next_individual_id = int(checkpoint["next_individual_id"])
            self.total_evaluations = history[-1].evaluations if history else 0
            if isinstance(self.backend, CachedEnergyBackend) and history:
                self.backend.cache.hits = history[-1].cache_hits
                self.backend.cache.misses = history[-1].cache_misses
            for island in islands:
                assign_rank_and_crowding(island, self.objective_names)
            best_objectives: dict[str, float] = {}
            for stats in history:
                for name, value in stats.objective_minima.items():
                    if value < best_objectives.get(name, float("inf")):
                        best_objectives[name] = value
            stagnant_generations = history[-1].stagnant_generations if history else 0
            archive_stagnant_generations = (
                history[-1].archive_stagnant_generations if history else 0
            )
            self.energy_backend_calls = history[-1].energy_backend_calls if history else 0
            self.energy_backend_calls_saved = (
                history[-1].energy_backend_calls_saved if history else 0
            )
            self.geometry_screen_checks = history[-1].geometry_screen_checks if history else 0

        def save_state(current_generation: int) -> None:
            flattened_population = [
                individual for island in islands for individual in island
            ]
            save_checkpoint(
                self.output_dir,
                current_generation,
                flattened_population,
                history,
                self.island_rngs[0],
                self.next_individual_id,
                self.fingerprint,
                list(self.archive.values()),
                islands=islands,
                island_rngs=self.island_rngs,
                migration_events=self.migration_events,
                operator_configuration=self._checkpoint_operator_configuration(),
            )

        save_state(generation)
        write_history(self.output_dir, history)
        write_migration_events(self.output_dir, self.migration_events)
        if history:
            initial_events = [
                event
                for event in self.migration_events
                if int(event.get("generation", -1)) == history[-1].generation
            ]
            self._report_progress(history[-1], islands, initial_events)

        end_generation = (
            generation
            if history and history[-1].stop_reason
            else self.config.generations
        )
        for current_generation in range(generation + 1, end_generation + 1):
            probabilities = self._operator_probabilities(current_generation - 1)
            operator_counter: Counter[str] = Counter()
            evolved: list[list[Individual]] = []
            for island_index, population in enumerate(islands):
                self.current_island = island_index
                self.rng = self.island_rngs[island_index]
                offspring = self._evaluate(
                    self._offspring(
                        population,
                        probabilities,
                        current_generation,
                        include_archive_duplicates=False,
                    )
                )
                operator_counter.update(child.operator for child in offspring)
                combined = [*population, *offspring]
                evolved.append(
                    environmental_selection(
                        combined,
                        self.config.population_size,
                        self.objective_names,
                    )
                )
            islands, migrations, migration_survivors = self._migrate_islands(
                evolved, current_generation
            )
            flattened = [individual for island in islands for individual in island]
            current_minima = self._objective_minima(flattened)
            improved = self._update_best_objectives(current_minima, best_objectives)
            island_diversities = [
                self._population_diversity(
                    [individual for individual in island if individual.valid]
                )
                for island in islands
            ]
            diversity = self._population_diversity(
                [individual for individual in flattened if individual.valid]
            )
            if improved:
                stagnant_generations = 0
            elif diversity <= self.config.early_stop_diversity_deg:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
            should_stop = (
                self.config.early_stopping
                and current_generation >= self.config.early_stop_min_generations
                and stagnant_generations >= self.config.early_stop_patience
            )
            if self._archive_added_since_stats:
                archive_stagnant_generations = 0
            else:
                archive_stagnant_generations += 1
            archive_should_stop = (
                self.config.archive_stagnation_patience > 0
                and current_generation >= self.config.early_stop_min_generations
                and archive_stagnant_generations
                >= self.config.archive_stagnation_patience
            )
            stop_reason = ""
            if should_stop:
                stop_reason = (
                    f"stagnation for {stagnant_generations} generations with diversity "
                    f"{diversity:.6f}° <= {self.config.early_stop_diversity_deg:.6f}°"
                )
            elif archive_should_stop:
                stop_reason = (
                    "archive without new unique individuals for "
                    f"{archive_stagnant_generations} generations"
                )
            stats_population = [
                Individual.from_dict(individual.to_dict()) for individual in flattened
            ]
            history.append(
                self._stats(
                    current_generation,
                    stats_population,
                    dict(operator_counter),
                    probabilities,
                    stagnant_generations,
                    stop_reason,
                    archive_stagnant_generations,
                    migrations,
                    migration_survivors,
                    island_diversities,
                )
            )
            write_history(self.output_dir, history)
            write_migration_events(self.output_dir, self.migration_events)
            if current_generation % self.config.checkpoint_every == 0:
                save_state(current_generation)
            latest = history[-1]
            generation_events = [
                event
                for event in self.migration_events
                if int(event.get("generation", -1)) == current_generation
            ]
            if not self._report_progress(latest, islands, generation_events):
                print(
                    f"generation={current_generation} valid={latest.valid}/{len(flattened)} "
                    f"pareto={latest.pareto_size} Emin={latest.min_energy_hartree:.10f} "
                    f"HBmin={latest.min_hbond_score:.6f} diversity={latest.diversity_deg:.2f}° "
                    f"Pmut={latest.mutation_probability:.4f} "
                    f"Pcross={latest.crossover_probability:.4f} "
                    f"migrants={migration_survivors}/{migrations}",
                    flush=True,
                )
            generation = current_generation
            if should_stop or archive_should_stop:
                print(f"early-stop: {stop_reason}", flush=True)
                break

        save_state(generation)
        write_migration_events(self.output_dir, self.migration_events)
        population = [individual for island in islands for individual in island]
        assign_rank_and_crowding(population, self.objective_names)
        archive = list(self.archive.values())
        structures = {
            individual.id: self.structure_for(individual)
            for individual in [*archive, *population]
            if individual.valid
        }
        write_final_results(self.output_dir, population, structures, history, archive)
        if self.auto_analyze:
            from .analysis import analyze_run

            analyze_run(
                self.output_dir,
                self.config.cluster_energy_window_kcal_mol,
                self.config.cluster_mean_threshold_deg,
                self.config.cluster_max_threshold_deg,
                self.config.clustering_source,
                self.config.clustering_method,
            )
        return population
