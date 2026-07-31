"""Compact, restartable run output."""

from __future__ import annotations

import ast
import csv
import json
import os
import platform
import random
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fragment_pose import FragmentPoseBlock
from .input import read_genes, write_xyz
from .geometry import genotype_key
from .models import (
    GenerationStats,
    Gene,
    Individual,
    Molecule,
    NativePoseCoordinate,
    RunConfig,
)
from .objectives import display_objective_value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_manifest(
    output_dir: str | Path,
    config: RunConfig,
    molecule_fingerprint: str,
    backend_signature: dict[str, Any],
    xyz_path: str | Path,
    genes_path: str | Path,
    descriptor_signature: dict[str, Any] | None = None,
    coordinates: Sequence[Gene | NativePoseCoordinate] | None = None,
    hbond_pi_metadata: Mapping[str, Any] | None = None,
    hbond_double_metadata: Mapping[str, Any] | None = None,
    disconnected_components_metadata: Mapping[str, Any] | None = None,
    fragment_pose_blocks: Sequence[FragmentPoseBlock] = (),
) -> None:
    root = Path(output_dir)
    payload = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "molecule_fingerprint": molecule_fingerprint,
        "backend": backend_signature,
        "descriptors": descriptor_signature or {},
        "config": asdict(config),
        "input": {"xyz": str(Path(xyz_path).resolve()), "genes": str(Path(genes_path).resolve())},
        "genes": [
            {
                "name": gene.name,
                "atoms": [index + 1 for index in gene.atoms],
                "periodicity": gene.periodicity,
                "periodic": gene.periodic,
                "kind": "native_pose" if isinstance(gene, NativePoseCoordinate) else "torsion",
                **(
                    {
                        "pose_name": gene.pose_name,
                        "component": gene.component,
                        "physical_bounds": [gene.lower, gene.upper],
                        "reference_value": gene.reference_value,
                        "reference_atoms": [index + 1 for index in gene.reference_atoms],
                        "moving_atoms": [index + 1 for index in gene.moving_atoms],
                        "units": gene.units,
                        "scan_points": gene.scan_points,
                    }
                    if isinstance(gene, NativePoseCoordinate) else {}
                ),
            }
            for gene in (coordinates if coordinates is not None else read_genes(genes_path))
        ],
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "objectives": list(config.objectives),
        "hbond_pi": dict(hbond_pi_metadata or {}),
        "hbond_=": dict(hbond_double_metadata or {}),
        "disconnected_components_penalty": dict(disconnected_components_metadata or {}),
        "reported_descriptors": (
            "Le colonne property_* sono riportate ma non partecipano alla dominanza NSGA-II."
        ),
        "constraints": [
            "rigid_torsions",
            "native_rigid_fragment_pose" if coordinates and any(
                isinstance(gene, NativePoseCoordinate) for gene in coordinates
            ) else "reference_topology",
            "geometric_prescreen",
        ],
    }
    if fragment_pose_blocks:
        native_pose_coordinates = [
            gene
            for gene in (coordinates or ())
            if isinstance(gene, NativePoseCoordinate)
        ]
        payload["active_variables"] = [
            {
                "name": gene.name,
                "lower": gene.lower,
                "upper": gene.upper,
                "periodic": gene.periodic,
                "periodicity": gene.periodicity,
                "units": gene.units,
                "scan_points": gene.scan_points,
            }
            for gene in native_pose_coordinates
        ]
        payload["fragment_pose_blocks"] = [
            asdict(block) for block in fragment_pose_blocks
        ]
    _atomic_json(root / "run_manifest.json", payload)


def _write_history_csv(path: Path, history: Sequence[GenerationStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "generation",
        "evaluations",
        "valid",
        "pareto_size",
        "min_energy_hartree",
        "min_hbond_score",
        "diversity_deg",
        "mutation_probability",
        "crossover_probability",
        "cache_hits",
        "cache_misses",
        "operator_counts",
        "objective_minima",
        "stagnant_generations",
        "stop_reason",
        "new_unique_individuals",
        "archive_size",
        "new_clusters",
        "duplicate_rejection_rate",
        "geometric_rejection_rate",
        "energy_backend_calls",
        "energy_backend_calls_saved",
        "archive_stagnant_generations",
        "geometry_screen_checks",
        "migrations",
        "migration_survivors",
        "island_diversities_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stats in history:
            row = stats.to_dict()
            row["operator_counts"] = json.dumps(row["operator_counts"], sort_keys=True)
            row["objective_minima"] = json.dumps(row["objective_minima"], sort_keys=True)
            row["island_diversities_deg"] = json.dumps(row["island_diversities_deg"])
            writer.writerow(row)


def write_history(output_dir: str | Path, history: Sequence[GenerationStats]) -> None:
    """Write the complete per-generation GA report.

    ``genetic_evolution.csv`` is the descriptive public filename.  The
    historical ``history.csv`` is kept byte-for-byte equivalent for restart
    compatibility and existing analysis scripts.
    """

    root = Path(output_dir)
    _write_history_csv(root / "genetic_evolution.csv", history)
    _write_history_csv(root / "history.csv", history)


def save_checkpoint(
    output_dir: str | Path,
    generation: int,
    population: Sequence[Individual],
    history: Sequence[GenerationStats],
    rng: random.Random,
    next_individual_id: int,
    run_fingerprint: str,
    archive: Sequence[Individual] = (),
    islands: Sequence[Sequence[Individual]] | None = None,
    island_rngs: Sequence[random.Random] | None = None,
    migration_events: Sequence[dict[str, Any]] = (),
    operator_configuration: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": 1,
        "generation": generation,
        "population": [individual.to_dict() for individual in population],
        "archive": [individual.to_dict() for individual in archive],
        "history": [stats.to_dict() for stats in history],
        "rng_state": repr(rng.getstate()),
        "next_individual_id": next_individual_id,
        "run_fingerprint": run_fingerprint,
        "islands": (
            [[individual.to_dict() for individual in island] for island in islands]
            if islands is not None
            else None
        ),
        "island_rng_states": (
            [repr(rng_item.getstate()) for rng_item in island_rngs]
            if island_rngs is not None
            else None
        ),
        "migration_events": list(migration_events),
        "operator_configuration": operator_configuration,
    }
    _atomic_json(Path(output_dir) / "checkpoint.json", payload)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("versione checkpoint non supportata")
    try:
        payload["rng_state"] = ast.literal_eval(payload["rng_state"])
    except (ValueError, SyntaxError) as exc:
        raise ValueError("stato RNG del checkpoint non valido") from exc
    payload["population"] = [Individual.from_dict(row) for row in payload["population"]]
    payload["archive"] = [Individual.from_dict(row) for row in payload.get("archive", [])]
    raw_islands = payload.get("islands")
    payload["islands"] = (
        [
            [Individual.from_dict(row) for row in island]
            for island in raw_islands
        ]
        if raw_islands is not None
        else None
    )
    raw_island_states = payload.get("island_rng_states")
    if raw_island_states is not None:
        try:
            payload["island_rng_states"] = [
                ast.literal_eval(state) for state in raw_island_states
            ]
        except (ValueError, SyntaxError) as exc:
            raise ValueError("stato RNG delle isole non valido nel checkpoint") from exc
    payload["history"] = [GenerationStats(**row) for row in payload.get("history", [])]
    return payload


def write_migration_events(
    output_dir: str | Path, events: Sequence[dict[str, Any]]
) -> None:
    path = Path(output_dir) / "migration_events.csv"
    fields = [
        "generation",
        "source_island",
        "destination_island",
        "source_individual_id",
        "migrant_individual_id",
        "survived",
        "alleles_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["alleles_deg"] = json.dumps(row["alleles_deg"])
            writer.writerow(row)


def _population_rows(
    population: Sequence[Individual], structure_paths: Mapping[int, str]
) -> list[dict[str, Any]]:
    if not population:
        return []
    finite_energies = [item.energy for item in population if item.valid]
    minimum_energy = min(finite_energies) if finite_energies else float("inf")
    rows: list[dict[str, Any]] = []
    objective_names = sorted(
        {name for item in population for name in item.objective_values}
    )
    property_names = sorted(
        {name for item in population for name in item.properties}
    )
    for item in sorted(population, key=lambda candidate: (candidate.rank, candidate.energy, candidate.hbond, candidate.id)):
        delta = (item.energy - minimum_energy) * 627.5094740631 if item.valid else float("inf")
        row = {
                "id": item.id,
                "valid": item.valid,
                "rank": item.rank,
                "crowding": item.crowding,
                "energy_hartree": item.energy,
                "delta_energy_kcal_mol": delta,
                "hbond_score": item.hbond,
                "hbond_count": item.hbond_count,
                "alleles_deg": json.dumps([round(value, 8) for value in item.alleles]),
                "parents": json.dumps(list(item.parents)),
                "operator": item.operator,
                "generation": item.generation,
                "island": item.island,
                "origin_island": item.origin_island,
                "genotype_key": json.dumps(genotype_key(item.alleles)),
                "fitness_values_minimized": json.dumps(item.objective_values, sort_keys=True),
                "properties": json.dumps(item.properties, sort_keys=True),
                "structure_file": structure_paths.get(item.id, ""),
                "error": item.error,
            }
        for name in objective_names:
            value = item.objective_values.get(name)
            row[f"objective_{name}"] = (
                display_objective_value(name, value) if value is not None else ""
            )
        for name in property_names:
            row[f"property_{name}"] = item.properties.get(name, "")
        rows.append(row)
    return rows


def _write_population_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "id",
        "valid",
        "rank",
        "crowding",
        "energy_hartree",
        "delta_energy_kcal_mol",
        "hbond_score",
        "hbond_count",
        "alleles_deg",
        "parents",
        "operator",
        "generation",
        "island",
        "origin_island",
        "genotype_key",
        "fitness_values_minimized",
        "properties",
        "structure_file",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_final_results(
    output_dir: str | Path,
    population: Sequence[Individual],
    structures: Mapping[int, Molecule],
    history: Sequence[GenerationStats] = (),
    archive: Sequence[Individual] = (),
) -> None:
    root = Path(output_dir)
    population_dir = root / "final_population"
    pareto_dir = root / "pareto_front"
    archive_dir = root / "evaluated_archive"
    for directory in (population_dir, pareto_dir, archive_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    structure_paths: dict[int, str] = {}
    pareto_paths: dict[int, str] = {}
    archive_paths: dict[int, str] = {}
    for individual in archive:
        molecule = structures.get(individual.id)
        if molecule is None or not individual.valid:
            continue
        filename = f"candidate_{individual.id:06d}.xyz"
        relative = str(Path("evaluated_archive") / filename)
        comment = (
            f"id={individual.id} generation={individual.generation} "
            f"energy_hartree={individual.energy:.12f} hbond_score={individual.hbond:.8f} "
            f"alleles_deg=" + ",".join(f"{value:.6f}" for value in individual.alleles)
        )
        write_xyz(root / relative, molecule, comment)
        archive_paths[individual.id] = relative
    for individual in population:
        molecule = structures.get(individual.id)
        if molecule is None or not individual.valid:
            continue
        filename = f"candidate_{individual.id:06d}.xyz"
        relative = str(Path("final_population") / filename)
        comment = (
            f"id={individual.id} energy_hartree={individual.energy:.12f} "
            f"hbond_score={individual.hbond:.8f} hbond_count={individual.hbond_count} "
            f"rank={individual.rank} alleles_deg="
            + ",".join(f"{value:.6f}" for value in individual.alleles)
        )
        write_xyz(root / relative, molecule, comment)
        structure_paths[individual.id] = relative
        if individual.rank == 0:
            pareto_relative = str(Path("pareto_front") / filename)
            shutil.copy2(root / relative, root / pareto_relative)
            pareto_paths[individual.id] = pareto_relative

    all_rows = _population_rows(population, structure_paths)
    pareto = [item for item in population if item.valid and item.rank == 0]
    pareto_rows = _population_rows(pareto, pareto_paths)
    _write_population_csv(root / "final_population.csv", all_rows)
    _write_population_csv(root / "pareto_front.csv", pareto_rows)
    _write_population_csv(
        root / "evaluated_archive.csv",
        _population_rows(archive, archive_paths),
    )

    summary = {
        "population_size": len(population),
        "valid_candidates": sum(1 for item in population if item.valid),
        "pareto_size": len(pareto),
        "best_energy_hartree": min((item.energy for item in population if item.valid), default=None),
        "best_hbond_score": min((item.hbond for item in population if item.valid), default=None),
        "objective_minima": history[-1].objective_minima if history else {},
        "stopped_early": bool(history and history[-1].stop_reason),
        "stop_reason": history[-1].stop_reason if history else "",
        "completed_generation": history[-1].generation if history else 0,
        "archive_size": len(archive),
        "island_count": len({item.island for item in population}),
        "population_per_island": {
            str(island): sum(1 for item in population if item.island == island)
            for island in sorted({item.island for item in population})
        },
        "migration_events": sum(stats.migrations for stats in history),
        "migration_survivors": sum(stats.migration_survivors for stats in history),
    }
    _atomic_json(root / "summary.json", summary)
