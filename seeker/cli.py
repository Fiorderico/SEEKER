"""Command-line interface for SEEKER."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .analysis import analyze_run
from .backends import (
    CachedEnergyBackend,
    EnergyBackend,
    EnergyCache,
    ExternalCommandBackend,
    PyScfBackend,
    XtbBackend,
)
from .engine import GeneticConformerSearch
from .fitness import (
    DisconnectedComponentsPenaltyModel,
    HydrogenDoubleBondModel,
    HydrogenPiModel,
)
from .geometry import build_bond_graph, prepare_genes
from .gaussian import single_point_gaussian_candidates
from .input import molecule_fingerprint, read_coordinate_specs, read_hbond_pi_config, read_xyz
from .models import FragmentPoseGene, Gene, HydrogenPiConfig, RunConfig
from .objectives import active_objectives, validate_objectives
from .output import load_checkpoint, write_manifest
from .postopt import optimize_gaussian_candidates, optimize_xtb_candidates
from .postopt_cluster import cluster_optimized_candidates


DEFAULT_EXTERNAL_REGEX = r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seeker",
        description="Multi-objective genetic conformer search: energy + hydrogen bonds.",
    )
    parser.add_argument("--version", action="version", version="SEEKER 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run or resume a genetic search")
    run.add_argument("--xyz", required=True, help="reference Cartesian geometry")
    run.add_argument(
        "--genes", required=True,
        help=(
            "GENE...=D(i,j,k,l) and POSE...=FRAGMENTS(reference;moving) coordinates"
        ),
    )
    run.add_argument("--output", required=True, help="run directory")
    run.add_argument("--charge", type=int, default=0)
    run.add_argument("--multiplicity", type=int, default=1)
    run.add_argument("--backend", choices=("xtb", "external", "pyscf"), default="xtb")
    run.add_argument("--resume", help="checkpoint.json to resume")
    run.add_argument("--validate-only", action="store_true", help="validate input and torsions without evaluating energies")
    run.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the live ANSI island dashboard when attached to a terminal",
    )

    run.add_argument("--population", type=int, default=32)
    run.add_argument("--offspring", type=int, default=32)
    run.add_argument("--generations", type=int, default=30)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 1) // 2)))
    run.add_argument(
        "--islands",
        type=int,
        default=1,
        help="number of independent populations; --population and --offspring apply per island",
    )
    run.add_argument(
        "--migration-interval",
        type=int,
        default=0,
        help="perform ring migration every N generations (0 disables migration)",
    )
    run.add_argument(
        "--migration-size",
        type=int,
        default=1,
        help="maximum migrants sent by each island",
    )
    run.add_argument(
        "--migration-selection",
        choices=("pareto", "random"),
        default="pareto",
        help="migrant selection: Pareto-diverse elitist or random",
    )
    run.add_argument(
        "--base-mutation-weight",
        "--mutation-probability",
        dest="base_mutation_weight",
        type=float,
        default=0.45,
        help="base mutation weight in exclusive operator selection",
    )
    run.add_argument(
        "--base-crossover-weight",
        "--crossover-probability",
        dest="base_crossover_weight",
        type=float,
        default=0.80,
        help="base crossover weight in exclusive operator selection",
    )
    run.add_argument("--mutation-weight-amplitude", type=float, default=0.10)
    run.add_argument("--crossover-weight-amplitude", type=float, default=0.10)
    run.add_argument("--operator-oscillations", type=int, default=2)
    run.add_argument(
        "--operator-schedule-file",
        help="versioned JSON containing one mutation/crossover probability pair per generation",
    )
    run.add_argument(
        "--mutation-operator", choices=("resample_one", "local_gaussian"), default="resample_one"
    )
    run.add_argument(
        "--crossover-operator", choices=("mixed_sbx", "uniform_gene"), default="mixed_sbx"
    )
    run.add_argument("--mutation-sigma-deg", type=float, default=20.0)
    run.add_argument("--periodicity-grid-step-deg", type=float, default=20.0)
    run.add_argument("--sbx-eta", type=float, default=15.0)
    run.add_argument("--duplicate-threshold-deg", type=float, default=3.0)
    run.add_argument("--duplicate-mean-threshold-deg", type=float)
    run.add_argument("--duplicate-max-threshold-deg", type=float)
    run.add_argument(
        "--initialization-strategy",
        choices=("random", "maximin", "lhs", "scan"),
        default="maximin",
    )
    run.add_argument("--initial-pool-factor", type=int, default=20)
    run.add_argument(
        "--initial-pool-sampling",
        choices=("prior", "latin_hypercube"),
        default="latin_hypercube",
    )
    run.add_argument(
        "--initial-scan-layout",
        choices=("tensor", "one-at-a-time"),
        default="tensor",
    )
    run.add_argument(
        "--initial-scan-grid",
        choices=("uniform", "periodicity-modes"),
        default="uniform",
    )
    run.add_argument(
        "--initial-scan-points-mode",
        choices=("fixed", "periodicity"),
        default="fixed",
    )
    run.add_argument("--initial-scan-points", type=int, default=3)
    run.add_argument("--topology-tolerance", type=float, default=0.45)
    run.add_argument("--hbond-cutoff", type=float, default=3.2)
    run.add_argument("--hbond-contact-threshold", type=float, default=-0.30)
    run.add_argument("--hh-clash-distance", type=float, default=1.40)
    run.add_argument("--geometric-prescreen", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--steric-hh-scale", type=float, default=0.55)
    run.add_argument("--steric-heavy-heavy-scale", type=float, default=0.55)
    run.add_argument("--steric-hydrogen-heavy-scale", type=float, default=0.50)
    run.add_argument("--steric-exclude-hops", type=int, default=3)
    run.add_argument("--checkpoint-every", type=int, default=1)
    run.add_argument("--max-duplicate-attempts", type=int, default=30)
    run.add_argument(
        "--save-generation-xyz",
        action="store_true",
        help="save evaluated and surviving structures for every generation",
    )
    run.add_argument(
        "--extra-objectives",
        default="",
        help=(
            "objectives beyond energy,hbond: "
            "hbond_pi,hbond_=,disconnected_components_penalty,"
            "rotational_a,rotational_b,rotational_c,"
            "rotor_prolate,rotor_oblate,rotor_spherical"
        ),
    )
    run.add_argument("--rotor-symmetry-sigma", type=float, default=0.15)
    run.add_argument("--rotor-anisotropy-sigma", type=float, default=0.15)
    run.add_argument("--early-stop", action="store_true")
    run.add_argument("--early-stop-patience", type=int, default=8)
    run.add_argument("--early-stop-min-delta", type=float, default=1.0e-6)
    run.add_argument("--early-stop-diversity-deg", type=float, default=8.0)
    run.add_argument("--early-stop-min-generations", type=int, default=0)
    run.add_argument("--archive-stagnation-patience", type=int, default=0)
    run.add_argument(
        "--clustering-source",
        choices=("archive", "final_population", "pareto_front"),
        default="archive",
    )
    run.add_argument(
        "--clustering-method",
        choices=(
            "complete_linkage",
            "periodicity_cells",
            "hybrid",
            "mixed_hybrid",
            "pose_hybrid",
        ),
        default="complete_linkage",
    )
    run.add_argument(
        "--auto-analyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run clustering in the same search invocation",
    )
    run.add_argument("--cluster-mean-threshold-deg", type=float, default=15.0)
    run.add_argument("--cluster-max-threshold-deg", type=float, default=15.0)
    run.add_argument("--cluster-energy-window", type=float, default=10.0)
    run.add_argument("--hybrid-max-candidates", type=int, default=16)
    run.add_argument("--hybrid-min-cluster-size", type=int, default=5)
    run.add_argument("--hybrid-min-samples", type=int, default=2)
    run.add_argument("--hybrid-energy-neighbors", type=int, default=8)
    run.add_argument("--hybrid-min-separation-deg", type=float, default=25.0)

    run.add_argument("--xtb-command")
    run.add_argument("--xtb-method", choices=("gfnff", "gfn0", "gfn1", "gfn2"), default="gfn2")
    run.add_argument("--xtb-threads", type=int, default=1, help="internal threads per xTB process")
    run.add_argument("--energy-timeout", type=float, default=600.0)
    run.add_argument("--external-command", help="template argv con placeholder {xyz}")
    run.add_argument("--external-regex", default=DEFAULT_EXTERNAL_REGEX)
    run.add_argument(
        "--external-unit",
        choices=("hartree", "kcal_mol", "kj_mol", "ev"),
        default="hartree",
    )
    run.add_argument("--pyscf-basis", default="sto-3g")

    analyze = subparsers.add_parser("analyze", help="cluster and summarize final results")
    analyze.add_argument("--run", required=True, help="directory produced by seeker run")
    analyze.add_argument("--max-delta-energy", type=float, default=10.0, help="window in kcal/mol")
    analyze.add_argument(
        "--torsion-mean-threshold-deg", "--torsion-threshold-deg",
        dest="torsion_mean_threshold_deg", type=float, default=15.0,
    )
    analyze.add_argument("--torsion-max-threshold-deg", type=float, default=15.0)
    analyze.add_argument(
        "--source", choices=("archive", "final_population", "pareto_front")
    )
    analyze.add_argument(
        "--method",
        choices=(
            "complete_linkage",
            "periodicity_cells",
            "hybrid",
            "mixed_hybrid",
            "pose_hybrid",
        ),
    )
    analyze.add_argument("--hybrid-max-candidates", type=int)
    analyze.add_argument("--hybrid-min-cluster-size", type=int)
    analyze.add_argument("--hybrid-min-samples", type=int)
    analyze.add_argument("--hybrid-energy-neighbors", type=int)
    analyze.add_argument("--hybrid-min-separation-deg", type=float)
    analyze.add_argument(
        "--pose-permutation-mode",
        choices=("equivalent", "ordered"),
        help="exchange identical POSE fragments or preserve block labels",
    )

    single_point = subparsers.add_parser(
        "single-point", help="B3LYP single points and pre-optimization energy filter"
    )
    single_point.add_argument("--input-dir", required=True)
    single_point.add_argument("--output-dir", required=True)
    single_point.add_argument("--backend", choices=("gaussian",), default="gaussian")
    single_point.add_argument("--gaussian-command")
    single_point.add_argument("--gaussian-nprocshared", type=int)
    single_point.add_argument("--gaussian-mem-gb", type=int)
    single_point.add_argument("--jobs", type=int)
    single_point.add_argument("--timeout", type=float, default=1800.0)
    single_point.add_argument("--charge", type=int, default=0)
    single_point.add_argument("--multiplicity", type=int, default=1)
    single_point.add_argument("--energy-window", type=float, default=10.0)

    optimize = subparsers.add_parser(
        "optimize", help="optimize post-clustering XYZ candidates with xTB or B3LYP"
    )
    optimize.add_argument("--input-dir", required=True)
    optimize.add_argument("--output-dir", required=True)
    optimize.add_argument("--backend", choices=("xtb", "gaussian"), default="xtb")
    optimize.add_argument("--xtb-command")
    optimize.add_argument("--gaussian-command")
    optimize.add_argument("--gaussian-nprocshared", type=int)
    optimize.add_argument("--gaussian-mem-gb", type=int)
    optimize.add_argument(
        "--xtb-method", choices=("gfnff", "gfn0", "gfn1", "gfn2"), default="gfn2"
    )
    optimize.add_argument("--xtb-threads", type=int, default=1)
    optimize.add_argument("--jobs", type=int)
    optimize.add_argument("--timeout", type=float)
    optimize.add_argument(
        "--opt-level", choices=("normal", "tight", "verytight"), default="tight"
    )
    optimize.add_argument("--charge", type=int, default=0)
    optimize.add_argument("--multiplicity", type=int, default=1)
    optimize.add_argument(
        "--conformer-change-rmsd",
        type=float,
        default=0.75,
        help="source-to-optimized RMSD above which the conformer is marked changed",
    )
    optimize.add_argument(
        "--dedup-rmsd-threshold",
        type=float,
        default=0.30,
        help="RMSD cutoff used to build the all-energy unique optimized set",
    )
    optimize.add_argument(
        "--comparison-atom-mode", choices=("heavy", "all"), default="all"
    )
    optimize.add_argument(
        "--permutation-mode",
        choices=("equivalent", "ordered"),
        default="equivalent",
        help="exchange graph-equivalent fragments/atoms or preserve XYZ ordering",
    )
    optimize.add_argument("--topology-tolerance", type=float, default=0.45)
    optimize.add_argument(
        "--constraint-mode", choices=("free", "active"), default="free"
    )
    optimize.add_argument("--run-manifest")

    postcluster = subparsers.add_parser(
        "cluster-optimized",
        help="cluster optimized XYZ structures and retain unique minima",
    )
    postcluster.add_argument("--optimization-csv", required=True)
    postcluster.add_argument("--output-dir", required=True)
    postcluster.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.30,
        help="complete-linkage RMSD cutoff in angstrom (default: 0.30)",
    )
    postcluster.add_argument("--energy-window", type=float, default=10.0)
    postcluster.add_argument(
        "--atom-mode",
        choices=("heavy", "all"),
        default="all",
        help="all preserves optimized H rotamers; heavy ignores them (default: all)",
    )
    postcluster.add_argument(
        "--permutation-mode",
        choices=("equivalent", "ordered"),
        default="equivalent",
        help="exchange graph-equivalent fragments/atoms or preserve XYZ ordering",
    )
    postcluster.add_argument("--topology-tolerance", type=float, default=0.45)
    from .launcher import add_launcher_parsers

    add_launcher_parsers(subparsers)
    return parser


def _config(
    args: argparse.Namespace,
    hbond_pi_config: HydrogenPiConfig | None = None,
    checkpoint_operators: dict[str, object] | None = None,
) -> RunConfig:
    operator_schedule: tuple[tuple[float, float], ...] = ()
    if args.operator_schedule_file:
        payload = json.loads(Path(args.operator_schedule_file).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or not isinstance(payload.get("probabilities"), list):
            raise ValueError("operator schedule JSON must have format_version=1 and probabilities[]")
        rows = []
        for expected_generation, row in enumerate(payload["probabilities"]):
            if int(row.get("generation", -1)) != expected_generation:
                raise ValueError("operator schedule generations must be contiguous and zero-based")
            rows.append((float(row["mutation_probability"]), float(row["crossover_probability"])))
        operator_schedule = tuple(rows)
    elif checkpoint_operators:
        operator_schedule = tuple(
            (float(pair[0]), float(pair[1]))
            for pair in checkpoint_operators.get("operator_schedule", [])  # type: ignore[union-attr]
        )
        args.mutation_operator = str(checkpoint_operators["mutation_operator"])
        args.crossover_operator = str(checkpoint_operators["crossover_operator"])
        args.mutation_sigma_deg = float(checkpoint_operators["mutation_sigma_deg"])
    return RunConfig(
        population_size=args.population,
        offspring_size=args.offspring,
        generations=args.generations,
        seed=args.seed,
        workers=args.workers,
        islands=args.islands,
        migration_interval=args.migration_interval,
        migration_size=args.migration_size,
        migration_selection=args.migration_selection,
        base_mutation_weight=args.base_mutation_weight,
        base_crossover_weight=args.base_crossover_weight,
        mutation_weight_amplitude=args.mutation_weight_amplitude,
        crossover_weight_amplitude=args.crossover_weight_amplitude,
        operator_oscillations=args.operator_oscillations,
        operator_schedule=operator_schedule,
        mutation_operator=args.mutation_operator,
        crossover_operator=args.crossover_operator,
        mutation_sigma_deg=args.mutation_sigma_deg,
        periodicity_grid_step_deg=args.periodicity_grid_step_deg,
        sbx_eta=args.sbx_eta,
        duplicate_threshold_deg=args.duplicate_threshold_deg,
        duplicate_mean_threshold_deg=args.duplicate_mean_threshold_deg,
        duplicate_max_threshold_deg=args.duplicate_max_threshold_deg,
        initialization_strategy=args.initialization_strategy,
        initial_pool_factor=args.initial_pool_factor,
        initial_pool_sampling=args.initial_pool_sampling,
        initial_scan_layout=args.initial_scan_layout,
        initial_scan_grid=args.initial_scan_grid,
        initial_scan_points_mode=args.initial_scan_points_mode,
        initial_scan_points=args.initial_scan_points,
        topology_tolerance=args.topology_tolerance,
        hbond_cutoff_angstrom=args.hbond_cutoff,
        hbond_contact_threshold=args.hbond_contact_threshold,
        hh_clash_distance_angstrom=args.hh_clash_distance,
        geometric_prescreen=args.geometric_prescreen,
        steric_hh_scale=args.steric_hh_scale,
        steric_heavy_heavy_scale=args.steric_heavy_heavy_scale,
        steric_hydrogen_heavy_scale=args.steric_hydrogen_heavy_scale,
        steric_exclude_hops=args.steric_exclude_hops,
        checkpoint_every=args.checkpoint_every,
        max_duplicate_attempts=args.max_duplicate_attempts,
        objectives=active_objectives(args.extra_objectives),
        hbond_pi_config=hbond_pi_config or HydrogenPiConfig(),
        rotor_symmetry_sigma=args.rotor_symmetry_sigma,
        rotor_anisotropy_sigma=args.rotor_anisotropy_sigma,
        early_stopping=args.early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_diversity_deg=args.early_stop_diversity_deg,
        early_stop_min_generations=args.early_stop_min_generations,
        archive_stagnation_patience=args.archive_stagnation_patience,
        clustering_source=args.clustering_source,
        clustering_method=args.clustering_method,
        cluster_mean_threshold_deg=args.cluster_mean_threshold_deg,
        cluster_max_threshold_deg=args.cluster_max_threshold_deg,
        cluster_energy_window_kcal_mol=args.cluster_energy_window,
        hybrid_max_candidates=args.hybrid_max_candidates,
        hybrid_min_cluster_size=args.hybrid_min_cluster_size,
        hybrid_min_samples=args.hybrid_min_samples,
        hybrid_energy_neighbors=args.hybrid_energy_neighbors,
        hybrid_min_separation_deg=args.hybrid_min_separation_deg,
    )


def _backend(args: argparse.Namespace, output_dir: Path) -> EnergyBackend:
    work_root = output_dir / "work"
    if args.backend == "xtb":
        backend: EnergyBackend = XtbBackend(
            command=args.xtb_command,
            method=args.xtb_method,
            threads=args.xtb_threads,
            timeout_seconds=args.energy_timeout,
            work_root=work_root,
        )
    elif args.backend == "external":
        if not args.external_command:
            raise ValueError("--backend external requires --external-command")
        backend = ExternalCommandBackend(
            command_template=args.external_command,
            energy_regex=args.external_regex,
            energy_unit=args.external_unit,
            timeout_seconds=args.energy_timeout,
            work_root=work_root,
        )
    else:
        backend = PyScfBackend(basis=args.pyscf_basis)
    return CachedEnergyBackend(backend, EnergyCache(output_dir / "energy_cache.sqlite"))


def _run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    checkpoint_path = Path(args.resume) if args.resume else None
    if not args.resume and (output_dir / "checkpoint.json").exists():
        raise FileExistsError(
            f"{output_dir} already contains a run; use --resume {output_dir / 'checkpoint.json'} or a new --output"
        )

    molecule = read_xyz(args.xyz, charge=args.charge, multiplicity=args.multiplicity)
    specs = read_coordinate_specs(args.genes)
    hbond_pi_config = read_hbond_pi_config(args.genes)
    checkpoint_operators = None
    if args.resume and not args.operator_schedule_file:
        raw_checkpoint = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        checkpoint_operators = raw_checkpoint.get("operator_configuration")
    config = _config(args, hbond_pi_config, checkpoint_operators)
    config.validate()
    validate_objectives(config.objectives)
    if hbond_pi_config.configured and not {
        "hbond_pi",
        "disconnected_components_penalty",
    }.intersection(config.objectives):
        print(
            "warning: HPI directives are valid but hbond_pi is not an active objective; "
            "they will be ignored",
            flush=True,
        )
    if len(config.objectives) > 4:
        print(
            "warning: more than four active objectives; NSGA-II may lose selection pressure "
            "in a many-objective regime",
            flush=True,
        )
    graph = build_bond_graph(molecule, config.topology_tolerance)
    poses = tuple(spec for spec in specs if isinstance(spec, FragmentPoseGene))
    if args.validate_only:
        HydrogenPiModel.from_reference(molecule, graph, hbond_pi_config)
        if "hbond_=" in config.objectives:
            HydrogenDoubleBondModel.from_reference(molecule, graph)
        if "disconnected_components_penalty" in config.objectives:
            DisconnectedComponentsPenaltyModel.from_reference(
                molecule,
                graph,
            )
    torsions = tuple(spec for spec in specs if isinstance(spec, Gene))
    if poses and torsions:
        raise ValueError(
            "POSE currently requires a pose-only coordinate set"
        )
    native_plan = None
    genes = torsions
    if poses:
        from .fragment_pose import prepare_native_fragment_poses

        native_plan = prepare_native_fragment_poses(molecule, poses, graph)
        genes = native_plan.coordinates
        coordinate_count = len(genes)
        coordinate_mode = "native-rigid-pose-batch"
        if config.initialization_strategy in {"lhs", "scan"}:
            raise ValueError(
                "native POSE supports random or maximin initialization; "
                "component-wise lhs/scan does not preserve the SE(3) bounds"
            )
        if (
            config.initialization_strategy == "maximin"
            and config.initial_pool_sampling == "latin_hypercube"
        ):
            config = replace(config, initial_pool_sampling="prior")
    else:
        coordinate_count = len(prepare_genes(genes, graph))
        coordinate_mode = "quaternion"
    print(
        f"input valid: atoms={len(molecule.atoms)} genes={coordinate_count} "
        f"coordinates={coordinate_mode} "
        f"charge={molecule.charge} multiplicity={molecule.multiplicity}",
        flush=True,
    )
    if args.validate_only:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _backend(args, output_dir)
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else None
    dashboard = None
    if args.tui:
        from .tui import TerminalDashboard

        dashboard = TerminalDashboard(
            config.generations,
            config.objectives,
            output_dir,
        )
    try:
        engine_kwargs = {
            "allele_structure_provider": native_plan.apply if native_plan else None,
            "allele_structure_batch_provider": (
                native_plan.apply_batch if native_plan else None
            ),
            "reference_alleles": native_plan.reference_alleles if native_plan else None,
            "auto_analyze": args.auto_analyze,
            "progress_callback": dashboard.update if dashboard else None,
        }
        if native_plan is None:
            engine = GeneticConformerSearch(
                molecule,
                genes,
                backend,
                config,
                output_dir,
                **engine_kwargs,
            )
        else:
            from .native_pose_engine import PoseGeneticConformerSearch

            engine = PoseGeneticConformerSearch(
                molecule,
                genes,
                backend,
                config,
                output_dir,
                pose_variables=native_plan.coordinates,
                pose_blocks=native_plan.pose_blocks,
                **engine_kwargs,
            )
        if dashboard is not None:
            dashboard.set_molecule_provider(engine.structure_for)
            dashboard.start_waiting(
                "Evaluating the initial population before generation 1 "
                f"({config.islands} island{'s' if config.islands != 1 else ''}, "
                f"{config.workers} worker{'s' if config.workers != 1 else ''})."
            )
        if checkpoint and checkpoint["run_fingerprint"] != engine.fingerprint:
            raise ValueError("checkpoint is incompatible with the current input, configuration, or backend")
        if checkpoint and int(checkpoint["generation"]) > config.generations:
            raise ValueError("--generations is lower than the generation saved in the checkpoint")
        write_manifest(
            output_dir,
            config,
            molecule_fingerprint(molecule, genes),
            backend.signature,
            args.xyz,
            args.genes,
            None,
            coordinates=genes,
            hbond_pi_metadata=(
                engine.hbond_pi_model.reference_metadata()
                if engine.hbond_pi_model is not None
                else None
            ),
            hbond_double_metadata=(
                engine.hbond_double_model.reference_metadata()
                if engine.hbond_double_model is not None
                else None
            ),
            disconnected_components_metadata=(
                engine.disconnected_components_model.reference_metadata()
                if engine.disconnected_components_model is not None
                else None
            ),
            fragment_pose_blocks=(native_plan.pose_blocks if native_plan else ()),
        )
        population = engine.run(checkpoint)
    finally:
        if dashboard is not None:
            dashboard.close()
        backend.close()
    valid = sum(1 for individual in population if individual.valid)
    pareto = sum(1 for individual in population if individual.valid and individual.rank == 0)
    print(f"completed: valid={valid} pareto={pareto} output={output_dir.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "configure":
            from .local_config import configure_interactively

            configure_interactively(args.config)
            return 0
        if args.command == "doctor":
            from .local_config import diagnostics, load_config

            configured = load_config(args.config) if args.config else None
            for item in diagnostics(configured):
                print(f"{item.status.upper():8} {item.name:16} {item.detail}")
            return 0
        if args.command == "launch":
            from .launcher import launch

            return launch(args)
        if args.command == "run":
            from .local_config import resolve_tools

            tools = resolve_tools()
            args.xtb_command = args.xtb_command or tools.xtb or "xtb"
            return _run(args)
        if args.command == "single-point":
            from .local_config import resolve_tools

            tools = resolve_tools()
            args.gaussian_command = args.gaussian_command or tools.gaussian or "g16"
            summary = single_point_gaussian_candidates(
                args.input_dir,
                args.output_dir,
                command=args.gaussian_command,
                jobs=args.jobs,
                nprocshared=args.gaussian_nprocshared,
                mem_gb=args.gaussian_mem_gb,
                timeout_seconds=args.timeout,
                charge=args.charge,
                multiplicity=args.multiplicity,
                energy_window_kcal_mol=args.energy_window,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if summary["succeeded"] > 0 else 1
        if args.command == "optimize":
            from .local_config import resolve_tools

            tools = resolve_tools()
            if args.backend == "xtb":
                if args.constraint_mode != "free":
                    raise ValueError(
                        "exact active-only constraints are available only with --backend gaussian"
                    )
                args.xtb_command = args.xtb_command or tools.xtb or "xtb"
                summary = optimize_xtb_candidates(
                    args.input_dir,
                    args.output_dir,
                    command=args.xtb_command,
                    method=args.xtb_method,
                    jobs=args.jobs or 1,
                    threads=args.xtb_threads,
                    timeout_seconds=600.0 if args.timeout is None else args.timeout,
                    opt_level=args.opt_level,
                    charge=args.charge,
                    multiplicity=args.multiplicity,
                    conformer_change_rmsd_angstrom=args.conformer_change_rmsd,
                    dedup_rmsd_threshold_angstrom=args.dedup_rmsd_threshold,
                    comparison_atom_mode=args.comparison_atom_mode,
                    permutation_mode=args.permutation_mode,
                    topology_tolerance=args.topology_tolerance,
                )
            else:
                args.gaussian_command = (
                    args.gaussian_command or tools.gaussian or "g16"
                )
                summary = optimize_gaussian_candidates(
                    args.input_dir,
                    args.output_dir,
                    command=args.gaussian_command,
                    jobs=args.jobs,
                    nprocshared=args.gaussian_nprocshared,
                    mem_gb=args.gaussian_mem_gb,
                    timeout_seconds=3600.0 if args.timeout is None else args.timeout,
                    charge=args.charge,
                    multiplicity=args.multiplicity,
                    constraint_mode=args.constraint_mode,
                    run_manifest=args.run_manifest,
                    conformer_change_rmsd_angstrom=args.conformer_change_rmsd,
                    dedup_rmsd_threshold_angstrom=args.dedup_rmsd_threshold,
                    comparison_atom_mode=args.comparison_atom_mode,
                    permutation_mode=args.permutation_mode,
                    topology_tolerance=args.topology_tolerance,
                )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if args.backend == "gaussian":
                return 0 if summary["succeeded"] > 0 else 1
            return 0 if summary["failed"] == 0 else 1
        if args.command == "cluster-optimized":
            summary = cluster_optimized_candidates(
                args.optimization_csv,
                args.output_dir,
                rmsd_threshold_angstrom=args.rmsd_threshold,
                energy_window_kcal_mol=args.energy_window,
                atom_mode=args.atom_mode,
                permutation_mode=args.permutation_mode,
                topology_tolerance=args.topology_tolerance,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        summary = analyze_run(
            args.run,
            max_delta_energy_kcal_mol=args.max_delta_energy,
            torsion_threshold_deg=args.torsion_mean_threshold_deg,
            torsion_max_threshold_deg=args.torsion_max_threshold_deg,
            clustering_source=args.source,
            clustering_method=args.method,
            hybrid_max_candidates=args.hybrid_max_candidates,
            hybrid_min_cluster_size=args.hybrid_min_cluster_size,
            hybrid_min_samples=args.hybrid_min_samples,
            hybrid_energy_neighbors=args.hybrid_energy_neighbors,
            hybrid_min_separation_deg=args.hybrid_min_separation_deg,
            pose_permutation_mode=args.pose_permutation_mode,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        print("\nSEEKER setup interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2
