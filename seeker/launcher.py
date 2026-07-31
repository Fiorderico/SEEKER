"""Portable interactive launcher for a source checkout or installed SEEKER."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Callable, IO, Sequence

from .fitness import HydrogenPiModel, DisconnectedComponentsPenaltyModel
from .gaussian import B3LYP_MODEL, recommend_gaussian_resources
from .geometry import build_bond_graph
from .input import read_coordinate_specs, read_hbond_pi_config, read_xyz
from .local_config import LocalConfig, config_path, configure_interactively, resolve_tools
from .models import FragmentPoseGene, Gene, Molecule
from .objectives import BASE_OBJECTIVES, OBJECTIVES, active_objectives
from .presets import recommend_search_preset


def _hbond_pi_recommendation(molecule: Molecule, genes: str | Path) -> str | None:
    """Describe an applicable X-H...pi objective, or return ``None``.

    Applicability is derived from the same topology and planarity checks used by
    the fitness engine.  A genetic ``RING(...)`` coordinate alone is deliberately
    insufficient: puckering coordinates may describe non-pi rings.
    """

    config = read_hbond_pi_config(genes)
    model = HydrogenPiModel.from_reference(
        molecule,
        build_bond_graph(molecule),
        config,
    )
    eligible_pairs = sum(len(indices) for indices in model.eligible_rings.values())
    if not model.rings or not eligible_pairs:
        return None
    explicit = sum(ring.source == "explicit" for ring in model.rings)
    automatic = len(model.rings) - explicit
    sources: list[str] = []
    if explicit:
        sources.append(f"{explicit} explicitly configured")
    if automatic:
        sources.append(f"{automatic} automatically detected")
    return (
        f"{len(model.rings)} pi ring(s) ({', '.join(sources)}) and "
        f"{eligible_pairs} eligible X-H/ring pair(s)"
    )


def default_output_path(xyz: str | Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "runs" / f"{Path(xyz).stem}_{timestamp}"


def _ask(
    label: str,
    default: str,
    input_fn: Callable[[str], str],
    *,
    choices: Sequence[str] = (),
    hint: str = "",
) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        explanation = hint or _prompt_hint(label)
        if choices:
            options = ", ".join(choices)
            explanation = f"{explanation} Allowed answers: {options}."
        print(f"{label}{suffix}")
        dim = (
            bool(getattr(sys.stdout, "isatty", lambda: False)())
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM", "") != "dumb"
        )
        rendered_hint = f"  {explanation}"
        print(f"\x1b[2;38;5;245m{rendered_hint}\x1b[0m" if dim else rendered_hint)
        answer = input_fn("> ").strip() or default
        if not choices or answer in choices:
            return answer
        print(f"Choose one of: {', '.join(choices)}")


def _prompt_hint(label: str) -> str:
    """Return a concise physical or operational explanation for a wizard field."""

    text = label.casefold()
    rules = (
        (("xyz file",), "Path to the reference Cartesian geometry in XYZ format."),
        (("genes file",), "Path to the GENES file defining the coordinates SEEKER may change."),
        (("output directory",), "A new directory that will contain this run and all generated artifacts."),
        (("validate inputs only",), "Check inputs and coordinate definitions without energies or genetic evolution."),
        (("energy backend",), "Select the program or driver used to evaluate candidate energies."),
        (("recommended preset",), "Apply the coordinate-family decision tree for population, islands, migration, and generations."),
        (("advanced scientific settings",), "Open detailed controls for sampling, fitness, operators, and clustering."),
        (("molecular charge",), "Total integer charge passed to the selected electronic-structure backend."),
        (("multiplicity",), "Spin multiplicity 2S+1; a closed-shell molecule normally uses 1."),
        (("parallel workers",), "Maximum number of candidate geometries evaluated concurrently."),
        (("random seed",), "Seed controlling reproducible initialization and genetic operations."),
        (("initialization",), "Choose how the first population covers the coordinate space."),
        (("scan layout",), "Tensor combines all grid axes; one-at-a-time changes one coordinate from the reference."),
        (("scan grid",), "Choose uniform coordinate spacing or modes derived from torsional periodicity."),
        (("uniform points",), "Number of grid values placed along each coordinate."),
        (("pool sampling",), "Distribution used to generate the oversized pool for maximin selection."),
        (("pool-size factor",), "Candidate-pool size as a multiple of the retained population."),
        (("population per island", "population size"), "Number of individuals retained independently on each island."),
        (("offspring",), "Number of new candidates proposed per island and generation."),
        (("generations",), "Maximum number of evolutionary iterations."),
        (("number of islands",), "Independent populations used to preserve alternative search directions."),
        (("migration interval",), "Number of generations between exchanges among islands."),
        (("migrants per island",), "Number of individuals transferred by each island at migration."),
        (("migrant selection",), "Choose whether migrants are Pareto-diverse candidates or random individuals."),
        (("mutation weight",), "Base relative probability of applying mutation."),
        (("crossover weight",), "Base relative probability of applying crossover."),
        (("oscillation amplitude",), "Amplitude of the sinusoidal change applied to this operator weight."),
        (("half-oscillations",), "Number of half-periods in the mutation/crossover schedule."),
        (("periodic-prior step",), "Angular resolution used to discretize periodic torsional priors."),
        (("sbx crossover eta",), "SBX locality parameter; larger values keep children closer to their parents."),
        (("duplicate mean threshold",), "Mean coordinate distance below which two candidates are duplicates."),
        (("duplicate maximum threshold",), "Maximum single-coordinate distance allowed for duplicate classification."),
        (("deduplication attempts",), "Maximum replacement attempts after proposing a duplicate candidate."),
        (("geometric prescreen",), "Reject broken topology and severe clashes before expensive energy calculations."),
        (("topology tolerance",), "Distance tolerance in angstrom used when inferring the reference bond graph."),
        (("h-bond cutoff",), "Maximum donor-acceptor distance considered by the geometric H-bond score."),
        (("h-bond contact threshold",), "Score threshold used to count a geometry as an H-bond contact."),
        (("h-h clash distance",), "Minimum allowed nonbonded hydrogen-hydrogen separation in angstrom."),
        (("steric scale",), "Scale applied to covalent radii in the corresponding clash test."),
        (("exclude pairs",), "Ignore atom pairs connected through this many bond-graph hops."),
        (("extra objectives",), "Comma-separated optional fitness objectives in addition to energy and H-bonding."),
        (
            ("x-h...pi ring-interaction",),
            "Reward geometries in which an N-H, O-H, or S-H donor points "
            "toward the center of an eligible pi ring.",
        ),
        (("rotor symmetry sigma",), "Width of the rotor-symmetry preference."),
        (("rotor anisotropy sigma",), "Width of the rotor-anisotropy preference."),
        (("early stopping",), "Stop after sustained fitness stagnation and low structural diversity."),
        (("patience generations",), "Consecutive stagnant generations required before early stopping."),
        (("fitness improvement",), "Changes below this value are ignored when measuring progress."),
        (("diversity threshold",), "Early stopping is considered only below this angular diversity."),
        (("minimum generation",), "Protect the initial exploration phase from early termination."),
        (("archive stagnation",), "Stop after this many generations without a new archive member; zero disables it."),
        (("clustering source",), "Select which discovered population is supplied to final clustering."),
        (("clustering method",), "Choose the structural grouping method used to select final representatives."),
        (("mean torsion threshold",), "Mean torsional separation used by complete-linkage clustering."),
        (("maximum torsion threshold",), "Largest allowed torsional difference within a cluster."),
        (("maximum final candidates",), "Hard cap on representatives retained for downstream refinement."),
        (("hdbscan minimum cluster size",), "Smallest dense group recognized as a cluster."),
        (("hdbscan min_samples",), "Local-density strictness used by HDBSCAN."),
        (("energy-graph neighbors",), "Number of energetic neighbors used to identify local minima."),
        (("discovery separation",), "Minimum structural separation between retained representatives."),
        (("clustering energy window",), "Retain candidates within this many kcal/mol of the minimum."),
        (("generation xyz",), "Write per-generation Cartesian snapshots; useful but potentially large."),
        (("energy timeout",), "Maximum wall time for one energy evaluation."),
        (("xtb method",), "Select the xTB Hamiltonian used for energies and optional optimization."),
        (("xtb threads",), "Internal xTB threads assigned to each concurrent geometry."),
        (("pyscf basis",), "Orbital basis set used by the PySCF backend."),
        (("external command",), "Command template for an external evaluator; it must contain {xyz}."),
        (("energy regex",), "Regular expression whose first capture group is the computed energy."),
        (("external energy unit",), "Unit produced by the external energy command."),
        (("optimize and rmsd",), "After clustering, optimize selected candidates and remove RMSD duplicates."),
        (("optimization accuracy",), "Geometry-convergence level used by xTB."),
        (("parallel xtb optimizations",), "Number of final xTB optimizations executed concurrently."),
        (("b3lyp single-point",), "Evaluate selected candidates at B3LYP before final optimization."),
        (("b3lyp filter",), "Keep candidates within this B3LYP energy window from the lowest single point."),
        (("final optimization backend",), "Choose xTB, B3LYP, or no final geometry optimization."),
        (("active-only",), "Freeze every intramolecular coordinate except explored D and intermolecular POSE degrees of freedom."),
        (("gaussian jobs",), "Maximum number of Gaussian calculations executed concurrently."),
        (("gaussian nprocshared",), "Shared-memory CPU cores assigned to each Gaussian calculation."),
        (("gaussian memory",), "Memory in GB assigned to each Gaussian calculation."),
        (("optimization timeout",), "Maximum wall time allowed for each final optimization."),
        (("final rmsd threshold",), "Structures closer than this aligned RMSD are considered duplicates."),
        (("rmsd atom mode",), "Choose whether RMSD includes all atoms or heavy atoms only."),
        (("execution mode",), "Foreground shows the live TUI; background writes run.log and run.pid."),
        (("launch seeker",), "Confirm the reviewed configuration and create the run directory."),
    )
    for needles, explanation in rules:
        if any(needle in text for needle in needles):
            return explanation
    return "Enter a value for this run, or press Enter to accept the displayed default."


def _yes_no(label: str, default: bool, input_fn: Callable[[str], str]) -> bool:
    answer = _ask(label, "yes" if default else "no", input_fn, choices=("yes", "no"))
    return answer == "yes"


def _ask_energy_backend(
    default: str,
    tools: LocalConfig,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    """Explain the concrete backend contract before asking for a driver."""

    pyscf_installed = importlib.util.find_spec("pyscf") is not None
    output_fn("")
    output_fn("Energy backends")
    output_fn(
        f"  {'[ready]' if tools.xtb else '[setup]'} xtb     "
        "xTB executable; fast semiempirical energies (recommended)."
    )
    output_fn(
        f"  {'[ready]' if pyscf_installed else '[setup]'} pyscf   "
        "In-process RHF/UHF energies; install SEEKER's optional pyscf dependency."
    )
    output_fn(
        "  [adapter] external Run any local executable you provide. SEEKER writes "
        "an XYZ path into the required {xyz} placeholder, also supports {charge}, "
        "{multiplicity}, and {workdir}, then extracts one energy from the output."
    )
    output_fn(
        "  'external' is an integration hook, not another quantum-chemistry program "
        "bundled with SEEKER."
    )
    return _ask(
        "Energy backend", default, input_fn,
        choices=("xtb", "pyscf", "external"),
        hint="Choose the driver that will evaluate every candidate geometry.",
    )


def _ask_optional_objectives(
    defaults: Sequence[str],
    *,
    hbond_pi_available: bool,
    disconnected_components_available: bool,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    """Show every fitness and explicitly select the optional active set."""

    default_names = tuple(dict.fromkeys(defaults))
    unavailable: dict[str, str] = {}
    if not hbond_pi_available:
        unavailable["hbond_pi"] = "no eligible donor/pi-ring pair in this input"
    if not disconnected_components_available:
        unavailable["disconnected_components_penalty"] = (
            "requires at least two covalent fragments"
        )
    output_fn("")
    output_fn("Fitness objectives")
    output_fn("  Always active (required by the current NSGA-II model):")
    for name in BASE_OBJECTIVES:
        output_fn(f"    [on]  {name:<32} {OBJECTIVES[name].description}")
    output_fn("  Optional (the state shown is the default for this run):")
    for name, definition in OBJECTIVES.items():
        if name in BASE_OBJECTIVES:
            continue
        if name in unavailable:
            state = "n/a"
            suffix = f" ({unavailable[name]})"
        else:
            state = "on" if name in default_names else "off"
            suffix = ""
        output_fn(f"    [{state:<3}] {name:<32} {definition.description}{suffix}")

    default_text = ",".join(default_names) or "none"
    while True:
        answer = _ask(
            "Enabled optional fitness objectives", default_text, input_fn,
            hint=(
                "Enter a comma-separated list from the available optional objectives; "
                "enter 'none' to disable all of them."
            ),
        ).strip()
        if not answer or answer.casefold() == "none":
            return ""
        try:
            selected = active_objectives(answer)[len(BASE_OBJECTIVES):]
        except ValueError as exc:
            output_fn(str(exc))
            continue
        blocked = [name for name in selected if name in unavailable]
        if blocked:
            for name in blocked:
                output_fn(f"Cannot enable {name}: {unavailable[name]}.")
            continue
        return ",".join(selected)


def _ask_int(
    label: str,
    default: int,
    minimum: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> int:
    while True:
        try:
            value = int(
                _ask(
                    label, str(default), input_fn,
                    hint=(
                        f"{_prompt_hint(label)} Enter an integer greater than or "
                        f"equal to {minimum}."
                    ),
                )
            )
        except ValueError:
            output_fn("Enter an integer.")
            continue
        if value >= minimum:
            return value
        output_fn(f"Enter an integer >= {minimum}.")


def _ask_float(
    label: str,
    default: float,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    *,
    minimum: float | None = None,
) -> float:
    while True:
        try:
            limit = (
                f"{_prompt_hint(label)} Enter a finite number."
                if minimum is None
                else (
                    f"{_prompt_hint(label)} Enter a number greater than or equal "
                    f"to {minimum}."
                )
            )
            value = float(_ask(label, str(default), input_fn, hint=limit))
        except ValueError:
            output_fn("Enter a number.")
            continue
        if minimum is None or value >= minimum:
            return value
        output_fn(f"Enter a number >= {minimum}.")


class _WizardChrome:
    """Keep a decorative SEEKER header fixed above the scrolling wizard."""

    header_rows = 8

    def __init__(self, stream: IO[str] = sys.stdout) -> None:
        self.stream = stream
        size = shutil.get_terminal_size((100, 28))
        self.width = size.columns
        self.height = size.lines
        self.enabled = (
            bool(getattr(stream, "isatty", lambda: False)())
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM", "") != "dumb"
            and self.width >= 80
            and self.height >= 20
        )
        self.started = False

    def start(self) -> None:
        if not self.enabled or self.started:
            return
        self.started = True
        width = self.width
        art = (
            "                    ●",
            "                    │",
            "                 ●──●──●  ╷",
            "                    ●   ╲●│",
            "                   ╱ ╲    │",
            "        _..---.._ ●   ●   │ _..---.._",
            "__..--´          `--..__..´           `--..__",
        )
        colors = {
            "sand": "\x1b[38;5;179m",
            "bright": "\x1b[38;5;223m",
            "muted": "\x1b[38;5;245m",
            "cyan": "\x1b[38;5;80m",
            "reset": "\x1b[0m",
        }
        self.stream.write("\x1b[2J\x1b[H")
        left = (
            f"{colors['bright']}S E E K E R{colors['reset']}",
            f"{colors['muted']}interactive multiobjective search{colors['reset']}",
        )
        for row, line in enumerate(left, 1):
            self.stream.write(f"\x1b[{row};3H{line}")
        art_width = max(len(line) for line in art)
        art_column = max(1, width - art_width + 1)
        for row, line in enumerate(art, 1):
            color = colors["cyan"] if row <= 5 else colors["sand"]
            self.stream.write(
                f"\x1b[{row};{art_column}H{color}{line}{colors['reset']}"
            )
        self.stream.write(
            f"\x1b[8;1H{colors['sand']}{'─' * width}{colors['reset']}"
            f"\x1b[9;{self.height}r\x1b[9;1H"
        )
        self.stream.flush()
        atexit.register(self.close)

    def close(self) -> None:
        if not self.started:
            return
        self.started = False
        try:
            atexit.unregister(self.close)
        except Exception:  # pragma: no cover - interpreter shutdown variations
            pass
        self.stream.write(f"\x1b[r\x1b[{self.height};1H\x1b[0m\n")
        self.stream.flush()


def _show_launcher_intro(
    *,
    interactive: bool,
    output_fn: Callable[[str], None] = print,
) -> _WizardChrome:
    """Show the traditional SEEKER intro for an interactive launch."""

    chrome = _WizardChrome()
    if not interactive:
        return chrome
    from .tui import play_intro

    # play_intro performs its own TTY, NO_COLOR, and SEEKER_ANIMATION checks.
    play_intro()
    chrome.start()
    if chrome.enabled:
        output_fn("Interactive setup started. Press Enter to accept each suggested value.\n")
    else:
        output_fn("╭────────────────────────────────────────────────────────────╮")
        output_fn("│ SEEKER · interactive genetic search and clustering      │")
        output_fn("╰────────────────────────────────────────────────────────────╯")
        output_fn("Press Enter to accept each suggested value.")
    return chrome


def build_launch_command(
    *,
    xyz: Path,
    genes: Path,
    output: Path,
    backend: str,
    tools: LocalConfig,
    charge: int = 0,
    multiplicity: int = 1,
    population: int = 48,
    offspring: int = 48,
    generations: int = 40,
    workers: int = 4,
    islands: int = 2,
    seed: int = 7,
    migration_interval: int | None = None,
    migration_size: int | None = None,
    migration_selection: str = "pareto",
    initialization_strategy: str = "lhs",
    initial_pool_factor: int = 20,
    initial_pool_sampling: str = "latin_hypercube",
    initial_scan_layout: str = "tensor",
    initial_scan_grid: str = "uniform",
    initial_scan_points_mode: str = "fixed",
    initial_scan_points: int = 3,
    base_mutation_weight: float = 0.75,
    base_crossover_weight: float = 0.55,
    mutation_weight_amplitude: float = 0.15,
    crossover_weight_amplitude: float = 0.10,
    operator_oscillations: int = 2,
    periodicity_grid_step_deg: float = 20.0,
    sbx_eta: float = 15.0,
    duplicate_mean_threshold_deg: float = 3.0,
    duplicate_max_threshold_deg: float = 3.0,
    max_duplicate_attempts: int = 30,
    geometric_prescreen: bool = True,
    topology_tolerance: float = 0.45,
    hbond_cutoff: float = 3.2,
    hbond_contact_threshold: float = -0.30,
    hh_clash_distance: float = 1.40,
    steric_hh_scale: float = 0.55,
    steric_heavy_heavy_scale: float = 0.55,
    steric_hydrogen_heavy_scale: float = 0.50,
    steric_exclude_hops: int = 3,
    extra_objectives: str = "",
    rotor_symmetry_sigma: float = 0.15,
    rotor_anisotropy_sigma: float = 0.15,
    early_stop: bool = False,
    early_stop_patience: int = 8,
    early_stop_min_delta: float = 1.0e-6,
    early_stop_diversity_deg: float = 8.0,
    early_stop_min_generations: int = 0,
    archive_stagnation_patience: int = 0,
    clustering_source: str = "archive",
    clustering_method: str | None = None,
    cluster_mean_threshold_deg: float = 15.0,
    cluster_max_threshold_deg: float = 15.0,
    cluster_energy_window: float = 10.0,
    hybrid_max_candidates: int = 40,
    hybrid_min_cluster_size: int = 5,
    hybrid_min_samples: int = 2,
    hybrid_energy_neighbors: int = 8,
    hybrid_min_separation_deg: float = 25.0,
    save_generation_xyz: bool = False,
    xtb_method: str = "gfn2",
    xtb_threads: int = 1,
    energy_timeout: float = 600.0,
    pyscf_basis: str = "sto-3g",
    external_command: str = "",
    external_regex: str = r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
    external_unit: str = "hartree",
    validate_only: bool = False,
    tui: bool = True,
) -> list[str]:
    specs = read_coordinate_specs(genes)
    has_pose = any(isinstance(item, FragmentPoseGene) for item in specs)
    has_torsion = any(isinstance(item, Gene) for item in specs)
    if has_pose and has_torsion:
        raise ValueError("POSE currently requires a pose-only coordinate set")
    method = clustering_method or (
        "pose_hybrid" if has_pose else "hybrid"
    )
    migration_interval = (10 if islands > 1 else 0) if migration_interval is None else migration_interval
    migration_size = (
        min(4, max(1, population - 1)) if migration_size is None else migration_size
    )
    command = [
        "run",
        "--xyz",
        str(xyz),
        "--genes",
        str(genes),
        "--output",
        str(output),
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
        "--backend",
        backend,
        "--population",
        str(population),
        "--offspring",
        str(offspring),
        "--generations",
        str(generations),
        "--workers",
        str(workers),
        "--islands",
        str(islands),
        "--migration-interval",
        str(migration_interval),
        "--migration-size",
        str(migration_size),
        "--migration-selection",
        migration_selection,
        "--seed",
        str(seed),
        "--initialization-strategy",
        initialization_strategy,
        "--initial-pool-factor", str(initial_pool_factor),
        "--initial-pool-sampling", initial_pool_sampling,
        "--initial-scan-layout", initial_scan_layout,
        "--initial-scan-grid", initial_scan_grid,
        "--initial-scan-points-mode", initial_scan_points_mode,
        "--initial-scan-points", str(initial_scan_points),
        "--base-mutation-weight", str(base_mutation_weight),
        "--base-crossover-weight", str(base_crossover_weight),
        "--mutation-weight-amplitude", str(mutation_weight_amplitude),
        "--crossover-weight-amplitude", str(crossover_weight_amplitude),
        "--operator-oscillations", str(operator_oscillations),
        "--periodicity-grid-step-deg", str(periodicity_grid_step_deg),
        "--sbx-eta", str(sbx_eta),
        "--duplicate-threshold-deg", str(duplicate_mean_threshold_deg),
        "--duplicate-mean-threshold-deg", str(duplicate_mean_threshold_deg),
        "--duplicate-max-threshold-deg", str(duplicate_max_threshold_deg),
        "--max-duplicate-attempts", str(max_duplicate_attempts),
        "--topology-tolerance", str(topology_tolerance),
        "--hbond-cutoff", str(hbond_cutoff),
        "--hbond-contact-threshold", str(hbond_contact_threshold),
        "--hh-clash-distance", str(hh_clash_distance),
        "--steric-hh-scale", str(steric_hh_scale),
        "--steric-heavy-heavy-scale", str(steric_heavy_heavy_scale),
        "--steric-hydrogen-heavy-scale", str(steric_hydrogen_heavy_scale),
        "--steric-exclude-hops", str(steric_exclude_hops),
        "--extra-objectives", extra_objectives,
        "--rotor-symmetry-sigma", str(rotor_symmetry_sigma),
        "--rotor-anisotropy-sigma", str(rotor_anisotropy_sigma),
        "--early-stop-patience", str(early_stop_patience),
        "--early-stop-min-delta", str(early_stop_min_delta),
        "--early-stop-diversity-deg", str(early_stop_diversity_deg),
        "--early-stop-min-generations", str(early_stop_min_generations),
        "--archive-stagnation-patience", str(archive_stagnation_patience),
        "--clustering-source",
        clustering_source,
        "--clustering-method",
        method,
        "--cluster-mean-threshold-deg", str(cluster_mean_threshold_deg),
        "--cluster-max-threshold-deg", str(cluster_max_threshold_deg),
        "--cluster-energy-window", str(cluster_energy_window),
        "--hybrid-max-candidates", str(hybrid_max_candidates),
        "--hybrid-min-cluster-size", str(hybrid_min_cluster_size),
        "--hybrid-min-samples", str(hybrid_min_samples),
        "--hybrid-energy-neighbors", str(hybrid_energy_neighbors),
        "--hybrid-min-separation-deg", str(hybrid_min_separation_deg),
        "--energy-timeout", str(energy_timeout),
        "--tui" if tui else "--no-tui",
    ]
    if not geometric_prescreen:
        command.append("--no-geometric-prescreen")
    if early_stop:
        command.append("--early-stop")
    if save_generation_xyz:
        command.append("--save-generation-xyz")
    if validate_only:
        command.append("--validate-only")
    if backend == "xtb":
        if not tools.xtb and not validate_only:
            raise RuntimeError("xTB is not configured; run `seeker configure`")
        if tools.xtb:
            command.extend(("--xtb-command", tools.xtb))
        command.extend(("--xtb-method", xtb_method, "--xtb-threads", str(xtb_threads)))
    elif backend == "pyscf":
        command.extend(("--pyscf-basis", pyscf_basis))
    elif backend == "external":
        if "{xyz}" not in external_command and not validate_only:
            raise RuntimeError(
                "the external backend requires a command template containing {xyz}"
            )
        command.extend(
            ("--external-command", external_command, "--external-regex", external_regex,
             "--external-unit", external_unit)
        )
    return command


def build_post_commands(
    output: Path,
    tools: LocalConfig,
    *,
    enabled: bool,
    single_points: bool = False,
    optimization_backend: str | None = None,
    method: str = "gfn2",
    threads: int = 1,
    jobs: int = 4,
    timeout: float = 1800.0,
    opt_level: str = "tight",
    charge: int = 0,
    multiplicity: int = 1,
    rmsd_threshold: float = 0.30,
    energy_window: float = 10.0,
    atom_mode: str = "all",
    permutation_mode: str = "equivalent",
    b3lyp_filter_window: float = 10.0,
    gaussian_jobs: int | None = None,
    gaussian_nprocshared: int | None = None,
    gaussian_mem_gb: int | None = None,
    gaussian_timeout: float = 3600.0,
    constraint_mode: str = "free",
    topology_tolerance: float = 0.45,
) -> list[list[str]]:
    final_backend = optimization_backend or ("xtb" if enabled else "none")
    if final_backend not in {"none", "xtb", "gaussian"}:
        raise ValueError("post-optimization backend must be none, xtb, or gaussian")
    if not single_points and final_backend == "none":
        return []
    if final_backend == "xtb" and not tools.xtb:
        raise RuntimeError("xTB is not configured; run `seeker configure`")
    if (single_points or final_backend == "gaussian") and not tools.gaussian:
        raise RuntimeError("Gaussian is not configured; run `seeker configure`")
    if final_backend != "gaussian" and constraint_mode != "free":
        raise ValueError("exact active-only constraints require B3LYP optimization")
    postopt = output / "postoptimization"
    commands: list[list[str]] = []
    optimization_input = output / "selected_candidates"
    if single_points:
        single_point_output = postopt / "single_points"
        command = [
            "single-point", "--input-dir", str(optimization_input),
            "--output-dir", str(single_point_output),
            "--gaussian-command", tools.gaussian,
            "--timeout", str(gaussian_timeout),
            "--charge", str(charge), "--multiplicity", str(multiplicity),
            "--energy-window", str(b3lyp_filter_window),
        ]
        if gaussian_jobs is not None:
            command.extend(("--jobs", str(gaussian_jobs)))
        if gaussian_nprocshared is not None:
            command.extend(("--gaussian-nprocshared", str(gaussian_nprocshared)))
        if gaussian_mem_gb is not None:
            command.extend(("--gaussian-mem-gb", str(gaussian_mem_gb)))
        commands.append(command)
        optimization_input = single_point_output / "filtered_xyz"
    if final_backend == "none":
        return commands
    if final_backend == "xtb":
        commands.append([
            "optimize", "--input-dir", str(output / "selected_candidates"),
            "--output-dir", str(postopt), "--backend", "xtb",
            "--xtb-command", tools.xtb,
            "--xtb-method", method, "--xtb-threads", str(threads),
            "--jobs", str(jobs), "--timeout", str(timeout),
            "--opt-level", opt_level, "--charge", str(charge),
            "--multiplicity", str(multiplicity),
            "--permutation-mode", permutation_mode,
        ])
        commands[-1][commands[-1].index("--input-dir") + 1] = str(optimization_input)
    else:
        command = [
            "optimize", "--input-dir", str(optimization_input),
            "--output-dir", str(postopt), "--backend", "gaussian",
            "--gaussian-command", tools.gaussian,
            "--timeout", str(gaussian_timeout),
            "--charge", str(charge), "--multiplicity", str(multiplicity),
            "--constraint-mode", constraint_mode,
            "--run-manifest", str(output / "run_manifest.json"),
            "--permutation-mode", permutation_mode,
            "--topology-tolerance", str(topology_tolerance),
        ]
        if gaussian_jobs is not None:
            command.extend(("--jobs", str(gaussian_jobs)))
        if gaussian_nprocshared is not None:
            command.extend(("--gaussian-nprocshared", str(gaussian_nprocshared)))
        if gaussian_mem_gb is not None:
            command.extend(("--gaussian-mem-gb", str(gaussian_mem_gb)))
        commands.append(command)
    commands.append([
            "cluster-optimized", "--optimization-csv", str(postopt / "optimization.csv"),
            "--output-dir", str(postopt / "clustering"),
            "--rmsd-threshold", str(rmsd_threshold),
            "--energy-window", str(energy_window), "--atom-mode", atom_mode,
            "--permutation-mode", permutation_mode,
        ])
    return commands


def write_launch_config(
    output: Path,
    command: Sequence[str],
    post_commands: Sequence[Sequence[str]] = (),
    *,
    execution: str = "foreground",
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    target = output / "launch_config.txt"
    payload = {
        "format": "seeker.launch.v2",
        "command": ["seeker", *command],
        "commands": [["seeker", *command]] + [
            ["seeker", *item] for item in post_commands
        ],
        "execution": execution,
        "working_directory": str(Path.cwd()),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def execute_launch_plan(path: str | Path) -> int:
    """Execute every command in a persisted launch plan in order."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    commands = payload.get("commands") or [payload.get("command")]
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"invalid launch plan: {path}")
    from .cli import main

    for raw in commands:
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"invalid command in launch plan: {path}")
        command = [str(item) for item in raw]
        if command[0] == "seeker":
            command = command[1:]
        status = main(command)
        if status:
            return status
    return 0


def launch(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    if getattr(args, "execute_plan", None):
        return execute_launch_plan(args.execute_plan)
    chrome = _show_launcher_intro(
        interactive=not args.non_interactive,
        output_fn=output_fn,
    )
    path = config_path()
    if not path.is_file() and not args.non_interactive:
        output_fn("No machine-local configuration was found; starting setup.")
        configure_interactively(path, input_fn=input_fn, output_fn=output_fn)
    tools = resolve_tools()

    xyz_text = args.xyz or _ask("XYZ file", "", input_fn)
    if not xyz_text:
        raise ValueError("an XYZ file is required")
    xyz = Path(xyz_text).expanduser().resolve()
    if not xyz.is_file():
        raise FileNotFoundError(f"XYZ file not found: {xyz}")
    genes_default = xyz.parent / "genes.txt"
    if not genes_default.is_file() and (xyz.parent / "GENES.txt").is_file():
        genes_default = xyz.parent / "GENES.txt"
    genes_text = args.genes or _ask("GENES file", str(genes_default), input_fn)
    genes = Path(genes_text).expanduser().resolve()
    if not genes.is_file():
        raise FileNotFoundError(f"GENES file not found: {genes}")

    molecule = read_xyz(xyz)
    specs = read_coordinate_specs(genes)
    has_pose = any(isinstance(item, FragmentPoseGene) for item in specs)
    has_torsion = any(isinstance(item, Gene) for item in specs)
    coordinate_count = sum(
        6 if isinstance(item, FragmentPoseGene)
        else 1
        for item in specs
    )
    output_fn(
        f"Input loaded: {len(molecule.atoms)} atoms · {coordinate_count} genetic coordinates"
    )
    hbond_pi_recommendation = _hbond_pi_recommendation(molecule, genes)
    hbond_pi_option = getattr(args, "hbond_pi", None)
    if hbond_pi_option is True and hbond_pi_recommendation is None:
        raise ValueError(
            "--hbond-pi was requested, but the reference has no eligible "
            "X-H donor/pi-ring pair"
        )

    if has_pose and has_torsion:
        raise ValueError("POSE currently requires a pose-only coordinate set")

    output = Path(
        args.output or _ask("New output directory", str(default_output_path(xyz)), input_fn)
    ).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    # Input-only validation remains available as the explicit CLI flag
    # --validate-only, but it is not part of the normal search wizard.
    validate_only = bool(args.validate_only)
    backend_default = "xtb"
    backend = args.backend or (
        backend_default
        if args.non_interactive
        else _ask_energy_backend(backend_default, tools, input_fn, output_fn)
    )

    extra_objectives = (
        "hbond_pi"
        if (
            hbond_pi_option is True
            or (hbond_pi_option is None and hbond_pi_recommendation is not None)
        )
        else ""
    )
    try:
        DisconnectedComponentsPenaltyModel.from_reference(
            molecule,
            build_bond_graph(molecule),
        )
        disconnected_components_available = True
    except ValueError:
        disconnected_components_available = False
    disconnected_components_option = getattr(args, "disconnected_components_penalty", None)
    if disconnected_components_option is True and not disconnected_components_available:
        raise ValueError(
            "--disconnected-components-penalty requires an intermolecular input with "
            "at least two covalent fragments"
        )
    if disconnected_components_option is True:
        extra_objectives = ",".join(
            item
            for item in (extra_objectives, "disconnected_components_penalty")
            if item
        )
    if not args.non_interactive:
        if hbond_pi_recommendation is not None:
            output_fn(f"X-H...pi applicability: {hbond_pi_recommendation}.")
        extra_objectives = _ask_optional_objectives(
            [item for item in extra_objectives.split(",") if item],
            hbond_pi_available=hbond_pi_recommendation is not None,
            disconnected_components_available=disconnected_components_available,
            input_fn=input_fn,
            output_fn=output_fn,
        )

    preset = recommend_search_preset(
        specs,
        objective_count=2
        + len([item for item in extra_objectives.split(",") if item]),
    )
    recommended = preset.launcher_settings()
    settings: dict[str, object] = {
        **recommended,
        "charge": 0,
        "multiplicity": 1,
        "workers": max(1, min(4, (os.cpu_count() or 1) // 2)),
        "seed": 7,
        "migration_selection": "pareto",
        "initial_pool_factor": 20,
        "initial_pool_sampling": "latin_hypercube",
        "initial_scan_layout": "tensor",
        "initial_scan_grid": "uniform",
        "initial_scan_points_mode": "fixed",
        "initial_scan_points": 3,
        "base_mutation_weight": 0.75,
        "base_crossover_weight": 0.55,
        "mutation_weight_amplitude": 0.15,
        "crossover_weight_amplitude": 0.10,
        "operator_oscillations": 2,
        "periodicity_grid_step_deg": 20.0,
        "sbx_eta": 15.0,
        "duplicate_mean_threshold_deg": 3.0,
        "duplicate_max_threshold_deg": 3.0,
        "max_duplicate_attempts": 30,
        "geometric_prescreen": True,
        "topology_tolerance": 0.45,
        "hbond_cutoff": 3.2,
        "hbond_contact_threshold": -0.30,
        "hh_clash_distance": 1.40,
        "steric_hh_scale": 0.55,
        "steric_heavy_heavy_scale": 0.55,
        "steric_hydrogen_heavy_scale": 0.50,
        "steric_exclude_hops": 3,
        "extra_objectives": extra_objectives,
        "rotor_symmetry_sigma": 0.15,
        "rotor_anisotropy_sigma": 0.15,
        "early_stop": False,
        "early_stop_patience": 8,
        "early_stop_min_delta": 1.0e-6,
        "early_stop_diversity_deg": 8.0,
        "early_stop_min_generations": 0,
        "archive_stagnation_patience": 0,
        "clustering_source": "archive",
        "clustering_method": (
            "pose_hybrid" if has_pose else "hybrid"
        ),
        "cluster_mean_threshold_deg": 15.0,
        "cluster_max_threshold_deg": 15.0,
        "cluster_energy_window": 10.0,
        "hybrid_min_cluster_size": 5,
        "hybrid_min_samples": 2,
        "hybrid_energy_neighbors": 8,
        "hybrid_min_separation_deg": 25.0,
        "save_generation_xyz": False,
        "xtb_method": "gfn2",
        "xtb_threads": 1,
        "energy_timeout": 600.0,
        "pyscf_basis": "sto-3g",
        "external_command": "",
        "external_regex": r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
        "external_unit": "hartree",
    }
    if has_pose:
        settings["initialization_strategy"] = "random"

    advanced = bool(getattr(args, "advanced", False))
    if not args.non_interactive:
        output_fn("")
        output_fn("Search preset")
        output_fn(f"Decision: {preset.family} / {preset.profile}")
        for reason in preset.decision_path:
            output_fn(f"  · {reason}")
        output_fn(
            f"Recommended: {settings['population']} individuals/island · "
            f"{settings['islands']} islands · {settings['generations']} generations"
        )
        output_fn(
            f"Maximum budget: about {preset.maximum_evaluations:,} energy evaluations "
            "before cache/deduplication"
        )
        if not _yes_no("Apply the recommended preset", True, input_fn):
            settings["population"] = _ask_int(
                "Population per island", int(settings["population"]), 2,
                input_fn, output_fn,
            )
            settings["offspring"] = _ask_int(
                "Offspring per generation", int(settings["offspring"]), 1,
                input_fn, output_fn,
            )
            settings["generations"] = _ask_int(
                "Generations", int(settings["generations"]), 0,
                input_fn, output_fn,
            )
            settings["islands"] = _ask_int(
                "Number of islands", int(settings["islands"]), 1,
                input_fn, output_fn,
            )
            settings["migration_size"] = min(
                4, max(1, int(settings["population"]) // 12)
            )
            settings["migration_interval"] = 10 if int(settings["islands"]) > 1 else 0
        advanced = advanced or _yes_no("Open advanced scientific settings", False, input_fn)

        settings["charge"] = _ask_int("Molecular charge", 0, -100, input_fn, output_fn)
        settings["multiplicity"] = _ask_int("Multiplicity", 1, 1, input_fn, output_fn)
        settings["workers"] = _ask_int(
            "Parallel workers", int(settings["workers"]), 1, input_fn, output_fn
        )
        settings["seed"] = _ask_int("Random seed", 7, 0, input_fn, output_fn)

    if advanced and not args.non_interactive:
        output_fn("\n1. Initial population and genetic operators")
        initial_choices = (
            ("maximin", "random")
            if has_pose
            else ("scan", "lhs", "maximin", "random")
        )
        settings["initialization_strategy"] = _ask(
            "Initialization", str(settings["initialization_strategy"]), input_fn,
            choices=initial_choices,
        )
        if settings["initialization_strategy"] == "scan":
            settings["initial_scan_layout"] = _ask(
                "SCAN layout", "tensor", input_fn, choices=("tensor", "one-at-a-time")
            )
            settings["initial_scan_grid"] = _ask(
                "SCAN grid", "periodicity-modes", input_fn,
                choices=("uniform", "periodicity-modes"),
            )
            settings["initial_scan_points_mode"] = (
                "periodicity" if settings["initial_scan_grid"] == "periodicity-modes" else "fixed"
            )
            if settings["initial_scan_points_mode"] == "fixed":
                settings["initial_scan_points"] = _ask_int(
                    "Uniform points per coordinate", 3, 2, input_fn, output_fn
                )
        elif settings["initialization_strategy"] == "maximin":
            settings["initial_pool_sampling"] = _ask(
                "Pool sampling", "latin_hypercube", input_fn,
                choices=("latin_hypercube", "prior"),
            )
            settings["initial_pool_factor"] = _ask_int(
                "Pool-size factor", 20, 1, input_fn, output_fn
            )
        if int(settings["islands"]) > 1:
            settings["migration_interval"] = _ask_int(
                "Migration interval", int(settings["migration_interval"]), 1, input_fn, output_fn
            )
            settings["migration_size"] = _ask_int(
                "Migrants per island", int(settings["migration_size"]), 1, input_fn, output_fn
            )
            if int(settings["migration_size"]) >= int(settings["population"]):
                raise ValueError("migrants must be fewer than the per-island population")
            settings["migration_selection"] = _ask(
                "Migrant selection", "pareto", input_fn, choices=("pareto", "random")
            )
        for key, label, default, minimum in (
            ("base_mutation_weight", "Base mutation weight", 0.75, 0.0),
            ("base_crossover_weight", "Base crossover weight", 0.55, 0.0),
            ("mutation_weight_amplitude", "Mutation oscillation amplitude", 0.15, 0.0),
            ("crossover_weight_amplitude", "Crossover oscillation amplitude", 0.10, 0.0),
            ("periodicity_grid_step_deg", "Periodic-prior step (degrees)", 20.0, 0.001),
            ("sbx_eta", "SBX crossover eta", 15.0, 0.001),
            ("duplicate_mean_threshold_deg", "Duplicate mean threshold (degrees)", 3.0, 0.0),
            ("duplicate_max_threshold_deg", "Duplicate maximum threshold (degrees)", 3.0, 0.0),
        ):
            settings[key] = _ask_float(label, default, input_fn, output_fn, minimum=minimum)
        settings["operator_oscillations"] = _ask_int(
            "Operator half-oscillations", 2, 0, input_fn, output_fn
        )
        settings["max_duplicate_attempts"] = _ask_int(
            "Maximum deduplication attempts", 30, 1, input_fn, output_fn
        )

        output_fn("\n2. Geometric prescreen and objectives")
        settings["geometric_prescreen"] = _yes_no("Geometric prescreen", True, input_fn)
        if settings["geometric_prescreen"]:
            for key, label, default in (
                ("topology_tolerance", "Topology tolerance (angstrom)", 0.45),
                ("hbond_cutoff", "H-bond cutoff (angstrom)", 3.2),
                ("hh_clash_distance", "H-H clash distance (angstrom)", 1.40),
                ("steric_hh_scale", "H-H steric scale", 0.55),
                ("steric_heavy_heavy_scale", "Heavy-heavy steric scale", 0.55),
                ("steric_hydrogen_heavy_scale", "H-heavy steric scale", 0.50),
            ):
                settings[key] = _ask_float(label, default, input_fn, output_fn, minimum=0.0)
            settings["hbond_contact_threshold"] = _ask_float(
                "H-bond contact threshold", -0.30, input_fn, output_fn
            )
            settings["steric_exclude_hops"] = _ask_int(
                "Exclude pairs through N bonds", 3, 1, input_fn, output_fn
            )
        objectives = str(settings["extra_objectives"])
        if "disconnected_components_penalty" in objectives:
            if not disconnected_components_available:
                raise ValueError(
                    "disconnected_components_penalty requires an intermolecular input with "
                    "at least two covalent fragments"
                )
        if any(name in objectives for name in ("rotor_prolate", "rotor_oblate", "rotor_spherical")):
            settings["rotor_symmetry_sigma"] = _ask_float(
                "Rotor symmetry sigma", 0.15, input_fn, output_fn, minimum=0.001
            )
            settings["rotor_anisotropy_sigma"] = _ask_float(
                "Rotor anisotropy sigma", 0.15, input_fn, output_fn, minimum=0.001
            )

        output_fn("\n3. Stopping and clustering")
        settings["early_stop"] = _yes_no("Enable fitness/diversity early stopping", False, input_fn)
        if settings["early_stop"]:
            settings["early_stop_patience"] = _ask_int("Patience generations", 8, 1, input_fn, output_fn)
            settings["early_stop_min_delta"] = _ask_float("Minimum fitness improvement", 1e-6, input_fn, output_fn, minimum=0.0)
            settings["early_stop_diversity_deg"] = _ask_float("Diversity threshold (degrees)", 8.0, input_fn, output_fn, minimum=0.0)
            settings["early_stop_min_generations"] = _ask_int("Minimum generation before stopping", 0, 0, input_fn, output_fn)
        settings["archive_stagnation_patience"] = _ask_int("Archive stagnation patience (0 disables)", 0, 0, input_fn, output_fn)
        settings["clustering_source"] = _ask("Clustering source", "archive", input_fn, choices=("archive", "final_population", "pareto_front"))
        allowed_methods = ("pose_hybrid",) if has_pose else ("hybrid", "periodicity_cells", "complete_linkage")
        settings["clustering_method"] = _ask("Clustering method", str(settings["clustering_method"]), input_fn, choices=allowed_methods)
        if settings["clustering_method"] == "complete_linkage":
            settings["cluster_mean_threshold_deg"] = _ask_float("Mean torsion threshold (degrees)", 15.0, input_fn, output_fn, minimum=0.0)
            settings["cluster_max_threshold_deg"] = _ask_float("Maximum torsion threshold (degrees)", 15.0, input_fn, output_fn, minimum=0.0)
        if "hybrid" in str(settings["clustering_method"]):
            settings["hybrid_max_candidates"] = _ask_int("Maximum final candidates", int(settings["hybrid_max_candidates"]), 1, input_fn, output_fn)
            settings["hybrid_min_cluster_size"] = _ask_int("HDBSCAN minimum cluster size", 5, 2, input_fn, output_fn)
            settings["hybrid_min_samples"] = _ask_int("HDBSCAN min_samples", 2, 1, input_fn, output_fn)
            settings["hybrid_energy_neighbors"] = _ask_int("Energy-graph neighbors", 8, 1, input_fn, output_fn)
            settings["hybrid_min_separation_deg"] = _ask_float("Minimum discovery separation (degrees)", 25.0, input_fn, output_fn, minimum=0.0)
        settings["cluster_energy_window"] = _ask_float("Clustering energy window (kcal/mol)", 10.0, input_fn, output_fn, minimum=0.0)
        settings["save_generation_xyz"] = _yes_no("Save generation XYZ snapshots", False, input_fn)

    if not args.non_interactive:
        output_fn("\nEnergy settings")
        settings["energy_timeout"] = _ask_float(
            "Energy timeout (s)", 600.0, input_fn, output_fn, minimum=0.1
        ) if advanced else 600.0
        if backend == "xtb":
            settings["xtb_method"] = _ask(
                "xTB method", "gfn2", input_fn, choices=("gfn2", "gfn1", "gfn0", "gfnff")
            )
            settings["xtb_threads"] = _ask_int("xTB threads per geometry", 1, 1, input_fn, output_fn) if advanced else 1
        elif backend == "pyscf":
            settings["pyscf_basis"] = _ask("PySCF basis", "sto-3g", input_fn)
        else:
            while "{xyz}" not in str(settings["external_command"]):
                settings["external_command"] = _ask(
                    "External command containing {xyz}", "", input_fn
                )
                if "{xyz}" not in str(settings["external_command"]):
                    output_fn(
                        "The external adapter needs a command template containing {xyz}."
                    )
            settings["external_regex"] = _ask(
                "Energy regex with one capture group",
                str(settings["external_regex"]), input_fn,
            )
            settings["external_unit"] = _ask("External energy unit", "hartree", input_fn, choices=("hartree", "kcal_mol", "kj_mol", "ev"))

    post_single_points = bool(getattr(args, "post_single_points", False))
    requested_post_backend = getattr(args, "post_optimize_backend", None)
    final_optimization_backend = (
        str(requested_post_backend)
        if requested_post_backend
        else "xtb"
        if bool(getattr(args, "post_optimize", False))
        else "none"
    )
    post_jobs = int(settings["workers"])
    post_timeout = 1800.0
    opt_level = "tight"
    rmsd_threshold = 0.30
    atom_mode = "all"
    b3lyp_filter_window = float(getattr(args, "b3lyp_filter_window", 10.0))
    constraint_mode = str(getattr(args, "post_constraint_mode", "free"))
    gaussian_defaults = recommend_gaussian_resources()
    gaussian_jobs = getattr(args, "gaussian_jobs", None) or gaussian_defaults.jobs
    gaussian_nproc = (
        getattr(args, "gaussian_nprocshared", None)
        or gaussian_defaults.nprocshared
    )
    gaussian_mem = getattr(args, "gaussian_mem_gb", None) or gaussian_defaults.mem_gb
    gaussian_timeout = float(getattr(args, "gaussian_timeout", 3600.0))
    if validate_only:
        post_single_points = False
        final_optimization_backend = "none"
    if not validate_only and not args.non_interactive:
        if tools.gaussian:
            post_single_points = _yes_no(
                "Apply a B3LYP single-point filter before optimization", False, input_fn
            )
            if post_single_points:
                b3lyp_filter_window = _ask_float(
                    "B3LYP filter window (kcal/mol)",
                    10.0,
                    input_fn,
                    output_fn,
                    minimum=0.0,
                )
        post_choices = tuple(
            name
            for name, available in (
                ("xtb", bool(tools.xtb)),
                ("gaussian", bool(tools.gaussian)),
                ("none", True),
            )
            if available
        )
        default_post_backend = "xtb" if tools.xtb else "none"
        final_optimization_backend = _ask(
            "Final optimization backend",
            default_post_backend,
            input_fn,
            choices=post_choices,
        )
        if final_optimization_backend == "gaussian":
            output_fn(f"  Fixed electronic structure model: {B3LYP_MODEL}")
            constraint_mode = (
                "active"
                if _yes_no(
                    "Optimize only explored D and intermolecular POSE degrees of freedom",
                    False,
                    input_fn,
                )
                else "free"
            )
        else:
            constraint_mode = "free"
        if final_optimization_backend == "xtb" and advanced:
            opt_level = _ask("xTB optimization accuracy", "tight", input_fn, choices=("normal", "tight", "verytight"))
            post_jobs = _ask_int("Parallel xTB optimizations", post_jobs, 1, input_fn, output_fn)
            post_timeout = _ask_float("Optimization timeout (s)", 1800.0, input_fn, output_fn, minimum=0.1)
        if post_single_points or final_optimization_backend == "gaussian":
            output_fn(
                "Gaussian resource recommendation: "
                f"{gaussian_jobs} jobs × {gaussian_nproc} cores × {gaussian_mem} GB "
                f"({gaussian_defaults.logical_cpus} CPUs detected)"
            )
            if advanced:
                gaussian_jobs = _ask_int(
                    "Parallel Gaussian jobs", gaussian_jobs, 1, input_fn, output_fn
                )
                gaussian_nproc = _ask_int(
                    "Gaussian nprocshared per job",
                    gaussian_nproc,
                    1,
                    input_fn,
                    output_fn,
                )
                gaussian_mem = _ask_int(
                    "Gaussian memory per job (GB)",
                    gaussian_mem,
                    1,
                    input_fn,
                    output_fn,
                )
                gaussian_timeout = _ask_float(
                    "Gaussian timeout per calculation (s)",
                    gaussian_timeout,
                    input_fn,
                    output_fn,
                    minimum=0.1,
                )
        if final_optimization_backend != "none" and advanced:
            rmsd_threshold = _ask_float("Final RMSD threshold (angstrom)", 0.30, input_fn, output_fn, minimum=0.0)
            atom_mode = _ask("RMSD atom mode", "all", input_fn, choices=("all", "heavy"))

    execution = "background" if bool(getattr(args, "background", False)) else "foreground"
    if not args.non_interactive and not validate_only:
        execution = _ask("Execution mode", execution, input_fn, choices=("foreground", "background"))

    command = build_launch_command(
        xyz=xyz,
        genes=genes,
        output=output,
        backend=backend,
        tools=tools,
        **settings,
        validate_only=validate_only,
        tui=not args.no_tui and execution == "foreground",
    )
    post_commands = build_post_commands(
        output,
        tools,
        enabled=final_optimization_backend == "xtb",
        single_points=post_single_points,
        optimization_backend=final_optimization_backend,
        method=str(settings["xtb_method"]),
        threads=int(settings["xtb_threads"]), jobs=post_jobs, timeout=post_timeout,
        opt_level=opt_level, charge=int(settings["charge"]),
        multiplicity=int(settings["multiplicity"]), rmsd_threshold=rmsd_threshold,
        energy_window=float(settings["cluster_energy_window"]), atom_mode=atom_mode,
        b3lyp_filter_window=b3lyp_filter_window,
        gaussian_jobs=gaussian_jobs,
        gaussian_nprocshared=gaussian_nproc,
        gaussian_mem_gb=gaussian_mem,
        gaussian_timeout=gaussian_timeout,
        constraint_mode=constraint_mode,
        topology_tolerance=float(settings["topology_tolerance"]),
    )
    output_fn("\nFinal configuration")
    output_fn("  coordinates       : native")
    output_fn(f"  search preset     : {preset.family} / {preset.profile}")
    output_fn(f"  population        : {settings['initialization_strategy']} · {settings['population']}/island")
    output_fn(f"  search            : {settings['islands']} islands · {settings['generations']} generations")
    output_fn(
        "  maximum budget    : "
        f"{int(settings['islands']) * (int(settings['population']) + int(settings['generations']) * int(settings['offspring'])):,} "
        "evaluations"
    )
    output_fn(f"  energy            : {backend}" + (f" / {settings['xtb_method']}" if backend == "xtb" else ""))
    output_fn(f"  extra objectives  : {settings['extra_objectives'] or 'none'}")
    output_fn(f"  clustering        : {settings['clustering_source']} / {settings['clustering_method']}")
    output_fn(
        f"  B3LYP prefilter   : {'yes' if post_single_points else 'no'}"
        + (f" / {b3lyp_filter_window:g} kcal/mol" if post_single_points else "")
    )
    output_fn(
        f"  post-optimization : {final_optimization_backend}"
        + (
            f" / {constraint_mode}"
            if final_optimization_backend == "gaussian"
            else ""
        )
    )
    output_fn(f"  execution         : {execution}")
    output_fn("Command: " + shlex.join(["seeker", *command]))
    for item in post_commands:
        output_fn("Then: " + shlex.join(["seeker", *item]))
    if not args.non_interactive and not _yes_no("Launch SEEKER", True, input_fn):
        output_fn("Launch cancelled; no directory was created.")
        chrome.close()
        return 0
    if args.dry_run:
        chrome.close()
        return 0
    chrome.close()
    plan = write_launch_config(output, command, post_commands, execution=execution)
    if execution == "background":
        log_path = output / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "seeker", "launch", "--execute-plan", str(plan)],
                cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (output / "run.pid").write_text(f"{process.pid}\n", encoding="utf-8")
        output_fn(f"SEEKER started in the background (PID {process.pid}).")
        output_fn(f"Log: {log_path}")
        output_fn(f"Follow: tail -f {shlex.quote(str(log_path))}")
        return 0

    try:
        return execute_launch_plan(plan)
    except BaseException:
        # Keep the reproducibility snapshot, but remove an otherwise empty
        # directory only when launch failed before producing any artifacts.
        entries = list(output.iterdir()) if output.is_dir() else []
        if entries == [output / "launch_config.txt"]:
            entries[0].unlink()
            output.rmdir()
        raise


def add_launcher_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    configure = subparsers.add_parser("configure", help="configure machine-local external tools")
    configure.add_argument("--config", type=Path)
    doctor = subparsers.add_parser("doctor", help="diagnose local dependencies")
    doctor.add_argument("--config", type=Path)
    launch_parser = subparsers.add_parser("launch", help="open the portable interactive launcher")
    launch_parser.add_argument("--xyz")
    launch_parser.add_argument("--genes")
    launch_parser.add_argument("--output")
    launch_parser.add_argument("--backend", choices=("xtb", "pyscf", "external"))
    launch_parser.add_argument("--advanced", action="store_true")
    launch_parser.add_argument(
        "--hbond-pi",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable the automatically recommended X-H...pi objective",
    )
    launch_parser.add_argument(
        "--disconnected-components-penalty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="penalize disconnected components in the full intermolecular interaction graph",
    )
    launch_parser.add_argument(
        "--post-optimize", action=argparse.BooleanOptionalAction, default=False
    )
    launch_parser.add_argument(
        "--post-single-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the fixed B3LYP single-point filter before optimization",
    )
    launch_parser.add_argument(
        "--post-optimize-backend", choices=("none", "xtb", "gaussian")
    )
    launch_parser.add_argument(
        "--post-constraint-mode", choices=("free", "active"), default="free"
    )
    launch_parser.add_argument("--b3lyp-filter-window", type=float, default=10.0)
    launch_parser.add_argument("--gaussian-jobs", type=int)
    launch_parser.add_argument("--gaussian-nprocshared", type=int)
    launch_parser.add_argument("--gaussian-mem-gb", type=int)
    launch_parser.add_argument("--gaussian-timeout", type=float, default=3600.0)
    launch_parser.add_argument("--background", action="store_true")
    launch_parser.add_argument("--execute-plan", type=Path, help=argparse.SUPPRESS)
    launch_parser.add_argument("--validate-only", action="store_true")
    launch_parser.add_argument("--no-tui", action="store_true")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("--non-interactive", action="store_true")
