"""Gaussian B3LYP post-processing for SEEKER candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from .geometry import BondGraph, build_bond_graph
from .input import read_coordinate_specs, read_xyz, write_xyz
from .models import Gene, Molecule


B3LYP_MODEL = "B3LYP/6-31+G* EmpiricalDispersion=GD3BJ"
HARTREE_TO_KCAL_MOL = 627.509474

_SCF_ENERGY = re.compile(
    r"SCF\s+Done:\s+E\([^)]*\)\s*=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GaussianResources:
    jobs: int
    nprocshared: int
    mem_gb: int
    logical_cpus: int
    total_memory_bytes: int


def _physical_memory_bytes() -> int:
    """Return physical RAM without adding a runtime dependency."""

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            capture_output=True,
            check=False,
        )
        value = int(completed.stdout.strip())
        if completed.returncode == 0 and value > 0:
            return value
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 4 * 1024**3


def recommend_gaussian_resources(
    logical_cpus: int | None = None,
    total_memory_bytes: int | None = None,
) -> GaussianResources:
    """Use at most roughly 75% of CPU and memory for concurrent Gaussian jobs."""

    cpus = max(1, int(logical_cpus or os.cpu_count() or 1))
    memory = max(1024**3, int(total_memory_bytes or _physical_memory_bytes()))
    usable_cpus = max(1, math.floor(cpus * 0.75))
    usable_memory_gb = max(1, math.floor(memory * 0.75 / 1024**3))
    cpu_jobs = usable_cpus // 2 if usable_cpus >= 2 else 1
    memory_jobs = usable_memory_gb // 4 if usable_memory_gb >= 4 else 1
    jobs = max(1, min(4, cpu_jobs, memory_jobs))
    nprocshared = max(1, usable_cpus // jobs)
    mem_gb = max(1, usable_memory_gb // jobs)
    return GaussianResources(jobs, nprocshared, mem_gb, cpus, memory)


def _validate_resources(jobs: int, nprocshared: int, mem_gb: int) -> None:
    if jobs < 1 or nprocshared < 1 or mem_gb < 1:
        raise ValueError("Gaussian jobs, nprocshared and memory must be at least 1")


def _resolve_executable(command: str) -> str:
    executable = shutil.which(command) if os.path.sep not in command else command
    if not executable or not Path(executable).is_file():
        raise ValueError(f"Gaussian executable not found: {command}")
    return str(Path(executable).resolve())


def _energy(text: str) -> float | None:
    matches = list(_SCF_ENERGY.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1).replace("D", "E").replace("d", "e"))


def _normal_termination(text: str) -> bool:
    return "normal termination of gaussian" in text.casefold()


def _last_orientation(text: str, source: Molecule) -> Molecule | None:
    lines = text.splitlines()
    blocks: list[list[tuple[float, float, float]]] = []
    for start, line in enumerate(lines):
        if "orientation:" not in line.casefold():
            continue
        separators: list[int] = []
        for index in range(start + 1, min(len(lines), start + len(source.atoms) + 20)):
            if re.fullmatch(r"\s*-{5,}\s*", lines[index]):
                separators.append(index)
                if len(separators) == 3:
                    break
        if len(separators) < 3:
            continue
        coordinates: list[tuple[float, float, float]] = []
        for row in lines[separators[1] + 1 : separators[2]]:
            fields = row.split()
            if len(fields) < 6:
                coordinates = []
                break
            try:
                coordinates.append(
                    (
                        float(fields[-3].replace("D", "E")),
                        float(fields[-2].replace("D", "E")),
                        float(fields[-1].replace("D", "E")),
                    )
                )
            except ValueError:
                coordinates = []
                break
        if len(coordinates) == len(source.atoms):
            blocks.append(coordinates)
    if not blocks:
        return None
    atoms = [atom.moved(position) for atom, position in zip(source.atoms, blocks[-1])]
    return source.with_atoms(atoms)


def _scientific_fingerprint(
    source: Path,
    *,
    route: str,
    charge: int,
    multiplicity: int,
    constraints: Sequence[str] = (),
    execution_signature: str = "",
) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(route.encode("utf-8"))
    digest.update(f"\n{charge} {multiplicity}\n".encode("ascii"))
    digest.update("\n".join(constraints).encode("utf-8"))
    digest.update(execution_signature.encode("utf-8"))
    return digest.hexdigest()


def _deck(
    molecule: Molecule,
    *,
    title: str,
    route: str,
    nprocshared: int,
    mem_gb: int,
    charge: int,
    multiplicity: int,
    checkpoint: str,
    constraints: Sequence[str] = (),
) -> str:
    lines = [
        f"%nprocshared={nprocshared}",
        f"%mem={mem_gb}GB",
        f"%chk={checkpoint}",
        f"#p {route}",
        "",
        title,
        "",
        f"{charge} {multiplicity}",
    ]
    lines.extend(
        f"{atom.element:<3s} {atom.x: .12f} {atom.y: .12f} {atom.z: .12f}"
        for atom in molecule.atoms
    )
    lines.append("")
    if constraints:
        lines.extend(constraints)
        lines.append("")
    return "\n".join(lines) + "\n"


def _run_gaussian(
    command: str,
    deck: str,
    work_dir: Path,
    timeout_seconds: float,
) -> tuple[int | str, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["GAUSS_SCRDIR"] = str(scratch)
    try:
        completed = subprocess.run(
            [command],
            cwd=work_dir,
            input=deck,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            env=env,
            check=False,
        )
        return completed.returncode, completed.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        return "timeout", output or ""


def _same_topology(left: BondGraph, right: BondGraph) -> bool:
    return len(left) == len(right) and all(a == b for a, b in zip(left, right))


def gaussian_active_constraints(
    source: Molecule,
    run_manifest: str | Path,
    *,
    topology_tolerance: float = 0.45,
) -> tuple[str, ...]:
    """Freeze all intramolecular coordinates except genetic torsional bonds."""

    manifest_path = Path(run_manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = payload.get("input", {})
    genes_path = Path(str(inputs.get("genes", ""))).expanduser()
    xyz_path = Path(str(inputs.get("xyz", ""))).expanduser()
    if not genes_path.is_file() or not xyz_path.is_file():
        raise ValueError("run manifest does not reference readable XYZ and GENES inputs")
    specs = read_coordinate_specs(genes_path)
    reference = read_xyz(xyz_path)
    if len(reference.atoms) != len(source.atoms) or [
        atom.element.upper() for atom in reference.atoms
    ] != [atom.element.upper() for atom in source.atoms]:
        raise ValueError("candidate atom count/order is incompatible with the run manifest")
    reference_graph = build_bond_graph(reference, topology_tolerance)
    source_graph = build_bond_graph(source, topology_tolerance)
    if not _same_topology(reference_graph, source_graph):
        raise ValueError("candidate topology is incompatible with the run manifest")
    active_bonds = {
        frozenset((item.atoms[1], item.atoms[2]))
        for item in specs
        if isinstance(item, Gene)
    }
    lines: list[str] = []
    for left, neighbours in enumerate(reference_graph):
        for right in sorted(index for index in neighbours if index > left):
            lines.append(f"B {left + 1} {right + 1} F")
    for center, neighbours in enumerate(reference_graph):
        for left, right in combinations(sorted(neighbours), 2):
            lines.append(f"A {left + 1} {center + 1} {right + 1} F")
    for center_left, neighbours in enumerate(reference_graph):
        for center_right in sorted(index for index in neighbours if index > center_left):
            if frozenset((center_left, center_right)) in active_bonds:
                continue
            left_candidates = sorted(reference_graph[center_left] - {center_right})
            right_candidates = sorted(reference_graph[center_right] - {center_left})
            pair = next(
                (
                    (left, right)
                    for left in left_candidates
                    for right in right_candidates
                    if len({left, center_left, center_right, right}) == 4
                ),
                None,
            )
            if pair is not None:
                lines.append(
                    f"D {pair[0] + 1} {center_left + 1} "
                    f"{center_right + 1} {pair[1] + 1} F"
                )
    return tuple(lines)


def _single_point_one(
    source: Path,
    output_dir: Path,
    *,
    command: str,
    nprocshared: int,
    mem_gb: int,
    timeout_seconds: float,
    charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    work_dir = output_dir / "work" / source.stem
    input_path = output_dir / "inputs" / f"{source.stem}.gjf"
    log_path = output_dir / "logs" / f"{source.stem}.log"
    result_path = work_dir / "result.json"
    route = f"{B3LYP_MODEL} NoSymm"
    fingerprint = _scientific_fingerprint(
        source,
        route=route,
        charge=charge,
        multiplicity=multiplicity,
        execution_signature=f"{command}|{nprocshared}|{mem_gb}",
    )
    if result_path.is_file() and log_path.is_file():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("status") == "ok":
                return {
                    "source": str(source),
                    "energy_hartree": f"{float(cached['energy_hartree']):.12f}",
                    "status": "ok",
                    "log": str(log_path),
                    "input": str(input_path),
                    "reused": True,
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    molecule = read_xyz(source, charge=charge, multiplicity=multiplicity)
    deck = _deck(
        molecule,
        title=f"SEEKER B3LYP single point {source.name}",
        route=route,
        nprocshared=nprocshared,
        mem_gb=mem_gb,
        charge=charge,
        multiplicity=multiplicity,
        checkpoint=f"{source.stem}.chk",
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(deck, encoding="utf-8")
    returncode, text = _run_gaussian(command, deck, work_dir, timeout_seconds)
    log_path.write_text(text, encoding="utf-8")
    energy = _energy(text)
    ok = returncode == 0 and energy is not None and _normal_termination(text)
    result_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "status": "ok" if ok else f"failed_{returncode}",
                "energy_hartree": energy,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(source),
        "energy_hartree": f"{energy:.12f}" if energy is not None else "",
        "status": "ok" if ok else f"failed_{returncode}",
        "log": str(log_path),
        "input": str(input_path),
        "reused": False,
    }


def single_point_gaussian_candidates(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    command: str,
    jobs: int | None = None,
    nprocshared: int | None = None,
    mem_gb: int | None = None,
    timeout_seconds: float = 1800.0,
    charge: int = 0,
    multiplicity: int = 1,
    energy_window_kcal_mol: float = 10.0,
) -> dict[str, int]:
    source_dir = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    sources = sorted(source_dir.glob("*.xyz"))
    if not sources:
        raise ValueError(f"no XYZ files for Gaussian single points in {source_dir}")
    if not math.isfinite(energy_window_kcal_mol) or energy_window_kcal_mol < 0.0:
        raise ValueError("B3LYP single-point energy window must be non-negative")
    recommended = recommend_gaussian_resources()
    actual_jobs = recommended.jobs if jobs is None else jobs
    actual_nproc = recommended.nprocshared if nprocshared is None else nprocshared
    actual_mem = recommended.mem_gb if mem_gb is None else mem_gb
    _validate_resources(actual_jobs, actual_nproc, actual_mem)
    executable = _resolve_executable(command)
    destination.mkdir(parents=True, exist_ok=True)
    filtered = destination / "filtered_xyz"
    if filtered.exists():
        shutil.rmtree(filtered)
    filtered.mkdir(parents=True)
    (destination / "single_point_config.json").write_text(
        json.dumps(
            {
                "schema": "seeker.gaussian.single-point.v1",
                "backend": "gaussian",
                "model": B3LYP_MODEL,
                "command": executable,
                "jobs": actual_jobs,
                "nprocshared": actual_nproc,
                "mem_gb": actual_mem,
                "timeout_seconds": timeout_seconds,
                "charge": charge,
                "multiplicity": multiplicity,
                "energy_window_kcal_mol": energy_window_kcal_mol,
                "input_dir": str(source_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=actual_jobs) as executor:
        futures = {
            executor.submit(
                _single_point_one,
                source,
                destination,
                command=executable,
                nprocshared=actual_nproc,
                mem_gb=actual_mem,
                timeout_seconds=timeout_seconds,
                charge=charge,
                multiplicity=multiplicity,
            ): source
            for source in sources
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:04d}/{len(sources):04d}] {row['status']:>14s} "
                f"{futures[future].name}",
                flush=True,
            )
    valid = [row for row in rows if row["status"] == "ok"]
    minimum = (
        min(float(row["energy_hartree"]) for row in valid) if valid else None
    )
    for row in rows:
        if row["status"] != "ok":
            row["delta_energy_kcal_mol"] = ""
            row["filter_status"] = "calculation_failed"
            row["filtered_xyz"] = ""
            continue
        assert minimum is not None
        delta = (float(row["energy_hartree"]) - minimum) * HARTREE_TO_KCAL_MOL
        kept = delta <= energy_window_kcal_mol + 1.0e-12
        filtered_path = filtered / Path(str(row["source"])).name
        if kept:
            shutil.copy2(row["source"], filtered_path)
        row["delta_energy_kcal_mol"] = f"{delta:.8f}"
        row["filter_status"] = "kept" if kept else "outside_window"
        row["filtered_xyz"] = str(filtered_path) if kept else ""
    rows.sort(key=lambda row: row["source"])
    with (destination / "single_points.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "source",
            "energy_hartree",
            "delta_energy_kcal_mol",
            "status",
            "filter_status",
            "filtered_xyz",
            "input",
            "log",
            "reused",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if not valid:
        raise RuntimeError("no valid Gaussian single-point result was produced")
    return {
        "total": len(rows),
        "succeeded": len(valid),
        "failed": len(rows) - len(valid),
        "kept": sum(row["filter_status"] == "kept" for row in rows),
        "outside_window": sum(
            row["filter_status"] == "outside_window" for row in rows
        ),
        "reused": sum(bool(row["reused"]) for row in rows),
    }


def optimize_gaussian_one(
    source: Path,
    output_dir: Path,
    *,
    command: str,
    nprocshared: int,
    mem_gb: int,
    timeout_seconds: float,
    charge: int,
    multiplicity: int,
    constraint_mode: str,
    run_manifest: str | Path | None,
    topology_tolerance: float,
) -> dict[str, Any]:
    molecule = read_xyz(source, charge=charge, multiplicity=multiplicity)
    constraints = (
        gaussian_active_constraints(
            molecule, run_manifest, topology_tolerance=topology_tolerance
        )
        if constraint_mode == "active" and run_manifest is not None
        else ()
    )
    route = (
        f"{B3LYP_MODEL} NoSymm Opt=ModRedundant"
        if constraints
        else f"{B3LYP_MODEL} NoSymm Opt"
    )
    fingerprint = _scientific_fingerprint(
        source,
        route=route,
        charge=charge,
        multiplicity=multiplicity,
        constraints=constraints,
        execution_signature=f"{command}|{nprocshared}|{mem_gb}",
    )
    work_dir = output_dir / "work" / source.stem
    input_path = output_dir / "inputs" / f"{source.stem}.gjf"
    log_path = output_dir / "logs" / f"{source.stem}.log"
    stable_xyz = work_dir / "optimized.xyz"
    result_path = work_dir / "result.json"
    optimized_path = (
        output_dir
        / "optimized_xyz"
        / ".unclassified"
        / f"{source.stem}_b3lypopt.xyz"
    )
    if result_path.is_file() and stable_xyz.is_file() and log_path.is_file():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("status") == "ok":
                optimized_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stable_xyz, optimized_path)
                return {
                    "source": str(source),
                    "optimized_xyz": str(optimized_path),
                    "energy_hartree": f"{float(cached['energy_hartree']):.12f}",
                    "status": "ok",
                    "log": str(log_path),
                    "input": str(input_path),
                    "backend": "gaussian",
                    "method": B3LYP_MODEL,
                    "constraint_mode": constraint_mode,
                    "reused": True,
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    deck = _deck(
        molecule,
        title=f"SEEKER B3LYP optimization {source.name}",
        route=route,
        nprocshared=nprocshared,
        mem_gb=mem_gb,
        charge=charge,
        multiplicity=multiplicity,
        checkpoint=f"{source.stem}.chk",
        constraints=constraints,
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(deck, encoding="utf-8")
    returncode, text = _run_gaussian(command, deck, work_dir, timeout_seconds)
    log_path.write_text(text, encoding="utf-8")
    energy = _energy(text)
    optimized = _last_orientation(text, molecule)
    converged = (
        "optimization completed" in text.casefold()
        or "stationary point found" in text.casefold()
    )
    ok = (
        returncode == 0
        and energy is not None
        and optimized is not None
        and converged
        and _normal_termination(text)
    )
    if ok and optimized is not None and energy is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        write_xyz(
            stable_xyz,
            optimized,
            f"SEEKER B3LYP optimized E_Ha={energy:.12f} source={source.name}",
        )
        optimized_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stable_xyz, optimized_path)
    result_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "status": "ok" if ok else f"failed_{returncode}",
                "energy_hartree": energy,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(source),
        "optimized_xyz": str(optimized_path) if ok else "",
        "energy_hartree": f"{energy:.12f}" if energy is not None else "",
        "status": "ok" if ok else f"failed_{returncode}",
        "log": str(log_path),
        "input": str(input_path),
        "backend": "gaussian",
        "method": B3LYP_MODEL,
        "constraint_mode": constraint_mode,
        "reused": False,
    }
