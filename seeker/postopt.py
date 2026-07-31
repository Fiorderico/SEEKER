"""Geometry optimization of the candidates selected by SEEKER."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .gaussian import (
    B3LYP_MODEL,
    gaussian_active_constraints,
    optimize_gaussian_one,
    recommend_gaussian_resources,
)
from .input import read_xyz, write_xyz
from .postopt_cluster import cluster_optimized_candidates, compare_structures


_XTB_ENERGY = re.compile(
    r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
    re.IGNORECASE,
)


def _method_args(method: str) -> list[str]:
    normalized = method.lower().replace("-", "")
    if normalized == "gfnff":
        return ["--gfnff"]
    if normalized in {"gfn0", "gfn1", "gfn2"}:
        return ["--gfn", normalized[-1]]
    raise ValueError(f"unsupported xTB method: {method}")


def _energy(text: str) -> float | None:
    matches = list(_XTB_ENERGY.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1).replace("D", "E").replace("d", "e"))


def _optimize_one(
    source: Path,
    output_dir: Path,
    *,
    command: str,
    method: str,
    threads: int,
    timeout_seconds: float,
    opt_level: str,
    charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    work_dir = output_dir / "work" / source.stem
    log_path = output_dir / "logs" / f"{source.stem}.log"
    optimized_path = (
        output_dir / "optimized_xyz" / ".unclassified" / f"{source.stem}_{method}opt.xyz"
    )
    # A failed xTB optimization can leave restart files behind.  They must not
    # affect an explicit retry of the same post-clustering candidate.
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.xyz"
    shutil.copy2(source, input_path)

    argv = [
        command,
        input_path.name,
        *_method_args(method),
        "--opt",
        opt_level,
        "--chrg",
        str(charge),
        "--uhf",
        str(max(0, multiplicity - 1)),
        "--parallel",
        str(threads),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        completed = subprocess.run(
            argv,
            cwd=work_dir,
            text=True,
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            env=env,
            check=False,
        )
        text = completed.stdout + "\n" + completed.stderr
        returncode: int | str = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        text = (stdout or "") + "\n" + (stderr or "")
        returncode = "timeout"
    log_path.write_text(text, encoding="utf-8")

    energy = _energy(text)
    raw_optimized = work_dir / "xtbopt.xyz"
    ok = returncode == 0 and energy is not None and raw_optimized.is_file()
    if ok:
        molecule = read_xyz(raw_optimized, charge=charge, multiplicity=multiplicity)
        write_xyz(
            optimized_path,
            molecule,
            f"SEEKER {method}-xTB optimized E_Ha={energy:.12f} source={source.name}",
        )
    return {
        "source": str(source),
        "optimized_xyz": str(optimized_path) if ok else "",
        "energy_hartree": f"{energy:.12f}" if energy is not None else "",
        "status": "ok" if ok else f"failed_{returncode}",
        "log": str(log_path),
        "input": "",
        "backend": "xtb",
        "method": method,
        "constraint_mode": "free",
        "reused": False,
    }


def optimize_xtb_candidates(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    command: str = "xtb",
    method: str = "gfn2",
    jobs: int = 1,
    threads: int = 1,
    timeout_seconds: float = 600.0,
    opt_level: str = "tight",
    charge: int = 0,
    multiplicity: int = 1,
    conformer_change_rmsd_angstrom: float = 0.75,
    dedup_rmsd_threshold_angstrom: float = 0.30,
    comparison_atom_mode: str = "all",
    permutation_mode: str = "equivalent",
    topology_tolerance: float = 0.45,
) -> dict[str, int]:
    source_dir = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    sources = sorted(source_dir.glob("*.xyz"))
    if not sources:
        raise ValueError(f"no XYZ files to optimize in {source_dir}")
    if jobs < 1 or threads < 1:
        raise ValueError("jobs and threads must be at least 1")
    if (
        not math.isfinite(conformer_change_rmsd_angstrom)
        or conformer_change_rmsd_angstrom <= 0.0
    ):
        raise ValueError("conformer-change RMSD threshold must be positive")
    if (
        not math.isfinite(dedup_rmsd_threshold_angstrom)
        or dedup_rmsd_threshold_angstrom <= 0.0
    ):
        raise ValueError("deduplication RMSD threshold must be positive")
    if not math.isfinite(topology_tolerance) or topology_tolerance <= 0.0:
        raise ValueError("topology tolerance must be positive")
    if comparison_atom_mode not in {"heavy", "all"}:
        raise ValueError("post-optimization RMSD atom mode must be heavy or all")
    if permutation_mode not in {"equivalent", "ordered"}:
        raise ValueError("permutation mode must be equivalent or ordered")
    executable = shutil.which(command) if os.path.sep not in command else command
    if not executable or not Path(executable).is_file():
        raise ValueError(f"xTB executable not found: {command}")
    destination.mkdir(parents=True, exist_ok=True)
    optimized_root = destination / "optimized_xyz"
    optimized_root.mkdir(parents=True, exist_ok=True)
    for previous_dir in (
        optimized_root / ".unclassified",
        optimized_root / "same_conformer",
        optimized_root / "changed_conformer",
        optimized_root / "unique_optimized_xyz",
    ):
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
    for legacy_xyz in optimized_root.glob("*.xyz"):
        legacy_xyz.unlink()
    deduplication_dir = destination / "deduplication"
    if deduplication_dir.exists():
        shutil.rmtree(deduplication_dir)
    (destination / "optimization_config.json").write_text(
        json.dumps(
            {
                "backend": "xtb",
                "command": str(Path(executable).resolve()),
                "method": method,
                "jobs": jobs,
                "threads_per_job": threads,
                "timeout_seconds": timeout_seconds,
                "optimization_level": opt_level,
                "charge": charge,
                "multiplicity": multiplicity,
                "input_dir": str(source_dir),
                "conformer_change_rmsd_angstrom": conformer_change_rmsd_angstrom,
                "dedup_rmsd_threshold_angstrom": dedup_rmsd_threshold_angstrom,
                "comparison_atom_mode": comparison_atom_mode,
                "permutation_mode": permutation_mode,
                "topology_tolerance": topology_tolerance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _optimize_one,
                source,
                destination,
                command=str(executable),
                method=method,
                threads=threads,
                timeout_seconds=timeout_seconds,
                opt_level=opt_level,
                charge=charge,
                multiplicity=multiplicity,
            ): source
            for source in sources
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{index:04d}/{len(sources):04d}] {row['status']:>14s} {futures[future].name}", flush=True)

    rows.sort(key=lambda row: row["source"])
    same_dir = optimized_root / "same_conformer"
    changed_dir = optimized_root / "changed_conformer"
    for classified_dir in (same_dir, changed_dir):
        if classified_dir.exists():
            shutil.rmtree(classified_dir)
        classified_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        optimized_value = str(row["optimized_xyz"])
        if not optimized_value:
            row.update(
                {
                    "optimization_rmsd_angstrom": "",
                    "same_topology": "",
                    "conformer_status": "",
                }
            )
            continue
        source = Path(str(row["source"]))
        unclassified = Path(optimized_value)
        comparison = compare_structures(
            read_xyz(source, charge=charge, multiplicity=multiplicity),
            read_xyz(unclassified, charge=charge, multiplicity=multiplicity),
            atom_mode=comparison_atom_mode,
            topology_tolerance=topology_tolerance,
            permutation_mode=permutation_mode,
        )
        same_conformer = (
            comparison.same_topology
            and comparison.rmsd_angstrom <= conformer_change_rmsd_angstrom
        )
        conformer_status = "same_conformer" if same_conformer else "changed_conformer"
        classified_path = (
            same_dir if same_conformer else changed_dir
        ) / unclassified.name
        shutil.move(str(unclassified), classified_path)
        row.update(
            {
                "optimized_xyz": str(classified_path),
                "optimization_rmsd_angstrom": f"{comparison.rmsd_angstrom:.8f}",
                "same_topology": comparison.same_topology,
                "conformer_status": conformer_status,
            }
        )
    with (destination / "optimization.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "source",
            "optimized_xyz",
            "energy_hartree",
            "status",
            "log",
            "input",
            "backend",
            "method",
            "constraint_mode",
            "reused",
            "optimization_rmsd_angstrom",
            "same_topology",
            "conformer_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    succeeded = sum(bool(row["optimized_xyz"]) for row in rows)
    same_count = sum(row.get("conformer_status") == "same_conformer" for row in rows)
    changed_count = sum(row.get("conformer_status") == "changed_conformer" for row in rows)
    unique_count = 0
    if succeeded:
        deduplication = cluster_optimized_candidates(
            destination / "optimization.csv",
            deduplication_dir,
            rmsd_threshold_angstrom=dedup_rmsd_threshold_angstrom,
            energy_window_kcal_mol=None,
            atom_mode=comparison_atom_mode,
            topology_tolerance=topology_tolerance,
            permutation_mode=permutation_mode,
            unique_output_dir=optimized_root / "unique_optimized_xyz",
        )
        unique_count = int(deduplication["unique_optimized_structures"])
    else:
        (optimized_root / "unique_optimized_xyz").mkdir(parents=True, exist_ok=True)
    unclassified_dir = optimized_root / ".unclassified"
    if unclassified_dir.exists():
        shutil.rmtree(unclassified_dir)
    if succeeded != len(rows):
        first_failure = next(row for row in rows if not row["optimized_xyz"])
        print(f"primo log fallito: {first_failure['log']}", flush=True)
    return {
        "total": len(rows),
        "succeeded": succeeded,
        "failed": len(rows) - succeeded,
        "same_conformer": same_count,
        "changed_conformer": changed_count,
        "unique_optimized": unique_count,
    }


def optimize_gaussian_candidates(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    command: str,
    jobs: int | None = None,
    nprocshared: int | None = None,
    mem_gb: int | None = None,
    timeout_seconds: float = 3600.0,
    charge: int = 0,
    multiplicity: int = 1,
    constraint_mode: str = "free",
    run_manifest: str | Path | None = None,
    conformer_change_rmsd_angstrom: float = 0.75,
    dedup_rmsd_threshold_angstrom: float = 0.30,
    comparison_atom_mode: str = "all",
    permutation_mode: str = "equivalent",
    topology_tolerance: float = 0.45,
) -> dict[str, int]:
    """Optimize selected candidates with the fixed SEEKER B3LYP model."""

    source_dir = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    sources = sorted(source_dir.glob("*.xyz"))
    if not sources:
        raise ValueError(f"no XYZ files to optimize in {source_dir}")
    if constraint_mode not in {"free", "active"}:
        raise ValueError("constraint mode must be free or active")
    if constraint_mode == "active" and run_manifest is None:
        raise ValueError("active-only Gaussian optimization requires --run-manifest")
    for label, value in (
        ("conformer-change RMSD threshold", conformer_change_rmsd_angstrom),
        ("deduplication RMSD threshold", dedup_rmsd_threshold_angstrom),
        ("topology tolerance", topology_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive")
    if constraint_mode == "active" and run_manifest is not None:
        gaussian_active_constraints(
            read_xyz(sources[0], charge=charge, multiplicity=multiplicity),
            run_manifest,
            topology_tolerance=topology_tolerance,
        )
    if comparison_atom_mode not in {"heavy", "all"}:
        raise ValueError("post-optimization RMSD atom mode must be heavy or all")
    if permutation_mode not in {"equivalent", "ordered"}:
        raise ValueError("permutation mode must be equivalent or ordered")
    recommended = recommend_gaussian_resources()
    actual_jobs = recommended.jobs if jobs is None else jobs
    actual_nproc = recommended.nprocshared if nprocshared is None else nprocshared
    actual_mem = recommended.mem_gb if mem_gb is None else mem_gb
    if actual_jobs < 1 or actual_nproc < 1 or actual_mem < 1:
        raise ValueError("Gaussian jobs, nprocshared and memory must be at least 1")
    executable = shutil.which(command) if os.path.sep not in command else command
    if not executable or not Path(executable).is_file():
        raise ValueError(f"Gaussian executable not found: {command}")
    executable = str(Path(executable).resolve())
    destination.mkdir(parents=True, exist_ok=True)
    optimized_root = destination / "optimized_xyz"
    optimized_root.mkdir(parents=True, exist_ok=True)
    for previous_dir in (
        optimized_root / ".unclassified",
        optimized_root / "same_conformer",
        optimized_root / "changed_conformer",
        optimized_root / "unique_optimized_xyz",
    ):
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
    deduplication_dir = destination / "deduplication"
    if deduplication_dir.exists():
        shutil.rmtree(deduplication_dir)
    (destination / "optimization_config.json").write_text(
        json.dumps(
            {
                "schema": "seeker.gaussian.optimization.v1",
                "backend": "gaussian",
                "command": executable,
                "model": B3LYP_MODEL,
                "jobs": actual_jobs,
                "nprocshared": actual_nproc,
                "mem_gb": actual_mem,
                "timeout_seconds": timeout_seconds,
                "charge": charge,
                "multiplicity": multiplicity,
                "constraint_mode": constraint_mode,
                "run_manifest": str(Path(run_manifest).resolve()) if run_manifest else "",
                "input_dir": str(source_dir),
                "conformer_change_rmsd_angstrom": conformer_change_rmsd_angstrom,
                "dedup_rmsd_threshold_angstrom": dedup_rmsd_threshold_angstrom,
                "comparison_atom_mode": comparison_atom_mode,
                "permutation_mode": permutation_mode,
                "topology_tolerance": topology_tolerance,
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
                optimize_gaussian_one,
                source,
                destination,
                command=executable,
                nprocshared=actual_nproc,
                mem_gb=actual_mem,
                timeout_seconds=timeout_seconds,
                charge=charge,
                multiplicity=multiplicity,
                constraint_mode=constraint_mode,
                run_manifest=run_manifest,
                topology_tolerance=topology_tolerance,
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

    rows.sort(key=lambda row: row["source"])
    same_dir = optimized_root / "same_conformer"
    changed_dir = optimized_root / "changed_conformer"
    same_dir.mkdir(parents=True, exist_ok=True)
    changed_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        optimized_value = str(row["optimized_xyz"])
        if not optimized_value:
            row.update(
                {
                    "optimization_rmsd_angstrom": "",
                    "same_topology": "",
                    "conformer_status": "",
                }
            )
            continue
        source = Path(str(row["source"]))
        unclassified = Path(optimized_value)
        comparison = compare_structures(
            read_xyz(source, charge=charge, multiplicity=multiplicity),
            read_xyz(unclassified, charge=charge, multiplicity=multiplicity),
            atom_mode=comparison_atom_mode,
            topology_tolerance=topology_tolerance,
            permutation_mode=permutation_mode,
        )
        same_conformer = (
            comparison.same_topology
            and comparison.rmsd_angstrom <= conformer_change_rmsd_angstrom
        )
        conformer_status = "same_conformer" if same_conformer else "changed_conformer"
        classified_path = (same_dir if same_conformer else changed_dir) / unclassified.name
        shutil.move(str(unclassified), classified_path)
        row.update(
            {
                "optimized_xyz": str(classified_path),
                "optimization_rmsd_angstrom": f"{comparison.rmsd_angstrom:.8f}",
                "same_topology": comparison.same_topology,
                "conformer_status": conformer_status,
            }
        )
    fields = [
        "source",
        "optimized_xyz",
        "energy_hartree",
        "status",
        "log",
        "input",
        "backend",
        "method",
        "constraint_mode",
        "reused",
        "optimization_rmsd_angstrom",
        "same_topology",
        "conformer_status",
    ]
    with (destination / "optimization.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    succeeded = sum(bool(row["optimized_xyz"]) for row in rows)
    same_count = sum(row.get("conformer_status") == "same_conformer" for row in rows)
    changed_count = sum(
        row.get("conformer_status") == "changed_conformer" for row in rows
    )
    unique_count = 0
    if succeeded:
        deduplication = cluster_optimized_candidates(
            destination / "optimization.csv",
            deduplication_dir,
            rmsd_threshold_angstrom=dedup_rmsd_threshold_angstrom,
            energy_window_kcal_mol=None,
            atom_mode=comparison_atom_mode,
            topology_tolerance=topology_tolerance,
            permutation_mode=permutation_mode,
            unique_output_dir=optimized_root / "unique_optimized_xyz",
        )
        unique_count = int(deduplication["unique_optimized_structures"])
    else:
        (optimized_root / "unique_optimized_xyz").mkdir(parents=True, exist_ok=True)
    unclassified_dir = optimized_root / ".unclassified"
    if unclassified_dir.exists():
        shutil.rmtree(unclassified_dir)
    return {
        "total": len(rows),
        "succeeded": succeeded,
        "failed": len(rows) - succeeded,
        "same_conformer": same_count,
        "changed_conformer": changed_count,
        "unique_optimized": unique_count,
        "reused": sum(bool(row["reused"]) for row in rows),
    }
