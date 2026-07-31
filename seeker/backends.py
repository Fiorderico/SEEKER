"""Pluggable low-level single-point energy backends and persistent cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .input import write_xyz
from .models import Molecule

HARTREE_PER_KCAL_MOL = 1.0 / 627.5094740631
HARTREE_PER_KJ_MOL = 1.0 / 2625.4996394799
HARTREE_PER_EV = 1.0 / 27.211386245988


class EnergyBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnergyResult:
    energy_hartree: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EnergyBackend(ABC):
    @property
    @abstractmethod
    def signature(self) -> dict[str, Any]:
        """Configuration that uniquely identifies the computed energy."""

    @abstractmethod
    def evaluate(self, molecule: Molecule) -> EnergyResult:
        """Return one single-point energy in Hartree."""

    def close(self) -> None:
        return None


def _to_hartree(value: float, unit: str) -> float:
    normalized = unit.strip().lower().replace("/", "_")
    factors = {
        "hartree": 1.0,
        "eh": 1.0,
        "au": 1.0,
        "kcal_mol": HARTREE_PER_KCAL_MOL,
        "kj_mol": HARTREE_PER_KJ_MOL,
        "ev": HARTREE_PER_EV,
    }
    if normalized not in factors:
        raise ValueError(f"unsupported energy unit: {unit}")
    return float(value) * factors[normalized]


class XtbBackend(EnergyBackend):
    ENERGY_PATTERN = re.compile(
        r"TOTAL\s+ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        command: str = "xtb",
        method: str = "gfn2",
        threads: int = 1,
        timeout_seconds: float = 600.0,
        work_root: str | Path | None = None,
    ) -> None:
        normalized = method.lower().replace("-", "")
        if normalized not in {"gfnff", "gfn0", "gfn1", "gfn2"}:
            raise ValueError(f"unsupported xTB method: {method}")
        self.command = command
        self.method = normalized
        self.threads = max(1, int(threads))
        self.timeout_seconds = float(timeout_seconds)
        self.work_root = Path(work_root) if work_root else None

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "backend": "xtb",
            "command": self.command,
            "method": self.method,
        }

    def _method_args(self) -> list[str]:
        if self.method == "gfnff":
            return ["--gfnff"]
        return ["--gfn", self.method[-1]]

    def evaluate(self, molecule: Molecule) -> EnergyResult:
        executable = shutil.which(self.command) if os.path.sep not in self.command else self.command
        if not executable or not Path(executable).exists():
            raise EnergyBackendError(f"xTB executable not found: {self.command}")
        if self.work_root:
            self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="seeker_xtb_", dir=str(self.work_root) if self.work_root else None
        ) as temporary:
            workdir = Path(temporary)
            xyz_path = workdir / "candidate.xyz"
            write_xyz(xyz_path, molecule, "SEEKER xTB single point")
            argv = [
                str(executable),
                xyz_path.name,
                *self._method_args(),
                "--sp",
                "--chrg",
                str(molecule.charge),
                "--uhf",
                str(max(0, molecule.multiplicity - 1)),
                "--parallel",
                str(self.threads),
            ]
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", str(self.threads))
            try:
                completed = subprocess.run(
                    argv,
                    cwd=workdir,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds if self.timeout_seconds > 0 else None,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise EnergyBackendError(f"xTB timed out after {self.timeout_seconds:g} s") from exc
            output = completed.stdout + "\n" + completed.stderr
            if completed.returncode != 0:
                tail = " | ".join(output.strip().splitlines()[-4:])
                raise EnergyBackendError(f"xTB exit {completed.returncode}: {tail[:600]}")
            matches = list(self.ENERGY_PATTERN.finditer(output))
            if not matches:
                raise EnergyBackendError("xTB did not produce a TOTAL ENERGY line")
            energy = float(matches[-1].group(1).replace("D", "E").replace("d", "e"))
            if not math.isfinite(energy):
                raise EnergyBackendError("non-finite xTB energy")
            return EnergyResult(energy, {"backend": "xtb", "method": self.method})


class ExternalCommandBackend(EnergyBackend):
    """Run a user command without a shell and parse one energy from its output.

    The command template must contain ``{xyz}``. Other placeholders are
    ``{charge}``, ``{multiplicity}`` and ``{workdir}``.
    """

    def __init__(
        self,
        command_template: str,
        energy_regex: str,
        energy_unit: str = "hartree",
        timeout_seconds: float = 600.0,
        work_root: str | Path | None = None,
    ) -> None:
        if "{xyz}" not in command_template:
            raise ValueError("--external-command must contain the {xyz} placeholder")
        self.command_template = command_template
        self.pattern = re.compile(energy_regex, re.MULTILINE)
        if self.pattern.groups < 1:
            raise ValueError("--external-regex must contain at least one capture group")
        _to_hartree(0.0, energy_unit)
        self.energy_regex = energy_regex
        self.energy_unit = energy_unit
        self.timeout_seconds = float(timeout_seconds)
        self.work_root = Path(work_root) if work_root else None

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "backend": "external",
            "command_template": self.command_template,
            "energy_regex": self.energy_regex,
            "energy_unit": self.energy_unit,
        }

    def evaluate(self, molecule: Molecule) -> EnergyResult:
        if self.work_root:
            self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="seeker_external_", dir=str(self.work_root) if self.work_root else None
        ) as temporary:
            workdir = Path(temporary)
            xyz_path = workdir / "candidate.xyz"
            write_xyz(xyz_path, molecule, "SEEKER external single point")
            values = {
                "xyz": str(xyz_path),
                "charge": str(molecule.charge),
                "multiplicity": str(molecule.multiplicity),
                "workdir": str(workdir),
            }
            argv = [token.format(**values) for token in shlex.split(self.command_template)]
            try:
                completed = subprocess.run(
                    argv,
                    cwd=workdir,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds if self.timeout_seconds > 0 else None,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise EnergyBackendError(f"external backend is not executable: {exc}") from exc
            output = completed.stdout + "\n" + completed.stderr
            if completed.returncode != 0:
                tail = " | ".join(output.strip().splitlines()[-4:])
                raise EnergyBackendError(f"external backend exited with {completed.returncode}: {tail[:600]}")
            matches = list(self.pattern.finditer(output))
            if not matches:
                raise EnergyBackendError("energy regex not found in external backend output")
            value = float(matches[-1].group(1).replace("D", "E").replace("d", "e"))
            energy = _to_hartree(value, self.energy_unit)
            if not math.isfinite(energy):
                raise EnergyBackendError("non-finite external energy")
            return EnergyResult(energy, {"backend": "external", "raw_value": value, "raw_unit": self.energy_unit})


class PyScfBackend(EnergyBackend):
    def __init__(self, basis: str = "sto-3g", convergence_tolerance: float = 1.0e-9) -> None:
        self.basis = basis
        self.convergence_tolerance = float(convergence_tolerance)

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "backend": "pyscf",
            "method": "rhf_or_uhf",
            "basis": self.basis,
            "conv_tol": self.convergence_tolerance,
        }

    def evaluate(self, molecule: Molecule) -> EnergyResult:
        try:
            from pyscf import gto, scf
        except ImportError as exc:
            raise EnergyBackendError("PySCF is not installed; use xTB or install the optional pyscf group") from exc

        mol = gto.Mole()
        mol.atom = [(atom.element, atom.position) for atom in molecule.atoms]
        mol.unit = "Angstrom"
        mol.basis = self.basis
        mol.charge = molecule.charge
        mol.spin = molecule.multiplicity - 1
        mol.verbose = 0
        try:
            mol.build()
            mean_field = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            mean_field.conv_tol = self.convergence_tolerance
            energy = float(mean_field.kernel())
        except Exception as exc:
            raise EnergyBackendError(f"PySCF failed: {exc}") from exc
        if not mean_field.converged or not math.isfinite(energy):
            raise EnergyBackendError("PySCF SCF did not converge")
        return EnergyResult(energy, {"backend": "pyscf", "basis": self.basis})


class EnergyCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS energies ("
            "cache_key TEXT PRIMARY KEY, energy REAL NOT NULL, metadata TEXT NOT NULL)"
        )
        self._connection.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> EnergyResult | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT energy, metadata FROM energies WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return EnergyResult(float(row[0]), json.loads(row[1]))

    def put(self, key: str, result: EnergyResult) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO energies(cache_key, energy, metadata) VALUES (?, ?, ?)",
                (key, result.energy_hartree, json.dumps(result.metadata, sort_keys=True)),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def energy_cache_key(molecule: Molecule, backend_signature: dict[str, Any]) -> str:
    payload = {
        "backend": backend_signature,
        "charge": molecule.charge,
        "multiplicity": molecule.multiplicity,
        "atoms": [
            [atom.element, round(atom.x, 8), round(atom.y, 8), round(atom.z, 8)]
            for atom in molecule.atoms
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CachedEnergyBackend(EnergyBackend):
    def __init__(self, backend: EnergyBackend, cache: EnergyCache) -> None:
        self.backend = backend
        self.cache = cache

    @property
    def signature(self) -> dict[str, Any]:
        return self.backend.signature

    def evaluate(self, molecule: Molecule) -> EnergyResult:
        key = energy_cache_key(molecule, self.signature)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.backend.evaluate(molecule)
        self.cache.put(key, result)
        return result

    def close(self) -> None:
        self.cache.close()
        self.backend.close()
