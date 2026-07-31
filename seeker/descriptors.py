"""Dependency-free molecular descriptors and optional Open Babel charges."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from .input import write_xyz
from .models import Molecule

Vector = tuple[float, float, float]

ATOMIC_MASS_AMU = {
    "H": 1.00782503223,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "F": 18.9984031627,
    "SI": 27.9769265347,
    "P": 30.9737619984,
    "S": 31.9720711744,
    "CL": 34.968852682,
    "BR": 78.9183376,
    "I": 126.904473,
}

ROTATIONAL_MHZ_FACTOR = 505379.008966721
E_ANGSTROM_TO_DEBYE = 4.803204712570263


def _mass(element: str) -> float:
    try:
        return ATOMIC_MASS_AMU[element.upper()]
    except KeyError as exc:
        raise ValueError(f"atomic mass unavailable for {element}") from exc


def _jacobi_eigh(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[Vector]]:
    """Eigenpairs of a real symmetric 3x3 matrix via Jacobi rotations."""

    values = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    vectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    for _ in range(64):
        first, second = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(values[pair[0]][pair[1]]))
        off_diagonal = values[first][second]
        if abs(off_diagonal) < 1.0e-14:
            break
        angle = 0.5 * math.atan2(
            2.0 * off_diagonal,
            values[second][second] - values[first][first],
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)

        app = values[first][first]
        aqq = values[second][second]
        apq = values[first][second]
        values[first][first] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        values[second][second] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        values[first][second] = values[second][first] = 0.0
        for index in range(3):
            if index in (first, second):
                continue
            aip = values[index][first]
            aiq = values[index][second]
            values[index][first] = values[first][index] = cosine * aip - sine * aiq
            values[index][second] = values[second][index] = sine * aip + cosine * aiq
        for index in range(3):
            vip = vectors[index][first]
            viq = vectors[index][second]
            vectors[index][first] = cosine * vip - sine * viq
            vectors[index][second] = sine * vip + cosine * viq

    order = sorted(range(3), key=lambda index: values[index][index])
    eigenvalues = [max(0.0, values[index][index]) for index in order]
    axes = [
        tuple(vectors[row][index] for row in range(3))
        for index in order
    ]
    return eigenvalues, axes


def principal_inertia(molecule: Molecule) -> tuple[list[float], list[Vector], Vector]:
    if not molecule.atoms:
        raise ValueError("empty molecule")
    masses = [_mass(atom.element) for atom in molecule.atoms]
    total_mass = sum(masses)
    center = tuple(
        sum(mass * atom.position[axis] for mass, atom in zip(masses, molecule.atoms)) / total_mass
        for axis in range(3)
    )
    tensor = [[0.0, 0.0, 0.0] for _ in range(3)]
    for mass, atom in zip(masses, molecule.atoms):
        x = atom.x - center[0]
        y = atom.y - center[1]
        z = atom.z - center[2]
        tensor[0][0] += mass * (y * y + z * z)
        tensor[1][1] += mass * (x * x + z * z)
        tensor[2][2] += mass * (x * x + y * y)
        tensor[0][1] -= mass * x * y
        tensor[0][2] -= mass * x * z
        tensor[1][2] -= mass * y * z
    tensor[1][0] = tensor[0][1]
    tensor[2][0] = tensor[0][2]
    tensor[2][1] = tensor[1][2]
    moments, axes = _jacobi_eigh(tensor)
    return moments, axes, center


def rotational_constants_mhz(molecule: Molecule) -> dict[str, float | str]:
    moments, _, _ = principal_inertia(molecule)
    ia, ib, ic = moments
    effective = [max(value, 1.0e-8) for value in moments]
    a, b, c = [ROTATIONAL_MHZ_FACTOR / value for value in effective]

    def almost(left: float, right: float, tolerance: float = 1.0e-3) -> bool:
        return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))

    if almost(ia, ib) and almost(ib, ic):
        classification = "spherical"
    elif almost(ia, 0.0) and almost(ib, ic):
        classification = "linear"
    elif almost(ia, ib) and ib < ic:
        classification = "oblate_symmetric"
    elif almost(ib, ic) and ia < ib:
        classification = "prolate_symmetric"
    else:
        classification = "near_prolate" if (ib - ia) < (ic - ib) else "near_oblate"
    return {
        "A_mhz": a,
        "B_mhz": b,
        "C_mhz": c,
        "Ia_amu_a2": ia,
        "Ib_amu_a2": ib,
        "Ic_amu_a2": ic,
        "rotor_classification": classification,
    }


def rotor_shape_scores(
    constants: dict[str, float | str],
    symmetry_sigma: float = 0.15,
    anisotropy_sigma: float = 0.15,
) -> dict[str, float]:
    a = float(constants["A_mhz"])
    b = float(constants["B_mhz"])
    c = float(constants["C_mhz"])
    ratio_ab = abs(math.log(a / b))
    ratio_bc = abs(math.log(b / c))
    spherical = math.exp(-math.hypot(ratio_ab, ratio_bc) / symmetry_sigma)
    prolate = (1.0 - math.exp(-ratio_ab / anisotropy_sigma)) * math.exp(-ratio_bc / symmetry_sigma)
    oblate = (1.0 - math.exp(-ratio_bc / anisotropy_sigma)) * math.exp(-ratio_ab / symmetry_sigma)
    return {
        "rotor_prolate": min(1.0, max(0.0, prolate)),
        "rotor_oblate": min(1.0, max(0.0, oblate)),
        "rotor_spherical": min(1.0, max(0.0, spherical)),
    }


def dipole_components_debye(molecule: Molecule, charges: Sequence[float]) -> dict[str, float]:
    if len(charges) != len(molecule.atoms):
        raise ValueError(
            f"inconsistent charge count: {len(charges)} for {len(molecule.atoms)} atoms"
        )
    _, axes, origin = principal_inertia(molecule)
    cartesian = [0.0, 0.0, 0.0]
    for charge, atom in zip(charges, molecule.atoms):
        for axis in range(3):
            cartesian[axis] += float(charge) * (atom.position[axis] - origin[axis])
    cartesian = [value * E_ANGSTROM_TO_DEBYE for value in cartesian]
    components = [
        abs(sum(cartesian[index] * axis[index] for index in range(3)))
        for axis in axes
    ]
    return {
        "dipole_debye": math.sqrt(sum(value * value for value in cartesian)),
        "mu_a_debye": components[0],
        "mu_b_debye": components[1],
        "mu_c_debye": components[2],
    }


class PartialChargeProvider(ABC):
    @property
    @abstractmethod
    def signature(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self, molecule: Molecule) -> list[float]:
        pass


class OpenBabelChargeProvider(PartialChargeProvider):
    def __init__(
        self,
        model: str,
        command: str = "obabel",
        work_root: str | Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("the Open Babel charge model cannot be empty")
        self.model = model.strip()
        self.command = command
        self.work_root = Path(work_root) if work_root else None
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("the Open Babel timeout must be positive and finite")
        self.timeout_seconds = timeout_seconds

    @property
    def signature(self) -> dict[str, Any]:
        return {
            "provider": "openbabel",
            "command": self.command,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }

    @staticmethod
    def parse_mol2_charges(text: str, expected_atoms: int) -> list[float]:
        charges: list[float] = []
        in_atoms = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if upper.startswith("@<TRIPOS>") and in_atoms:
                break
            if not in_atoms:
                continue
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"MOL2 ATOM line has no charge: {line}")
            charges.append(float(fields[-1]))
        if len(charges) != expected_atoms:
            raise ValueError(
                f"Open Babel returned {len(charges)} charges for {expected_atoms} atoms"
            )
        return charges

    @staticmethod
    def validate_charge_sum(
        charges: list[float],
        molecular_charge: int,
        tolerance: float = 0.15,
    ) -> list[float]:
        residual = abs(sum(charges) - molecular_charge)
        if residual > tolerance:
            raise ValueError(
                "Open Babel partial-charge sum "
                f"({sum(charges):.4f}) is incompatible with the molecular charge "
                f"({molecular_charge:+d}); provide consistent charges before using "
                "dipole calculations"
            )
        return charges

    def evaluate(self, molecule: Molecule) -> list[float]:
        executable = shutil.which(self.command) if os.path.sep not in self.command else self.command
        if not executable or not Path(executable).exists():
            raise RuntimeError(f"Open Babel executable not found: {self.command}")
        if self.work_root:
            self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="seeker_obcharge_",
            dir=str(self.work_root) if self.work_root else None,
        ) as temporary:
            workdir = Path(temporary)
            xyz_path = workdir / "candidate.xyz"
            mol2_path = workdir / "candidate.mol2"
            write_xyz(xyz_path, molecule, "SEEKER partial charges")
            try:
                completed = subprocess.run(
                    [
                        str(executable),
                        "-ixyz",
                        str(xyz_path),
                        "-omol2",
                        "-O",
                        str(mol2_path),
                        "--partialcharge",
                        self.model,
                    ],
                    cwd=workdir,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Open Babel exceeded the {self.timeout_seconds:g} s timeout"
                ) from exc
            if completed.returncode != 0 or not mol2_path.exists():
                message = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"Open Babel did not assign charges with '{self.model}': {message[:600]}"
                )
            charges = self.parse_mol2_charges(
                mol2_path.read_text(encoding="utf-8", errors="replace"),
                len(molecule.atoms),
            )
            return self.validate_charge_sum(charges, molecule.charge)
