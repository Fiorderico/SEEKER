"""Native XYZ plus torsion and rigid-fragment gene input handling."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .geometry import canonical_element
from .models import (
    Atom,
    FragmentPoseGene,
    Gene,
    HydrogenPiConfig,
    HydrogenPiParameters,
    HydrogenPiRingSpec,
    Molecule,
    NativePoseCoordinate,
)

GENE_PATTERN = re.compile(
    r"^\s*(?P<name>GENE[\w-]*)"
    r"(?:\s*\(\s*periodicity\s*=\s*(?P<periodicity>\d+)\s*\))?"
    r"\s*=\s*D\s*\(\s*(?P<i>\d+)\s*[,;]\s*(?P<j>\d+)\s*[,;]"
    r"\s*(?P<k>\d+)\s*[,;]\s*(?P<l>\d+)\s*\)",
    re.IGNORECASE,
)

RING_PATTERN = re.compile(
    r"^\s*(?P<name>RING[\w-]*)"
    r"(?:\s*\((?P<options>[^)]*)\))?"
    r"\s*=\s*RING\s*\((?P<atoms>[^)]*)\)\s*$",
    re.IGNORECASE,
)

FRAGMENT_POSE_PATTERN = re.compile(
    r"^\s*(?P<name>POSE[\w-]*)"
    r"(?:\s*\((?P<options>[^)]*)\))?"
    r"\s*=\s*FRAGMENTS\s*\((?P<fragments>[^)]*)\)\s*$",
    re.IGNORECASE,
)

HPI_MODE_PATTERN = re.compile(
    r"^\s*HPI\s*\(\s*mode\s*=\s*(?P<mode>auto|explicit)\s*\)\s*$",
    re.IGNORECASE,
)

HPI_RING_PATTERN = re.compile(
    r"^\s*(?P<kind>HPI_RING|HPI_EXCLUDE)_(?P<name>[\w-]+)"
    r"\s*=\s*RING\s*\((?P<atoms>[^)]*)\)\s*$",
    re.IGNORECASE,
)

HPI_PARAMETER_PATTERN = re.compile(
    r"^\s*HPI_PARAM_(?P<donor>OH|NH|SH)\s*\((?P<options>[^)]*)\)\s*$",
    re.IGNORECASE,
)


def _is_hpi_directive(line: str) -> bool:
    return bool(
        HPI_MODE_PATTERN.fullmatch(line)
        or HPI_RING_PATTERN.fullmatch(line)
        or HPI_PARAMETER_PATTERN.fullmatch(line)
    )


def _parse_hpi_parameters(
    source: Path,
    line_number: int,
    donor: str,
    raw: str,
    base: HydrogenPiParameters,
) -> HydrogenPiParameters:
    values: dict[str, float] = {}
    aliases = {
        "z0": "z0_angstrom",
        "sigma_z": "sigma_z_angstrom",
        "rho_c": "rho_c_angstrom",
        "sigma_beta": "sigma_beta_degrees",
        "weight": "weight",
    }
    for field in raw.split(","):
        if "=" not in field:
            raise ValueError(
                f"invalid HPI_PARAM_{donor} option at {source}:{line_number}: {field.strip()}"
            )
        key, value = field.split("=", 1)
        normalized = key.strip().lower().replace("-", "_")
        if normalized not in aliases:
            raise ValueError(
                f"unrecognized HPI_PARAM_{donor} option at {source}:{line_number}: {key.strip()}"
            )
        canonical = aliases[normalized]
        if canonical in values:
            raise ValueError(f"duplicate HPI_PARAM_{donor} option: {key.strip()}")
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid HPI_PARAM_{donor} value at {source}:{line_number}: {value.strip()}"
            ) from exc
        values[canonical] = number
    if not values:
        raise ValueError(f"HPI_PARAM_{donor} requires at least one parameter")
    result = replace(base, **values)
    result.validate(donor)
    return result


def read_hbond_pi_config(path: str | Path) -> HydrogenPiConfig:
    """Read non-genetic HPI directives while ignoring coordinate definitions."""

    source = Path(path)
    default = HydrogenPiConfig()
    mode = "auto"
    mode_seen = False
    included: list[HydrogenPiRingSpec] = []
    excluded: list[HydrogenPiRingSpec] = []
    parameters = {"OH": default.oh, "NH": default.nh, "SH": default.sh}
    parameter_seen: set[str] = set()
    configured = False

    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line or not line.upper().startswith("HPI"):
            continue
        configured = True
        mode_match = HPI_MODE_PATTERN.fullmatch(line)
        ring_match = HPI_RING_PATTERN.fullmatch(line)
        parameter_match = HPI_PARAMETER_PATTERN.fullmatch(line)
        if not mode_match and not ring_match and not parameter_match:
            raise ValueError(f"invalid HPI directive at {source}:{line_number}: {raw_line}")
        if mode_match:
            if mode_seen:
                raise ValueError(f"duplicate HPI mode at {source}:{line_number}")
            mode = mode_match.group("mode").lower()
            mode_seen = True
            continue
        if ring_match:
            label = ring_match.group("name").upper()
            kind = ring_match.group("kind").upper()
            atoms = _parse_atom_selection(
                source,
                line_number,
                f"{kind}_{label}",
                ring_match.group("atoms").replace(";", ","),
            )
            if not 5 <= len(atoms) <= 7:
                raise ValueError(f"{kind}_{label} requires between 5 and 7 atoms")
            spec = HydrogenPiRingSpec(label, atoms)
            (included if kind == "HPI_RING" else excluded).append(spec)
            continue
        donor = parameter_match.group("donor").upper()
        if donor in parameter_seen:
            raise ValueError(f"duplicate HPI_PARAM_{donor} directive")
        parameter_seen.add(donor)
        parameters[donor] = _parse_hpi_parameters(
            source,
            line_number,
            donor,
            parameter_match.group("options"),
            parameters[donor],
        )

    config = HydrogenPiConfig(
        mode=mode,
        included_rings=tuple(included),
        excluded_rings=tuple(excluded),
        oh=parameters["OH"],
        nh=parameters["NH"],
        sh=parameters["SH"],
        configured=configured,
    )
    config.validate()
    return config


def read_xyz(path: str | Path, charge: int = 0, multiplicity: int = 1) -> Molecule:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"malformed XYZ: {source}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"non-numeric first XYZ line: {source}") from exc
    if atom_count < 1 or len(lines) < atom_count + 2:
        raise ValueError(f"inconsistent atom count in {source}")

    atoms: list[Atom] = []
    for line_number, line in enumerate(lines[2 : atom_count + 2], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ line {line_number} in {source}")
        try:
            atoms.append(
                Atom(
                    canonical_element(fields[0]),
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"invalid XYZ line {line_number} in {source}: {line}") from exc
    if multiplicity < 1:
        raise ValueError("multiplicity must be at least 1")
    return Molecule(tuple(atoms), lines[1].strip(), int(charge), int(multiplicity))


def write_xyz(path: str | Path, molecule: Molecule, comment: str = "") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [str(len(molecule.atoms)), comment or molecule.comment]
    rows.extend(
        f"{atom.element:<2s} {atom.x: .10f} {atom.y: .10f} {atom.z: .10f}"
        for atom in molecule.atoms
    )
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _parse_atom_selection(
    source: Path, line_number: int, name: str, raw: str
) -> tuple[int, ...]:
    """Parse a one-based comma list containing atoms or inclusive ranges."""

    atoms: list[int] = []
    for field in raw.split(","):
        token = field.strip()
        if not token:
            continue
        if "-" in token:
            parts = [item.strip() for item in token.split("-", 1)]
            try:
                first, last = (int(item) for item in parts)
            except ValueError as exc:
                raise ValueError(
                    f"invalid atom range for {name} at {source}:{line_number}: {token}"
                ) from exc
            if first < 1 or last < first:
                raise ValueError(
                    f"invalid atom range for {name} at {source}:{line_number}: {token}"
                )
            atoms.extend(range(first - 1, last))
        else:
            try:
                atom = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"invalid atom index for {name} at {source}:{line_number}: {token}"
                ) from exc
            if atom < 1:
                raise ValueError(f"atom indices are one-based for {name}")
            atoms.append(atom - 1)
    if not atoms or len(set(atoms)) != len(atoms):
        raise ValueError(f"{name} requires a nonempty list of distinct atoms")
    return tuple(atoms)


def _parse_fragment_pose_options(
    source: Path, line_number: int, name: str, raw: str | None
) -> tuple[tuple[float, float], float, float, int]:
    options: dict[str, str] = {}
    if raw:
        for field in raw.split(","):
            if "=" not in field:
                raise ValueError(
                    f"invalid POSE option at {source}:{line_number}: {field.strip()}"
                )
            key, value = field.split("=", 1)
            normalized = key.strip().lower().replace("-", "_")
            if normalized in options:
                raise ValueError(f"duplicate POSE option for {name}: {normalized}")
            options[normalized] = value.strip()
    aliases = {
        "rotation": "orientation",
        "body_rotation": "orientation",
        "placement": "direction",
    }
    for alias, canonical in aliases.items():
        if alias in options:
            if canonical in options:
                raise ValueError(f"duplicate POSE option for {name}: {canonical}")
            options[canonical] = options.pop(alias)
    unknown = set(options) - {"distance", "direction", "orientation", "points"}
    if unknown:
        raise ValueError(f"unrecognized POSE options for {name}: {', '.join(sorted(unknown))}")
    if "distance" not in options:
        raise ValueError(f"{name} requires distance=min:max in angstrom")
    fields = [item.strip() for item in re.split(r"[:;]", options["distance"])]
    if len(fields) != 2:
        raise ValueError(f"distance must be min:max in angstrom for {name}")
    try:
        lower, upper = (float(item.removesuffix("A").removesuffix("a")) for item in fields)
        direction = float(options.get("direction", "180").lower().removesuffix("deg"))
        orientation = float(options.get("orientation", "180").lower().removesuffix("deg"))
        points = int(options.get("points", "3"))
    except ValueError as exc:
        raise ValueError(f"invalid numeric POSE option for {name}") from exc
    if not 0.0 < lower < upper:
        raise ValueError(f"distance requires 0 < min < max for {name}")
    if not 0.0 <= direction <= 180.0:
        raise ValueError(f"direction must be between 0 and 180 degrees for {name}")
    if not 0.0 <= orientation <= 180.0:
        raise ValueError(f"orientation must be between 0 and 180 degrees for {name}")
    if points < 2:
        raise ValueError(f"points must be at least 2 for {name}")
    return (lower, upper), direction, orientation, points


def read_coordinate_specs(
    path: str | Path,
) -> tuple[Gene | FragmentPoseGene, ...]:
    source = Path(path)
    genes: list[Gene | FragmentPoseGene] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.upper().startswith("HPI"):
            if not _is_hpi_directive(line):
                raise ValueError(f"invalid HPI directive at {source}:{line_number}: {raw_line}")
            continue
        match = GENE_PATTERN.fullmatch(line)
        ring_match = RING_PATTERN.fullmatch(line)
        pose_match = FRAGMENT_POSE_PATTERN.fullmatch(line)
        if not match and not ring_match and not pose_match:
            raise ValueError(f"unrecognized gene at {source}:{line_number}: {raw_line}")
        name = (match or ring_match or pose_match).group("name").upper()
        if name in seen_names:
            raise ValueError(f"duplicate gene name in {source}: {name}")
        seen_names.add(name)
        if ring_match:
            raise ValueError(
                f"SEEKER does not support RING coordinates at {source}:{line_number}"
            )
        if match:
            atoms = (
                int(match.group("i")) - 1,
                int(match.group("j")) - 1,
                int(match.group("k")) - 1,
                int(match.group("l")) - 1,
            )
            periodicity = int(match.group("periodicity") or 1)
            if periodicity < 1:
                raise ValueError(f"invalid periodicity for {name}")
            genes.append(Gene(name, atoms, periodicity))
        else:
            fragments = [item.strip() for item in pose_match.group("fragments").split(";")]
            if len(fragments) != 2 or not all(fragments):
                raise ValueError(
                    f"{name} requires FRAGMENTS(reference_atoms;moving_atoms)"
                )
            reference_atoms = _parse_atom_selection(
                source, line_number, name, fragments[0]
            )
            moving_atoms = _parse_atom_selection(
                source, line_number, name, fragments[1]
            )
            if set(reference_atoms) & set(moving_atoms):
                raise ValueError(f"{name} fragment atom lists must be disjoint")
            bounds, direction, orientation, points = _parse_fragment_pose_options(
                source, line_number, name, pose_match.group("options")
            )
            genes.append(
                FragmentPoseGene(
                    name,
                    reference_atoms,
                    moving_atoms,
                    bounds,
                    direction,
                    orientation,
                    points,
                )
            )
    if not genes:
        raise ValueError(f"no genes found in {source}")
    return tuple(genes)


def read_genes(path: str | Path) -> tuple[Gene, ...]:
    specs = read_coordinate_specs(path)
    poses = [item.name for item in specs if isinstance(item, FragmentPoseGene)]
    if poses:
        raise ValueError(
            "read_genes accepts torsions only; use read_coordinate_specs for: "
            + ", ".join(poses)
        )
    return tuple(item for item in specs if isinstance(item, Gene))


def molecule_fingerprint(
    molecule: Molecule,
    genes: Sequence[Gene | NativePoseCoordinate],
) -> str:
    payload = {
        "charge": molecule.charge,
        "multiplicity": molecule.multiplicity,
        "atoms": [
            [atom.element, round(atom.x, 10), round(atom.y, 10), round(atom.z, 10)]
            for atom in molecule.atoms
        ],
        "genes": [
            {
                "name": gene.name,
                "atoms": list(gene.atoms),
                "periodicity": gene.periodicity,
                "periodic": gene.periodic,
                "kind": "native_pose" if isinstance(gene, NativePoseCoordinate) else "torsion",
                **(
                    {
                        "pose_name": gene.pose_name,
                        "component": gene.component,
                        "lower": gene.lower,
                        "upper": gene.upper,
                        "reference_value": gene.reference_value,
                        "reference_atoms": list(gene.reference_atoms),
                        "moving_atoms": list(gene.moving_atoms),
                        "units": gene.units,
                        "scan_points": gene.scan_points,
                    }
                    if isinstance(gene, NativePoseCoordinate) else {}
                ),
            }
            for gene in genes
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
