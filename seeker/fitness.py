"""Independent geometric hydrogen-bond objective."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .geometry import BondGraph, VDW_RADII, angle_deg, distance, graph_distance_leq
from .models import (
    HydrogenDoubleBondParameters,
    HydrogenPiConfig,
    HydrogenPiParameters,
    Molecule,
)

DONOR_ACCEPTOR_ELEMENTS = {"N", "O", "S"}

PAIR_EPSILON = {
    ("O", "N"): 1.10,
    ("N", "O"): 1.10,
    ("O", "O"): 1.20,
    ("S", "O"): 1.00,
    ("S", "N"): 1.00,
    ("O", "S"): 1.20,
    ("N", "S"): 1.20,
}

PI_RING_ELEMENTS = {"C", "N", "O", "S"}

# Conservative geometry windows (angstrom) for common formal double bonds.
# XYZ has no bond-order field: the lower bounds reject triple bonds, while the
# C-C/C-N upper bounds deliberately sit below ordinary aromatic bond lengths.
DOUBLE_BOND_LENGTH_RANGES = {
    ("C", "C"): (1.24, 1.38),
    ("C", "N"): (1.20, 1.32),
    ("C", "O"): (1.14, 1.29),
    ("C", "P"): (1.54, 1.72),
    ("C", "S"): (1.48, 1.68),
    ("C", "SI"): (1.60, 1.80),
    ("N", "N"): (1.16, 1.30),
    ("N", "O"): (1.10, 1.26),
    ("N", "S"): (1.42, 1.61),
    ("O", "P"): (1.38, 1.54),
    ("O", "S"): (1.35, 1.53),
}

HBOND_DOUBLE_PARAMETERS = {
    "OH": HydrogenDoubleBondParameters(2.30, 0.30, 0.75, 20.0),
    "NH": HydrogenDoubleBondParameters(2.40, 0.35, 0.80, 25.0),
    "SH": HydrogenDoubleBondParameters(2.55, 0.40, 0.90, 30.0),
}


@dataclass(frozen=True)
class HydrogenDoubleBondSite:
    atoms: tuple[int, int]
    elements: tuple[str, str]
    reference_length_angstrom: float


def _double_bond_sites(
    molecule: Molecule, graph: BondGraph
) -> tuple[HydrogenDoubleBondSite, ...]:
    elements = [atom.element.upper() for atom in molecule.atoms]
    sites: list[HydrogenDoubleBondSite] = []
    for left, neighbours in enumerate(graph):
        for right in sorted(index for index in neighbours if index > left):
            left_element, right_element = elements[left], elements[right]
            pair: tuple[str, str] = (
                (left_element, right_element)
                if left_element <= right_element
                else (right_element, left_element)
            )
            bounds = DOUBLE_BOND_LENGTH_RANGES.get(pair)
            if bounds is None:
                continue
            bond_length = distance(
                molecule.atoms[left].position, molecule.atoms[right].position
            )
            if bounds[0] <= bond_length <= bounds[1]:
                sites.append(
                    HydrogenDoubleBondSite((left, right), pair, bond_length)
                )
    return tuple(sites)


@dataclass(frozen=True)
class HydrogenDoubleBondModel:
    """Continuous X-H...double-bond fitness aimed at the bond midpoint."""

    donors: tuple[tuple[int, int, str], ...]
    sites: tuple[HydrogenDoubleBondSite, ...]
    eligible_sites: dict[int, tuple[int, ...]]
    parameters: dict[str, HydrogenDoubleBondParameters]
    reference_graph: BondGraph

    @classmethod
    def from_reference(
        cls, molecule: Molecule, graph: BondGraph
    ) -> "HydrogenDoubleBondModel":
        elements = [atom.element.upper() for atom in molecule.atoms]
        donors = tuple(
            (donor, hydrogen, f"{elements[donor]}H")
            for donor, element in enumerate(elements)
            if element in DONOR_ACCEPTOR_ELEMENTS
            for hydrogen in sorted(graph[donor])
            if elements[hydrogen] == "H"
        )
        sites = _double_bond_sites(molecule, graph)
        eligible = {
            hydrogen: tuple(
                site_index
                for site_index, site in enumerate(sites)
                if not any(
                    graph_distance_leq(graph, donor, atom, max_hops=2)
                    for atom in site.atoms
                )
            )
            for donor, hydrogen, _donor_type in donors
        }
        parameters = dict(HBOND_DOUBLE_PARAMETERS)
        for donor_type, values in parameters.items():
            values.validate(donor_type)
        return cls(donors, sites, eligible, parameters, graph)

    def reference_metadata(self) -> dict[str, object]:
        return {
            "double_bonds": [
                {
                    "atoms": [index + 1 for index in site.atoms],
                    "elements": list(site.elements),
                    "reference_length_angstrom": site.reference_length_angstrom,
                }
                for site in self.sites
            ],
            "donors": [
                {
                    "donor": donor + 1,
                    "hydrogen": hydrogen + 1,
                    "donor_type": donor_type,
                    "eligible_double_bonds": [
                        [index + 1 for index in self.sites[site_index].atoms]
                        for site_index in self.eligible_sites.get(hydrogen, ())
                    ],
                }
                for donor, hydrogen, donor_type in self.donors
            ],
            "parameters": {
                donor_type: {
                    "r0_angstrom": values.r0_angstrom,
                    "sigma_r_angstrom": values.sigma_r_angstrom,
                    "axial_c_angstrom": values.axial_c_angstrom,
                    "sigma_beta_degrees": values.sigma_beta_degrees,
                    "weight": values.weight,
                }
                for donor_type, values in self.parameters.items()
            },
            "detection_length_ranges_angstrom": {
                "=".join(pair): list(bounds)
                for pair, bounds in DOUBLE_BOND_LENGTH_RANGES.items()
            },
        }

    def evaluate(self, molecule: Molecule) -> tuple[float, int, int, dict[str, object]]:
        total = 0.0
        possible_count = 0
        favorable_count = 0
        best_contacts: list[dict[str, object]] = []
        active_contacts: list[dict[str, object]] = []

        for donor, hydrogen, donor_type in self.donors:
            parameters = self.parameters[donor_type]
            h_position = np.asarray(molecule.atoms[hydrogen].position, dtype=float)
            x_position = np.asarray(molecule.atoms[donor].position, dtype=float)
            best: dict[str, object] | None = None
            best_score = -1.0
            for site_index in self.eligible_sites.get(hydrogen, ()):
                site = self.sites[site_index]
                left = np.asarray(molecule.atoms[site.atoms[0]].position, dtype=float)
                right = np.asarray(molecule.atoms[site.atoms[1]].position, dtype=float)
                bond_vector = right - left
                bond_length = float(np.linalg.norm(bond_vector))
                if bond_length <= 1.0e-14:
                    continue
                center = 0.5 * (left + right)
                displacement = h_position - center
                radius = float(np.linalg.norm(displacement))
                hx = x_position - h_position
                hc = center - h_position
                denominator = float(np.linalg.norm(hx) * np.linalg.norm(hc))
                if radius <= 1.0e-14 or denominator <= 1.0e-14:
                    continue
                axial = abs(float(np.dot(displacement, bond_vector / bond_length)))
                cosine_beta = float(np.dot(hx, hc)) / denominator
                beta = math.degrees(math.acos(max(-1.0, min(1.0, cosine_beta))))
                approach = math.degrees(
                    math.acos(max(-1.0, min(1.0, axial / radius)))
                )
                exponent = (
                    -0.5
                    * ((radius - parameters.r0_angstrom) / parameters.sigma_r_angstrom) ** 2
                    - (axial / parameters.axial_c_angstrom) ** 4
                    - 0.5
                    * ((180.0 - beta) / parameters.sigma_beta_degrees) ** 2
                )
                score = math.exp(exponent) if exponent > -745.0 else 0.0
                possible = 1.7 < radius < 3.3 and axial < 0.9 and beta > 120.0
                favorable = 2.0 < radius < 2.9 and axial < 0.5 and beta > 150.0
                contact: dict[str, object] = {
                    "donor": donor + 1,
                    "hydrogen": hydrogen + 1,
                    "donor_type": donor_type,
                    "double_bond_atoms": [index + 1 for index in site.atoms],
                    "double_bond_elements": list(site.elements),
                    "center_angstrom": center.tolist(),
                    "r_hcenter_angstrom": radius,
                    "axial_offset_angstrom": axial,
                    "beta_deg": beta,
                    "approach_deg": approach,
                    "score": score,
                    "possible": possible,
                    "favorable": favorable,
                }
                if possible:
                    active_contacts.append(contact)
                if score > best_score:
                    best = contact
                    best_score = score
            if best is None:
                continue
            total += parameters.weight * best_score * best_score
            possible_count += int(bool(best["possible"]))
            favorable_count += int(bool(best["favorable"]))
            best_contacts.append(best)

        details: dict[str, object] = {
            "double_bonds": [
                {
                    "atoms": [index + 1 for index in site.atoms],
                    "elements": list(site.elements),
                }
                for site in self.sites
            ],
            "best_contacts": best_contacts,
            "active_contacts": active_contacts,
            "possible_contacts": possible_count,
            "favorable_contacts": favorable_count,
        }
        return total, possible_count, favorable_count, details


@dataclass(frozen=True)
class HydrogenPiRing:
    name: str
    atoms: tuple[int, ...]
    source: str = "auto"


def _canonical_cycle(cycle: tuple[int, ...]) -> tuple[int, ...]:
    rotations: list[tuple[int, ...]] = []
    for oriented in (cycle, tuple(reversed(cycle))):
        rotations.extend(
            oriented[offset:] + oriented[:offset] for offset in range(len(oriented))
        )
    return min(rotations)


def _is_chordless(cycle: tuple[int, ...], graph: BondGraph) -> bool:
    size = len(cycle)
    for left_index, left in enumerate(cycle):
        for right_index in range(left_index + 1, size):
            right = cycle[right_index]
            adjacent = right_index == left_index + 1 or (
                left_index == 0 and right_index == size - 1
            )
            if not adjacent and right in graph[left]:
                return False
    return True


def chordless_cycles(
    graph: BondGraph, minimum_size: int = 5, maximum_size: int = 7
) -> tuple[tuple[int, ...], ...]:
    """Enumerate unique chordless cycles without relying on an arbitrary SSSR."""

    found: set[tuple[int, ...]] = set()
    for start in range(len(graph)):
        stack: list[tuple[int, tuple[int, ...]]] = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for neighbour in sorted(graph[node], reverse=True):
                if neighbour == start:
                    if minimum_size <= len(path) <= maximum_size:
                        cycle = _canonical_cycle(path)
                        if _is_chordless(cycle, graph):
                            found.add(cycle)
                    continue
                if len(path) >= maximum_size or neighbour <= start or neighbour in path:
                    continue
                stack.append((neighbour, (*path, neighbour)))
    return tuple(sorted(found, key=lambda item: (len(item), item)))


def _cycle_from_atom_set(atoms: tuple[int, ...], graph: BondGraph) -> tuple[int, ...]:
    if not 5 <= len(atoms) <= 7:
        raise ValueError("HPI rings must contain between 5 and 7 atoms")
    if len(set(atoms)) != len(atoms):
        raise ValueError("HPI ring atoms must be distinct")
    if any(index < 0 or index >= len(graph) for index in atoms):
        raise ValueError("HPI ring contains an atom index outside input.xyz")
    selected = set(atoms)
    local = {
        atom: tuple(sorted(neighbour for neighbour in graph[atom] if neighbour in selected))
        for atom in selected
    }
    if any(len(neighbours) != 2 for neighbours in local.values()):
        raise ValueError("HPI ring atoms must induce one simple chordless cycle")
    start = min(selected)
    previous: int | None = None
    current = start
    ordered = [start]
    while True:
        choices = [item for item in local[current] if item != previous]
        if previous is None:
            next_atom = min(choices)
        else:
            next_atom = choices[0]
        if next_atom == start:
            break
        if next_atom in ordered:
            raise ValueError("HPI ring atoms do not form one simple cycle")
        ordered.append(next_atom)
        previous, current = current, next_atom
    if len(ordered) != len(selected):
        raise ValueError("HPI ring atoms do not form one connected cycle")
    return _canonical_cycle(tuple(ordered))


def _ring_plane(
    molecule: Molecule, atoms: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    coordinates = np.asarray([molecule.atoms[index].position for index in atoms], dtype=float)
    centroid = coordinates.mean(axis=0)
    centered = coordinates - centroid
    covariance = centered.T @ centered / len(atoms)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[1]) <= 1.0e-10:
        return None
    normal = eigenvectors[:, 0]
    distances = np.abs(centered @ normal)
    rms = float(np.sqrt(np.mean(distances * distances)))
    maximum = float(np.max(distances))
    return centroid, normal, rms, maximum


def _is_automatic_pi_ring(
    molecule: Molecule, graph: BondGraph, cycle: tuple[int, ...]
) -> bool:
    elements = [atom.element.upper() for atom in molecule.atoms]
    if any(elements[index] not in PI_RING_ELEMENTS for index in cycle):
        return False
    for index in cycle:
        heavy_neighbours = sum(elements[item] != "H" for item in graph[index])
        hydrogen_neighbours = sum(elements[item] == "H" for item in graph[index])
        if heavy_neighbours > 3 or hydrogen_neighbours > 1:
            return False
    plane = _ring_plane(molecule, cycle)
    return bool(plane is not None and plane[2] <= 0.10 and plane[3] <= 0.20)


@dataclass(frozen=True)
class HydrogenPiModel:
    """Continuous geometric fitness for N/O/S-H...pi contacts."""

    donors: tuple[tuple[int, int, str], ...]
    rings: tuple[HydrogenPiRing, ...]
    eligible_rings: dict[int, tuple[int, ...]]
    parameters: dict[str, HydrogenPiParameters]
    reference_graph: BondGraph

    def reference_metadata(self) -> dict[str, object]:
        return {
            "rings": [
                {
                    "name": ring.name,
                    "atoms": [index + 1 for index in ring.atoms],
                    "source": ring.source,
                }
                for ring in self.rings
            ],
            "donors": [
                {
                    "donor": donor + 1,
                    "hydrogen": hydrogen + 1,
                    "donor_type": donor_type,
                    "eligible_rings": [
                        self.rings[index].name
                        for index in self.eligible_rings.get(hydrogen, ())
                    ],
                }
                for donor, hydrogen, donor_type in self.donors
            ],
            "parameters": {
                donor_type: {
                    "z0_angstrom": values.z0_angstrom,
                    "sigma_z_angstrom": values.sigma_z_angstrom,
                    "rho_c_angstrom": values.rho_c_angstrom,
                    "sigma_beta_degrees": values.sigma_beta_degrees,
                    "weight": values.weight,
                }
                for donor_type, values in self.parameters.items()
            },
        }

    @classmethod
    def from_reference(
        cls,
        molecule: Molecule,
        graph: BondGraph,
        config: HydrogenPiConfig | None = None,
    ) -> "HydrogenPiModel":
        settings = config or HydrogenPiConfig()
        settings.validate()
        elements = [atom.element.upper() for atom in molecule.atoms]
        donors = tuple(
            (donor, hydrogen, f"{elements[donor]}H")
            for donor, element in enumerate(elements)
            if element in DONOR_ACCEPTOR_ELEMENTS
            for hydrogen in sorted(graph[donor])
            if elements[hydrogen] == "H"
        )

        automatic_cycles = tuple(
            cycle
            for cycle in chordless_cycles(graph)
            if _is_automatic_pi_ring(molecule, graph, cycle)
        )
        automatic = {
            frozenset(cycle): HydrogenPiRing(
                "AUTO_" + "_".join(str(index + 1) for index in cycle),
                cycle,
                "auto",
            )
            for cycle in automatic_cycles
        }

        explicit: dict[frozenset[int], HydrogenPiRing] = {}
        for spec in settings.included_rings:
            cycle = _cycle_from_atom_set(spec.atoms, graph)
            plane = _ring_plane(molecule, cycle)
            if plane is None:
                raise ValueError(f"HPI_RING_{spec.name} has a degenerate mean plane")
            if plane[2] > 0.10 or plane[3] > 0.20:
                warnings.warn(
                    f"HPI_RING_{spec.name} is forced active despite reference planarity "
                    f"RMS={plane[2]:.3f} Å max={plane[3]:.3f} Å",
                    UserWarning,
                    stacklevel=2,
                )
            explicit[frozenset(cycle)] = HydrogenPiRing(spec.name, cycle, "explicit")

        if settings.mode == "explicit":
            selected = explicit
        else:
            selected = dict(automatic)
            for spec in settings.excluded_rings:
                cycle = _cycle_from_atom_set(spec.atoms, graph)
                key = frozenset(cycle)
                if key not in automatic:
                    raise ValueError(
                        f"HPI_EXCLUDE_{spec.name} does not match an automatically detected pi ring"
                    )
                selected.pop(key, None)
            selected.update(explicit)

        rings = tuple(
            sorted(selected.values(), key=lambda ring: (len(ring.atoms), ring.atoms, ring.name))
        )
        eligible: dict[int, tuple[int, ...]] = {}
        for donor, hydrogen, _kind in donors:
            eligible[hydrogen] = tuple(
                ring_index
                for ring_index, ring in enumerate(rings)
                if not any(
                    graph_distance_leq(graph, donor, atom, max_hops=2)
                    for atom in ring.atoms
                )
            )
        return cls(
            donors,
            rings,
            eligible,
            {"OH": settings.oh, "NH": settings.nh, "SH": settings.sh},
            graph,
        )

    def evaluate(self, molecule: Molecule) -> tuple[float, int, int, dict[str, object]]:
        total = 0.0
        possible_count = 0
        favorable_count = 0
        best_contacts: list[dict[str, object]] = []
        active_contacts: list[dict[str, object]] = []

        planes = [_ring_plane(molecule, ring.atoms) for ring in self.rings]
        for donor, hydrogen, donor_type in self.donors:
            parameters = self.parameters[donor_type]
            h_position = np.asarray(molecule.atoms[hydrogen].position, dtype=float)
            x_position = np.asarray(molecule.atoms[donor].position, dtype=float)
            best: dict[str, object] | None = None
            for ring_index in self.eligible_rings.get(hydrogen, ()):
                plane = planes[ring_index]
                if plane is None:
                    continue
                centroid, normal, _rms, _maximum = plane
                displacement = h_position - centroid
                radius = float(np.linalg.norm(displacement))
                hx = x_position - h_position
                hc = centroid - h_position
                denominator = float(np.linalg.norm(hx) * np.linalg.norm(hc))
                if radius <= 1.0e-14 or denominator <= 1.0e-14:
                    continue
                projection = float(np.dot(displacement, normal))
                z = abs(projection)
                rho = float(np.linalg.norm(displacement - projection * normal))
                cosine_beta = float(np.dot(hx, hc)) / denominator
                beta = math.degrees(math.acos(max(-1.0, min(1.0, cosine_beta))))
                theta = math.degrees(
                    math.acos(max(-1.0, min(1.0, z / radius)))
                )
                exponent = (
                    -0.5
                    * ((z - parameters.z0_angstrom) / parameters.sigma_z_angstrom) ** 2
                    - (rho / parameters.rho_c_angstrom) ** 4
                    - 0.5
                    * ((180.0 - beta) / parameters.sigma_beta_degrees) ** 2
                )
                score = math.exp(exponent) if exponent > -745.0 else 0.0
                possible = 1.7 < z < 3.3 and rho < 2.0 and beta > 120.0
                favorable = 2.0 < z < 2.8 and rho < 1.5 and beta > 150.0
                ring = self.rings[ring_index]
                contact: dict[str, object] = {
                    "donor": donor + 1,
                    "hydrogen": hydrogen + 1,
                    "donor_type": donor_type,
                    "ring": ring.name,
                    "ring_atoms": [index + 1 for index in ring.atoms],
                    "ring_source": ring.source,
                    "r_hpi_angstrom": radius,
                    "z_angstrom": z,
                    "rho_angstrom": rho,
                    "beta_deg": beta,
                    "theta_deg": theta,
                    "score": score,
                    "possible": possible,
                    "favorable": favorable,
                }
                if possible:
                    active_contacts.append(contact)
                if best is None or score > float(best["score"]):
                    best = contact
            if best is None:
                continue
            score = float(best["score"])
            total += parameters.weight * score * score
            possible_count += int(bool(best["possible"]))
            favorable_count += int(bool(best["favorable"]))
            best_contacts.append(best)

        details: dict[str, object] = {
            "rings": [
                {
                    "name": ring.name,
                    "atoms": [index + 1 for index in ring.atoms],
                    "source": ring.source,
                }
                for ring in self.rings
            ],
            "best_contacts": best_contacts,
            "active_contacts": active_contacts,
            "possible_contacts": possible_count,
            "favorable_contacts": favorable_count,
        }
        return total, possible_count, favorable_count, details


@dataclass(frozen=True)
class HydrogenBondModel:
    donors: tuple[int, ...]
    donor_hydrogens: dict[int, tuple[int, ...]]
    acceptors: tuple[int, ...]
    reference_graph: BondGraph
    cutoff_angstrom: float = 3.2
    contact_threshold: float = -0.30
    hh_clash_distance_angstrom: float = 1.40

    @classmethod
    def from_reference(
        cls,
        molecule: Molecule,
        graph: BondGraph,
        cutoff_angstrom: float = 3.2,
        contact_threshold: float = -0.30,
        hh_clash_distance_angstrom: float = 1.40,
    ) -> "HydrogenBondModel":
        elements = [atom.element.upper() for atom in molecule.atoms]
        donors: list[int] = []
        donor_hydrogens: dict[int, tuple[int, ...]] = {}
        for index, element in enumerate(elements):
            if element not in DONOR_ACCEPTOR_ELEMENTS:
                continue
            hydrogens = tuple(neighbour for neighbour in graph[index] if elements[neighbour] == "H")
            if hydrogens:
                donors.append(index)
                donor_hydrogens[index] = hydrogens
        acceptors = tuple(
            index for index, element in enumerate(elements) if element in DONOR_ACCEPTOR_ELEMENTS
        )
        return cls(
            tuple(donors),
            donor_hydrogens,
            acceptors,
            graph,
            cutoff_angstrom,
            contact_threshold,
            hh_clash_distance_angstrom,
        )

    def evaluate(self, molecule: Molecule) -> tuple[float, int, dict[str, object]]:
        score = 0.0
        contacts: list[dict[str, float | int]] = []
        atoms = molecule.atoms
        elements = [atom.element.upper() for atom in atoms]

        for donor in self.donors:
            donor_element = elements[donor]
            for hydrogen in self.donor_hydrogens.get(donor, ()):
                for acceptor in self.acceptors:
                    if acceptor == donor:
                        continue
                    if graph_distance_leq(self.reference_graph, donor, acceptor, max_hops=2):
                        continue
                    h_a = distance(atoms[hydrogen].position, atoms[acceptor].position)
                    if h_a > self.cutoff_angstrom:
                        continue
                    acceptor_element = elements[acceptor]
                    r_d = VDW_RADII.get(donor_element, 1.70)
                    r_a = VDW_RADII.get(acceptor_element, 1.70)
                    equilibrium = 0.60 * (r_d + r_a)
                    epsilon = PAIR_EPSILON.get((donor_element, acceptor_element), 1.0)
                    raw = hbond_pair_energy(h_a, equilibrium, alpha=15.0, epsilon=epsilon)
                    dha_angle = angle_deg(
                        atoms[donor].position,
                        atoms[hydrogen].position,
                        atoms[acceptor].position,
                    )
                    orientation = max(0.0, min(1.0, (dha_angle - 90.0) / 90.0)) ** 2
                    contribution = raw if raw >= 0.0 else raw * orientation
                    score += contribution
                    contacts.append(
                        {
                            "donor": donor + 1,
                            "hydrogen": hydrogen + 1,
                            "acceptor": acceptor + 1,
                            "distance_angstrom": h_a,
                            "angle_deg": dha_angle,
                            "score": contribution,
                        }
                    )

        clashes: list[dict[str, float | int]] = []
        all_hydrogens = [index for index, element in enumerate(elements) if element == "H"]
        for offset, first in enumerate(all_hydrogens):
            for second in all_hydrogens[offset + 1 :]:
                if graph_distance_leq(self.reference_graph, first, second, max_hops=4):
                    continue
                h_h = distance(atoms[first].position, atoms[second].position)
                if h_h < self.hh_clash_distance_angstrom:
                    score += 1000.0
                    clashes.append({"first": first + 1, "second": second + 1, "distance_angstrom": h_h})

        active_contacts = [
            contact
            for contact in contacts
            if float(contact["score"]) <= self.contact_threshold
            and float(contact["angle_deg"]) >= 120.0
        ]
        details: dict[str, object] = {"contacts": contacts, "active_contacts": active_contacts, "clashes": clashes}
        return score, len(active_contacts), details

@dataclass(frozen=True)
class DisconnectedComponentsPenaltyModel:
    """Penalize disconnected components in the complete interaction graph.

    The graph contains every atom. Reference covalent bonds are permanent
    edges, while active hydrogen bonds and H...pi contacts add configuration-
    dependent edges.  A connected system has zero penalty; otherwise the
    penalty is the minimum number of additional edges needed to connect it,
    namely ``number_of_components - 1``.
    """

    reference_graph: BondGraph
    covalent_fragments: tuple[tuple[int, ...], ...]
    atom_to_fragment: tuple[int, ...]

    @classmethod
    def from_reference(
        cls,
        molecule: Molecule,
        graph: BondGraph,
    ) -> "DisconnectedComponentsPenaltyModel":
        if len(graph) != len(molecule.atoms):
            raise ValueError("interaction graph size does not match input.xyz")
        fragments = cls._connected_components(graph)
        if len(fragments) < 2:
            raise ValueError(
                "disconnected_components_penalty requires an intermolecular input with "
                "at least two covalent fragments"
            )
        atom_to_fragment = [-1] * len(graph)
        for fragment_index, fragment in enumerate(fragments):
            for atom in fragment:
                atom_to_fragment[atom] = fragment_index
        return cls(graph, fragments, tuple(atom_to_fragment))

    @staticmethod
    def _connected_components(graph: BondGraph) -> tuple[tuple[int, ...], ...]:
        unseen = set(range(len(graph)))
        components: list[tuple[int, ...]] = []
        while unseen:
            start = min(unseen)
            stack = [start]
            component: set[int] = set()
            while stack:
                atom = stack.pop()
                if atom in component:
                    continue
                component.add(atom)
                stack.extend(graph[atom] - component)
            unseen.difference_update(component)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def reference_metadata(self) -> dict[str, object]:
        primary = max(
            range(len(self.covalent_fragments)),
            key=lambda index: (len(self.covalent_fragments[index]), -index),
        )
        return {
            "covalent_fragments": [
                {"index": index + 1, "atoms": [atom + 1 for atom in fragment]}
                for index, fragment in enumerate(self.covalent_fragments)
            ],
            "primary_fragment": primary + 1,
            "penalty_definition": "connected_components_minus_one",
            "interaction_edges": [
                "reference_covalent_bond",
                "standard_active_hbond",
                "geometrically_possible_hbond_pi",
            ],
        }

    def evaluate_interactions(
        self,
        active_hbond_contacts: object,
        hbond_pi_details: object,
    ) -> tuple[float, int, int, dict[str, object]]:
        adjacency = [set(neighbours) for neighbours in self.reference_graph]
        hbond_edges: set[tuple[int, int]] = set()
        hbond_contacts = (
            active_hbond_contacts if isinstance(active_hbond_contacts, list) else []
        )
        for raw_contact in hbond_contacts:
            if not isinstance(raw_contact, dict):
                continue
            hydrogen = int(raw_contact.get("hydrogen", 0)) - 1
            acceptor = int(raw_contact.get("acceptor", 0)) - 1
            if not (0 <= hydrogen < len(adjacency) and 0 <= acceptor < len(adjacency)):
                continue
            edge = tuple(sorted((hydrogen, acceptor)))
            hbond_edges.add(edge)
            adjacency[hydrogen].add(acceptor)
            adjacency[acceptor].add(hydrogen)

        pi_edges: set[tuple[int, int]] = set()
        active_pi_contacts: list[dict[str, object]] = []
        pi_contacts: object = []
        if isinstance(hbond_pi_details, dict):
            pi_contacts = hbond_pi_details.get("active_contacts", [])
        for raw_contact in pi_contacts if isinstance(pi_contacts, list) else []:
            if not isinstance(raw_contact, dict) or not bool(raw_contact.get("possible")):
                continue
            hydrogen = int(raw_contact.get("hydrogen", 0)) - 1
            raw_ring = raw_contact.get("ring_atoms", [])
            ring_atoms = raw_ring if isinstance(raw_ring, list) else []
            if not 0 <= hydrogen < len(adjacency):
                continue
            valid_ring_atoms: list[int] = []
            for raw_atom in ring_atoms:
                ring_atom = int(raw_atom) - 1
                if not 0 <= ring_atom < len(adjacency):
                    continue
                valid_ring_atoms.append(ring_atom)
                edge = tuple(sorted((hydrogen, ring_atom)))
                pi_edges.add(edge)
                adjacency[hydrogen].add(ring_atom)
                adjacency[ring_atom].add(hydrogen)
            if valid_ring_atoms:
                active_pi_contacts.append(
                    {
                        "hydrogen": hydrogen + 1,
                        "ring": raw_contact.get("ring", ""),
                        "ring_atoms": [atom + 1 for atom in valid_ring_atoms],
                        "interaction_site": "ring_center",
                    }
                )

        components = self._connected_components(
            tuple(frozenset(items) for items in adjacency)
        )
        penalty = float(max(0, len(components) - 1))
        primary_fragment = max(
            range(len(self.covalent_fragments)),
            key=lambda index: (len(self.covalent_fragments[index]), -index),
        )
        primary_atom = self.covalent_fragments[primary_fragment][0]
        primary_component = next(
            index for index, component in enumerate(components) if primary_atom in component
        )
        component_fragments = [
            sorted({self.atom_to_fragment[atom] for atom in component})
            for component in components
        ]
        detached_fragments = sum(
            len(fragments)
            for index, fragments in enumerate(component_fragments)
            if index != primary_component
        )
        details: dict[str, object] = {
            "penalty": penalty,
            "connected": len(components) == 1,
            "component_count": len(components),
            "missing_links": int(penalty),
            "components": [
                {
                    "index": index + 1,
                    "atoms": [atom + 1 for atom in component],
                    "covalent_fragments": [item + 1 for item in component_fragments[index]],
                    "contains_primary_fragment": index == primary_component,
                }
                for index, component in enumerate(components)
            ],
            "detached_fragment_count": detached_fragments,
            "active_hbond_edges": [
                [left + 1, right + 1] for left, right in sorted(hbond_edges)
            ],
            "active_hbond_pi_contacts": active_pi_contacts,
            "active_hbond_pi_atomic_edges": len(pi_edges),
        }
        return penalty, len(components), detached_fragments, details


def hbond_pair_energy(
    distance_angstrom: float,
    equilibrium_angstrom: float,
    alpha: float = 15.0,
    epsilon: float = 1.0,
) -> float:
    if equilibrium_angstrom <= 1.0e-12:
        return 0.0
    ratio = distance_angstrom / equilibrium_angstrom
    first_exp = math.exp(min(700.0, alpha * (1.0 - ratio)))
    second_exp = math.exp(min(700.0, 0.5 * alpha * (1.0 - ratio)))
    return epsilon * (first_exp - (ratio * ratio - 2.0 * ratio + 3.0) * second_exp)
